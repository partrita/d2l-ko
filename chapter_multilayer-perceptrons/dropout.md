```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 드롭아웃 (Dropout)
:label:`sec_dropout`


우리가 좋은 예측 모델에서 기대하는 것이 무엇인지 잠시 생각해 봅시다. 
우리는 보지 못한 데이터에서 잘 수행되기를 원합니다. 
고전적인 일반화 이론은 훈련 성능과 테스트 성능 사이의 격차를 줄이기 위해 단순한 모델을 목표로 해야 한다고 제안합니다. 
단순함은 차원의 수가 적은 형태로 올 수 있습니다. 
우리는 :numref:`sec_generalization_basics`에서 선형 모델의 단항 기저 함수를 논의할 때 이것을 탐구했습니다. 
또한 :numref:`sec_weight_decay`에서 가중치 감쇠($\ell_2$ 정규화)를 논의할 때 보았듯이 파라미터의 (역) 노름도 단순함의 유용한 척도를 나타냅니다. 
단순함의 또 다른 유용한 개념은 부드러움(smoothness), 즉 함수가 입력의 작은 변화에 민감하지 않아야 한다는 것입니다. 
예를 들어 이미지를 분류할 때 픽셀에 약간의 무작위 노이즈를 추가해도 대부분 무해할 것으로 예상합니다.

:citet:`Bishop.1995`는 입력 노이즈를 사용한 훈련이 티호노프(Tikhonov) 정규화와 동일하다는 것을 증명하면서 이 아이디어를 공식화했습니다. 
이 작업은 함수가 부드러워야 한다(따라서 단순해야 한다)는 요구 사항과 입력의 섭동(perturbation)에 탄력적이어야 한다는 요구 사항 사이에 명확한 수학적 연결을 도출했습니다.

그 후, :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`는 Bishop의 아이디어를 네트워크의 내부 레이어에도 적용하는 영리한 아이디어를 개발했습니다. 
*드롭아웃(dropout)*이라고 불리는 그들의 아이디어는 순전파 동안 각 내부 레이어를 계산하는 동안 노이즈를 주입하는 것을 포함하며, 신경망 훈련을 위한 표준 기술이 되었습니다. 
이 방법은 훈련 중에 문자 그대로 일부 뉴런을 *떨어뜨리기(drop out)* 때문에 *드롭아웃*이라고 불립니다. 
훈련 전반에 걸쳐 각 반복에서 표준 드롭아웃은 다음 레이어를 계산하기 전에 각 레이어의 노드 중 일부를 0으로 만드는 것으로 구성됩니다.

분명히 하자면, 우리는 Bishop과의 연결을 통해 우리 자신의 내러티브를 강요하고 있습니다. 
드롭아웃에 대한 원본 논문은 유성 생식에 대한 놀라운 비유를 통해 직관을 제공합니다. 
저자들은 신경망 과대적합이 각 레이어가 이전 레이어의 특정 활성화 패턴에 의존하는 상태로 특징지어지며, 이 상태를 *공동 적응(co-adaptation)*이라고 부릅니다. 
그들은 유성 생식이 공동 적응 유전자를 분해한다고 주장되는 것처럼 드롭아웃이 공동 적응을 분해한다고 주장합니다. 
이 이론에 대한 정당성은 확실히 논쟁의 여지가 있지만, 드롭아웃 기술 자체는 지속성이 있음이 입증되었으며 대부분의 딥러닝 라이브러리에 다양한 형태의 드롭아웃이 구현되어 있습니다. 


핵심 과제는 이 노이즈를 어떻게 주입하느냐입니다. 
한 가지 아이디어는 *편향되지 않은(unbiased)* 방식으로 주입하여 각 레이어의 기댓값(다른 레이어는 고정된 상태에서)이 노이즈가 없을 때 취했을 값과 같아지도록 하는 것입니다. 
Bishop의 연구에서 그는 선형 모델의 입력에 가우스 노이즈를 추가했습니다. 
각 훈련 반복에서 그는 평균이 0인 분포 $\epsilon \sim \mathcal{N}(0,\sigma^2)$에서 샘플링된 노이즈를 입력 $\mathbf{x}$에 추가하여 섭동된 점 $\mathbf{x}' = \mathbf{x} + \epsilon$을 산출했습니다. 
기댓값에서 $E[\mathbf{x}'] = \mathbf{x}$입니다.

표준 드롭아웃 정규화에서는 각 레이어의 노드 중 일부를 0으로 만든 다음 유지된(떨어뜨리지 않은) 노드의 비율로 정규화하여 각 레이어의 편향을 *제거(debiases)*합니다. 
즉, *드롭아웃 확률* $p$를 사용하여 각 중간 활성화 $h$는 다음과 같이 확률 변수 $h'$로 대체됩니다.

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ 확률 } p \\
    \frac{h}{1-p} & \textrm{ 그 외}
\end{cases}
\end{aligned}
$$

설계상 기댓값은 변경되지 않습니다. 즉, $E[h'] = h$.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## 실제 드롭아웃 (Dropout in Practice)

:numref:`fig_mlp`의 은닉층과 5개의 은닉 유닛이 있는 MLP를 상기해 보십시오. 
은닉층에 드롭아웃을 적용하여 각 은닉 유닛을 확률 $p$로 0으로 만들면, 그 결과는 원래 뉴런의 부분 집합만 포함하는 네트워크로 볼 수 있습니다. 
:numref:`fig_dropout2`에서 $h_2$와 $h_5$가 제거되었습니다. 
결과적으로 출력의 계산은 더 이상 $h_2$나 $h_5$에 의존하지 않으며 역전파를 수행할 때 각각의 기울기도 사라집니다. 
이런 식으로 출력 레이어의 계산은 $h_1, \ldots, h_5$의 어느 한 요소에 과도하게 의존할 수 없습니다.

![드롭아웃 전후의 MLP.](../img/dropout2.svg)
:label:`fig_dropout2`

일반적으로 테스트 시에는 드롭아웃을 비활성화합니다. 
훈련된 모델과 새로운 예제가 주어지면, 우리는 어떤 노드도 떨어뜨리지 않으므로 정규화할 필요도 없습니다. 
그러나 몇 가지 예외가 있습니다: 일부 연구자들은 신경망 예측의 *불확실성*을 추정하기 위한 휴리스틱으로 테스트 시 드롭아웃을 사용합니다: 예측이 서로 다른 많은 드롭아웃 출력에 걸쳐 일치한다면 네트워크가 더 확신한다고 말할 수 있습니다.

## 밑바닥부터 구현하기 (Implementation from Scratch)

단일 레이어에 대한 드롭아웃 함수를 구현하려면 레이어가 가진 차원 수만큼 베르누이(이진) 확률 변수에서 샘플을 추출해야 합니다. 
여기서 확률 변수는 확률 $1-p$로 값 $1$(유지)을, 확률 $p$로 $0$(제거)을 취합니다. 
이를 구현하는 쉬운 방법 중 하나는 먼저 균등 분포 $U[0, 1]$에서 샘플을 추출하는 것입니다. 
그런 다음 해당 샘플이 $p$보다 큰 노드를 유지하고 나머지는 떨어뜨릴 수 있습니다.

다음 코드에서 우리는 위에서 설명한 대로 나머지를 재조정하여 (**확률 `dropout`으로 텐서 입력 `X`의 요소를 떨어뜨리는 `dropout_layer` 함수를 구현**)합니다: 생존자들을 `1.0-dropout`으로 나눕니다.

```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

우리는 [**몇 가지 예제에서 `dropout_layer` 함수를 테스트**]할 수 있습니다. 
다음 코드 줄에서는 입력 `X`를 각각 확률 0, 0.5, 1로 드롭아웃 연산에 통과시킵니다.

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### 모델 정의하기

아래 모델은 각 은닉층의 출력(활성화 함수 다음)에 드롭아웃을 적용합니다. 
각 레이어에 대해 드롭아웃 확률을 별도로 설정할 수 있습니다. 
일반적인 선택은 입력 레이어에 가까울수록 낮은 드롭아웃 확률을 설정하는 것입니다. 
우리는 훈련 중에만 드롭아웃이 활성화되도록 합니다.

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**훈련 (Training)**]

다음은 앞서 설명한 MLP의 훈련과 유사합니다.

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**간결한 구현**]

고수준 API를 사용하면 각 완전 연결 레이어 뒤에 `Dropout` 레이어를 추가하고 생성자에 유일한 인수로 드롭아웃 확률을 전달하기만 하면 됩니다. 
훈련 중에 `Dropout` 레이어는 지정된 드롭아웃 확률에 따라 이전 레이어의 출력(또는 동등하게 후속 레이어의 입력)을 무작위로 떨어뜨립니다. 
훈련 모드가 아닐 때, `Dropout` 레이어는 테스트 중에 데이터를 단순히 통과시킵니다.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
`Module.apply()`를 사용할 때 드롭아웃 레이어가 있는 네트워크는 PRNGKey가 필요하고, 이 RNG 시드의 이름은 명시적으로 `dropout`이어야 하므로 손실 함수를 재정의해야 한다는 점에 유의하십시오. 이 키는 Flax의 `dropout` 레이어에서 내부적으로 무작위 드롭아웃 마스크를 생성하는 데 사용됩니다. 훈련 루프의 모든 에폭마다 고유한 `dropout_rng` 키를 사용하는 것이 중요합니다. 그렇지 않으면 생성된 드롭아웃 마스크가 확률적이지 않고 에폭 실행 간에 다르지 않게 됩니다. 이 `dropout_rng`는 `TrainState` 객체(:numref:`oo-design-training`에 정의된 `d2l.Trainer` 클래스에 있음)에 속성으로 저장될 수 있으며 매 에폭마다 새 `dropout_rng`로 대체됩니다. 우리는 이미 :numref:`sec_linear_scratch`에 정의된 `fit_epoch` 메서드로 이를 처리했습니다.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # 나중에 사용될 예정(예: 배치 정규화)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # 반환된 빈 딕셔너리는 나중에 사용될(예: 배치 정규화) 보조 데이터를 위한 플레이스홀더입니다.
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

다음으로 [**모델을 훈련합니다**].

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## 요약 (Summary)

차원의 수와 가중치 벡터의 크기를 제어하는 것 외에도, 드롭아웃은 과대적합을 피하기 위한 또 다른 도구입니다. 종종 도구들은 함께 사용됩니다. 
드롭아웃은 훈련 중에만 사용된다는 점에 유의하십시오: 
활성화 $h$를 기댓값 $h$를 가진 확률 변수로 대체합니다.


## 연습 문제 (Exercises)

1. 첫 번째와 두 번째 레이어의 드롭아웃 확률을 변경하면 어떻게 됩니까? 특히 두 레이어의 확률을 바꾸면 어떻게 됩니까? 이 질문에 답하기 위한 실험을 설계하고, 결과를 정량적으로 설명하고, 정성적인 시사점을 요약하십시오.
2. 에폭 수를 늘리고 드롭아웃을 사용할 때와 사용하지 않을 때 얻은 결과를 비교하십시오.
3. 드롭아웃이 적용될 때와 적용되지 않을 때 각 은닉층에서 활성화의 분산은 얼마입니까? 두 모델에 대해 이 양이 시간이 지남에 따라 어떻게 진화하는지 보여주는 플롯을 그리십시오.
4. 테스트 시에 드롭아웃을 일반적으로 사용하지 않는 이유는 무엇입니까?
5. 이 섹션의 모델을 예로 들어 드롭아웃과 가중치 감쇠 사용의 효과를 비교하십시오. 드롭아웃과 가중치 감쇠를 동시에 사용하면 어떻게 됩니까? 결과가 가산적입니까? 수익이 감소합니까(혹은 더 나쁩니까)? 서로 상쇄합니까?
6. 활성화 대신 가중치 행렬의 개별 가중치에 드롭아웃을 적용하면 어떻게 됩니까?
7. 표준 드롭아웃 기술과 다른, 각 레이어에 무작위 노이즈를 주입하는 다른 기술을 발명하십시오. Fashion-MNIST 데이터셋(고정된 아키텍처에 대해)에서 드롭아웃보다 성능이 뛰어난 방법을 개발할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17987)
:end_tab: