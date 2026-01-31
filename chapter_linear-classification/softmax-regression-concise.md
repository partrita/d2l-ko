```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 회귀의 간결한 구현 (Concise Implementation of Softmax Regression)
:label:`sec_softmax_concise`



고수준 딥러닝 프레임워크가 선형 회귀 구현을 쉽게 만들어주었듯이(:numref:`sec_linear_concise` 참조), 여기에서도 마찬가지로 편리합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
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

## 모델 정의하기 (Defining the Model)

:numref:`sec_linear_concise`에서처럼, 내장 레이어를 사용하여 완전 연결 레이어를 구성합니다. 내장된 `__call__` 메서드는 네트워크를 입력에 적용해야 할 때마다 `forward`를 호출합니다.

:begin_tab:`mxnet`
입력 `X`가 4차 텐서이더라도, 내장된 `Dense` 레이어는 첫 번째 축을 따른 차원을 유지하면서 자동으로 `X`를 2차 텐서로 변환합니다.
:end_tab:

:begin_tab:`pytorch`
우리는 `Flatten` 레이어를 사용하여 첫 번째 축을 따른 차원을 유지하면서 4차 텐서 `X`를 2차 텐서로 변환합니다.

:end_tab:

:begin_tab:`tensorflow`
우리는 `Flatten` 레이어를 사용하여 첫 번째 축을 따른 차원을 유지하면서 4차 텐서 `X`를 변환합니다.
:end_tab:

:begin_tab:`jax`
Flax는 `@nn.compact` 데코레이터를 사용하여 네트워크 클래스를 더 간결하게 작성할 수 있게 해줍니다. `@nn.compact`를 사용하면 데이터 클래스에 표준 `setup` 메서드를 정의할 필요 없이, 단일 "순방향 패스(forward pass)" 메서드 내에 모든 네트워크 로직을 작성할 수 있습니다.
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """소프트맥스 회귀 모델입니다."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """소프트맥스 회귀 모델입니다."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## 소프트맥스 다시 보기 (Softmax Revisited)
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch`에서 우리는 모델의 출력을 계산하고 크로스 엔트로피 손실을 적용했습니다. 이는 수학적으로는 완전히 합리적이지만, 지수 계산에서의 수치적 언더플로와 오버플로 때문에 계산적으로는 위험합니다.

소프트맥스 함수가 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$를 통해 확률을 계산한다는 점을 상기하십시오. 만약 일부 $o_k$가 매우 크면(즉, 매우 큰 양수이면), $\exp(o_k)$는 특정 데이터 유형이 가질 수 있는 가장 큰 숫자보다 커질 수 있습니다. 이를 *오버플로(overflow)*라고 합니다. 마찬가지로 모든 인수가 매우 큰 음수이면 *언더플로(underflow)*가 발생합니다. 예를 들어 단정밀도 부동 소수점 숫자는 대략 $10^{-38}$에서 $10^{38}$의 범위를 커버합니다. 따라서 $\mathbf{o}$의 가장 큰 항이 구간 $[-90, 90]$을 벗어나면 결과가 안정적이지 않습니다. 이 문제를 해결하는 방법은 모든 항목에서 $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$를 빼는 것입니다:

$$ 
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} = 
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} = 
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$ 

구조상 모든 $j$에 대해 $o_j - \bar{o} \leq 0$임을 압니다. 따라서 $q$-클래스 분류 문제의 경우 분모는 구간 $[1, q]$에 포함됩니다. 게다가 분자는 1을 넘지 않으므로 수치적 오버플로를 방지합니다. 수치적 언더플로는 $\exp(o_j - \bar{o})$가 수치적으로 0으로 평가될 때만 발생합니다. 그럼에도 불구하고 나중에 $\log \hat{y}_j$를 $\log 0$으로 계산하고 싶을 때 문제가 발생할 수 있습니다. 특히 역전파에서 무시무시한 `NaN` (Not a Number) 결과로 가득 찬 화면을 보게 될 수도 있습니다.

다행히도 지수 함수를 계산하고 있더라도 궁극적으로는 그들의 로그를 취하려 한다는 사실(크로스 엔트로피 손실을 계산할 때) 덕분에 구원받을 수 있습니다. 소프트맥스와 크로스 엔트로피를 결합함으로써 수치 안정성 문제를 완전히 벗어날 수 있습니다. 다음과 같습니다:

$$ 
\log \hat{y}_j = 
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} = 
 o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$ 

이는 오버플로와 언더플로를 모두 방지합니다. 모델에 의한 출력 확률을 평가하고 싶을 때를 대비해 기존의 소프트맥스 함수를 계속 유용하게 사용할 것입니다. 하지만 새 손실 함수에 소프트맥스 확률을 전달하는 대신, [**"LogSumExp 트릭"(https://en.wikipedia.org/wiki/LogSumExp)과 같은 영리한 일을 수행하는 크로스 엔트로피 손실 함수 내부에서 로짓(logits)을 전달하고 소프트맥스와 그 로그를 한 번에 계산합니다.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # 나중에 사용될 예정(예: 배치 정규화)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # 반환된 빈 딕셔너리는 나중에 사용될(예: 배치 정규화) 보조 데이터를 위한 플레이스홀더입니다.
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## 훈련 (Training)

다음으로 모델을 훈련합니다. 784차원 특성 벡터로 평탄화된 Fashion-MNIST 이미지를 사용합니다.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

이전처럼 이 알고리즘은 합리적으로 정확한 솔루션으로 수렴하지만, 이번에는 이전보다 훨씬 적은 줄의 코드를 사용합니다.


## 요약 (Summary)

고수준 API는 수치 안정성과 같이 사용자에게 잠재적으로 위험한 측면을 숨기는 데 매우 편리합니다. 게다가 사용자가 아주 적은 줄의 코드로 모델을 간결하게 설계할 수 있게 해줍니다. 이는 축복이자 저주입니다. 분명한 이점은 통계학 수업을 한 번도 들어본 적 없는 엔지니어들에게도(실제로 그들이 이 책의 대상 독자 중 일부입니다) 접근성을 매우 높여준다는 것입니다. 하지만 날카로운 모서리를 숨기는 데는 대가가 따릅니다: 스스로 새로운 그리고 다른 구성 요소를 추가하려는 의욕을 꺾게 되는데, 이를 수행하기 위한 근육 기억이 거의 없기 때문입니다. 게다가 프레임워크의 보호 패딩이 모든 코너 케이스를 완벽하게 커버하지 못할 때마다 무언가를 *고치는* 것을 더 어렵게 만듭니다. 이 역시 익숙함의 부족 때문입니다.

따라서 우리는 여러분이 이어지는 많은 구현의 가감 없는 버전과 우아한 버전 *둘 다*를 검토할 것을 강력히 권장합니다. 우리는 이해의 용이성을 강조하지만, 그럼에도 불구하고 구현은 대개 상당히 우수한 성능을 보여줍니다(합성곱은 여기서 큰 예외입니다). 어떤 프레임워크도 제공할 수 없는 새로운 것을 발명할 때 여러분이 이러한 토대 위에 구축할 수 있도록 하는 것이 우리의 의도입니다.


## 연습 문제 (Exercises)

1. 딥러닝은 FP64 배정밀도(매우 드물게 사용됨), FP32 단정밀도, BFLOAT16(압축된 표현에 좋음), FP16(매우 불안정함), TF32(NVIDIA의 새로운 형식), INT8을 포함한 많은 다양한 숫자 형식을 사용합니다. 결과가 수치적 언더플로나 오버플로로 이어지지 않는 지수 함수의 가장 작은 인수와 가장 큰 인수를 계산하십시오.
2. INT8은 1에서 255 사이의 0이 아닌 숫자로 구성된 매우 제한된 형식입니다. 더 많은 비트를 사용하지 않고 어떻게 동적 범위를 확장할 수 있을까요? 표준 곱셈과 덧셈이 여전히 작동합니까?
3. 훈련 에폭 수를 늘리십시오. 왜 잠시 후에 검증 정확도가 떨어질 수 있을까요? 이를 어떻게 고칠 수 있을까요?
4. 학습률을 높이면 어떻게 됩니까? 여러 학습률에 대한 손실 곡선을 비교하십시오. 어떤 것이 더 잘 작동합니까? 언제 그렇습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17983)
:end_tab:

```