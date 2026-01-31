```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 밑바닥부터 시작하는 소프트맥스 회귀 구현 (Softmax Regression Implementation from Scratch)
:label:`sec_softmax_scratch`

소프트맥스 회귀는 매우 기초적이기 때문에, 여러분이 이를 직접 구현하는 방법을 알아야 한다고 믿습니다. 여기서는 모델의 소프트맥스 관련 측면을 정의하는 것으로 제한하고, 훈련 루프를 포함한 다른 구성 요소들은 선형 회귀 섹션의 것들을 재사용하겠습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
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
import jax
from jax import numpy as jnp
from functools import partial
```

## 소프트맥스 (The Softmax)

가장 중요한 부분인 스칼라를 확률로 매핑하는 것부터 시작해 봅시다. 복습을 위해 :numref:`subsec_lin-alg-reduction` 및 :numref:`subsec_lin-alg-non-reduction`에서 논의한 텐서의 특정 차원을 따른 합계 연산자를 상기해 보십시오. [**행렬 `X`가 주어졌을 때 우리는 (기본적으로) 모든 요소를 합산하거나 동일한 축에 있는 요소들만 합산할 수 있습니다.**] `axis` 변수를 사용하면 행 합계와 열 합계를 계산할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

소프트맥스를 계산하려면 세 단계가 필요합니다: 
(i) 각 항의 지수화; 
(ii) 각 예제에 대한 정규화 상수를 계산하기 위해 각 행에 대한 합계; 
(iii) 각 행을 정규화 상수로 나누어 결과의 합이 1이 되도록 함:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$ 
**)

분모(의 로그)를 (로그) *분할 함수(partition function)*라고 합니다. 이는 통계 물리학에서 열역학적 앙상블의 모든 가능한 상태를 합산하기 위해 도입되었습니다. 구현은 간단합니다:

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # 여기서 브로드캐스팅 메커니즘이 적용됩니다.
```

임의의 입력 `X`에 대해 [**우리는 각 요소를 음이 아닌 숫자로 바꿉니다. 각 행의 합은 확률에 필요한 대로 1이 됩니다.**] 주의: 위 코드는 매우 크거나 작은 인수에 대해 강건하지 *않습니다*. 무슨 일이 일어나는지 설명하기에는 충분하지만, 심각한 용도로 이 코드를 그대로 사용해서는 안 됩니다. 딥러닝 프레임워크에는 이러한 보호 기능이 내장되어 있으며 앞으로는 내장된 소프트맥스를 사용할 것입니다.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## 모델 (The Model)

이제 [**소프트맥스 회귀 모델**]을 구현하는 데 필요한 모든 것을 갖추었습니다. 선형 회귀 예제에서와 같이, 각 인스턴스는 고정 길이 벡터로 표현됩니다. 원시 데이터가 $28 \times 28$ 픽셀 이미지로 구성되므로, [**각 이미지를 평탄화(flatten)하여 길이가 784인 벡터로 취급합니다.**] 나중 장에서는 공간 구조를 더 만족스러운 방식으로 활용하는 합성곱 신경망을 소개할 것입니다.


소프트맥스 회귀에서 우리 네트워크의 출력 수는 클래스의 수와 같아야 합니다. (**우리 데이터셋에는 10개의 클래스가 있으므로, 우리 네트워크는 10의 출력 차원을 갖습니다.**) 결과적으로 우리의 가중치는 $784 \times 10$ 행렬이 되고 편향을 위한 $1 \times 10$ 행 벡터가 추가됩니다. 선형 회귀와 마찬가지로 가중치 `W`를 가우스 노이즈로 초기화합니다. 편향은 0으로 초기화됩니다.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

아래 코드는 네트워크가 각 입력을 출력에 매핑하는 방식을 정의합니다. 모델을 통해 데이터를 전달하기 전에 배치의 각 $28 \times 28$ 픽셀 이미지를 `reshape`을 사용하여 벡터로 평탄화합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## 크로스 엔트로피 손실 (The Cross-Entropy Loss)

다음으로 크로스 엔트로피 손실 함수를 구현해야 합니다(:numref:`subsec_softmax-regression-loss-func`에서 소개됨). 이는 모든 딥러닝에서 가장 흔한 손실 함수일 것입니다. 현재로서는 분류 문제로 쉽게 던질 수 있는 딥러닝 응용 분야가 회귀 문제로 처리하는 것이 더 나은 분야보다 훨씬 많습니다.

크로스 엔트로피는 실제 레이블에 할당된 예측 확률의 음의 로그 우도를 취한다는 점을 상기하십시오. 효율성을 위해 Python for-루프를 피하고 대신 인덱싱을 사용합니다. 특히 $\mathbf{y}$의 원-핫 인코딩을 통해 $\hat{\mathbf{y}}$에서 일치하는 항을 선택할 수 있습니다.

작동 방식을 확인하기 위해 [**3개 클래스에 대한 예측 확률이 있는 2개의 예제와 그에 대응하는 레이블 `y`를 포함한 샘플 데이터 `y_hat`을 생성합니다.**] 올바른 레이블은 각각 0과 2입니다(즉, 첫 번째와 세 번째 클래스). [**`y`를 `y_hat`의 확률 인덱스로 사용하여**] 항을 효율적으로 뽑아낼 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
이제 선택된 확률의 로그에 대해 평균을 내어 (**크로스 엔트로피 손실 함수를 구현**)할 수 있습니다.
:end_tab:

:begin_tab:`jax`
이제 선택된 확률의 로그에 대해 평균을 내어 (**크로스 엔트로피 손실 함수를 구현**)할 수 있습니다.

JAX 구현의 속도를 높이기 위해 `jax.jit`을 활용하고 `loss`가 순수 함수(pure function)임을 보장하기 위해, `loss` 함수를 불순하게 만들 수 있는 전역 변수나 함수의 사용을 피하고자 `cross_entropy` 함수를 `loss` 내부에 다시 정의했습니다. `jax.jit`과 순수 함수에 대한 관심 있는 독자들은 [JAX 문서](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)를 참조하십시오.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # 반환된 빈 딕셔너리는 나중에 사용될(예: 배치 정규화) 보조 데이터를 위한 플레이스홀더입니다.
    return cross_entropy(y_hat, y), {}
```

## 훈련 (Training)

:numref:`sec_linear_scratch`에서 정의된 `fit` 메서드를 재사용하여 [**모델을 10 에폭 동안 훈련합니다.**] 에폭 수(`max_epochs`), 미니배치 크기(`batch_size`), 학습률(`lr`)은 조정 가능한 하이퍼파라미터입니다. 즉, 이러한 값들은 기본 훈련 루프 중에 학습되지 않지만, 훈련 및 일반화 성능 측면에서 우리 모델의 성능에 여전히 영향을 미칩니다. 실제로는 데이터의 *검증(validation)* 분할을 기반으로 이러한 값들을 선택하고, 최종적으로 *테스트(test)* 분할에서 최종 모델을 평가하고 싶을 것입니다. :numref:`subsec_generalization-model-selection`에서 논의했듯이, 우리는 Fashion-MNIST의 테스트 데이터를 검증 세트로 간주하여 이 분할에 대한 검증 손실과 검증 정확도를 보고할 것입니다.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 예측 (Prediction)

훈련이 완료되었으므로, 이제 우리 모델은 [**일부 이미지를 분류**]할 준비가 되었습니다.

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

우리는 모델이 *잘못* 분류한 이미지들에 더 관심이 있습니다. 실제 레이블(텍스트 출력의 첫 번째 줄)과 모델의 예측(텍스트 출력의 두 번째 줄)을 비교하여 시각화해 보겠습니다.

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## 요약 (Summary)

이제 우리는 선형 회귀와 분류 문제를 해결하는 데 어느 정도 경험을 쌓기 시작했습니다. 이를 통해 우리는 1960-1970년대의 최첨단 통계 모델링 수준에 도달했습니다. 다음 섹션에서는 딥러닝 프레임워크를 활용하여 이 모델을 훨씬 더 효율적으로 구현하는 방법을 보여드리겠습니다.

## 연습 문제 (Exercises)

1. 이 섹션에서는 소프트맥스 연산의 수학적 정의를 기반으로 소프트맥스 함수를 직접 구현했습니다. :numref:`sec_softmax`에서 논의했듯이 이는 수치적 불안정성을 유발할 수 있습니다.
    1. 입력 값 중 하나가 100일 때 `softmax`가 여전히 올바르게 작동하는지 테스트하십시오.
    2. 모든 입력 중 가장 큰 값이 -100보다 작을 때 `softmax`가 여전히 올바르게 작동하는지 테스트하십시오.
    3. 인수의 가장 큰 항목에 대한 상대적인 값을 살펴봄으로써 해결책을 구현하십시오.
2. 크로스 엔트로피 손실 함수의 정의 $\sum_i y_i \log \hat{y}_i$를 따르는 `cross_entropy` 함수를 구현하십시오.
    1. 이 섹션의 코드 예제에서 시도해 보십시오.
    2. 왜 더 느리게 실행된다고 생각하십니까?
    3. 이 함수를 사용해야 할까요? 어떤 경우에 사용하는 것이 타당할까요?
    4. 무엇을 주의해야 할까요? 힌트: 로그의 정의역을 고려하십시오.
3. 항상 가장 가능성 높은 레이블을 반환하는 것이 좋은 아이디어일까요? 예를 들어 의료 진단에서도 이렇게 하시겠습니까? 이를 어떻게 해결하려고 시도하시겠습니까?
4. 일부 특성을 기반으로 다음 단어를 예측하기 위해 소프트맥스 회귀를 사용하고 싶다고 가정해 봅시다. 어휘 사전(vocabulary)이 클 때 어떤 문제가 발생할 수 있을까요?
5. 이 섹션 코드의 하이퍼파라미터를 실험해 보십시오. 특히:
    1. 학습률을 변경함에 따라 검증 손실이 어떻게 변하는지 플롯하십시오.
    2. 미니배치 크기를 변경함에 따라 검증 및 훈련 손실이 변합니까? 효과가 나타나기 전까지 얼마나 크거나 작게 조정해야 합니까?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17982)
:end_tab:

```