```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 가중치 감쇠 (Weight Decay)
:label:`sec_weight_decay`

이제 과대적합 문제를 규명했으므로, 첫 번째 *정규화(regularization)* 기술을 소개할 수 있습니다. 
더 많은 훈련 데이터를 수집함으로써 언제나 과대적합을 완화할 수 있다는 점을 상기하십시오. 
그러나 이는 비용이 많이 들고 시간이 오래 걸리거나, 우리의 통제 밖일 수 있어 단기적으로는 불가능할 수 있습니다. 
지금은 이미 리소스가 허용하는 만큼의 고품질 데이터를 확보했다고 가정하고, 데이터셋이 주어진 것으로 간주될 때 사용할 수 있는 도구에 집중해 보겠습니다.

다항식 회귀 예제(:numref:`subsec_polynomial-curve-fitting`)에서 적합된 다항식의 차수를 조정하여 모델의 용량을 제한할 수 있었다는 점을 상기하십시오. 
실제로 특성 수를 제한하는 것은 과대적합을 완화하기 위한 대중적인 기술입니다. 
하지만 단순히 특성을 버리는 것은 너무 무딘 도구일 수 있습니다. 
다항식 회귀 예제를 계속해서 생각해보면, 고차원 입력에서 어떤 일이 일어날지 고려해 보십시오. 
다변수 데이터로의 다항식의 자연스러운 확장을 *단항식(monomials)*이라고 하며, 이는 단순히 변수의 거듭제곱들의 곱입니다. 
단항식의 차수는 거듭제곱의 합입니다. 예를 들어, $x_1^2 x_2$와 $x_3 x_5^2$는 모두 3차 단항식입니다.

$d$차 항의 수는 $d$가 커짐에 따라 급격히 늘어납니다. 
$k$개의 변수가 주어졌을 때 $d$차 단항식의 수는 ${k - 1 + d} \choose {k - 1}$입니다. 
차수가 2에서 3으로 조금만 변해도 모델의 복잡도는 비약적으로 증가합니다. 
따라서 우리는 종종 함수 복잡도를 조정하기 위한 더 미세한 도구가 필요합니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## 노름과 가중치 감쇠 (Norms and Weight Decay)

(**파라미터 수를 직접 조작하는 대신, *가중치 감쇠(weight decay)*는 파라미터가 취할 수 있는 값을 제한함으로써 작동합니다.**)
딥러닝 이외의 분야에서 미니배치 확률적 경사 하강법으로 최적화될 때 더 흔히 $\ell_2$ 정규화라고 불리는 가중치 감쇠는, 파라미터화된 머신러닝 모델을 정규화하기 위해 가장 널리 사용되는 기술일 것입니다. 
이 기술은 모든 함수 $f$ 중에서 함수 $f = 0$ (모든 입력에 값 0을 할당)이 어떤 의미에서 가장 *단순*하며, 파라미터가 0에서 떨어진 거리로 함수의 복잡도를 측정할 수 있다는 기본적인 직관에 근거합니다. 
하지만 함수와 0 사이의 거리를 정확히 어떻게 측정해야 할까요? 
정답은 하나가 아닙니다. 사실 함수 해석학의 일부와 바나흐 공간(Banach spaces) 이론을 포함한 수학의 전체 분야가 이러한 문제를 다루는 데 헌신하고 있습니다.

하나의 간단한 해석은 선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$의 복잡도를 그 가중치 벡터의 어떤 노름, 예를 들어 $\| \mathbf{w} \|^2$으로 측정하는 것일 수 있습니다. 
우리는 :numref:`subsec_lin-algebra-norms`에서 더 일반적인 $\ell_p$ 노름의 특수한 경우인 $\ell_2$ 노름과 $\ell_1$ 노름을 소개했습니다. 
가중치 벡터를 작게 유지하는 가장 일반적인 방법은 손실을 최소화하는 문제에 노름을 페널티 항으로 추가하는 것입니다. 
따라서 우리는 원래의 목표인 *훈련 레이블에 대한 예측 손실 최소화*를 새로운 목표인 *예측 손실과 페널티 항의 합 최소화*로 대체합니다. 
이제 가중치 벡터가 너무 커지면, 학습 알고리즘은 훈련 오차를 최소화하는 것보다 가중치 노름 $\| \mathbf{w} \|^2$을 최소화하는 데 집중할 수 있습니다. 
그것이 바로 우리가 원하는 것입니다. 
이를 코드로 설명하기 위해 :numref:`sec_linear_regression`의 선형 회귀 예제를 다시 가져와 보겠습니다. 
거기서 우리의 손실은 다음과 같이 주어졌습니다.

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$ 

$\\mathbf{x}^{(i)}$는 특성, $y^{(i)}$는 데이터 예제 $i$에 대한 레이블, $(\mathbf{w}, b)$는 각각 가중치와 편향 파라미터임을 상기하십시오. 
가중치 벡터의 크기에 페널티를 주려면 어떻게든 손실 함수에 $\| \mathbf{w} \|^2$을 추가해야 하지만, 모델이 표준 손실과 이 새로운 가산 페널티 사이에서 어떻게 절충(trade-off)해야 할까요? 
실제로는 검증 데이터를 사용하여 맞추는 음이 아닌 하이퍼파라미터인 *정규화 상수* $\lambda$를 통해 이 절충을 특성화합니다:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$ 


$\\lambda = 0$인 경우 원래의 손실 함수를 복구합니다. 
$\\lambda > 0$인 경우 $\| \mathbf{w} \|$의 크기를 제한합니다. 
관례적으로 2로 나눕니다: 이차 함수의 도함수를 취할 때 2와 1/2이 상쇄되어 업데이트 식이 깔끔하고 단순해지기 때문입니다. 
기민한 독자는 왜 표준 노름(유클리드 거리)이 아니라 제곱 노름을 사용하는지 궁금할 수 있습니다. 
우리는 계산적 편의를 위해 이렇게 합니다. $\ell_2$ 노름을 제곱함으로써 제곱근을 제거하고 가중치 벡터의 각 성분의 제곱 합만 남깁니다. 
이는 페널티의 도함수를 계산하기 쉽게 만듭니다: 합의 도함수는 도함수의 합과 같기 때문입니다.


게다가 왜 애초에 $\ell_1$ 노름 등이 아니라 $\ell_2$ 노름을 사용하는지 물을 수도 있습니다. 
사실 다른 선택들도 유효하며 통계학 전반에서 인기가 있습니다. 
$\\ell_2$ 정규화된 선형 모델은 고전적인 *릿지 회귀(ridge regression)* 알고리즘을 구성하는 반면, $\\ell_1$ 정규화된 선형 회귀는 통계학에서 마찬가지로 근본적인 방법으로 흔히 *라소 회귀(lasso regression)*로 알려져 있습니다. 
$\\ell_2$ 노름을 사용하는 한 가지 이유는 가중치 벡터의 큰 성분에 대해 과도한 페널티를 부여하기 때문입니다. 
이는 학습 알고리즘이 더 많은 수의 특성에 가중치를 고르게 분산시키는 모델을 선호하게 만듭니다. 
실제로 이는 단일 변수의 측정 오차에 대해 모델을 더 강건하게 만들 수 있습니다. 
대조적으로, $\\ell_1$ 페널티는 다른 가중치를 0으로 만듦으로써 작은 특성 세트에 가중치를 집중시키는 모델로 이어집니다. 
이는 다른 이유로 바람직할 수 있는 *특성 선택(feature selection)*을 위한 효과적인 방법을 제공합니다. 
예를 들어 모델이 몇 가지 특성에만 의존한다면, 다른 (버려진) 특성에 대한 데이터를 수집, 저장 또는 전송할 필요가 없기 때문입니다. 

:eqref:`eq_linreg_batch_update`와 동일한 표기법을 사용하여, $\\ell_2$ 정규화된 회귀에 대한 미니배치 확률적 경사 하강법 업데이트는 다음과 같습니다:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$ 

이전과 마찬가지로, 추정치가 관찰값과 다른 정도에 따라 $\\mathbf{w}$를 업데이트합니다. 
하지만 우리는 또한 $\\mathbf{w}$의 크기를 0을 향해 수축시킵니다. 
그래서 이 방법을 때때로 "가중치 감쇠"라고 부릅니다: 페널티 항만 주어졌을 때, 최적화 알고리즘이 훈련의 각 단계에서 가중치를 *감쇠(decays)*시키기 때문입니다. 
특성 선택과 대조적으로, 가중치 감쇠는 함수의 복잡도를 연속적으로 조정하는 메커니즘을 제공합니다. 
작은 $\\lambda$ 값은 덜 제한된 $\\mathbf{w}$에 해당하고, 큰 $\\lambda$ 값은 $\\mathbf{w}$를 더 상당히 제한합니다. 
해당하는 편향 페널티 $b^2$을 포함할지 여부는 구현마다 다를 수 있으며 신경망의 레이어마다 다를 수 있습니다. 
종종 우리는 편향 항을 정규화하지 않습니다. 
게다가 다른 최적화 알고리즘의 경우 $\\ell_2$ 정규화가 가중치 감쇠와 동일하지 않을 수 있지만, 가중치의 크기를 줄여 정규화한다는 아이디어는 여전히 유효합니다.

## 고차원 선형 회귀 (High-Dimensional Linear Regression)

간단한 합성 예제를 통해 가중치 감쇠의 이점을 설명할 수 있습니다.

먼저, 이전처럼 데이터를 생성합니다:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ 여기서 } 
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**) 

이 합성 데이터셋에서 레이블은 입력의 기저 선형 함수에 평균 0, 표준 편차 0.01의 가우스 노이즈가 더해져 주어집니다. 
설명을 위해 문제의 차원을 $d = 200$으로 늘리고 예제가 20개뿐인 작은 훈련 세트를 사용하여 과대적합의 효과를 두드러지게 만들 수 있습니다.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## 밑바닥부터 구현하기 (Implementation from Scratch)

이제 가중치 감쇠를 밑바닥부터 구현해 보겠습니다. 
미니배치 확률적 경사 하강법이 우리의 최적화기이므로, 원래 손실 함수에 제곱 $\ell_2$ 페널티를 추가하기만 하면 됩니다.

### (**$\\ell_2$ 노름 페널티 정의하기**)

이 페널티를 구현하는 가장 편리한 방법은 모든 항을 제자리에서 제곱하고 합산하는 것입니다.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### 모델 정의하기

최종 모델에서 선형 회귀와 제곱 손실은 :numref:`sec_linear_scratch` 이후로 바뀌지 않았으므로, `d2l.LinearRegressionScratch`의 서브클래스를 정의하기만 하면 됩니다. 유일한 변경 사항은 이제 손실에 페널티 항이 포함된다는 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

다음 코드는 20개 예제가 있는 훈련 세트에서 모델을 맞추고 100개 예제가 있는 검증 세트에서 평가합니다.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('w의 L2 노름:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('w의 L2 노름:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**정규화 없이 훈련하기**]

이제 `lambd = 0`으로 이 코드를 실행하여 가중치 감쇠를 비활성화합니다. 
훈련 오차는 감소하지만 검증 오차는 감소하지 않는 심각한 과대적합이 발생합니다. 이는 과대적합의 교과서적인 사례입니다.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**가중치 감쇠 사용하기**]

아래에서는 상당한 가중치 감쇠를 사용하여 실행합니다. 
훈련 오차는 증가하지만 검증 오차는 감소하는 것을 보십시오. 
이것이 바로 우리가 정규화에서 기대하는 효과입니다.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**간결한 구현**]

가중치 감쇠는 신경망 최적화에서 어디에나 존재하기 때문에, 딥러닝 프레임워크는 이를 특히 편리하게 만들어 최적화 알고리즘 자체에 가중치 감쇠를 통합하여 모든 손실 함수와 함께 쉽게 사용할 수 있게 합니다. 
게다가 이 통합은 계산적인 이점을 제공하여, 추가적인 계산 오버헤드 없이 알고리즘에 가중치 감쇠를 추가하는 구현 트릭을 가능하게 합니다. 
업데이트의 가중치 감쇠 부분은 각 파라미터의 현재 값에만 의존하기 때문에, 최적화기는 어쨌든 각 파라미터를 한 번 건드려야 합니다.

:begin_tab:`mxnet`
아래에서는 `Trainer`를 인스턴스화할 때 `wd`를 통해 가중치 감쇠 하이퍼파라미터를 직접 지정합니다. 
기본적으로 Gluon은 가중치와 편향을 동시에 감쇠시킵니다. 
하이퍼파라미터 `wd`는 모델 파라미터를 업데이트할 때 `wd_mult`와 곱해진다는 점에 유의하십시오. 
따라서 `wd_mult`를 0으로 설정하면 편향 파라미터 $b$는 감쇠하지 않습니다.
:end_tab:

:begin_tab:`pytorch`
아래에서는 최적화기를 인스턴스화할 때 `weight_decay`를 통해 가중치 감쇠 하이퍼파라미터를 직접 지정합니다. 
기본적으로 PyTorch는 가중치와 편향을 동시에 감쇠시키지만, 서로 다른 정책에 따라 서로 다른 파라미터를 처리하도록 최적화기를 구성할 수 있습니다. 
여기서는 가중치(`net.weight` 파라미터)에 대해서만 `weight_decay`를 설정하므로 편향(`net.bias` 파라미터)은 감쇠하지 않습니다.
:end_tab:

:begin_tab:`tensorflow`
아래에서는 가중치 감쇠 하이퍼파라미터 `wd`를 사용하여 $\\ell_2$ 정규화기를 만들고, `kernel_regularizer` 인수를 통해 레이어의 가중치에 적용합니다.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # 가중치 감쇠는 optax.sgd 내에서 직접 사용할 수 없지만,
        # optax를 사용하면 여러 변환을 함께 연결할 수 있습니다.
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**플롯은 가중치 감쇠를 밑바닥부터 구현했을 때와 비슷해 보입니다**]. 
하지만 이 버전은 더 빠르게 실행되고 구현하기 더 쉬우며, 더 큰 문제를 다루고 이 작업이 일상화됨에 따라 이러한 이점은 더욱 두드러질 것입니다.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('w의 L2 노름:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('w의 L2 노름:', float(l2_penalty(model.get_w_b()[0])))
```

지금까지 우리는 단순한 선형 함수가 무엇인지에 대한 한 가지 개념을 다루었습니다. 
하지만 단순한 비선형 함수에 대해서도 상황은 훨씬 더 복잡할 수 있습니다. 이를 확인하기 위해, [재생 커널 힐베르트 공간(RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 개념을 사용하면 선형 함수를 위해 도입된 도구를 비선형 맥락에서 적용할 수 있습니다. 
불행히도 RKHS 기반 알고리즘은 크고 고차원적인 데이터로 잘 확장되지 않는 경향이 있습니다. 
이 책에서 우리는 심층 네트워크의 모든 레이어에 가중치 감쇠를 적용하는 일반적인 휴리스틱을 종종 채택할 것입니다.

## 요약 (Summary)

정규화는 과대적합을 다루기 위한 일반적인 방법입니다. 고전적인 정규화 기술은 (훈련 시) 손실 함수에 페널티 항을 추가하여 학습된 모델의 복잡도를 줄입니다. 
모델을 단순하게 유지하기 위한 한 가지 특별한 선택은 $\\ell_2$ 페널티를 사용하는 것입니다. 이는 미니배치 확률적 경사 하강법 알고리즘의 업데이트 단계에서 가중치 감쇠로 이어집니다. 
실제로 가중치 감쇠 기능은 딥러닝 프레임워크의 최적화기에서 제공됩니다. 
동일한 훈련 루프 내에서 서로 다른 파라미터 세트가 서로 다른 업데이트 동작을 가질 수 있습니다.



## 연습 문제 (Exercises)

1. 이 섹션의 추정 문제에서 $\\lambda$ 값을 실험해 보십시오. $\\lambda$의 함수로서 훈련 및 검증 정확도를 플롯하십시오. 무엇을 관찰하셨습니까?
2. 검증 세트를 사용하여 $\\lambda$의 최적 값을 찾으십시오. 정말 최적의 값인가요? 이것이 중요한가요?
3. $\|\mathbf{w}\|^2$ 대신 페널티로 $\sum_i |w_i|$를 사용한다면 업데이트 식은 어떻게 보일까요 ($\\ell_1$ 정규화)?
4. 우리는 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$임을 알고 있습니다. 행렬에 대해서도 유사한 식을 찾을 수 있습니까 (:numref:`subsec_lin-algebra-norms`의 프로베니우스 노름 참조)?
5. 훈련 오차와 일반화 오차의 관계를 검토하십시오. 가중치 감쇠, 훈련량 증가, 적절한 복잡도의 모델 사용 외에 과대적합을 처리하는 데 도움이 될 수 있는 다른 방법은 무엇이 있을까요?
6. 베이지안 통계에서는 $P(w \mid x) \propto P(x \mid w) P(w)$를 통해 사후 확률에 도달하기 위해 사전 확률과 우도의 곱을 사용합니다. $P(w)$를 정규화와 어떻게 연관 지을 수 있을까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17979)
:end_tab:

```