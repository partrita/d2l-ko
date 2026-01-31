```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 회귀의 간결한 구현 (Concise Implementation of Linear Regression)
:label:`sec_linear_concise`

딥러닝은 지난 10년 동안 일종의 캄브리아기 폭발을 목격했습니다. 
수많은 기술, 응용 프로그램 및 알고리즘의 수는 이전 수십 년의 발전을 훨씬 능가합니다. 
이는 여러 요인의 행운 섞인 조합 덕분이며, 그중 하나는 여러 오픈 소스 딥러닝 프레임워크가 제공하는 강력한 무료 도구입니다. 
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`, DistBelief :cite:`Dean.Corrado.Monga.ea.2012`, Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`는 널리 채택된 그러한 모델의 1세대라고 할 수 있습니다. 
Lisp와 유사한 프로그래밍 경험을 제공했던 초기(중요한) 작품인 SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`와 대조적으로, 현대 프레임워크는 자동 미분과 Python의 편리함을 제공합니다. 
이러한 프레임워크를 통해 우리는 경사 기반 학습 알고리즘을 구현하는 반복적인 작업을 자동화하고 모듈화할 수 있습니다.

:numref:`sec_linear_scratch`에서 우리는 (i) 데이터 저장 및 선형 대수를 위한 텐서, (ii) 기울기 계산을 위한 자동 미분에만 의존했습니다. 
실제로 데이터 반복자, 손실 함수, 최적화기 및 신경망 레이어는 매우 일반적이기 때문에 현대 라이브러리는 우리를 위해 이러한 구성 요소도 구현합니다. 
이 섹션에서는 딥러닝 프레임워크의 고수준 API를 사용하여 :numref:`sec_linear_scratch`의 (**선형 회귀 모델을 간결하게 구현하는 방법을 보여드리겠습니다.**)

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
import numpy as np
import torch
from torch import nn
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
import jax
from jax import numpy as jnp
import optax
```

## 모델 정의하기 (Defining the Model)

:numref:`sec_linear_scratch`에서 선형 회귀를 밑바닥부터 구현했을 때, 우리는 모델 파라미터를 명시적으로 정의하고 기본 선형 대수 연산을 사용하여 출력을 생성하는 계산을 코딩했습니다. 
이를 수행하는 방법은 반드시 알고 *있어야* 합니다. 
하지만 모델이 복잡해지고 거의 매일 이 작업을 수행해야 한다면 도구의 도움을 받는 것이 기쁠 것입니다. 
상황은 자체 블로그를 밑바닥부터 코딩하는 것과 비슷합니다. 
한두 번 해보는 것은 보람 있고 유익하지만, 바퀴를 재발명하는 데 한 달을 보낸다면 서투른 웹 개발자가 될 것입니다.

표준 연산의 경우, 우리는 모델을 구성하는 레이어에 집중할 수 있게 해주는 [**프레임워크의 미리 정의된 레이어를 사용할 수 있습니다.**] 
그 구현에 대해 걱정할 필요가 없습니다. 
:numref:`fig_single_neuron`에 설명된 단일 레이어 네트워크의 아키텍처를 상기해 보십시오. 
각 입력이 행렬-벡터 곱셈을 통해 각 출력에 연결되기 때문에 이 레이어를 *완전 연결(fully connected)*이라고 합니다.

:begin_tab:`mxnet`
Gluon에서 완전 연결 레이어는 `Dense` 클래스에 정의되어 있습니다. 
단일 스칼라 출력을 생성하고 싶으므로 그 숫자를 1로 설정합니다. 
편의를 위해 Gluon은 각 레이어의 입력 모양을 지정할 것을 요구하지 않는다는 점에 유의할 가치가 있습니다. 
따라서 이 선형 레이어에 몇 개의 입력이 들어가는지 Gluon에 말해줄 필요가 없습니다. 
나중에 `net(X)`를 실행할 때처럼 모델을 통해 처음으로 데이터를 전달할 때, Gluon은 자동으로 각 레이어의 입력 수를 추론하여 올바른 모델을 인스턴스화합니다. 
이것이 어떻게 작동하는지는 나중에 더 자세히 설명하겠습니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 완전 연결 레이어는 `Linear` 및 `LazyLinear` 클래스(버전 1.8.0부터 사용 가능)에 정의되어 있습니다. 
후자는 사용자가 *오직* 출력 차원만 지정할 수 있게 해주는 반면, 전자는 추가로 이 레이어에 몇 개의 입력이 들어가는지 묻습니다. 
입력 모양을 지정하는 것은 불편하고 (합성곱 레이어와 같이) 복잡한 계산이 필요할 수 있습니다. 
따라서 단순함을 위해 가능할 때마다 이러한 "lazy" 레이어를 사용할 것입니다. 
:end_tab:

:begin_tab:`tensorflow`
Keras에서 완전 연결 레이어는 `Dense` 클래스에 정의되어 있습니다. 
단일 스칼라 출력을 생성하고 싶으므로 그 숫자를 1로 설정합니다. 
편의를 위해 Keras는 각 레이어의 입력 모양을 지정할 것을 요구하지 않는다는 점에 유의할 가치가 있습니다. 
이 선형 레이어에 몇 개의 입력이 들어가는지 Keras에 말해줄 필요가 없습니다. 
나중에 `net(X)`를 실행할 때처럼 모델을 통해 처음으로 데이터를 전달하려고 할 때, Keras는 자동으로 각 레이어의 입력 수를 추론합니다. 
이것이 어떻게 작동하는지는 나중에 더 자세히 설명하겠습니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """고수준 API로 구현된 선형 회귀 모델입니다."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """고수준 API로 구현된 선형 회귀 모델입니다."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

`forward` 메서드에서는 단순히 미리 정의된 레이어의 내장 `__call__` 메서드를 호출하여 출력을 계산합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## 손실 함수 정의하기 (Defining the Loss Function)

:begin_tab:`mxnet`
`loss` 모듈은 많은 유용한 손실 함수를 정의합니다. 
속도와 편의를 위해 자체 구현은 포기하고 대신 내장된 `loss.L2Loss`를 선택합니다. 
이 함수가 반환하는 `loss`는 각 예제에 대한 제곱 오차이므로, `mean`을 사용하여 미니배치에 대해 손실을 평균냅니다.
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` 클래스는 평균 제곱 오차를 계산합니다(:eqref:`eq_mse`에서 1/2 계수 제외).**] 
기본적으로 `MSELoss`는 예제에 대한 평균 손실을 반환합니다. 
직접 구현하는 것보다 빠르고 사용하기 쉽습니다.
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` 클래스는 평균 제곱 오차를 계산합니다(:eqref:`eq_mse`에서 1/2 계수 제외). 
기본적으로 예제에 대한 평균 손실을 반환합니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## 최적화 알고리즘 정의하기 (Defining the Optimization Algorithm)

:begin_tab:`mxnet`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로 Gluon은 `Trainer` 클래스를 통해 이 알고리즘의 다양한 변형과 함께 이를 지원합니다. 
Gluon의 `Trainer` 클래스는 최적화 알고리즘을 의미하며, :numref:`sec_oo-design`에서 만든 `Trainer` 클래스는 훈련 메서드, 즉 모델 파라미터를 업데이트하기 위해 최적화기를 반복적으로 호출하는 것을 포함한다는 점에 유의하십시오. 
`Trainer`를 인스턴스화할 때 최적화할 파라미터(우리 모델 `net`에서 `net.collect_params()`를 통해 얻을 수 있음), 사용하려는 최적화 알고리즘(`sgd`), 그리고 최적화 알고리즘에 필요한 하이퍼파라미터 딕셔너리를 지정합니다.
:end_tab:

:begin_tab:`pytorch`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로 PyTorch는 `optim` 모듈에서 이 알고리즘의 다양한 변형과 함께 이를 지원합니다. 
우리가 (**`SGD` 인스턴스를 인스턴스화할 때,**) 최적화할 파라미터(우리 모델에서 `self.parameters()`를 통해 얻을 수 있음)와 최적화 알고리즘에 필요한 학습률(`self.lr`)을 지정합니다.
:end_tab:

:begin_tab:`tensorflow`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로 Keras는 `optimizers` 모듈에서 이 알고리즘의 다양한 변형과 함께 이를 지원합니다.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## 훈련 (Training)

딥러닝 프레임워크의 고수준 API를 통해 우리 모델을 표현하면 더 적은 줄의 코드가 필요하다는 것을 눈치챘을 것입니다. 
파라미터를 개별적으로 할당하거나, 손실 함수를 정의하거나, 미니배치 SGD를 구현할 필요가 없었습니다. 
훨씬 더 복잡한 모델로 작업하기 시작하면 고수준 API의 이점은 상당히 커질 것입니다.

이제 모든 기본적인 부분이 준비되었으므로, [**훈련 루프 자체는 밑바닥부터 구현한 것과 동일합니다.**] 
따라서 모델을 훈련하기 위해 (:numref:`oo-design-training`에서 소개된) `fit` 메서드를 호출하기만 하면 됩니다. 이 메서드는 :numref:`sec_linear_scratch`에서의 `fit_epoch` 메서드 구현에 의존합니다.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

아래에서는 [**유한한 데이터에서 훈련하여 학습한 모델 파라미터와 우리 데이터셋을 생성한 실제 파라미터를 비교**]합니다. 
파라미터에 액세스하려면 우리가 필요한 레이어의 가중치와 편향에 액세스합니다. 
밑바닥부터 구현한 경우와 마찬가지로, 추정된 파라미터가 실제 대응물과 가깝다는 점에 유의하십시오.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'w 추정 오차: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'b 추정 오차: {data.b - b}')
```

## 요약 (Summary)

이 섹션은 MXNet :cite:`Chen.Li.Li.ea.2015`, JAX :cite:`Frostig.Johnson.Leary.2018`, PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`와 같은 현대 딥러닝 프레임워크가 제공하는 편리함을 활용한 (이 책에서의) 첫 번째 심층 네트워크 구현을 포함합니다. 
우리는 데이터 로딩, 레이어 정의, 손실 함수, 최적화기 및 훈련 루프를 위해 프레임워크 기본값을 사용했습니다. 
프레임워크가 필요한 모든 기능을 제공할 때마다 이를 사용하는 것이 일반적으로 좋은 아이디어입니다. 이러한 구성 요소의 라이브러리 구현은 성능을 위해 고도로 최적화되고 신뢰성을 위해 적절하게 테스트되는 경향이 있기 때문입니다. 
동시에 이러한 모듈들을 직접 구현할 수 *있음*을 잊지 마십시오. 
이는 현재 어떤 라이브러리에도 존재할 수 없는 새로운 구성 요소를 발명하게 될, 모델 개발의 최첨단에서 살고자 하는 야심 찬 연구자들에게 특히 중요합니다.

:begin_tab:`mxnet`
Gluon에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고, `nn` 모듈은 많은 수의 신경망 레이어를 정의하며, `loss` 모듈은 많은 일반적인 손실 함수를 정의합니다. 
게다가 `initializer`는 파라미터 초기화를 위한 많은 선택권을 제공합니다. 
사용자에게 편리하게도, 차원과 저장소는 자동으로 추론됩니다. 
이러한 지연 초기화(lazy initialization)의 결과로, 파라미터가 인스턴스화(및 초기화)되기 전에 액세스하려고 시도해서는 안 됩니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고, `nn` 모듈은 많은 수의 신경망 레이어와 일반적인 손실 함수를 정의합니다. 
우리는 `_`로 끝나는 메서드로 파라미터 값을 대체하여 초기화할 수 있습니다. 
네트워크의 입력 차원을 지정해야 한다는 점에 유의하십시오. 
지금은 사소해 보이지만, 많은 레이어를 가진 복잡한 네트워크를 설계할 때 상당한 연쇄 효과를 가질 수 있습니다. 
이러한 네트워크를 파라미터화하는 방법에 대한 신중한 고려가 이식성을 허용하기 위해 필요합니다.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고, `keras` 모듈은 많은 수의 신경망 레이어와 일반적인 손실 함수를 정의합니다. 
게다가 `initializers` 모듈은 모델 파라미터 초기화를 위한 다양한 메서드를 제공합니다. 
네트워크의 차원과 저장소는 자동으로 추론됩니다(하지만 초기화되기 전에 파라미터에 액세스하려고 시도하지 않도록 주의하십시오).
:end_tab:

## 연습 문제 (Exercises)

1. 미니배치에 대한 총 손실을 미니배치에 대한 손실 평균으로 바꾸면 학습률을 어떻게 변경해야 합니까?
2. 프레임워크 문서를 검토하여 어떤 손실 함수가 제공되는지 확인하십시오. 특히 제곱 손실을 후버(Huber)의 강건한 손실 함수로 대체해 보십시오. 즉, 다음 손실 함수를 사용하십시오.
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
3. 모델 가중치의 기울기에 어떻게 액세스합니까?
4. 학습률과 에폭 수를 변경하면 해에 어떤 영향을 미칩니까? 계속 개선되나요?
5. 생성된 데이터의 양을 변경함에 따라 해가 어떻게 변합니까?
    1. 데이터 양의 함수로서 $\hat{\mathbf{w}} - \mathbf{w}$ 및 $\hat{b} - b$에 대한 추정 오차를 플롯하십시오. 힌트: 데이터 양을 선형적이 아니라 로그적으로 증가시키십시오. 즉, 1000, 2000, ..., 10,000이 아니라 5, 10, 20, 50, ..., 10,000으로 하십시오.
    2. 힌트의 제안이 왜 적절할까요?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17977)
:end_tab:

```