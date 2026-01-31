```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 사용자 정의 레이어 (Custom Layers)

딥러닝의 성공 요인 중 하나는 다양한 작업에 적합한 아키텍처를 설계하기 위해 창의적인 방식으로 구성할 수 있는 광범위한 레이어를 사용할 수 있다는 점입니다. 
예를 들어 연구자들은 이미지, 텍스트를 처리하고 순차 데이터를 반복하며 동적 프로그래밍을 수행하기 위한 레이어를 발명했습니다. 
조만간 딥러닝 프레임워크에 아직 존재하지 않는 레이어가 필요하게 될 것입니다. 
이러한 경우 사용자 정의 레이어를 구축해야 합니다. 
이 섹션에서는 그 방법을 보여드립니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## (**파라미터 없는 레이어 (Layers without Parameters)**)

시작하기 위해 자체 파라미터가 없는 사용자 정의 레이어를 구성해 보겠습니다. 
:numref:`sec_model_construction`의 모듈 소개를 기억한다면 익숙해 보일 것입니다. 
다음 `CenteredLayer` 클래스는 단순히 입력에서 평균을 뺍니다. 
이를 구축하려면 기본 레이어 클래스를 상속하고 순전파 함수를 구현하기만 하면 됩니다.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

데이터를 공급하여 레이어가 의도한 대로 작동하는지 확인해 봅시다.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

이제 [**더 복잡한 모델을 구성하는 데 우리 레이어를 구성 요소로 통합**]할 수 있습니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

추가적인 정상성 확인으로, 네트워크를 통해 무작위 데이터를 보내고 평균이 실제로 0인지 확인할 수 있습니다. 
부동 소수점 숫자를 다루고 있기 때문에 양자화로 인해 매우 작은 0이 아닌 숫자가 보일 수 있습니다.

:begin_tab:`jax`
여기서는 네트워크의 출력과 파라미터를 모두 반환하는 `init_with_output` 메서드를 활용합니다. 이 경우 우리는 출력에만 집중합니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**파라미터가 있는 레이어 (Layers with Parameters)**]

이제 단순한 레이어를 정의하는 방법을 알았으니, 훈련을 통해 조정할 수 있는 파라미터가 있는 레이어를 정의하는 것으로 넘어가겠습니다. 
기본적인 관리 기능을 제공하는 내장 함수를 사용하여 파라미터를 생성할 수 있습니다. 
특히 모델 파라미터의 액세스, 초기화, 공유, 저장 및 로드를 관리합니다. 
이렇게 하면 다른 이점들 중에서도 모든 사용자 정의 레이어에 대해 사용자 정의 직렬화 루틴을 작성할 필요가 없습니다.

이제 완전 연결 레이어의 자체 버전을 구현해 봅시다. 
이 레이어에는 가중치를 나타내는 파라미터 하나와 편향을 위한 파라미터 하나, 총 두 개의 파라미터가 필요합니다. 
이 구현에서는 기본적으로 ReLU 활성화를 포함합니다. 
이 레이어는 `in_units`와 `units`라는 두 개의 입력 인수가 필요하며, 각각 입력 및 출력 수를 나타냅니다.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
다음으로 `MyDense` 클래스를 인스턴스화하고 모델 파라미터에 액세스합니다.
:end_tab:

:begin_tab:`pytorch`
다음으로 `MyLinear` 클래스를 인스턴스화하고 모델 파라미터에 액세스합니다.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

우리는 [**사용자 정의 레이어를 사용하여 순전파 계산을 직접 수행**]할 수 있습니다.

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

또한 (**사용자 정의 레이어를 사용하여 모델을 구성**)할 수도 있습니다. 
일단 가지고 있으면 내장 완전 연결 레이어처럼 사용할 수 있습니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## 요약 (Summary)

기본 레이어 클래스를 통해 사용자 정의 레이어를 설계할 수 있습니다. 이를 통해 라이브러리의 기존 레이어와 다르게 동작하는 유연한 새 레이어를 정의할 수 있습니다. 
일단 정의되면 사용자 정의 레이어는 임의의 맥락과 아키텍처에서 호출될 수 있습니다. 
레이어는 로컬 파라미터를 가질 수 있으며, 이는 내장 함수를 통해 생성될 수 있습니다.


## 연습 문제 (Exercises)

1. 입력을 받아 텐서 축소를 계산하는 레이어를 설계하십시오. 즉, $y_k = \sum_{i, j} W_{ijk} x_i x_j$를 반환합니다.
2. 데이터의 푸리에 계수의 앞쪽 절반을 반환하는 레이어를 설계하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17993)
:end_tab: