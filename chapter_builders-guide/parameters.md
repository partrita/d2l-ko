```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 파라미터 관리 (Parameter Management)

아키텍처를 선택하고 하이퍼파라미터를 설정했으면 훈련 루프로 진행합니다. 
여기서 우리의 목표는 손실 함수를 최소화하는 파라미터 값을 찾는 것입니다. 
훈련 후에는 향후 예측을 위해 이러한 파라미터가 필요합니다. 
또한 다른 맥락에서 재사용하거나, 다른 소프트웨어에서 실행될 수 있도록 모델을 디스크에 저장하거나, 과학적 이해를 얻기 위해 검사하기 위해 파라미터를 추출하고 싶을 때가 있습니다.

대부분의 경우, 우리는 파라미터가 선언되고 조작되는 구체적인 세부 사항을 무시하고 딥러닝 프레임워크에 무거운 작업을 맡길 수 있습니다. 
하지만 표준 레이어가 쌓인 아키텍처에서 벗어날 때, 파라미터 선언 및 조작의 세부 사항으로 들어가야 할 때가 있습니다. 
이 섹션에서는 다음 내용을 다룹니다:

* 디버깅, 진단 및 시각화를 위한 파라미터 액세스.
* 서로 다른 모델 구성 요소 간의 파라미터 공유.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

(**은닉층이 하나 있는 MLP에 집중하는 것으로 시작합니다.**)

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # 기본 초기화 방법 사용

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**파라미터 액세스 (Parameter Access)**]
:label:`subsec_param-access`

이미 알고 있는 모델에서 파라미터에 액세스하는 방법부터 시작해 봅시다.

:begin_tab:`mxnet, pytorch, tensorflow`
모델이 `Sequential` 클래스를 통해 정의되면, 모델을 리스트인 것처럼 인덱싱하여 모든 레이어에 먼저 액세스할 수 있습니다. 
각 레이어의 파라미터는 해당 속성에 편리하게 위치해 있습니다.
:end_tab:

:begin_tab:`jax`
이전에 정의한 모델에서 관찰했을 수 있듯이 Flax와 JAX는 모델과 파라미터를 분리합니다. 
모델이 `Sequential` 클래스를 통해 정의되면, 먼저 네트워크를 초기화하여 파라미터 딕셔너리를 생성해야 합니다. 
이 딕셔너리의 키를 통해 모든 레이어의 파라미터에 액세스할 수 있습니다.
:end_tab:

다음과 같이 두 번째 완전 연결 레이어의 파라미터를 검사할 수 있습니다.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

이 완전 연결 레이어에는 두 개의 파라미터가 포함되어 있음을 알 수 있습니다. 
각각 해당 레이어의 가중치와 편향에 해당합니다.


### [**타겟 파라미터 (Targeted Parameters)**]

각 파라미터는 파라미터 클래스의 인스턴스로 표현됩니다. 
파라미터로 유용한 작업을 하려면 먼저 기본 수치 값에 액세스해야 합니다. 
이를 수행하는 방법에는 여러 가지가 있습니다. 
일부는 더 간단하고 다른 일부는 더 일반적입니다. 
다음 코드는 두 번째 신경망 레이어에서 편향을 추출하여 파라미터 클래스 인스턴스를 반환하고, 
더 나아가 해당 파라미터의 값에 액세스합니다.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
파라미터는 값, 기울기 및 추가 정보를 포함하는 복잡한 객체입니다. 
그렇기 때문에 값을 명시적으로 요청해야 합니다.

값 외에도 각 파라미터는 기울기에 액세스할 수 있게 해줍니다. 이 네트워크에 대해 아직 역전파를 호출하지 않았으므로 초기 상태입니다.
:end_tab:

:begin_tab:`jax`
다른 프레임워크와 달리 JAX는 신경망 파라미터에 대한 기울기를 추적하지 않고 대신 파라미터와 네트워크가 분리됩니다. 
사용자가 계산을 Python 함수로 표현하고 동일한 목적을 위해 `grad` 변환을 사용할 수 있게 합니다.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**한꺼번에 모든 파라미터 (All Parameters at Once)**]

모든 파라미터에 대해 작업을 수행해야 할 때, 하나씩 액세스하는 것은 지루할 수 있습니다. 
더 복잡한, 예를 들어 중첩된 모듈로 작업할 때 상황은 특히 다루기 어려워질 수 있습니다. 
각 하위 모듈의 파라미터를 추출하기 위해 전체 트리를 재귀적으로 탐색해야 하기 때문입니다. 아래에서는 모든 레이어의 파라미터에 액세스하는 것을 보여줍니다.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**묶인 파라미터 (Tied Parameters)**]

종종 여러 레이어 간에 파라미터를 공유하고 싶을 때가 있습니다. 
우아하게 수행하는 방법을 알아봅시다. 
다음에서는 완전 연결 레이어를 할당한 다음 해당 파라미터를 사용하여 다른 레이어의 파라미터를 설정합니다. 
여기서 파라미터에 액세스하기 전에 순전파 `net(X)`를 실행해야 합니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# 공유 레이어의 파라미터를 참조할 수 있도록 이름을 지정해야 합니다
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# 파라미터가 동일한지 확인
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# 단순히 같은 값을 갖는 것이 아니라 실제로 같은 객체인지 확인
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# 공유 레이어의 파라미터를 참조할 수 있도록 이름을 지정해야 합니다
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# 파라미터가 동일한지 확인
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 단순히 같은 값을 갖는 것이 아니라 실제로 같은 객체인지 확인
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras는 약간 다르게 동작합니다. 중복 레이어를 자동으로 제거합니다
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# 파라미터가 다른지 확인
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# 공유 레이어의 파라미터를 참조할 수 있도록 이름을 지정해야 합니다
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# 파라미터가 다른지 확인
print(len(params['params']) == 3)
```

이 예제는 두 번째와 세 번째 레이어의 파라미터가 묶여 있음을 보여줍니다. 
그것들은 단순히 같은 것이 아니라 정확히 같은 텐서로 표현됩니다. 
따라서 파라미터 중 하나를 변경하면 다른 하나도 변경됩니다.

:begin_tab:`mxnet, pytorch, tensorflow`
파라미터가 묶여 있을 때 기울기는 어떻게 되는지 궁금할 수 있습니다. 
모델 파라미터에 기울기가 포함되어 있으므로, 
역전파 중에 두 번째 은닉층과 세 번째 은닉층의 기울기가 함께 더해집니다.
:end_tab:


## 요약 (Summary)

우리는 모델 파라미터에 액세스하고 묶는 몇 가지 방법을 가지고 있습니다.


## 연습 문제 (Exercises)

1. :numref:`sec_model_construction`에 정의된 `NestMLP` 모델을 사용하여 다양한 레이어의 파라미터에 액세스하십시오.
2. 공유 파라미터 레이어를 포함하는 MLP를 구성하고 훈련하십시오. 훈련 과정 동안 각 레이어의 모델 파라미터와 기울기를 관찰하십시오.
3. 파라미터를 공유하는 것이 왜 좋은 아이디어입니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17990)
:end_tab: