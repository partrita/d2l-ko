```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 지연 초기화 (Lazy Initialization)
:label:`sec_lazy_init`

지금까지 우리는 네트워크를 설정하는 데 있어 다소 엉성했던 것처럼 보일 수 있습니다. 
구체적으로, 다음과 같은 직관적이지 않고 작동하지 않아야 할 것 같은 일들을 했습니다:

* 입력 차원을 지정하지 않고 네트워크 아키텍처를 정의했습니다.
* 이전 레이어의 출력 차원을 지정하지 않고 레이어를 추가했습니다.
* 모델에 몇 개의 파라미터가 포함되어야 하는지 결정하기에 충분한 정보를 제공하기도 전에 파라미터를 "초기화"했습니다.

코드가 실행된다는 것 자체가 놀라울 수 있습니다. 
결국 딥러닝 프레임워크가 네트워크의 입력 차원이 무엇인지 알 수 있는 방법은 없습니다. 
여기서 트릭은 프레임워크가 *초기화를 지연(defers initialization)*하여, 모델을 통해 데이터를 처음 전달할 때까지 기다렸다가 즉석에서 각 레이어의 크기를 추론한다는 것입니다.


나중에 합성곱 신경망을 다룰 때, 이 기술은 더욱 편리해질 것입니다. 
입력 차원(예: 이미지의 해상도)이 후속 각 레이어의 차원에 영향을 미치기 때문입니다. 
따라서 코드를 작성할 때 차원 값을 알 필요 없이 파라미터를 설정할 수 있는 능력은 모델을 지정하고 나중에 수정하는 작업을 크게 단순화할 수 있습니다. 
다음으로 초기화 메커니즘을 더 깊이 살펴봅니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

시작하기 위해 MLP를 인스턴스화해 봅시다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

이 시점에서 네트워크는 입력 레이어 가중치의 차원을 알 수 없습니다. 
입력 차원이 아직 알려지지 않았기 때문입니다.

:begin_tab:`mxnet, pytorch, tensorflow`
결과적으로 프레임워크는 아직 어떤 파라미터도 초기화하지 않았습니다. 
아래에서 파라미터에 액세스하려고 시도하여 이를 확인합니다.
:end_tab:

:begin_tab:`jax`
:numref:`subsec_param-access`에서 언급했듯이 Jax와 Flax에서는 파라미터와 네트워크 정의가 분리되어 있으며 사용자가 둘 다 수동으로 처리합니다. Flax 모델은 상태 비저장(stateless)이므로 `parameters` 속성이 없습니다.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
파라미터 객체는 존재하지만 각 레이어에 대한 입력 차원이 -1로 나열되어 있음에 유의하십시오. 
MXNet은 파라미터 차원이 아직 알려지지 않았음을 나타내기 위해 특수 값 -1을 사용합니다. 
이 시점에서 `net[0].weight.data()`에 액세스하려고 시도하면 파라미터에 액세스하기 전에 네트워크를 초기화해야 한다는 런타임 오류가 발생합니다. 
이제 `initialize` 메서드를 통해 파라미터를 초기화하려고 시도하면 어떻게 되는지 봅시다.
:end_tab:

:begin_tab:`tensorflow`
각 레이어 객체는 존재하지만 가중치는 비어 있다는 점에 유의하십시오. 
가중치가 아직 초기화되지 않았으므로 `net.get_weights()`를 사용하면 오류가 발생합니다.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
보시다시피 아무것도 바뀌지 않았습니다. 
입력 차원을 알 수 없을 때 initialize 호출은 실제로 파라미터를 초기화하지 않습니다. 
대신 이 호출은 파라미터를 초기화하고 싶다는 의사를 (선택적으로 어떤 분포에 따라) MXNet에 등록합니다.
:end_tab:

이제 네트워크를 통해 데이터를 전달하여 프레임워크가 마침내 파라미터를 초기화하도록 해봅시다.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

입력 차원이 20이라는 것을 알게 되자마자, 프레임워크는 20 값을 대입하여 첫 번째 레이어의 가중치 행렬 모양을 식별할 수 있습니다. 
첫 번째 레이어의 모양을 인식한 후, 프레임워크는 두 번째 레이어로, 그리고 계산 그래프를 통해 모든 모양이 알려질 때까지 계속 진행합니다. 
이 경우 첫 번째 레이어만 지연 초기화가 필요하지만, 프레임워크는 순차적으로 초기화합니다. 
모든 파라미터 모양이 알려지면 프레임워크는 마침내 파라미터를 초기화할 수 있습니다.

:begin_tab:`pytorch`
다음 메서드는 모든 파라미터 모양을 추론하기 위해 네트워크를 통해 더미 입력을 전달하여 예행 연습을 하고, 
그 후 파라미터를 초기화합니다. 
기본 무작위 초기화가 필요하지 않을 때 나중에 사용될 것입니다.
:end_tab:

:begin_tab:`jax`
Flax의 파라미터 초기화는 항상 수동으로 수행되며 사용자가 처리합니다. 
다음 메서드는 더미 입력과 키 딕셔너리를 인수로 받습니다. 
이 키 딕셔너리에는 모델 파라미터를 초기화하기 위한 rng와 드롭아웃 레이어가 있는 모델의 드롭아웃 마스크를 생성하기 위한 드롭아웃 rng가 있습니다. 드롭아웃에 대한 자세한 내용은 나중에 :numref:`sec_dropout`에서 다룰 것입니다. 
결국 메서드는 모델을 초기화하고 파라미터를 반환합니다. 
우리는 이전 섹션에서도 내부적으로 이를 사용해 왔습니다.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input 튜플 언팩됨
    return params
```

## 요약 (Summary)

지연 초기화는 편리할 수 있으며, 프레임워크가 파라미터 모양을 자동으로 추론하도록 하여 아키텍처를 쉽게 수정하고 일반적인 오류 원인을 제거할 수 있습니다. 
모델을 통해 데이터를 전달하여 프레임워크가 최종적으로 파라미터를 초기화하도록 할 수 있습니다.


## 연습 문제 (Exercises)

1. 첫 번째 레이어에만 입력 차원을 지정하고 후속 레이어에는 지정하지 않으면 어떻게 됩니까? 즉시 초기화됩니까?
2. 일치하지 않는 차원을 지정하면 어떻게 됩니까?
3. 입력 차원이 다양한 경우 어떻게 해야 합니까? 힌트: 파라미터 묶기를 살펴보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17992)
:end_tab: