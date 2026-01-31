```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 파일 I/O (File I/O)

지금까지 데이터 처리 방법과 딥러닝 모델 구축, 훈련, 테스트 방법에 대해 논의했습니다. 
하지만 언젠가는 학습된 모델에 충분히 만족하여 다양한 맥락에서 나중에 사용하기 위해 결과를 저장하고 싶을 것입니다(배포 시 예측을 수행하기 위해). 
또한 긴 훈련 프로세스를 실행할 때, 서버의 전원 코드를 건드려 며칠 동안의 계산을 잃지 않도록 중간 결과를 주기적으로 저장(체크포인트)하는 것이 가장 좋습니다. 
따라서 개별 가중치 벡터와 전체 모델을 모두 로드하고 저장하는 방법을 배워야 할 때입니다. 
이 섹션에서는 두 가지 문제를 모두 다룹니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**텐서 로드 및 저장 (Loading and Saving Tensors)**)

개별 텐서의 경우 `load` 및 `save` 함수를 직접 호출하여 읽고 쓸 수 있습니다. 
두 함수 모두 이름을 제공해야 하며, `save`는 저장할 변수를 입력으로 요구합니다.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

이제 저장된 파일에서 데이터를 다시 메모리로 읽어올 수 있습니다.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

우리는 [**텐서 리스트를 저장하고 다시 메모리로 읽어올 수 있습니다.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

심지어 [**문자열에서 텐서로 매핑하는 딕셔너리를 쓰고 읽을 수도 있습니다.**] 
이는 모델의 모든 가중치를 읽거나 쓰고 싶을 때 편리합니다.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**모델 파라미터 로드 및 저장 (Loading and Saving Model Parameters)**]

개별 가중치 벡터(또는 다른 텐서)를 저장하는 것은 유용하지만, 전체 모델을 저장(하고 나중에 로드)하려는 경우 매우 지루해집니다. 
결국 수백 개의 파라미터 그룹이 곳곳에 흩어져 있을 수 있습니다. 
이러한 이유로 딥러닝 프레임워크는 전체 네트워크를 로드하고 저장하는 내장 기능을 제공합니다. 
주목해야 할 중요한 세부 사항은 이것이 전체 모델이 아닌 모델 *파라미터*를 저장한다는 것입니다. 
예를 들어 3개 레이어의 MLP가 있는 경우 아키텍처를 별도로 지정해야 합니다. 
그 이유는 모델 자체에 임의의 코드가 포함될 수 있어 자연스럽게 직렬화할 수 없기 때문입니다. 
따라서 모델을 복원하려면 코드에서 아키텍처를 생성한 다음 디스크에서 파라미터를 로드해야 합니다. 
(**익숙한 MLP로 시작해 봅시다.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

다음으로, "mlp.params"라는 이름으로 [**모델의 파라미터를 파일로 저장**]합니다.

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

모델을 복구하기 위해 원래 MLP 모델의 클론을 인스턴스화합니다. 
모델 파라미터를 무작위로 초기화하는 대신, [**파일에 저장된 파라미터를 직접 읽습니다**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

두 인스턴스 모두 동일한 모델 파라미터를 가지므로 동일한 입력 `X`의 계산 결과는 같아야 합니다. 
이것을 확인해 봅시다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## 요약 (Summary)

`save` 및 `load` 함수는 텐서 객체에 대한 파일 I/O를 수행하는 데 사용할 수 있습니다. 
파라미터 딕셔너리를 통해 네트워크의 전체 파라미터 세트를 저장하고 로드할 수 있습니다. 
아키텍처 저장은 파라미터가 아닌 코드로 수행해야 합니다.

## 연습 문제 (Exercises)

1. 훈련된 모델을 다른 장치에 배포할 필요가 없더라도 모델 파라미터를 저장하는 것의 실질적인 이점은 무엇입니까?
2. 다른 아키텍처를 가진 네트워크에 통합하기 위해 네트워크의 일부만 재사용하고 싶다고 가정해 봅시다. 이전 네트워크의 처음 두 레이어를 새 네트워크에서 사용하려면 어떻게 해야 합니까?
3. 네트워크 아키텍처와 파라미터를 어떻게 저장하시겠습니까? 아키텍처에 어떤 제약을 두겠습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17994)
:end_tab:
