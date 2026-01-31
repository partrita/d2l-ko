```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU (GPUs)
:label:`sec_use_gpu`

:numref:`tab_intro_decade`에서 우리는 지난 20년 동안의 급격한 계산 성장을 설명했습니다. 
간단히 말해서 GPU 성능은 2000년 이후 매 10년마다 1000배씩 증가했습니다. 
이것은 엄청난 기회를 제공하지만 동시에 그러한 성능에 대한 상당한 수요가 있었음을 시사합니다.


이 섹션에서는 연구를 위해 이러한 계산 성능을 활용하는 방법에 대해 논의하기 시작합니다. 
먼저 단일 GPU를 사용하는 방법을 다루고, 나중에 다중 GPU와 다중 서버(다중 GPU 포함)를 사용하는 방법을 다룹니다.

구체적으로 단일 NVIDIA GPU를 계산에 사용하는 방법을 논의합니다. 
먼저 NVIDIA GPU가 하나 이상 설치되어 있는지 확인하십시오. 
그런 다음 [NVIDIA 드라이버 및 CUDA](https://developer.nvidia.com/cuda-downloads)를 다운로드하고 프롬프트에 따라 적절한 경로를 설정하십시오. 
이러한 준비가 완료되면 `nvidia-smi` 명령을 사용하여 (**그래픽 카드 정보를 볼 수 있습니다**).

:begin_tab:`mxnet`
MXNet 텐서가 NumPy `ndarray`와 거의 똑같아 보인다는 것을 눈치챘을 것입니다. 
하지만 몇 가지 중요한 차이점이 있습니다. 
NumPy와 MXNet을 구별하는 주요 특징 중 하나는 다양한 하드웨어 장치 지원입니다.

MXNet에서 모든 배열에는 컨텍스트(context)가 있습니다. 
지금까지는 기본적으로 모든 변수와 관련 계산이 CPU에 할당되었습니다. 
일반적으로 다른 컨텍스트는 다양한 GPU일 수 있습니다. 
여러 서버에 작업을 배포할 때 상황은 더욱 복잡해질 수 있습니다. 
배열을 컨텍스트에 지능적으로 할당함으로써 장치 간 데이터 전송에 소요되는 시간을 최소화할 수 있습니다. 
예를 들어 GPU가 있는 서버에서 신경망을 훈련할 때 일반적으로 모델의 파라미터가 GPU에 상주하는 것을 선호합니다.

다음으로 MXNet의 GPU 버전이 설치되어 있는지 확인해야 합니다. 
CPU 버전의 MXNet이 이미 설치되어 있는 경우 먼저 제거해야 합니다. 
예를 들어 `pip uninstall mxnet` 명령을 사용한 다음 CUDA 버전에 따라 해당 MXNet 버전을 설치하십시오. 
CUDA 10.0이 설치되어 있다고 가정하면 `pip install mxnet-cu100`을 통해 CUDA 10.0을 지원하는 MXNet 버전을 설치할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 모든 배열에는 장치(device)가 있습니다. 우리는 종종 이를 *컨텍스트(context)*라고 부릅니다. 
지금까지는 기본적으로 모든 변수와 관련 계산이 CPU에 할당되었습니다. 
일반적으로 다른 컨텍스트는 다양한 GPU일 수 있습니다. 
여러 서버에 작업을 배포할 때 상황은 더욱 복잡해질 수 있습니다. 
배열을 컨텍스트에 지능적으로 할당함으로써 장치 간 데이터 전송에 소요되는 시간을 최소화할 수 있습니다. 
예를 들어 GPU가 있는 서버에서 신경망을 훈련할 때 일반적으로 모델의 파라미터가 GPU에 상주하는 것을 선호합니다.
:end_tab:

이 섹션의 프로그램을 실행하려면 최소 두 개의 GPU가 필요합니다. 
대부분의 데스크톱 컴퓨터에는 과도할 수 있지만 클라우드(예: AWS EC2 멀티 GPU 인스턴스 사용)에서는 쉽게 사용할 수 있습니다. 
거의 모든 다른 섹션은 다중 GPU를 *요구하지 않지만*, 여기서는 단순히 장치 간 데이터 흐름을 설명하고자 합니다.

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

## [**컴퓨팅 장치 (Computing Devices)**]

저장 및 계산을 위해 CPU 및 GPU와 같은 장치를 지정할 수 있습니다. 
기본적으로 텐서는 메인 메모리에 생성된 다음 계산에 CPU를 사용합니다.

:begin_tab:`mxnet`
MXNet에서 CPU와 GPU는 `cpu()`와 `gpu()`로 나타낼 수 있습니다. 
`cpu()`(또는 괄호 안의 정수)는 모든 물리적 CPU와 메모리를 의미한다는 점에 유의해야 합니다. 
이는 MXNet의 계산이 모든 CPU 코어를 사용하려고 시도한다는 것을 의미합니다. 
그러나 `gpu()`는 하나의 카드와 해당 메모리만 나타냅니다. 
GPU가 여러 개 있는 경우 `gpu(i)`를 사용하여 $i$번째 GPU를 나타냅니다($i$는 0부터 시작). 
또한 `gpu(0)`과 `gpu()`는 동일합니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 CPU와 GPU는 `torch.device('cpu')`와 `torch.device('cuda')`로 나타낼 수 있습니다. 
`cpu` 장치는 모든 물리적 CPU와 메모리를 의미한다는 점에 유의해야 합니다. 
이는 PyTorch의 계산이 모든 CPU 코어를 사용하려고 시도한다는 것을 의미합니다. 
그러나 `gpu` 장치는 하나의 카드와 해당 메모리만 나타냅니다. 
GPU가 여러 개 있는 경우 `torch.device(f'cuda:{i}')`를 사용하여 $i$번째 GPU를 나타냅니다($i$는 0부터 시작). 
또한 `gpu:0`과 `gpu`는 동일합니다.
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """CPU 장치를 가져옵니다."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """GPU 장치를 가져옵니다."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """CPU 장치를 가져옵니다."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """GPU 장치를 가져옵니다."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

우리는 (**사용 가능한 GPU 수를 쿼리**)할 수 있습니다.

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """사용 가능한 GPU 수를 가져옵니다."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """사용 가능한 GPU 수를 가져옵니다."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # GPU 백엔드를 찾을 수 없음

num_gpus()
```

이제 [**요청한 GPU가 존재하지 않더라도 코드를 실행할 수 있게 해주는 두 가지 편리한 함수를 정의합니다.**]

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """존재하면 gpu(i)를, 그렇지 않으면 cpu()를 반환합니다."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """모든 사용 가능한 GPU를 반환하거나, GPU가 없으면 [cpu(),]를 반환합니다."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## 텐서와 GPU (Tensors and GPUs)

:begin_tab:`pytorch`
기본적으로 텐서는 CPU에 생성됩니다. 
우리는 [**텐서가 위치한 장치를 쿼리**]할 수 있습니다.
:end_tab:

:begin_tab:`mxnet`
기본적으로 텐서는 CPU에 생성됩니다. 
우리는 [**텐서가 위치한 장치를 쿼리**]할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow, jax`
기본적으로 텐서는 GPU/TPU를 사용할 수 있으면 생성되고, 그렇지 않으면 CPU가 사용됩니다. 
우리는 [**텐서가 위치한 장치를 쿼리**]할 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

여러 항에 대해 연산하고자 할 때마다, 그것들이 동일한 장치에 있어야 한다는 점을 기억하는 것이 중요합니다. 
예를 들어 두 텐서를 더하는 경우, 두 인수가 동일한 장치에 있는지 확인해야 합니다. 그렇지 않으면 프레임워크는 결과를 어디에 저장해야 할지, 심지어 어디서 계산을 수행해야 할지 결정할 수 없습니다.

### GPU에 저장하기 (Storage on the GPU)

[**텐서를 GPU에 저장**]하는 방법에는 여러 가지가 있습니다. 
예를 들어 텐서를 생성할 때 저장 장치를 지정할 수 있습니다. 
다음으로 첫 번째 `gpu`에 텐서 변수 `X`를 생성합니다. 
GPU에 생성된 텐서는 해당 GPU의 메모리만 소비합니다. 
`nvidia-smi` 명령을 사용하여 GPU 메모리 사용량을 볼 수 있습니다. 
일반적으로 GPU 메모리 한도를 초과하는 데이터를 생성하지 않도록 해야 합니다.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# 기본적으로 JAX는 사용 가능한 경우 배열을 GPU 또는 TPU에 넣습니다
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

최소 두 개의 GPU가 있다고 가정하면, 다음 코드는 (**두 번째 GPU에 무작위 텐서 `Y`를 생성합니다.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### 복사하기 (Copying)

[**`X + Y`를 계산하려면, 이 연산을 어디서 수행할지 결정해야 합니다.**] 
예를 들어 :numref:`fig_copyto`에 표시된 것처럼 `X`를 두 번째 GPU로 전송하여 거기서 연산을 수행할 수 있습니다. 
단순히 `X`와 `Y`를 더하지 *마십시오*. 예외가 발생할 것입니다. 
런타임 엔진은 무엇을 해야 할지 모릅니다: 동일한 장치에서 데이터를 찾을 수 없어 실패합니다. 
`Y`가 두 번째 GPU에 있으므로 두 개를 더하기 전에 `X`를 거기로 옮겨야 합니다.

![데이터를 복사하여 동일한 장치에서 연산을 수행합니다.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

이제 [**데이터(`Z`와 `Y` 모두)가 동일한 GPU에 있으므로, 더할 수 있습니다.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보십시오. 
여전히 `Z.copyto(gpu(1))`을 호출하면 어떻게 될까요? 
변수가 이미 원하는 장치에 있음에도 불구하고 복사본을 만들고 새 메모리를 할당할 것입니다. 
코드가 실행되는 환경에 따라 두 변수가 이미 동일한 장치에 있을 수 있는 경우가 있습니다. 
따라서 변수가 현재 다른 장치에 있는 경우에만 복사본을 만들고 싶습니다. 
이러한 경우 `as_in_ctx`를 호출할 수 있습니다. 
변수가 이미 지정된 장치에 있다면 아무 작업도 수행하지 않습니다. 
특별히 복사본을 만들고 싶지 않다면 `as_in_ctx`가 선택할 수 있는 방법입니다.
:end_tab:

:begin_tab:`pytorch`
하지만 변수 `Z`가 이미 두 번째 GPU에 있었다면 어떨까요? 
여전히 `Z.cuda(1)`을 호출하면 어떻게 될까요? 
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환할 것입니다.
:end_tab:

:begin_tab:`tensorflow`
변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보십시오. 
동일한 장치 범위에서 `Z2 = Z`를 호출하면 어떻게 될까요? 
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환할 것입니다.
:end_tab:

:begin_tab:`jax`
변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보십시오. 
동일한 장치 범위에서 `Z2 = Z`를 호출하면 어떻게 될까요? 
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환할 것입니다.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### 부가적인 참고 사항 (Side Notes)

사람들은 빠를 것이라고 기대하기 때문에 GPU를 사용하여 머신러닝을 수행합니다. 
하지만 장치 간에 변수를 전송하는 것은 느립니다: 계산보다 훨씬 느립니다. 
따라서 우리는 여러분이 느린 작업을 수행하기를 원한다는 것을 100% 확신하기를 바랍니다. 
딥러닝 프레임워크가 충돌 없이 자동으로 복사를 수행했다면 여러분은 느린 코드를 작성했다는 것을 깨닫지 못했을 수 있습니다.

데이터 전송은 느릴 뿐만 아니라 병렬화도 훨씬 어렵게 만듭니다. 더 많은 작업을 진행하기 전에 데이터가 전송될 때까지(또는 수신될 때까지) 기다려야 하기 때문입니다. 
이것이 복사 작업에 각별한 주의를 기울여야 하는 이유입니다. 
경험 법칙에 따르면, 작은 작업 여러 개는 하나의 큰 작업보다 훨씬 나쁩니다. 
또한 여러분이 무엇을 하고 있는지 알지 못한다면, 코드에 흩어져 있는 많은 단일 작업보다 한 번에 여러 작업을 수행하는 것이 훨씬 낫습니다. 
한 장치가 다른 장치를 기다려야 다른 작업을 수행할 수 있는 경우 이러한 작업이 차단될 수 있기 때문입니다. 
마치 줄을 서서 커피를 주문하는 것보다 전화로 미리 주문하고 갔을 때 준비되어 있는 것을 확인하는 것과 비슷합니다.

마지막으로, 텐서를 인쇄하거나 텐서를 NumPy 형식으로 변환할 때 데이터가 메인 메모리에 없으면 프레임워크는 먼저 메인 메모리로 복사하여 추가적인 전송 오버헤드를 발생시킵니다. 
설상가상으로 이제 Python이 완료될 때까지 모든 것을 기다리게 만드는 무시무시한 전역 인터프리터 록(Global Interpreter Lock)의 적용을 받습니다.


## [**신경망과 GPU (Neural Networks and GPUs)**]

마찬가지로 신경망 모델도 장치를 지정할 수 있습니다. 
다음 코드는 모델 파라미터를 GPU에 넣습니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # 더미 입력
params = net.init(key2, x)  # 초기화 호출
```

다음 장에서 GPU에서 모델을 실행하는 더 많은 예를 볼 수 있을 것입니다. 단순히 모델이 계산적으로 좀 더 집약적이 될 것이기 때문입니다.

예를 들어 입력이 GPU에 있는 텐서인 경우, 모델은 동일한 GPU에서 결과를 계산합니다.

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

(**모델 파라미터가 동일한 GPU에 저장되어 있는지 확인**)해 봅시다.

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

트레이너가 GPU를 지원하도록 합니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

간단히 말해서 모든 데이터와 파라미터가 동일한 장치에 있는 한 모델을 효율적으로 학습할 수 있습니다. 다음 장에서 몇 가지 그러한 예를 볼 것입니다.

## 요약 (Summary)

CPU 또는 GPU와 같은 저장 및 계산용 장치를 지정할 수 있습니다.
  기본적으로 데이터는 메인 메모리에 생성된 다음 계산을 위해 CPU를 사용합니다.
딥러닝 프레임워크는 계산을 위한 모든 입력 데이터가 CPU이든 동일한 GPU이든 동일한 장치에 있어야 합니다.
주의 없이 데이터를 이동하면 상당한 성능 저하가 발생할 수 있습니다.
  전형적인 실수는 다음과 같습니다: GPU에서 모든 미니배치에 대한 손실을 계산하고 명령줄에서 사용자에게 다시 보고(또는 NumPy `ndarray`에 기록)하면 모든 GPU를 멈추게 하는 전역 인터프리터 록이 트리거됩니다.
  GPU 내부에 로깅을 위한 메모리를 할당하고 더 큰 로그만 이동하는 것이 훨씬 낫습니다.

## 연습 문제 (Exercises)

1. 큰 행렬의 곱셈과 같은 더 큰 계산 작업을 시도해보고 CPU와 GPU 사이의 속도 차이를 확인하십시오. 계산 횟수가 적은 작업은 어떻습니까?
2. GPU에서 모델 파라미터를 어떻게 읽고 써야 합니까?
3. $100 	imes 100$ 행렬의 행렬-행렬 곱셈 1000개를 계산하는 데 걸리는 시간을 측정하고, 한 번에 하나씩 결과 출력 행렬의 프로베니우스 노름을 기록하십시오. GPU에 로그를 유지하고 최종 결과만 전송하는 것과 비교하십시오.
4. 동시에 두 개의 GPU에서 두 개의 행렬-행렬 곱셈을 수행하는 데 걸리는 시간을 측정하십시오. 하나의 GPU에서 순차적으로 계산하는 것과 비교하십시오. 힌트: 거의 선형적인 확장을 볼 수 있어야 합니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/270)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17995)
:end_tab:

```