```
```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 레이어와 모듈 (Layers and Modules)
:label:`sec_model_construction`

우리가 신경망을 처음 소개했을 때, 단일 출력이 있는 선형 모델에 집중했습니다. 
여기서 전체 모델은 단 하나의 뉴런으로 구성됩니다. 
단일 뉴런은 (i) 입력 세트를 받고; (ii) 해당하는 스칼라 출력을 생성하며; (iii) 관심 있는 목적 함수를 최적화하기 위해 업데이트할 수 있는 관련 파라미터 세트를 가지고 있습니다. 
그 후, 다중 출력을 가진 네트워크에 대해 생각하기 시작했을 때, 벡터화된 산술을 활용하여 전체 뉴런 레이어를 특징지었습니다. 
개별 뉴런과 마찬가지로 레이어는 (i) 입력 세트를 받고, (ii) 해당하는 출력을 생성하며, (iii) 조정 가능한 파라미터 세트로 설명됩니다. 
소프트맥스 회귀를 작업할 때, 단일 레이어 자체가 모델이었습니다. 
그러나 그 후 MLP를 소개했을 때도 여전히 모델이 이와 동일한 기본 구조를 유지한다고 생각할 수 있었습니다.

흥미롭게도 MLP의 경우, 전체 모델과 구성 레이어 모두 이 구조를 공유합니다. 
전체 모델은 원시 입력(특성)을 받아 출력(예측)을 생성하고 파라미터(모든 구성 레이어의 결합된 파라미터)를 보유합니다. 
마찬가지로 각 개별 레이어는 입력(이전 레이어에서 제공)을 섭취하여 출력(후속 레이어에 대한 입력)을 생성하고, 후속 레이어에서 역방향으로 흐르는 신호에 따라 업데이트되는 조정 가능한 파라미터 세트를 보유합니다.


뉴런, 레이어, 모델이 우리 업무를 수행하기에 충분한 추상화를 제공한다고 생각할 수 있지만, 개별 레이어보다는 크지만 전체 모델보다는 작은 구성 요소에 대해 이야기하는 것이 편리할 때가 많습니다. 
예를 들어 컴퓨터 비전에서 매우 인기 있는 ResNet-152 아키텍처는 수백 개의 레이어를 보유하고 있습니다. 
이 레이어들은 *레이어 그룹*의 반복되는 패턴으로 구성됩니다. 이러한 네트워크를 한 번에 한 레이어씩 구현하는 것은 지루할 수 있습니다. 
이 우려는 단지 가상적인 것이 아닙니다. 이러한 디자인 패턴은 실제로 일반적입니다. 
위에서 언급한 ResNet 아키텍처는 인식 및 검출 모두에 대해 2015 ImageNet 및 COCO 컴퓨터 비전 대회에서 우승했으며 :cite:`He.Zhang.Ren.ea.2016` 많은 비전 작업에서 여전히 선호되는 아키텍처입니다. 
레이어가 다양한 반복 패턴으로 배열된 유사한 아키텍처는 이제 자연어 처리 및 음성을 포함한 다른 도메인에서도 어디에나 있습니다.

이러한 복잡한 네트워크를 구현하기 위해 신경망 *모듈(module)* 개념을 도입합니다. 
모듈은 단일 레이어, 여러 레이어로 구성된 구성 요소, 또는 전체 모델 자체를 설명할 수 있습니다! 
모듈 추상화로 작업하는 것의 한 가지 이점은 더 큰 아티팩트로 결합될 수 있으며 종종 재귀적으로 결합된다는 것입니다. 이것은 :numref:`fig_blocks`에 설명되어 있습니다. 필요에 따라 임의의 복잡도를 가진 모듈을 생성하는 코드를 정의함으로써, 놀랍도록 간결한 코드를 작성하면서도 복잡한 신경망을 구현할 수 있습니다.

![여러 레이어가 모듈로 결합되어 더 큰 모델의 반복 패턴을 형성합니다.](../img/blocks.svg)
:label:`fig_blocks`


프로그래밍 관점에서 모듈은 *클래스(class)*로 표현됩니다. 
모든 서브클래스는 입력을 출력으로 변환하는 순전파 메서드를 정의해야 하며 필요한 모든 파라미터를 저장해야 합니다. 
일부 모듈은 파라미터가 전혀 필요하지 않다는 점에 유의하십시오. 
마지막으로 모듈은 기울기 계산을 위해 역전파 메서드를 보유해야 합니다. 
다행히도 자체 모듈을 정의할 때 자동 미분(:numref:`sec_autograd`에서 소개됨)이 제공하는 무대 뒤의 마법 덕분에 파라미터와 순전파 메서드만 걱정하면 됩니다.

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
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**시작하기 위해 MLP를 구현하는 데 사용했던 코드**] (:numref:`sec_mlp`)를 다시 살펴봅니다. 
다음 코드는 256개 유닛과 ReLU 활성화가 있는 하나의 완전 연결 은닉층과, 10개 유닛(활성화 함수 없음)이 있는 완전 연결 출력 레이어로 구성된 네트워크를 생성합니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key는 jax.random.PRNGKey(random_seed)를 반환하는 d2l 저장 함수입니다
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
이 예제에서 우리는 `nn.Sequential`을 인스턴스화하여 모델을 구성하고 반환된 객체를 `net` 변수에 할당했습니다. 
다음으로 `add` 메서드를 반복적으로 호출하여 실행되어야 할 순서대로 레이어를 추가합니다. 
간단히 말해서 `nn.Sequential`은 Gluon에서 *모듈*을 나타내는 클래스인 `Block`의 특별한 종류를 정의합니다. 
구성 `Block`의 정렬된 목록을 유지합니다. 
`add` 메서드는 목록에 각 후속 `Block`을 추가하는 것을 용이하게 합니다. 
각 레이어는 `Dense` 클래스의 인스턴스이며 그 자체로 `Block`의 서브클래스라는 점에 유의하십시오. 
순전파(`forward`) 메서드도 놀랍도록 간단합니다. 목록에 있는 각 `Block`을 연결하여 각 출력을 다음 입력으로 전달합니다. 
지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출해 왔다는 점에 유의하십시오. 
실제로는 `Block` 클래스의 `__call__` 메서드를 통해 달성되는 멋진 Python 트릭인 `net.forward(X)`의 약어일 뿐입니다.
:end_tab:

:begin_tab:`pytorch`
이 예제에서 우리는 실행되어야 할 순서대로 레이어를 인수로 전달하여 `nn.Sequential`을 인스턴스화함으로써 모델을 구성했습니다. 
간단히 말해서, (**`nn.Sequential`은 특별한 종류의 `Module`을 정의합니다**). 
`Module`은 PyTorch에서 모듈을 나타내는 클래스입니다. 
구성 `Module`의 정렬된 목록을 유지합니다. 
두 완전 연결 레이어 각각은 `Linear` 클래스의 인스턴스이며 그 자체로 `Module`의 서브클래스라는 점에 유의하십시오. 
순전파(`forward`) 메서드도 놀랍도록 간단합니다. 목록에 있는 각 모듈을 연결하여 각 출력을 다음 입력으로 전달합니다. 
지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출해 왔다는 점에 유의하십시오. 
실제로는 `net.__call__(X)`의 약어일 뿐입니다.
:end_tab:

:begin_tab:`tensorflow`
이 예제에서 우리는 실행되어야 할 순서대로 레이어를 인수로 전달하여 `keras.models.Sequential`을 인스턴스화함으로써 모델을 구성했습니다. 
간단히 말해서, `Sequential`은 특별한 종류의 `keras.Model`을 정의합니다. 
`Model`은 Keras에서 모듈을 나타내는 클래스입니다. 
구성 `Model`의 정렬된 목록을 유지합니다. 
두 완전 연결 레이어 각각은 `Dense` 클래스의 인스턴스이며 그 자체로 `Model`의 서브클래스라는 점에 유의하십시오. 
순전파(`call`) 메서드도 놀랍도록 간단합니다. 목록에 있는 각 모듈을 연결하여 각 출력을 다음 입력으로 전달합니다. 
지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출해 왔다는 점에 유의하십시오. 
실제로는 모듈 클래스의 `__call__` 메서드를 통해 달성되는 멋진 Python 트릭인 `net.call(X)`의 약어일 뿐입니다.
:end_tab:

## [**사용자 정의 모듈 (A Custom Module)**]

모듈이 작동하는 방식에 대한 직관을 개발하는 가장 쉬운 방법은 아마도 직접 구현해 보는 것일 겁니다. 
그렇게 하기 전에, 각 모듈이 제공해야 하는 기본 기능을 간략하게 요약합니다:

1. 입력 데이터를 순전파 메서드의 인수로 섭취합니다.
2. 순전파 메서드가 값을 반환하도록 하여 출력을 생성합니다. 출력은 입력과 다른 모양을 가질 수 있습니다. 예를 들어 위 모델의 첫 번째 완전 연결 레이어는 임의 차원의 입력을 섭취하지만 차원 256의 출력을 반환합니다.
3. 역전파 메서드를 통해 액세스할 수 있는 입력에 대한 출력의 기울기를 계산합니다. 일반적으로 이는 자동으로 발생합니다.
4. 순전파 계산을 실행하는 데 필요한 파라미터를 저장하고 액세스를 제공합니다.
5. 필요에 따라 모델 파라미터를 초기화합니다.


다음 스니펫에서는 256개 은닉 유닛이 있는 하나의 은닉층과 10차원 출력 레이어가 있는 MLP에 해당하는 모듈을 밑바닥부터 코딩합니다. 
아래의 `MLP` 클래스는 모듈을 나타내는 클래스를 상속한다는 점에 유의하십시오. 
우리는 부모 클래스의 메서드에 크게 의존하며 자체 생성자(Python의 `__init__` 메서드)와 순전파 메서드만 제공할 것입니다.

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # 필요한 초기화를 수행하기 위해 MLP 부모 클래스 nn.Block의 생성자를 호출합니다
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # 모델의 순전파, 즉 입력 X를 기반으로 필요한 모델 출력을 반환하는 방법을 정의합니다
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # 필요한 초기화를 수행하기 위해 부모 클래스 nn.Module의 생성자를 호출합니다
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # 모델의 순전파, 즉 입력 X를 기반으로 필요한 모델 출력을 반환하는 방법을 정의합니다
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # 필요한 초기화를 수행하기 위해 부모 클래스 tf.keras.Model의 생성자를 호출합니다
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # 모델의 순전파, 즉 입력 X를 기반으로 필요한 모델 출력을 반환하는 방법을 정의합니다
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # 레이어 정의
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # 모델의 순전파, 즉 입력 X를 기반으로 필요한 모델 출력을 반환하는 방법을 정의합니다
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

먼저 순전파 메서드에 집중해 봅시다. 
`X`를 입력으로 받아 활성화 함수가 적용된 은닉 표현을 계산하고 로짓을 출력한다는 점에 유의하십시오. 
이 `MLP` 구현에서 두 레이어는 모두 인스턴스 변수입니다. 
이것이 왜 합리적인지 알기 위해 두 개의 MLP `net1`과 `net2`를 인스턴스화하고 서로 다른 데이터로 훈련한다고 상상해 보십시오. 
당연히 우리는 이들이 두 개의 서로 다른 학습된 모델을 나타낼 것으로 기대할 것입니다.

우리는 생성자에서 [**MLP의 레이어를 인스턴스화하고**] (**이후에**) 순전파 메서드를 호출할 때마다 (**이 레이어들을 호출합니다**). 
몇 가지 주요 세부 사항에 유의하십시오. 
먼저, 사용자 정의 `__init__` 메서드는 `super().__init__()`를 통해 부모 클래스의 `__init__` 메서드를 호출하여 대부분의 모듈에 적용되는 상용구 코드를 다시 작성하는 고통을 덜어줍니다. 
그런 다음 두 개의 완전 연결 레이어를 인스턴스화하여 `self.hidden`과 `self.out`에 할당합니다. 
새 레이어를 구현하지 않는 한 역전파 메서드나 파라미터 초기화에 대해 걱정할 필요가 없다는 점에 유의하십시오. 
시스템이 이러한 메서드를 자동으로 생성합니다. 
한 번 시도해 봅시다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

모듈 추상화의 주요 장점은 다용도성입니다. 
우리는 모듈을 서브클래스화하여 레이어(완전 연결 레이어 클래스 등), 전체 모델(위의 `MLP` 클래스 등) 또는 중간 복잡도의 다양한 구성 요소를 만들 수 있습니다. 
우리는 합성곱 신경망을 다룰 때와 같이 앞으로 나올 장들에서 이 다용도성을 활용할 것입니다.


## [**Sequential 모듈 (The Sequential Module)**]
:label:`subsec_model-construction-sequential`

이제 `Sequential` 클래스가 어떻게 작동하는지 자세히 살펴볼 수 있습니다. 
`Sequential`은 다른 모듈들을 데이지 체인(daisy-chain) 방식으로 연결하도록 설계되었음을 상기하십시오. 
우리만의 단순화된 `MySequential`을 구축하려면 두 가지 주요 메서드만 정의하면 됩니다:

1. 모듈을 하나씩 리스트에 추가하는 메서드.
2. 추가된 것과 동일한 순서로 모듈 체인을 통해 입력을 전달하는 순전파 메서드.

다음 `MySequential` 클래스는 기본 `Sequential` 클래스와 동일한 기능을 제공합니다.

```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # 여기서 block은 Block 서브클래스의 인스턴스이며 고유한 이름을 가지고 있다고 가정합니다.
        # Block 클래스의 멤버 변수 _children에 저장하며 그 타입은 OrderedDict입니다.
        # MySequential 인스턴스가 initialize 메서드를 호출하면 시스템은
        # _children의 모든 멤버를 자동으로 초기화합니다
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict는 멤버가 추가된 순서대로 순회될 것임을 보장합니다
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` 메서드는 단일 블록을 정렬된 딕셔너리 `_children`에 추가합니다. 
모든 Gluon `Block`이 왜 `_children` 속성을 가지고 있는지, 그리고 왜 우리가 직접 Python 리스트를 정의하지 않고 그것을 사용했는지 궁금할 수 있습니다. 
간단히 말해서 `_children`의 주된 장점은 블록의 파라미터 초기화 중에 Gluon이 `_children` 딕셔너리 내부를 살펴보고 파라미터도 초기화해야 하는 하위 블록을 찾는다는 것입니다.
:end_tab:

:begin_tab:`pytorch`
`__init__` 메서드에서 우리는 `add_modules` 메서드를 호출하여 모든 모듈을 추가합니다. 이 모듈들은 나중에 `children` 메서드로 액세스할 수 있습니다. 
이런 식으로 시스템은 추가된 모듈을 알게 되고 각 모듈의 파라미터를 적절하게 초기화할 것입니다.
:end_tab:

`MySequential`의 순전파 메서드가 호출되면 추가된 각 모듈이 추가된 순서대로 실행됩니다. 
이제 `MySequential` 클래스를 사용하여 MLP를 다시 구현할 수 있습니다.

```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

이 `MySequential` 사용법은 이전에 `Sequential` 클래스에 대해 작성한 코드(:numref:`sec_mlp`에 설명됨)와 동일합니다.


## [**순전파 메서드에서 코드 실행하기 (Executing Code in the Forward Propagation Method)**]

`Sequential` 클래스는 모델 구성을 쉽게 만들어주며, 우리만의 클래스를 정의할 필요 없이 새로운 아키텍처를 조립할 수 있게 해줍니다. 
그러나 모든 아키텍처가 단순한 데이지 체인인 것은 아닙니다. 
더 큰 유연성이 필요할 때는 우리만의 블록을 정의하고 싶을 것입니다. 
예를 들어 순전파 메서드 내에서 Python의 제어 흐름을 실행하고 싶을 수 있습니다. 
더욱이 미리 정의된 신경망 레이어에만 의존하지 않고 임의의 수학적 연산을 수행하고 싶을 수도 있습니다.

지금까지 우리 네트워크의 모든 연산이 네트워크의 활성화와 파라미터에 작용했다는 것을 눈치채셨을 겁니다. 
하지만 때로는 이전 레이어의 결과도 아니고 업데이트 가능한 파라미터도 아닌 항을 포함하고 싶을 때가 있습니다. 
우리는 이를 *상수 파라미터(constant parameters)*라고 부릅니다. 
예를 들어 함수 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^{\top} \mathbf{x}$를 계산하는 레이어를 원한다고 가정해 봅시다. 여기서 $\mathbf{x}$는 입력, $\mathbf{w}$는 파라미터, $c$는 최적화 중에 업데이트되지 않는 지정된 상수입니다. 
따라서 다음과 같이 `FixedHiddenMLP` 클래스를 구현합니다.

```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # get_constant 메서드로 생성된 무작위 가중치 파라미터는
        # 훈련 중에 업데이트되지 않습니다(즉, 상수 파라미터)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # 생성된 상수 파라미터와 relu 및 dot 함수 사용
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # 완전 연결 레이어 재사용. 이는 두 완전 연결 레이어와 파라미터를 공유하는 것과 동일합니다
        X = self.dense(X)
        # 제어 흐름
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 기울기를 계산하지 않아 훈련 중에 일정하게 유지되는 무작위 가중치 파라미터
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # 완전 연결 레이어 재사용. 이는 두 완전 연결 레이어와 파라미터를 공유하는 것과 동일합니다
        X = self.linear(X)
        # 제어 흐름
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # tf.constant로 생성된 무작위 가중치 파라미터는
        # 훈련 중에 업데이트되지 않습니다(즉, 상수 파라미터)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # 생성된 상수 파라미터와 relu 및 matmul 함수 사용
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 완전 연결 레이어 재사용. 이는 두 완전 연결 레이어와 파라미터를 공유하는 것과 동일합니다
        X = self.dense(X)
        # 제어 흐름
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # 기울기를 계산하지 않아 훈련 중에 일정하게 유지되는 무작위 가중치 파라미터
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # 완전 연결 레이어 재사용. 이는 두 완전 연결 레이어와 파라미터를 공유하는 것과 동일합니다
        X = self.dense(X)
        # 제어 흐름
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

이 모델에서 우리는 인스턴스화 시 무작위로 초기화되고 이후에는 일정하게 유지되는 가중치(`self.rand_weight`)를 가진 은닉층을 구현합니다. 
이 가중치는 모델 파라미터가 아니므로 역전파에 의해 업데이트되지 않습니다. 
그런 다음 네트워크는 이 "고정된" 레이어의 출력을 완전 연결 레이어를 통해 전달합니다.

출력을 반환하기 전에 우리 모델이 특이한 작업을 수행했다는 점에 유의하십시오. 
우리는 while 루프를 실행하여 $\ell_1$ 노름이 1보다 크다는 조건을 테스트하고, 조건을 만족할 때까지 출력 벡터를 2로 나누었습니다. 
마지막으로 `X` 항목의 합계를 반환했습니다. 
우리가 아는 한, 이 연산을 수행하는 표준 신경망은 없습니다. 
이 특정 작업은 실제 작업에서 유용하지 않을 수 있습니다. 
우리의 요점은 신경망 계산 흐름에 임의의 코드를 통합하는 방법을 보여주는 것뿐입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

우리는 [**모듈을 조립하는 다양한 방법을 혼합하고 일치**]시킬 수 있습니다. 
다음 예제에서는 모듈을 창의적인 방식으로 중첩합니다.

```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## 요약 (Summary)

개별 레이어는 모듈이 될 수 있습니다.
많은 레이어가 모듈을 구성할 수 있습니다.
많은 모듈이 모듈을 구성할 수 있습니다.

모듈은 코드를 포함할 수 있습니다.
모듈은 파라미터 초기화 및 역전파를 포함한 많은 관리 작업을 처리합니다.
레이어와 모듈의 순차적 연결은 `Sequential` 모듈에 의해 처리됩니다.


## 연습 문제 (Exercises)

1. `MySequential`을 변경하여 모듈을 Python 리스트에 저장하면 어떤 종류의 문제가 발생합니까?
2. 두 개의 모듈을 인수로 받아, 예를 들어 `net1`과 `net2`, 순전파에서 두 네트워크의 연결된 출력을 반환하는 모듈을 구현하십시오. 이를 *병렬 모듈(parallel module)*이라고도 합니다.
3. 동일한 네트워크의 여러 인스턴스를 연결하고 싶다고 가정해 봅시다. 동일한 모듈의 여러 인스턴스를 생성하는 팩토리 함수를 구현하고 그로부터 더 큰 네트워크를 구축하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/264)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17989)
:end_tab:

```