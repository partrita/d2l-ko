```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 구현을 위한 객체 지향 설계 (Object-Oriented Design for Implementation)
:label:`sec_oo-design`

선형 회귀 소개에서 우리는 데이터, 모델, 손실 함수, 최적화 알고리즘을 포함한 다양한 구성 요소를 살펴보았습니다. 
사실 선형 회귀는 가장 단순한 머신러닝 모델 중 하나입니다. 
하지만 이를 훈련하는 데는 이 책의 다른 모델들이 요구하는 것과 동일한 구성 요소가 많이 사용됩니다. 
따라서 구현 세부 사항을 깊이 파고들기 전에, 우리가 전체적으로 사용하는 일부 API를 설계하는 것이 가치가 있습니다. 
딥러닝의 구성 요소를 객체로 취급하여, 이러한 객체와 그 상호 작용을 위한 클래스를 정의하는 것부터 시작할 수 있습니다. 
구현을 위한 이러한 객체 지향 설계는 설명을 크게 간소화할 것이며, 여러분의 프로젝트에서도 사용하고 싶어질 것입니다.


[PyTorch Lightning](https://www.pytorchlightning.ai/)과 같은 오픈 소스 라이브러리에서 영감을 받아, 높은 수준에서 세 가지 클래스를 갖고자 합니다: 
(i) `Module`은 모델, 손실, 최적화 방법을 포함합니다; 
(ii) `DataModule`은 훈련 및 검증을 위한 데이터 로더를 제공합니다; 
(iii) 두 클래스는 다양한 하드웨어 플랫폼에서 모델을 훈련할 수 있게 해주는 `Trainer` 클래스를 사용하여 결합됩니다. 
이 책의 대부분의 코드는 `Module`과 `DataModule`을 조정합니다. `Trainer` 클래스는 GPU, CPU, 병렬 훈련, 최적화 알고리즘을 논의할 때만 언급할 것입니다.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## 유틸리티 (Utilities)
:label:`oo-design-utilities`

Jupyter 노트북에서 객체 지향 프로그래밍을 단순화하기 위해 몇 가지 유틸리티가 필요합니다. 한 가지 어려운 점은 클래스 정의가 상당히 긴 코드 블록이 되는 경향이 있다는 것입니다. 노트북 가독성을 위해서는 설명이 섞인 짧은 코드 조각이 요구되는데, 이는 Python 라이브러리에서 흔히 볼 수 있는 프로그래밍 스타일과 호환되지 않습니다. 첫 번째 유틸리티 함수를 사용하면 클래스가 생성된 *후에* 함수를 클래스의 메서드로 등록할 수 있습니다. 사실, 클래스의 인스턴스를 생성한 *후에도* 그렇게 할 수 있습니다! 이를 통해 클래스의 구현을 여러 코드 블록으로 나눌 수 있습니다.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """함수를 생성된 클래스의 메서드로 등록합니다."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

사용법을 빠르게 살펴봅시다. `do` 메서드를 가진 클래스 `A`를 구현하려고 합니다. 동일한 코드 블록에 `A`와 `do`에 대한 코드를 모두 넣는 대신, 먼저 클래스 `A`를 선언하고 인스턴스 `a`를 생성할 수 있습니다.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

다음으로 평소처럼 `do` 메서드를 정의하되, 클래스 `A`의 범위 내에서 정의하지 않습니다. 대신, 이 메서드를 클래스 `A`를 인수로 하는 `add_to_class`로 데코레이션합니다. 그렇게 함으로써, 메서드는 마치 `A` 정의의 일부로 포함된 것처럼 `A`의 멤버 변수에 액세스할 수 있습니다. 인스턴스 `a`에 대해 호출할 때 어떤 일이 일어나는지 봅시다.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('클래스 속성 "b"는', self.b)

a.do()
```

두 번째는 클래스의 `__init__` 메서드에 있는 모든 인수를 클래스 속성으로 저장하는 유틸리티 클래스입니다. 이를 통해 추가 코드 없이 생성자 호출 서명을 암시적으로 확장할 수 있습니다.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """하이퍼파라미터의 기본 클래스입니다."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

이 구현은 :numref:`sec_utils`로 미룹니다. 이를 사용하기 위해 `HyperParameters`를 상속받고 `__init__` 메서드에서 `save_hyperparameters`를 호출하는 클래스를 정의합니다.

```{.python .input}
%%tab all
# d2l에 저장된 완전히 구현된 HyperParameters 클래스를 호출합니다.
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('self.c가 존재하지 않음 =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

마지막 유틸리티를 사용하면 실험이 진행되는 동안 대화식으로 실험 진행 상황을 플롯할 수 있습니다. 훨씬 더 강력하고 복잡한 [TensorBoard](https://www.tensorflow.org/tensorboard)에 대한 경의를 표하며 이름을 `ProgressBoard`라고 명명했습니다. 구현은 :numref:`sec_utils`로 미룹니다. 지금은 단순히 작동하는 모습을 봅시다.

`draw` 메서드는 그림에 점 `(x, y)`를 플롯하며, 범례에 `label`이 지정됩니다. 선택적 인수 `every_n`은 그림에 $1/n$개의 점만 표시하여 선을 부드럽게 만듭니다. 그 값은 원래 그림의 $n$개 이웃 점의 평균입니다.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """데이터 포인트를 애니메이션으로 플롯하는 보드입니다."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

다음 예제에서는 서로 다른 부드러움으로 `sin`과 `cos`을 그립니다. 이 코드 블록을 실행하면 선이 애니메이션으로 자라나는 것을 볼 수 있습니다.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## 모델 (Models)
:label:`subsec_oo-design-models`

`Module` 클래스는 우리가 구현할 모든 모델의 기본 클래스입니다. 최소한 세 가지 메서드가 필요합니다. 첫 번째 `__init__`은 학습 가능한 파라미터를 저장하고, `training_step` 메서드는 데이터 배치를 받아 손실 값을 반환하며, 마지막으로 `configure_optimizers`는 학습 가능한 파라미터를 업데이트하는 데 사용되는 최적화 방법 또는 그 리스트를 반환합니다. 선택적으로 평가 측정치를 보고하기 위해 `validation_step`을 정의할 수 있습니다. 
때때로 우리는 출력을 계산하는 코드를 더 재사용하기 쉽게 별도의 `forward` 메서드에 넣습니다.

:begin_tab:`jax`
Python 3.7에서 [dataclasses](https://docs.python.org/3/library/dataclasses.html)가 도입되면서, `@dataclass`로 데코레이션된 클래스는 `__init__` 및 `__repr__`과 같은 매직 메서드를 자동으로 추가합니다. 멤버 변수는 유형 주석(type annotations)을 사용하여 정의됩니다. 모든 Flax 모듈은 Python 3.7 데이터 클래스입니다.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """모델의 기본 클래스입니다."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), '신경망이 정의되었습니다.'
        return self.net(X)

    def plot(self, key, value, train):
        """점을 애니메이션으로 플롯합니다."""
        assert hasattr(self, 'trainer'), 'Trainer가 초기화되지 않았습니다.'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())), 
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """모델의 기본 클래스입니다."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # Python 데이터 클래스를 사용할 때는 save_hyperparam이 필요하지 않습니다.
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # 매 실행마다 새 플롯이 생성되도록 default_factory를 사용합니다.
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), '신경망이 정의되었습니다.'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX 및 Flax에는 forward-method와 같은 구문이 없습니다. Flax는 순방향 패스를 위해 setup과
        # 내장 __call__ 매직 메서드를 사용합니다. 일관성을 위해 여기에 추가합니다.
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), '신경망이 정의되었습니다.'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """점을 애니메이션으로 플롯합니다."""
        assert hasattr(self, 'trainer'), 'Trainer가 초기화되지 않았습니다.'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """나중에 :numref:`sec_lazy_init`에서 정의될 예정입니다."""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
`Module`이 Gluon의 신경망 기본 클래스인 `nn.Block`의 서브클래스임을 알 수 있습니다. 
이는 신경망을 처리하는 데 편리한 기능을 제공합니다. 예를 들어, `forward(self, X)`와 같은 `forward` 메서드를 정의하면 인스턴스 `a`에 대해 `a(X)`로 이 메서드를 호출할 수 있습니다. 이는 내장된 `__call__` 메서드에서 `forward` 메서드를 호출하기 때문입니다. `nn.Block`에 대한 자세한 내용과 예제는 :numref:`sec_model_construction`에서 확인할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
`Module`이 PyTorch의 신경망 기본 클래스인 `nn.Module`의 서브클래스임을 알 수 있습니다. 
이는 신경망을 처리하는 데 편리한 기능을 제공합니다. 예를 들어, `forward(self, X)`와 같은 `forward` 메서드를 정의하면 인스턴스 `a`에 대해 `a(X)`로 이 메서드를 호출할 수 있습니다. 이는 내장된 `__call__` 메서드에서 `forward` 메서드를 호출하기 때문입니다. `nn.Module`에 대한 자세한 내용과 예제는 :numref:`sec_model_construction`에서 확인할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
`Module`이 TensorFlow의 신경망 기본 클래스인 `tf.keras.Model`의 서브클래스임을 알 수 있습니다. 
이는 신경망을 처리하는 데 편리한 기능을 제공합니다. 예를 들어, 내장된 `__call__` 메서드에서 `call` 메서드를 호출합니다. 여기서는 `call`을 `forward` 메서드로 리디렉션하고 그 인수를 클래스 속성으로 저장합니다. 우리는 코드를 다른 프레임워크 구현과 더 유사하게 만들기 위해 이렇게 합니다.
:end_tab:

:begin_tab:`jax`
`Module`이 Flax의 신경망 기본 클래스인 `linen.Module`의 서브클래스임을 알 수 있습니다. 
이는 신경망을 처리하는 데 편리한 기능을 제공합니다. 예를 들어, 모델 파라미터를 처리하고, 코드를 단순화하기 위해 `nn.compact` 데코레이터를 제공하며, `__call__` 메서드 등을 호출합니다. 
여기서도 `__call__`을 `forward` 메서드로 리디렉션합니다. 우리는 코드를 다른 프레임워크 구현과 더 유사하게 만들기 위해 이렇게 합니다.
:end_tab:

## 데이터 (Data)
:label:`oo-design-data`

`DataModule` 클래스는 데이터의 기본 클래스입니다. 상당히 자주 `__init__` 메서드는 데이터를 준비하는 데 사용됩니다. 여기에는 필요한 경우 다운로드 및 전처리가 포함됩니다. `train_dataloader`는 훈련 데이터셋에 대한 데이터 로더를 반환합니다. 데이터 로더는 사용될 때마다 데이터 배치를 생성하는 (Python) 제너레이터입니다. 이 배치는 손실을 계산하기 위해 `Module`의 `training_step` 메서드에 공급됩니다. 검증 데이터셋 로더를 반환하는 선택적인 `val_dataloader`가 있습니다. 이는 `Module`의 `validation_step` 메서드에 대한 데이터 배치를 생성한다는 점을 제외하고는 동일한 방식으로 동작합니다.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """데이터의 기본 클래스입니다."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## 훈련 (Training)
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
`Trainer` 클래스는 `DataModule`에 지정된 데이터를 사용하여 `Module` 클래스의 학습 가능한 파라미터를 훈련합니다. 핵심 메서드는 `fit`으로, `Module`의 인스턴스인 `model`과 `DataModule`의 인스턴스인 `data`라는 두 개의 인수를 받습니다. 그런 다음 모델을 훈련하기 위해 전체 데이터셋을 `max_epochs`번 반복합니다. 이전과 마찬가지로, 이 메서드의 구현은 나중 장으로 미룰 것입니다.
:end_tab:

:begin_tab:`jax`
`Trainer` 클래스는 `DataModule`에 지정된 데이터를 사용하여 학습 가능한 파라미터 `params`를 훈련합니다. 핵심 메서드는 `fit`으로, `Module`의 인스턴스인 `model`, `DataModule`의 인스턴스인 `data`, 그리고 JAX `PRNGKeyArray`인 `key`라는 세 개의 인수를 받습니다. 여기서는 인터페이스를 단순화하기 위해 `key` 인수를 선택 사항으로 만들었지만, JAX와 Flax에서 모델 파라미터를 항상 루트 키로 전달하고 초기화하는 것이 좋습니다. 그런 다음 모델을 훈련하기 위해 전체 데이터셋을 `max_epochs`번 반복합니다. 이전과 마찬가지로, 이 메서드의 구현은 나중 장으로 미룰 것입니다.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """데이터로 모델을 훈련하기 위한 기본 클래스입니다."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, '아직 GPU를 지원하지 않습니다.'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # 여기서 batch_stats는 나중에 사용됩니다(예: 배치 정규화)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax는 내부적으로 단일 상태 객체 TrainState를 위해 optax를 사용합니다.
            # 드롭아웃 및 배치 정규화 섹션에서 자세히 논의될 것입니다.
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## 요약 (Summary)

향후 딥러닝 구현을 위한 객체 지향 설계를 강조하기 위해, 위의 클래스들은 단순히 그 객체들이 어떻게 데이터를 저장하고 서로 상호 작용하는지를 보여줍니다. 우리는 책의 나머지 부분에서 `@add_to_class` 등을 통해 이러한 클래스들의 구현을 계속 풍부하게 만들 것입니다. 
게다가, 완전히 구현된 이러한 클래스들은 [D2L 라이브러리](https://github.com/d2l-ai/d2l-en/tree/master/d2l)에 저장되어 있으며, 이는 딥러닝을 위한 구조화된 모델링을 쉽게 만들어주는 *경량 툴킷*입니다. 특히, 프로젝트 간에 거의 아무것도 바꾸지 않고도 많은 구성 요소를 재사용할 수 있게 해줍니다. 예를 들어, 최적화기만, 모델만, 또는 데이터셋만 교체할 수 있습니다. 이러한 정도의 모듈성은 간결함과 단순함 측면에서 책 전체에 걸쳐 이점을 제공하며(이것이 우리가 이를 추가한 이유입니다), 여러분의 프로젝트에서도 동일한 역할을 할 수 있습니다. 


## 연습 문제 (Exercises)

1. [D2L 라이브러리](https://github.com/d2l-ai/d2l-en/tree/master/d2l)에 저장된 위 클래스들의 전체 구현을 찾아보십시오. 딥러닝 모델링에 좀 더 익숙해지면 구현을 자세히 살펴볼 것을 강력히 권장합니다.
2. `B` 클래스에서 `save_hyperparameters` 문을 제거하십시오. 여전히 `self.a`와 `self.b`를 출력할 수 있습니까? 선택 사항: `HyperParameters` 클래스의 전체 구현을 깊이 파고들었다면 이유를 설명할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17974)
:end_tab:

```