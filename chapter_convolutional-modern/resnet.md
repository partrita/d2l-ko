```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 잔차 네트워크 (ResNet)와 ResNeXt (Residual Networks (ResNet) and ResNeXt)
:label:`sec_resnet`

점점 더 깊은 네트워크를 설계함에 따라 레이어를 추가하는 것이 네트워크의 복잡성과 표현력을 어떻게 증가시킬 수 있는지 이해하는 것이 필수적이 되었습니다. 
더 중요한 것은 레이어를 추가하는 것이 네트워크를 단순히 다르게 만드는 것이 아니라 엄격하게 더 표현력 있게 만드는 네트워크를 설계하는 능력입니다. 
약간의 진전을 이루려면 약간의 수학이 필요합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## 함수 클래스 (Function Classes)

특정 네트워크 아키텍처(학습률 및 기타 하이퍼파라미터 설정과 함께)가 도달할 수 있는 함수 클래스 $\mathcal{F}$를 고려하십시오. 
즉, 모든 $f \in \mathcal{F}$에 대해 적절한 데이터셋에 대한 훈련을 통해 얻을 수 있는 파라미터 세트(예: 가중치 및 편향)가 존재합니다. 
$f^*$가 우리가 정말로 찾고 싶은 "진실" 함수라고 가정해 봅시다. 
그것이 $\mathcal{F}$에 있다면 우리는 좋은 상태이지만 일반적으로 그렇게 운이 좋지는 않을 것입니다. 
대신 우리는 $\mathcal{F}$ 내에서 최선의 선택인 $f^*_\mathcal{F}$를 찾으려고 노력할 것입니다. 
예를 들어, 
특성 $\mathbf{X}$와 레이블 $\mathbf{y}$가 있는 데이터셋이 주어지면, 
우리는 다음 최적화 문제를 해결하여 그것을 찾으려고 시도할 수 있습니다:

$$f^*_\mathcal{F} \stackrel{\textrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \textrm{ subject to } f \in \mathcal{F}.$$ 

우리는 정규화 :cite:`tikhonov1977solutions,morozov2012methods`가 $\mathcal{F}$의 복잡도를 제어하고 일관성을 달성할 수 있음을 알고 있으므로, 더 큰 크기의 훈련 데이터는 일반적으로 더 나은 $f^*_\mathcal{F}$로 이어집니다. 
우리가 다른 더 강력한 아키텍처 $\mathcal{F}'$를 설계하면 더 나은 결과를 얻어야 한다고 가정하는 것이 합리적입니다. 즉, 우리는 $f^*_{\mathcal{F}'}$가 $f^*_{\mathcal{F}}$보다 "더 낫기"를 기대합니다. 그러나 $\mathcal{F} \not\subseteq \mathcal{F}'$이면 이런 일이 일어날 것이라는 보장이 없습니다. 사실 $f^*_{\mathcal{F}'}$는 더 나쁠 수도 있습니다. 
:numref:`fig_functionclasses`에서 설명한 것처럼, 중첩되지 않은 함수 클래스의 경우 더 큰 함수 클래스가 항상 "진실" 함수 $f^*$에 더 가까이 이동하는 것은 아닙니다. 예를 들어, 
:numref:`fig_functionclasses`의 왼쪽에서 $\mathcal{F}_3$은 $\mathcal{F}_1$보다 $f^*$에 가깝지만, $\mathcal{F}_6$은 멀어지고 복잡도를 더 높이면 $f^*$와의 거리를 줄일 수 있다는 보장이 없습니다. 
:numref:`fig_functionclasses`의 오른쪽과 같이 $\mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_6$인 중첩 함수 클래스를 사용하면 중첩되지 않은 함수 클래스의 앞서 언급한 문제를 피할 수 있습니다.


![중첩되지 않은 함수 클래스의 경우, 더 큰(영역으로 표시됨) 함수 클래스가 "진실" 함수($\mathit{f}^*$)에 더 가까워진다는 보장이 없습니다. 중첩 함수 클래스에서는 이런 일이 발생하지 않습니다.](../img/functionclasses.svg)
:label:`fig_functionclasses`

따라서 더 큰 함수 클래스가 더 작은 함수 클래스를 포함하는 경우에만 이를 늘리면 네트워크의 표현력이 엄격하게 증가한다는 것이 보장됩니다. 
심층 신경망의 경우, 
새로 추가된 레이어를 항등 함수 $f(\mathbf{x}) = \mathbf{x}$로 훈련할 수 있다면 새 모델은 원래 모델만큼 효과적일 것입니다. 새 모델이 훈련 데이터셋에 맞는 더 나은 솔루션을 얻을 수 있으므로, 추가된 레이어는 훈련 오류를 줄이는 것을 더 쉽게 만들 수 있습니다.

이것은 :citet:`He.Zhang.Ren.ea.2016`가 매우 깊은 컴퓨터 비전 모델을 작업할 때 고려한 질문이었습니다. 
제안된 *잔차 네트워크* (*ResNet*)의 핵심에는 모든 추가 레이어가 그 요소 중 하나로 항등 함수를 더 쉽게 포함해야 한다는 아이디어가 있습니다. 
이러한 고려 사항은 다소 심오하지만 *잔차 블록(residual block)*이라는 놀랍도록 간단한 해결책으로 이어졌습니다. 
이를 통해 ResNet은 2015년 ImageNet 대규모 시각 인식 챌린지에서 우승했습니다. 이 디자인은 심층 신경망을 구축하는 방법에 지대한 영향을 미쳤습니다. 예를 들어 잔차 블록은 순환 네트워크에 추가되었습니다 :cite:`prakash2016neural,kim2017residual`. 마찬가지로 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`는 이를 사용하여 많은 레이어의 네트워크를 효율적으로 쌓습니다. 그래프 신경망 :cite:`Kipf.Welling.2016`에도 사용되며 기본 개념으로서 컴퓨터 비전에서 광범위하게 사용되었습니다 :cite:`Redmon.Farhadi.2018,Ren.He.Girshick.ea.2015`. 
잔차 네트워크 이전에 고속도로 네트워크(highway networks) :cite:`srivastava2015highway`가 있었는데, 항등 함수 주변의 우아한 파라미터화는 없지만 동기의 일부를 공유합니다.


## (**잔차 블록**)
:label:`subsec_residual-blks`

:numref:`fig_residual_block`에 묘사된 대로 신경망의 국소 부분에 집중해 봅시다. 입력을 $\mathbf{x}$로 표시합니다. 
우리는 학습을 통해 얻고자 하는 기저 매핑 $f(\mathbf{x})$가 상단의 활성화 함수에 입력으로 사용된다고 가정합니다. 
왼쪽에서, 
점선 상자 안의 부분은 $f(\mathbf{x})$를 직접 학습해야 합니다. 
오른쪽에서, 
점선 상자 안의 부분은 *잔차 매핑(residual mapping)* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$를 학습해야 하며, 여기서 잔차 블록이라는 이름이 유래했습니다. 
항등 매핑 $f(\mathbf{x}) = \mathbf{x}$가 원하는 기저 매핑인 경우, 잔차 매핑은 $g(\mathbf{x}) = 0$이 되므로 학습하기가 더 쉽습니다: 
점선 상자 안의 상부 가중치 레이어(예: 완전 연결 레이어 및 합성곱 레이어)의 가중치와 편향을 0으로 밀어 넣기만 하면 됩니다. 
오른쪽 그림은 ResNet의 *잔차 블록*을 보여주며, 레이어 입력 $\mathbf{x}$를 덧셈 연산자로 전달하는 실선을 *잔차 연결(residual connection)* (또는 *숏컷 연결(shortcut connection)*)이라고 합니다. 
잔차 블록을 사용하면 입력이 레이어를 가로질러 잔차 연결을 통해 더 빠르게 순전파될 수 있습니다. 
사실, 
잔차 블록은 다중 분기 Inception 블록의 특수한 경우로 생각할 수 있습니다: 
그중 하나가 항등 매핑인 두 개의 분기가 있습니다.

![일반 블록(왼쪽)에서 점선 상자 안의 부분은 매핑 $\mathit{f}(\mathbf{x})$를 직접 학습해야 합니다. 잔차 블록(오른쪽)에서 점선 상자 안의 부분은 잔차 매핑 $\mathit{g}(\mathbf{x}) = \mathit{f}(\mathbf{x}) - \mathbf{x}$를 학습해야 하므로 항등 매핑 $\mathit{f}(\mathbf{x}) = \mathbf{x}$를 학습하기가 더 쉽습니다.](../img/residual-block.svg)
:label:`fig_residual_block`


ResNet은 VGG의 전체 $3\times 3$ 합성곱 레이어 설계를 가지고 있습니다. 잔차 블록에는 동일한 수의 출력 채널을 가진 두 개의 $3\times 3$ 합성곱 레이어가 있습니다. 각 합성곱 레이어 뒤에는 배치 정규화 레이어와 ReLU 활성화 함수가 옵니다. 그런 다음 이 두 합성곱 연산을 건너뛰고 최종 ReLU 활성화 함수 바로 앞에 입력을 직접 더합니다. 
이러한 종류의 설계는 두 합성곱 레이어의 출력이 입력과 동일한 모양이어야 함께 더할 수 있음을 요구합니다. 채널 수를 변경하려면 덧셈 연산을 위해 입력을 원하는 모양으로 변환하는 추가 $1\times 1$ 합성곱 레이어를 도입해야 합니다. 아래 코드를 살펴보겠습니다.

```{.python .input}
%%tab mxnet
class Residual(nn.Block):  #@save
    """ResNet 모델의 잔차 블록."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class Residual(nn.Module):  #@save
    """ResNet 모델의 잔차 블록."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
%%tab tensorflow
class Residual(tf.keras.Model):  #@save
    """ResNet 모델의 잔차 블록."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
%%tab jax
class Residual(nn.Module):  #@save
    """ResNet 모델의 잔차 블록."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

이 코드는 두 가지 유형의 네트워크를 생성합니다: `use_1x1conv=False`일 때마다 ReLU 비선형성을 적용하기 전에 입력을 출력에 더하는 것과, 더하기 전에 $1 \times 1$ 합성곱을 통해 채널과 해상도를 조정하는 것입니다. :numref:`fig_resnet_block`은 이를 보여줍니다.

![덧셈 연산을 위해 입력을 원하는 모양으로 변환하는 $1 \times 1$ 합성곱이 있거나 없는 ResNet 블록.](../img/resnet-block.svg)
:label:`fig_resnet_block`

이제 $1 \times 1$ 합성곱이 필요하지 않은 [**입력과 출력의 모양이 같은 상황**]을 살펴보겠습니다.

```{.python .input}
%%tab mxnet, pytorch
if tab.selected('mxnet'):
    blk = Residual(3)
    blk.initialize()
if tab.selected('pytorch'):
    blk = Residual(3)
X = d2l.randn(4, 3, 6, 6)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = Residual(3)
X = d2l.normal((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = Residual(3)
X = jax.random.normal(d2l.get_key(), (4, 6, 6, 3))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

[**출력 채널 수를 늘리면서 출력 높이와 너비를 절반으로 줄이는**] 옵션도 있습니다. 
이 경우 `use_1x1conv=True`를 통해 $1 \times 1$ 합성곱을 사용합니다. 이것은 `strides=2`를 통해 공간 차원을 줄이기 위해 각 ResNet 블록의 시작 부분에서 유용합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
if tab.selected('mxnet'):
    blk.initialize()
blk(X).shape
```

```{.python .input}
%%tab jax
blk = Residual(6, use_1x1conv=True, strides=(2, 2))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## [**ResNet 모델**]

ResNet의 처음 두 레이어는 앞서 설명한 GoogLeNet의 것과 동일합니다: 64개의 출력 채널과 스트라이드 2가 있는 $7\times 7$ 합성곱 레이어 다음에는 스트라이드 2가 있는 $3\times 3$ 최대 풀링 레이어가 이어집니다. 차이점은 ResNet의 각 합성곱 레이어 뒤에 추가된 배치 정규화 레이어입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class ResNet(d2l.Classifier):
    arch: tuple
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                  padding='same')])
```

GoogLeNet은 Inception 블록으로 구성된 4개의 모듈을 사용합니다. 
그러나 ResNet은 잔차 블록으로 구성된 4개의 모듈을 사용하며, 각 모듈은 동일한 수의 출력 채널을 가진 여러 잔차 블록을 사용합니다. 
첫 번째 모듈의 채널 수는 입력 채널 수와 동일합니다. 스트라이드 2의 최대 풀링 레이어가 이미 사용되었으므로 높이와 너비를 줄일 필요가 없습니다. 후속 각 모듈의 첫 번째 잔차 블록에서 채널 수는 이전 모듈의 채널 수에 비해 두 배가 되고 높이와 너비는 반으로 줄어듭니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab jax
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True,
                                strides=(2, 2), training=self.training))
        else:
            blk.append(Residual(num_channels, training=self.training))
    return nn.Sequential(blk)
```

그런 다음 모든 모듈을 ResNet에 추가합니다. 여기서 각 모듈에는 두 개의 잔차 블록이 사용됩니다. 마지막으로 GoogLeNet과 마찬가지로 전역 평균 풀링 레이어를 추가하고 완전 연결 레이어 출력을 추가합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
# %%tab jax
@d2l.add_to_class(ResNet)
def create_net(self):
    net = nn.Sequential([self.b1()])
    for i, b in enumerate(self.arch):
        net.layers.extend([self.block(*b, first_block=(i==0))])
    net.layers.extend([nn.Sequential([
        # Flax는 GlobalAvg2D 레이어를 제공하지 않습니다
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

각 모듈에는 4개의 합성곱 레이어가 있습니다($1\times 1$ 합성곱 레이어 제외). 첫 번째 $7\times 7$ 합성곱 레이어와 마지막 완전 연결 레이어와 함께 총 18개의 레이어가 있습니다. 따라서 이 모델은 일반적으로 ResNet-18로 알려져 있습니다. 
모듈에서 채널 수와 잔차 블록 수를 다르게 구성하여 더 깊은 152-레이어 ResNet-152와 같은 다양한 ResNet 모델을 만들 수 있습니다. ResNet의 주요 아키텍처는 GoogLeNet과 유사하지만 ResNet의 구조는 더 간단하고 수정하기 쉽습니다. 이러한 모든 요인으로 인해 ResNet은 빠르고 광범위하게 사용되었습니다. :numref:`fig_resnet18`은 전체 ResNet-18을 보여줍니다.

![ResNet-18 아키텍처.](../img/resnet18-90.svg)
:label:`fig_resnet18`

ResNet을 훈련하기 전에 [**ResNet의 다양한 모듈에서 입력 모양이 어떻게 변하는지 관찰**]해 보겠습니다. 모든 이전 아키텍처와 마찬가지로 전역 평균 풀링 레이어가 모든 특성을 집계하는 지점까지 해상도는 감소하는 반면 채널 수는 증가합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)
```

```{.python .input}
%%tab jax
class ResNet18(ResNet):
    arch: tuple = ((2, 64), (2, 128), (2, 256), (2, 512))
    lr: float = 0.1
    num_classes: int = 10
```

```{.python .input}
%%tab pytorch, mxnet
ResNet18().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
ResNet18().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
ResNet18(training=False).layer_summary((1, 96, 96, 1))
```

## [**훈련 (Training)**]

이전과 마찬가지로 Fashion-MNIST 데이터셋에서 ResNet을 훈련합니다. ResNet은 꽤 강력하고 유연한 아키텍처입니다. 훈련 및 검증 손실을 포착한 플롯은 두 그래프 사이에 상당한 격차를 보여주며 훈련 손실이 훨씬 낮습니다. 이 정도의 유연성을 가진 네트워크의 경우, 더 많은 훈련 데이터가 격차를 줄이고 정확도를 높이는 데 뚜렷한 이점을 제공할 것입니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = ResNet18(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = ResNet18(lr=0.01)
    trainer.fit(model, data)
```

## ResNeXt
:label:`subsec_resnext`

ResNet 설계에서 직면하는 과제 중 하나는 주어진 블록 내에서 비선형성과 차원성 사이의 트레이드오프입니다. 즉, 레이어 수를 늘리거나 합성곱의 너비를 늘려 비선형성을 더할 수 있습니다. 대안 전략은 블록 간에 정보를 전달할 수 있는 채널 수를 늘리는 것입니다. 불행히도 후자는 $c_\textrm{i}$ 채널을 섭취하고 $c_\textrm{o}$ 채널을 방출하는 계산 비용이 $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$에 비례하므로 이차적 페널티가 따릅니다(:numref:`sec_channels`의 논의 참조).

:numref:`fig_inception`의 Inception 블록에서 영감을 얻을 수 있습니다. 정보가 별도의 그룹으로 블록을 통해 흐릅니다. 다중 독립 그룹 아이디어를 :numref:`fig_resnet_block`의 ResNet 블록에 적용하여 ResNeXt의 설계가 이루어졌습니다 :cite:`Xie.Girshick.Dollar.ea.2017`.
다른 점은 Inception의 뒤죽박죽 변환과 달리, 
ResNeXt는 모든 분기에서 *동일한* 변환을 채택하여 각 분기의 수동 튜닝 필요성을 최소화합니다.

![ResNeXt 블록. $\mathit{g}$ 그룹이 있는 그룹화된 합성곱을 사용하는 것은 조밀한 합성곱보다 $\mathit{g}$배 빠릅니다. 중간 채널 수 $\mathit{b}$가 $\mathit{c}$보다 작으면 병목 잔차 블록입니다.](../img/resnext-block.svg)
:label:`fig_resnext_block`

$c_\textrm{i}$에서 $c_\textrm{o}$ 채널로의 합성곱을 $c_\textrm{i}/g$ 크기의 $g$ 그룹 중 하나로 나누어 $c_\textrm{o}/g$ 크기의 $g$ 출력을 생성하는 것을 적절하게도 *그룹화된 합성곱(grouped convolution)*이라고 합니다. 계산 비용(비례적으로)은 $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$에서 $\mathcal{O}(g \cdot (c_\textrm{i}/g) \cdot (c_\textrm{o}/g)) = \mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$로 줄어듭니다. 즉, $g$배 더 빠릅니다. 더 좋은 점은 출력을 생성하는 데 필요한 파라미터 수도 $c_\textrm{i} \times c_\textrm{o}$ 행렬에서 $(c_\textrm{i}/g) \times (c_\textrm{o}/g)$ 크기의 $g$개의 더 작은 행렬로 줄어들어 역시 $g$배 감소한다는 것입니다. 다음에서는 $c_\textrm{i}$와 $c_\textrm{o}$가 모두 $g$로 나누어떨어진다고 가정합니다. 

이 설계의 유일한 과제는 $g$ 그룹 간에 정보가 교환되지 않는다는 것입니다. :numref:`fig_resnext_block`의 ResNeXt 블록은 두 가지 방식으로 이를 수정합니다: $3 \times 3$ 커널을 사용한 그룹화된 합성곱이 두 개의 $1 \times 1$ 합성곱 사이에 끼어 있습니다. 두 번째 것은 채널 수를 다시 변경하는 이중 역할을 합니다. 이점은 $1 \times 1$ 커널에 대해 $\mathcal{O}(c \cdot b)$ 비용만 지불하고 $3 \times 3$ 커널에 대해 $\mathcal{O}(b^2 / g)$ 비용으로 해결할 수 있다는 것입니다. :numref:`subsec_residual-blks`의 잔차 블록 구현과 유사하게 잔차 연결은 $1 \times 1$ 합성곱으로 대체(따라서 일반화)됩니다.

:numref:`fig_resnext_block`의 오른쪽 그림은 결과 네트워크 블록에 대한 훨씬 더 간결한 요약을 제공합니다. 이것은 또한 :numref:`sec_cnn-design`의 일반적인 현대 CNN 설계에서 중요한 역할을 할 것입니다. 그룹화된 합성곱 아이디어는 AlexNet 구현 :cite:`Krizhevsky.Sutskever.Hinton.2012`으로 거슬러 올라갑니다. 제한된 메모리를 가진 두 개의 GPU에 네트워크를 분산할 때 구현은 각 GPU를 부작용 없이 자체 채널로 처리했습니다. 

`ResNeXtBlock` 클래스의 다음 구현은 `groups` ($g$)를 인수로 취하며, `bot_channels` ($b$) 중간(병목) 채널을 갖습니다. 마지막으로 표현의 높이와 너비를 줄여야 할 때 `use_1x1conv=True, strides=2`로 설정하여 스트라이드 $2$를 추가합니다.

```{.python .input}
%%tab mxnet
class ResNeXtBlock(nn.Block):  #@save
    """ResNeXt 블록."""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class ResNeXtBlock(nn.Module):  #@save
    """ResNeXt 블록."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.python .input}
%%tab tensorflow
class ResNeXtBlock(tf.keras.Model):  #@save
    """ResNeXt 블록."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)
```

```{.python .input}
%%tab jax
class ResNeXtBlock(nn.Module):  #@save
    """ResNeXt 블록."""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=bot_channels//self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
```

사용법은 앞에서 논의한 `ResNetBlock`의 사용법과 전적으로 유사합니다. 예를 들어 (`use_1x1conv=False, strides=1`)을 사용하면 입력과 출력의 모양이 같습니다. 대안으로 `use_1x1conv=True, strides=2`를 설정하면 출력 높이와 너비가 절반으로 줄어듭니다.

```{.python .input}
%%tab mxnet, pytorch
blk = ResNeXtBlock(32, 16, 1)
if tab.selected('mxnet'):
    blk.initialize()
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = ResNeXtBlock(32, 16, 1)
X = d2l.normal((4, 96, 96, 32))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = ResNeXtBlock(32, 16, 1)
X = jnp.zeros((4, 96, 96, 32))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## 요약 및 토론 (Summary and Discussion)

중첩된 함수 클래스는 용량을 추가할 때 미묘하게 *다른* 함수 클래스 대신 엄격하게 *더 강력한* 함수 클래스를 얻을 수 있도록 하므로 바람직합니다. 이를 달성하는 한 가지 방법은 추가 레이어가 단순히 입력을 출력으로 통과시키도록 하는 것입니다. 잔차 연결은 이를 가능하게 합니다. 결과적으로 이는 단순 함수의 귀납적 편향을 $f(\mathbf{x}) = 0$ 형태에서 $f(\mathbf{x}) = \mathbf{x}$와 같은 형태로 변경합니다.


잔차 매핑은 가중치 레이어의 파라미터를 0으로 미는 것과 같이 항등 함수를 더 쉽게 학습할 수 있습니다. 우리는 잔차 블록을 사용하여 효과적인 *심층* 신경망을 훈련할 수 있습니다. 입력은 레이어 전체의 잔차 연결을 통해 더 빠르게 순전파될 수 있습니다. 결과적으로 훨씬 더 깊은 네트워크를 훈련할 수 있습니다. 예를 들어 원래 ResNet 논문 :cite:`He.Zhang.Ren.ea.2016`은 최대 152개 레이어를 허용했습니다. 잔차 네트워크의 또 다른 이점은 훈련 과정 *중*에 항등 함수로 초기화된 레이어를 추가할 수 있다는 것입니다. 결국 레이어의 기본 동작은 데이터를 변경하지 않고 통과시키는 것입니다. 이는 경우에 따라 매우 큰 네트워크의 훈련을 가속화할 수 있습니다. 

잔차 연결 이전에, 
게이팅 유닛이 있는 우회 경로가 도입되어 
100개 이상의 레이어를 가진 고속도로 네트워크를 효과적으로 훈련했습니다 
:cite:`srivastava2015highway`. 
우회 경로로 항등 함수를 사용하여 
ResNet은 여러 컴퓨터 비전 작업에서 놀라울 정도로 잘 수행되었습니다. 
잔차 연결은 합성곱 또는 순차적 성격의 후속 심층 신경망 설계에 큰 영향을 미쳤습니다. 
나중에 소개하겠지만, 
Transformer 아키텍처 :cite:`Vaswani.Shazeer.Parmar.ea.2017`는 
잔차 연결을 채택(다른 설계 선택과 함께)하며 
언어, 비전, 음성, 강화 학습과 같이 다양한 분야에 
널리 퍼져 있습니다.

ResNeXt는 합성곱 신경망 설계가 시간이 지남에 따라 어떻게 진화했는지 보여주는 예입니다: 계산을 더 절약하고 활성화 크기(채널 수)와 절충함으로써 저렴한 비용으로 더 빠르고 정확한 네트워크를 가능하게 합니다. 그룹화된 합성곱을 보는 다른 방법은 합성곱 가중치에 대한 블록 대각 행렬을 생각하는 것입니다. 더 효율적인 네트워크로 이어지는 꽤 많은 "트릭"이 있다는 점에 유의하십시오. 예를 들어 ShiftNet :cite:`wu2018shift`은 단순히 채널에 이동된 활성화를 추가하여 $3 \times 3$ 합성곱의 효과를 모방하여 이번에는 계산 비용 없이 향상된 함수 복잡성을 제공합니다. 

지금까지 논의한 설계의 공통적인 특징은 네트워크 설계가 상당히 수동적이며 주로 "올바른" 네트워크 하이퍼파라미터를 찾기 위해 설계자의 독창성에 의존한다는 것입니다. 분명히 실현 가능하지만 인간 시간 측면에서 비용이 많이 들고 결과가 어떤 의미에서 최적이라는 보장이 없습니다. :numref:`sec_cnn-design`에서는 더 자동화된 방식으로 고품질 네트워크를 얻기 위한 여러 전략을 논의할 것입니다. 특히 RegNetX/Y 모델 :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`로 이어진 *네트워크 설계 공간*의 개념을 검토할 것입니다.

## 연습 문제 (Exercises)

1. :numref:`fig_inception`의 Inception 블록과 잔차 블록의 주요 차이점은 무엇입니까? 계산, 정확도, 설명할 수 있는 함수 클래스 측면에서 어떻게 비교됩니까?
2. 네트워크의 다양한 변형을 구현하려면 ResNet 논문 :cite:`He.Zhang.Ren.ea.2016`의 표 1을 참조하십시오. 
3. 더 깊은 네트워크를 위해 ResNet은 모델 복잡성을 줄이기 위해 "병목" 아키텍처를 도입합니다. 구현해 보십시오.
4. ResNet의 후속 버전에서 저자는 "합성곱, 배치 정규화, 활성화" 구조를 "배치 정규화, 활성화, 합성곱" 구조로 변경했습니다. 이 개선을 직접 수행하십시오. 자세한 내용은 :citet:`He.Zhang.Ren.ea.2016*1`의 그림 1을 참조하십시오.
5. 함수 클래스가 중첩되더라도 왜 함수의 복잡성을 무한정 늘릴 수 없습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/8737)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18006)
:end_tab: