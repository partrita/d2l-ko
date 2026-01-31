```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 블록을 사용하는 네트워크 (VGG) (Networks Using Blocks (VGG))
:label:`sec_vgg`

AlexNet은 심층 CNN이 좋은 결과를 얻을 수 있다는 경험적 증거를 제공했지만, 후속 연구자들이 새로운 네트워크를 설계하는 데 지침이 될 일반적인 템플릿을 제공하지는 않았습니다. 
다음 섹션에서는 심층 네트워크를 설계하는 데 일반적으로 사용되는 몇 가지 휴리스틱 개념을 소개합니다.

이 분야의 발전은 칩 설계의 VLSI(초고밀도 집적 회로) 발전과 유사합니다. 
엔지니어들은 트랜지스터 배치에서 논리 요소, 논리 블록으로 이동했습니다 :cite:`Mead.1980`. 
마찬가지로 신경망 아키텍처의 설계는 점점 더 추상적으로 성장하여, 연구자들은 개별 뉴런 측면에서 생각하는 것에서 전체 레이어로, 그리고 이제는 레이어의 반복 패턴인 블록으로 이동했습니다. 10년 후, 이제는 연구자들이 전체 훈련된 모델을 사용하여 관련성은 있지만 다른 작업을 위해 용도 변경하는 것으로 발전했습니다. 이러한 대규모 사전 훈련된 모델을 일반적으로 *파운데이션 모델(foundation models)*이라고 부릅니다 :cite:`bommasi2021opportunities`.

네트워크 설계로 돌아갑시다. 블록을 사용한다는 아이디어는 옥스퍼드 대학의 VGG(Visual Geometry Group)에서 그들의 이름을 딴 *VGG* 네트워크로 처음 등장했습니다 :cite:`Simonyan.Zisserman.2014`. 
루프와 서브루틴을 사용하여 최신 딥러닝 프레임워크로 코드에서 이러한 반복 구조를 쉽게 구현할 수 있습니다.

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
import jax
```

## (**VGG 블록**)
:label:`subsec_vgg-blocks`

CNN의 기본 빌딩 블록은 다음의 시퀀스입니다:
(i) 해상도를 유지하기 위한 패딩이 있는 합성곱 레이어,
(ii) ReLU와 같은 비선형성,
(iii) 해상도를 줄이기 위한 최대 풀링과 같은 풀링 레이어. 
이 접근 방식의 문제 중 하나는 공간 해상도가 매우 빠르게 감소한다는 것입니다. 특히, 
이것은 모든 차원($d$)이 소진되기 전에 네트워크에 $\log_2 d$ 합성곱 레이어라는 엄격한 제한을 부과합니다. 예를 들어 ImageNet의 경우, 이 방법으로는 8개 이상의 합성곱 레이어를 가질 수 없습니다.

:citet:`Simonyan.Zisserman.2014`의 핵심 아이디어는 블록 형태의 최대 풀링을 통한 다운샘플링 사이에 *여러* 합성곱을 사용하는 것이었습니다. 그들은 주로 깊은 네트워크와 넓은 네트워크 중 어느 것이 더 잘 수행되는지에 관심이 있었습니다. 예를 들어 두 번의 $3 \times 3$ 합성곱을 연속적으로 적용하면 단일 $5 \times 5$ 합성곱과 동일한 픽셀을 터치합니다. 동시에 후자는 세 번의 $3 \times 3$ 합성곱($3 \cdot 9 \cdot c^2$)과 거의 같은 수의 파라미터($25 \cdot c^2$)를 사용합니다. 
상당히 상세한 분석에서 그들은 깊고 좁은 네트워크가 얕은 네트워크보다 훨씬 성능이 뛰어나다는 것을 보여주었습니다. 이로 인해 딥러닝은 일반적인 응용 프로그램을 위해 100개 이상의 레이어가 있는 더 깊은 네트워크를 추구하게 되었습니다. 
$3 \times 3$ 합성곱을 쌓는 것은 나중의 심층 네트워크에서 금본위제가 되었습니다(최근 :citet:`liu2022convnet`에 의해 재검토된 설계 결정). 결과적으로 작은 합성곱을 위한 빠른 구현은 GPU의 필수 요소가 되었습니다 :cite:`lavin2016fast`.

VGG로 돌아가서: VGG 블록은 패딩 1(높이와 너비 유지)이 있는 $3\times3$ 커널을 가진 합성곱 *시퀀스*와 스트라이드 2인 $2 \times 2$ 최대 풀링 레이어(각 블록 후 높이와 너비 절반으로 줄임)로 구성됩니다. 
아래 코드에서는 하나의 VGG 블록을 구현하기 위해 `vgg_block`이라는 함수를 정의합니다.

아래 함수는 합성곱 레이어 수 `num_convs`와 출력 채널 수 `num_channels`에 해당하는 두 개의 인수를 취합니다.

```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**VGG 네트워크**]
:label:`subsec_vgg-network`

AlexNet 및 LeNet과 마찬가지로, VGG 네트워크는 두 부분으로 나눌 수 있습니다: 
첫 번째는 대부분 합성곱 및 풀링 레이어로 구성되고, 두 번째는 AlexNet과 동일한 완전 연결 레이어로 구성됩니다. 
주요 차이점은 합성곱 레이어가 차원을 변경하지 않는 비선형 변환으로 그룹화되고, 그 뒤에 :numref:`fig_vgg`에 묘사된 대로 해상도 감소 단계가 따른다는 것입니다.

![AlexNet에서 VGG로. 주요 차이점은 VGG는 레이어 블록으로 구성되는 반면 AlexNet의 레이어는 모두 개별적으로 설계되었다는 점입니다.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

네트워크의 합성곱 부분은 :numref:`fig_vgg`의 여러 VGG 블록(`vgg_block` 함수에도 정의됨)을 연속적으로 연결합니다. 
이러한 합성곱 그룹화는 지난 10년 동안 거의 변하지 않은 패턴이지만, 구체적인 연산 선택은 상당한 수정을 거쳤습니다. 
변수 `arch`는 튜플 목록(블록당 하나)으로 구성되며, 각 튜플에는 합성곱 레이어 수와 출력 채널 수라는 두 개의 값이 포함되어 있습니다. 이는 `vgg_block` 함수를 호출하는 데 필요한 인수와 정확히 일치합니다. 따라서 VGG는 특정 구현보다는 네트워크 *패밀리*를 정의합니다. 특정 네트워크를 구축하기 위해 우리는 단순히 `arch`를 반복하여 블록을 구성합니다.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

원래 VGG 네트워크에는 5개의 합성곱 블록이 있었으며, 그중 처음 두 개는 각각 하나의 합성곱 레이어를 갖고 나중 세 개는 각각 두 개의 합성곱 레이어를 포함합니다. 
첫 번째 블록에는 64개의 출력 채널이 있으며, 각 후속 블록은 출력 채널 수가 512에 도달할 때까지 두 배로 늘립니다. 
이 네트워크는 8개의 합성곱 레이어와 3개의 완전 연결 레이어를 사용하므로 종종 VGG-11이라고 불립니다.

```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

보시다시피 각 블록에서 높이와 너비를 반으로 줄여, 네트워크의 완전 연결 부분에서 처리하기 위해 표현을 평탄화하기 전에 최종적으로 높이와 너비 7에 도달합니다. 
:citet:`Simonyan.Zisserman.2014`는 VGG의 여러 다른 변형을 설명했습니다. 
사실 새로운 아키텍처를 도입할 때 속도-정확도 트레이드오프가 다른 네트워크 *패밀리*를 제안하는 것이 표준이 되었습니다.

## 훈련 (Training)

[**VGG-11은 AlexNet보다 계산적으로 더 까다로으므로 더 적은 수의 채널을 가진 네트워크를 구성합니다.**] 
이는 Fashion-MNIST에서 훈련하기에 충분합니다. 
[**모델 훈련**] 과정은 :numref:`sec_alexnet`의 AlexNet과 유사합니다. 
다시 검증 손실과 훈련 손실이 밀접하게 일치하여 과대적합이 적다는 것을 관찰하십시오.

```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## 요약 (Summary)

VGG가 최초의 진정한 현대적 합성곱 신경망이라고 주장할 수 있습니다. AlexNet이 딥러닝을 대규모로 효과적으로 만드는 구성 요소 중 많은 것을 도입했지만, 다중 합성곱 블록과 깊고 좁은 네트워크에 대한 선호도와 같은 핵심 속성을 도입한 것은 틀림없이 VGG입니다. 또한 실제로 유사하게 파라미터화된 모델의 전체 패밀리인 첫 번째 네트워크로, 실무자에게 복잡성과 속도 간의 충분한 트레이드오프를 제공합니다. 이곳은 또한 현대 딥러닝 프레임워크가 빛을 발하는 곳이기도 합니다. 네트워크를 지정하기 위해 XML 구성 파일을 생성할 필요가 없으며, 간단한 Python 코드를 통해 해당 네트워크를 조립할 수 있습니다.

최근에는 ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021`이 대규모 병렬 계산을 통해 훨씬 더 얕은 아키텍처를 사용하여 경쟁력 있는 성능을 달성할 수 있음을 입증했습니다. 이는 흥미로운 발전이며 미래의 아키텍처 설계에 영향을 미칠 것이라는 희망이 있습니다. 하지만 이 장의 나머지 부분에서는 지난 10년 동안의 과학적 진보의 길을 따를 것입니다.

## 연습 문제 (Exercises)


1. AlexNet과 비교할 때 VGG는 계산 측면에서 훨씬 느리고 GPU 메모리도 더 많이 필요합니다.
    1. AlexNet과 VGG에 필요한 파라미터 수를 비교하십시오.
    2. 합성곱 레이어와 완전 연결 레이어에서 사용되는 부동 소수점 연산 수를 비교하십시오. 
    3. 완전 연결 레이어로 인해 발생하는 계산 비용을 어떻게 줄일 수 있습니까?
2. 네트워크의 다양한 레이어와 관련된 차원을 표시할 때, 네트워크에 11개의 레이어가 있음에도 불구하고 8개의 블록(및 일부 보조 변환)과 관련된 정보만 표시됩니다. 나머지 3개의 레이어는 어디로 갔습니까?
3. VGG 논문 :cite:`Simonyan.Zisserman.2014`의 표 1을 사용하여 VGG-16 또는 VGG-19와 같은 다른 일반적인 모델을 구성하십시오.
4. Fashion-MNIST의 해상도를 $28 \times 28$에서 $224 \times 224$ 차원으로 8배 업샘플링하는 것은 매우 낭비적입니다. 대신 입력을 위해 네트워크 아키텍처와 해상도 변환을 수정해 보십시오(예: 56 또는 84 차원으로). 네트워크의 정확도를 떨어뜨리지 않고 할 수 있습니까? 다운샘플링 전에 더 많은 비선형성을 추가하는 아이디어에 대해서는 VGG 논문 :cite:`Simonyan.Zisserman.2014`을 참조하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/277)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18002)
:end_tab: