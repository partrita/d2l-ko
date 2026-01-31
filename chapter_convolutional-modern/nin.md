```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 네트워크 인 네트워크 (NiN) (Network in Network (NiN))
:label:`sec_nin`

LeNet, AlexNet, VGG는 모두 공통적인 설계 패턴을 공유합니다: 
일련의 합성곱 및 풀링 레이어를 통해 *공간* 구조를 활용하여 특성을 추출하고 완전 연결 레이어를 통해 표현을 후처리합니다. 
AlexNet과 VGG가 LeNet보다 개선된 점은 주로 이 후반부 네트워크가 이 두 모듈을 어떻게 넓히고 깊게 만들었는지에 있습니다.

이 설계는 두 가지 주요 과제를 제기합니다. 
첫째, 아키텍처 끝의 완전 연결 레이어는 엄청난 수의 파라미터를 소비합니다. 예를 들어 VGG-11과 같은 단순한 모델조차도 단일 정밀도(FP32)에서 거의 400MB의 RAM을 차지하는 괴물 같은 행렬을 필요로 합니다. 이는 특히 모바일 및 임베디드 장치에서 계산에 상당한 장애물이 됩니다. 결국 최고급 휴대폰조차도 8GB 이상의 RAM을 자랑하지 않습니다. VGG가 발명되었을 당시에는 이것이 10배나 적었습니다(iPhone 4S는 512MB였습니다). 따라서 이미지 분류기에 메모리의 대부분을 소비하는 것을 정당화하기 어려웠을 것입니다.

둘째, 비선형성 정도를 높이기 위해 네트워크 초기에 완전 연결 레이어를 추가하는 것도 마찬가지로 불가능합니다. 그렇게 하면 공간 구조가 파괴되고 잠재적으로 훨씬 더 많은 메모리가 필요하기 때문입니다.

*네트워크 인 네트워크* (*NiN*) 블록 :cite:`Lin.Chen.Yan.2013`은 하나의 간단한 전략으로 두 문제를 모두 해결할 수 있는 대안을 제공합니다. 
그것들은 매우 간단한 통찰력을 바탕으로 제안되었습니다: (i) $1 \times 1$ 합성곱을 사용하여 채널 활성화에 국소 비선형성을 추가하고 (ii) 전역 평균 풀링을 사용하여 마지막 표현 레이어의 모든 위치에 걸쳐 통합합니다. 전역 평균 풀링은 추가된 비선형성이 없었다면 효과적이지 않았을 것입니다. 이에 대해 자세히 알아봅시다.

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
from jax import numpy as jnp
```

## (**NiN 블록**)

:numref:`subsec_1x1`을 상기하십시오. 거기서 우리는 합성곱 레이어의 입력과 출력이 예제, 채널, 높이, 너비에 해당하는 축을 가진 4차원 텐서로 구성된다고 말했습니다. 
또한 완전 연결 레이어의 입력과 출력은 일반적으로 예제와 특성에 해당하는 2차원 텐서임을 상기하십시오. 
NiN 뒤에 있는 아이디어는 각 픽셀 위치(각 높이와 너비에 대해)에 완전 연결 레이어를 적용하는 것입니다. 
결과로 나오는 $1 \times 1$ 합성곱은 각 픽셀 위치에서 독립적으로 작동하는 완전 연결 레이어로 생각할 수 있습니다.

:numref:`fig_nin`은 VGG와 NiN, 그리고 그들의 블록 간의 주요 구조적 차이점을 보여줍니다. 
NiN 블록의 차이점(초기 합성곱 뒤에 $1 \times 1$ 합성곱이 이어지지만 VGG는 $3 \times 3$ 합성곱을 유지함)과 더 이상 거대한 완전 연결 레이어가 필요하지 않은 끝부분의 차이점에 유의하십시오.

![VGG와 NiN, 그리고 그들의 블록 아키텍처 비교.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**NiN 모델**]

NiN은 AlexNet과 동일한 초기 합성곱 크기를 사용합니다(그 직후에 제안되었습니다). 
커널 크기는 각각 $11\times 11$, $5\times 5$, $3\times 3$이며 출력 채널 수는 AlexNet과 일치합니다. 각 NiN 블록 뒤에는 스트라이드 2와 윈도우 모양 $3\times 3$인 최대 풀링 레이어가 이어집니다.

NiN과 AlexNet 및 VGG의 두 번째 중요한 차이점은 NiN이 완전 연결 레이어를 완전히 피한다는 것입니다. 
대신 NiN은 레이블 클래스 수와 동일한 출력 채널 수를 가진 NiN 블록을 사용하고 *전역* 평균 풀링 레이어가 이어져 로짓 벡터를 산출합니다. 
이 설계는 잠재적으로 훈련 시간이 증가하는 대신 필요한 모델 파라미터 수를 크게 줄입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global avg pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

우리는 [**각 블록의 출력 모양**]을 보기 위해 데이터 예제를 생성합니다.

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**훈련 (Training)**]

이전과 마찬가지로 Fashion-MNIST를 사용하여 AlexNet 및 VGG에 사용했던 것과 동일한 최적화기로 모델을 훈련합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## 요약 (Summary)

NiN은 AlexNet 및 VGG보다 파라미터 수가 훨씬 적습니다. 이는 주로 거대한 완전 연결 레이어가 필요하지 않기 때문입니다. 대신 네트워크 본문의 마지막 단계 이후 모든 이미지 위치에 걸쳐 집계하기 위해 전역 평균 풀링을 사용합니다. 이것은 비용이 많이 드는 (학습된) 축소 연산의 필요성을 없애고 단순 평균으로 대체합니다. 당시 연구자들을 놀라게 했던 것은 이 평균화 연산이 정확도에 해를 끼치지 않는다는 사실이었습니다. (많은 채널을 가진) 저해상도 표현에 걸친 평균화는 네트워크가 처리할 수 있는 평행 이동 불변성의 양도 추가한다는 점에 유의하십시오. 

넓은 커널을 가진 더 적은 합성곱을 선택하고 이를 $1 \times 1$ 합성곱으로 대체하는 것은 더 적은 파라미터를 향한 탐구를 더욱 돕습니다. 주어진 위치 내의 채널 전반에 걸쳐 상당한 양의 비선형성을 제공할 수 있습니다. $1 \times 1$ 합성곱과 전역 평균 풀링 모두 후속 CNN 설계에 큰 영향을 미쳤습니다. 

## 연습 문제 (Exercises)

1. NiN 블록당 두 개의 $1\times 1$ 합성곱 레이어가 있는 이유는 무엇입니까? 그 수를 3개로 늘리십시오. 1개로 줄이십시오. 무엇이 변경됩니까?
2. $1 \times 1$ 합성곱을 $3 \times 3$ 합성곱으로 대체하면 어떻게 변경됩니까? 
3. 전역 평균 풀링을 완전 연결 레이어로 대체하면 어떻게 됩니까(속도, 정확도, 파라미터 수)?
4. NiN의 리소스 사용량을 계산하십시오.
    1. 파라미터 수는 얼마입니까?
    2. 계산량은 얼마입니까?
    3. 훈련 중 필요한 메모리 양은 얼마입니까?
    4. 예측 중 필요한 메모리 양은 얼마입니까?
5. $384 \times 5 \times 5$ 표현을 $10 \times 5 \times 5$ 표현으로 한 번에 줄이는 것의 가능한 문제는 무엇입니까?
6. VGG-11, VGG-16, VGG-19로 이어진 VGG의 구조적 설계 결정을 사용하여 NiN과 유사한 네트워크 패밀리를 설계하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18003)
:end_tab: