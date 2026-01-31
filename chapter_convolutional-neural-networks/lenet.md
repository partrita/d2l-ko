```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성곱 신경망 (LeNet) (Convolutional Neural Networks (LeNet))
:label:`sec_lenet`

우리는 이제 완전히 기능하는 CNN을 조립하는 데 필요한 모든 재료를 갖추었습니다. 
이전의 이미지 데이터와의 만남에서, 우리는 Fashion-MNIST 데이터셋의 의류 사진에 선형 모델과 소프트맥스 회귀(:numref:`sec_softmax_scratch`) 및 MLP(:numref:`sec_mlp-implementation`)를 적용했습니다. 
이러한 데이터를 처리하기 위해 우리는 먼저 각 이미지를 $28\times28$ 행렬에서 고정 길이 $784$차원 벡터로 평탄화한 다음, 완전 연결 레이어에서 처리했습니다. 
이제 합성곱 레이어를 다룰 수 있게 되었으므로, 이미지의 공간 구조를 유지할 수 있습니다. 
완전 연결 레이어를 합성곱 레이어로 대체하는 추가적인 이점으로, 훨씬 적은 파라미터를 필요로 하는 더 간결한 모델을 누릴 수 있습니다.

이 섹션에서는 컴퓨터 비전 작업에서의 성능으로 폭넓은 주목을 받은 최초의 발표된 CNN 중 하나인 *LeNet*을 소개합니다. 
이 모델은 당시 AT&T 벨 연구소의 연구원이었던 얀 르쿤(Yann LeCun)이 이미지 내 손글씨 숫자를 인식하기 위해 소개했습니다(그리고 그의 이름을 따서 명명되었습니다) :cite:`LeCun.Bottou.Bengio.ea.1998`. 
이 작업은 기술을 개발해 온 10년 연구의 정점을 나타냈습니다. 
르쿤의 팀은 역전파를 통해 CNN을 성공적으로 훈련한 최초의 연구를 발표했습니다 :cite:`LeCun.Boser.Denker.ea.1989`.

당시 LeNet은 지도 학습의 지배적인 접근 방식이었던 서포트 벡터 머신의 성능과 일치하는 뛰어난 결과를 달성했으며, 숫자당 1% 미만의 오류율을 달성했습니다. 
LeNet은 결국 ATM 기계에서 예금을 처리하기 위해 숫자를 인식하도록 조정되었습니다. 
오늘날까지도 일부 ATM은 1990년대에 얀 르쿤과 그의 동료 레옹 보투(Leon Bottou)가 작성한 코드를 실행하고 있습니다!

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from types import FunctionType
```

## LeNet

높은 수준에서 볼 때, (**LeNet (LeNet-5)은 두 부분으로 구성됩니다:
(i) 두 개의 합성곱 레이어로 구성된 합성곱 인코더; 그리고
(ii) 세 개의 완전 연결 레이어로 구성된 밀집 블록(dense block)**). 
아키텍처는 :numref:`img_lenet`에 요약되어 있습니다.

![LeNet의 데이터 흐름. 입력은 손글씨 숫자이고 출력은 10가지 가능한 결과에 대한 확률입니다.](../img/lenet.svg)
:label:`img_lenet`

각 합성곱 블록의 기본 단위는 합성곱 레이어, 시그모이드 활성화 함수, 그리고 후속 평균 풀링 연산입니다. 
ReLU와 최대 풀링이 더 잘 작동하지만, 당시에는 아직 발견되지 않았다는 점에 유의하십시오. 
각 합성곱 레이어는 $5\times 5$ 커널과 시그모이드 활성화 함수를 사용합니다. 
이 레이어들은 공간적으로 배열된 입력을 다수의 2차원 특성 맵으로 매핑하며, 일반적으로 채널 수를 늘립니다. 
첫 번째 합성곱 레이어는 6개의 출력 채널을 갖고, 두 번째는 16개를 갖습니다. 
각 $2\times2$ 풀링 연산(스트라이드 2)은 공간 다운샘플링을 통해 차원을 $4$배로 줄입니다. 
합성곱 블록은 (배치 크기, 채널 수, 높이, 너비)로 주어진 모양의 출력을 방출합니다.

합성곱 블록의 출력을 밀집 블록으로 전달하려면, 미니배치의 각 예제를 평탄화해야 합니다. 
즉, 이 4차원 입력을 완전 연결 레이어가 예상하는 2차원 입력으로 변환합니다: 
상기시키자면, 우리가 원하는 2차원 표현은 첫 번째 차원을 사용하여 미니배치의 예제를 인덱싱하고 두 번째 차원을 사용하여 각 예제의 평면 벡터 표현을 제공합니다. 
LeNet의 밀집 블록에는 각각 120, 84, 10개의 출력을 가진 세 개의 완전 연결 레이어가 있습니다. 
여전히 분류를 수행하고 있으므로 10차원 출력 레이어는 가능한 출력 클래스 수에 해당합니다.

LeNet 내부에서 무슨 일이 일어나고 있는지 진정으로 이해하는 데까지는 약간의 노력이 필요했을 수 있지만, 다음 코드 스니펫이 현대 딥러닝 프레임워크로 이러한 모델을 구현하는 것이 놀랍도록 간단하다는 것을 확신시켜 주기를 바랍니다. 
우리는 `Sequential` 블록을 인스턴스화하고 적절한 레이어를 함께 연결하기만 하면 됩니다. 
:numref:`subsec_xavier`에서 소개된 Xavier 초기화를 사용합니다.

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """CNN 가중치 초기화."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 모델."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 모델."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # 평탄화
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

우리는 가우스 활성화 레이어를 소프트맥스 레이어로 대체한다는 점에서 LeNet 재현에 약간의 자유를 취했습니다. 
이는 무엇보다도 가우스 디코더가 오늘날 거의 사용되지 않는다는 사실 때문에 구현을 크게 단순화합니다. 
그 외에는 이 네트워크가 원래 LeNet-5 아키텍처와 일치합니다.

:begin_tab:`pytorch, mxnet, tensorflow`
네트워크 내부에서 어떤 일이 일어나는지 봅시다. 
단일 채널(흑백) $28 \times 28$ 이미지를 네트워크에 통과시키고 각 레이어에서 출력 모양을 인쇄함으로써, 
[**모델을 검사**]하여 그 연산이 :numref:`img_lenet_vert`에서 기대하는 것과 일치하는지 확인할 수 있습니다.
:end_tab:

:begin_tab:`jax`
네트워크 내부에서 어떤 일이 일어나는지 봅시다. 
단일 채널(흑백) $28 \times 28$ 이미지를 네트워크에 통과시키고 각 레이어에서 출력 모양을 인쇄함으로써, 
[**모델을 검사**]하여 그 연산이 :numref:`img_lenet_vert`에서 기대하는 것과 일치하는지 확인할 수 있습니다. 
Flax는 네트워크의 레이어와 파라미터를 요약하는 멋진 메서드인 `nn.tabulate`를 제공합니다. 
여기서는 `bind` 메서드를 사용하여 바운드 모델을 생성합니다. 
변수들은 이제 `d2l.Module` 클래스에 바인딩됩니다. 즉, 이 바운드 모델은 상태 저장(stateful) 객체가 되어 `Sequential` 객체 속성 `net`과 그 안의 `layers`에 액세스하는 데 사용할 수 있습니다. 
`bind` 메서드는 대화형 실험에만 사용해야 하며 `apply` 메서드를 직접 대체하는 것은 아님에 유의하십시오.
:end_tab:

![LeNet-5를 위한 압축 표기법.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

합성곱 블록 전체의 각 레이어에서 표현의 높이와 너비가 (이전 레이어에 비해) 줄어든다는 점에 유의하십시오. 
첫 번째 합성곱 레이어는 $5 \times 5$ 커널을 사용하여 발생할 수 있는 높이와 너비의 감소를 보상하기 위해 2픽셀의 패딩을 사용합니다. 
참고로 원래 MNIST OCR 데이터셋의 $28 \times 28$ 픽셀 이미지 크기는 $32 \times 32$ 픽셀 크기의 원본 스캔에서 2픽셀 행(및 열)을 *다듬은(trimming)* 결과입니다. 
이는 주로 메가바이트가 중요했던 시절에 공간을 절약(30% 감소)하기 위해 수행되었습니다.

대조적으로, 두 번째 합성곱 레이어는 패딩을 생략하므로 높이와 너비가 모두 4픽셀씩 줄어듭니다. 
레이어 스택을 올라가면서 채널 수는 입력의 1개에서 첫 번째 합성곱 레이어 후 6개, 두 번째 합성곱 레이어 후 16개로 레이어마다 증가합니다. 
그러나 각 풀링 레이어는 높이와 너비를 반으로 줄입니다. 
마지막으로 각 완전 연결 레이어는 차원 수를 줄여 최종적으로 클래스 수와 일치하는 차원의 출력을 방출합니다.


## 훈련 (Training)

이제 모델을 구현했으므로, [**LeNet-5 모델이 Fashion-MNIST에서 어떻게 수행되는지 확인하기 위해 실험을 실행**]해 보겠습니다.

CNN은 파라미터가 더 적지만, 각 파라미터가 훨씬 더 많은 곱셈에 참여하기 때문에 비슷하게 깊은 MLP보다 계산 비용이 더 많이 들 수 있습니다. 
GPU에 액세스할 수 있다면 지금이 훈련 속도를 높이기 위해 실행에 옮길 좋은 때입니다. 
`d2l.Trainer` 클래스가 모든 세부 사항을 처리한다는 점에 유의하십시오. 
기본적으로 사용 가능한 장치에서 모델 파라미터를 초기화합니다. 
MLP와 마찬가지로 손실 함수는 교차 엔트로피이며 미니배치 확률적 경사 하강법을 통해 최소화합니다.

```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## 요약 (Summary)

우리는 이 장에서 상당한 진전을 이루었습니다. 1980년대의 MLP에서 1990년대와 2000년대 초반의 CNN으로 이동했습니다. 제안된 아키텍처, 예를 들어 LeNet-5 형태는 오늘날까지도 의미가 있습니다. Fashion-MNIST에서 LeNet-5로 달성할 수 있는 오류율을 MLP로 가능한 최고의 오류율(:numref:`sec_mlp-implementation`) 및 ResNet(:numref:`sec_resnet`)과 같은 훨씬 더 진보된 아키텍처의 오류율과 비교해 볼 가치가 있습니다. LeNet은 전자보다는 후자와 훨씬 더 유사합니다. 우리가 보게 될 주요 차이점 중 하나는 더 많은 양의 계산이 훨씬 더 복잡한 아키텍처를 가능하게 했다는 것입니다.

두 번째 차이점은 우리가 LeNet을 구현할 수 있었던 상대적인 용이성입니다. 예전에는 몇 달 간의 C++ 및 어셈블리 코드 가치가 있는 엔지니어링 과제였고, 초기 Lisp 기반 딥러닝 도구인 SN :cite:`Bottou.Le-Cun.1988`을 개선하기 위한 엔지니어링, 그리고 마지막으로 모델 실험이 이제는 몇 분 만에 달성될 수 있습니다. 딥러닝 모델 개발을 엄청나게 민주화한 것은 바로 이러한 놀라운 생산성 향상입니다. 다음 장에서는 이 토끼굴을 따라 내려가 어디로 가는지 볼 것입니다.

## 연습 문제 (Exercises)

1. LeNet을 현대화해 봅시다. 다음 변경 사항을 구현하고 테스트하십시오:
    1. 평균 풀링을 최대 풀링으로 대체하십시오.
    2. 소프트맥스 레이어를 ReLU로 대체하십시오.
2. 최대 풀링 및 ReLU 외에도 정확도를 향상시키기 위해 LeNet 스타일 네트워크의 크기를 변경해 보십시오.
    1. 합성곱 윈도우 크기를 조정하십시오.
    2. 출력 채널 수를 조정하십시오.
    3. 합성곱 레이어 수를 조정하십시오.
    4. 완전 연결 레이어 수를 조정하십시오.
    5. 학습률 및 기타 훈련 세부 사항(예: 초기화 및 에폭 수)을 조정하십시오.
3. 개선된 네트워크를 원본 MNIST 데이터셋에서 사용해 보십시오.
4. 다른 입력(예: 스웨터 및 코트)에 대한 LeNet의 첫 번째 및 두 번째 레이어의 활성화를 표시하십시오.
5. 네트워크에 상당히 다른 이미지(예: 고양이, 자동차, 심지어 무작위 노이즈)를 공급하면 활성화에 어떤 일이 발생합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18000)
:end_tab:

```