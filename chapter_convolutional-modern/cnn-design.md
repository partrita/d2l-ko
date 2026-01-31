```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성곱 네트워크 아키텍처 설계 (Designing Convolution Network Architectures)
:label:`sec_cnn-design`

이전 섹션들에서는 컴퓨터 비전을 위한 현대 네트워크 설계에 대해 살펴보았습니다. 
많은 아키텍처가 과학자들의 직관에 크게 의존했으며, 인간의 창의성에 크게 의존하고 심층 네트워크가 제공하는 설계 공간에 대한 체계적인 탐구에는 훨씬 덜 의존했습니다. 
그럼에도 불구하고 이 *네트워크 엔지니어링* 접근 방식은 엄청난 성공을 거두었습니다. 

AlexNet(:numref:`sec_alexnet`)이 ImageNet에서 기존 컴퓨터 비전 모델을 능가한 이래로, 
동일한 패턴에 따라 설계된 합성곱 블록을 쌓아 매우 깊은 네트워크를 구축하는 것이 인기를 얻었습니다. 
특히 $3 \times 3$ 합성곱은 VGG 네트워크(:numref:`sec_vgg`)에 의해 대중화되었습니다. 
NiN(:numref:`sec_nin`)은 $1 \times 1$ 합성곱조차도 국소 비선형성을 추가함으로써 유익할 수 있음을 보여주었습니다. 
또한 NiN은 모든 위치에 걸쳐 집계함으로써 네트워크 헤드에서 정보를 집계하는 문제를 해결했습니다. 
GoogLeNet(:numref:`sec_googlenet`)은 Inception 블록에서 VGG와 NiN의 장점을 결합하여 다양한 합성곱 너비의 다중 분기를 추가했습니다. 
ResNet(:numref:`sec_resnet`)은 항등 매핑($f(x) = 0$에서)으로 귀납적 편향을 변경했습니다. 
이로써 매우 깊은 네트워크가 가능해졌습니다. 
거의 10년이 지난 지금도 ResNet 설계는 여전히 인기가 있으며, 이는 그 설계의 증거입니다. 
마지막으로 ResNeXt(:numref:`subsec_resnext`)는 그룹화된 합성곱을 추가하여 파라미터와 계산 간의 더 나은 트레이드오프를 제공했습니다. 
비전을 위한 Transformer의 전신인 Squeeze-and-Excitation Networks (SENets)는 위치 간의 효율적인 정보 전송을 가능하게 합니다 :cite:`Hu.Shen.Sun.2018`. 
이는 채널별 전역 주의 함수를 계산하여 달성되었습니다.

지금까지 *신경 아키텍처 검색(neural architecture search, NAS)* :cite:`zoph2016neural,liu2018darts`을 통해 얻은 네트워크는 생략했습니다. 
우리는 그 비용이 일반적으로 엄청나며 무차별 대입 검색, 유전 알고리즘, 강화 학습 또는 다른 형태의 하이퍼파라미터 최적화에 의존하기 때문에 그렇게 하기로 결정했습니다. 
고정된 검색 공간이 주어지면 NAS는 반환된 성능 추정을 기반으로 아키텍처를 자동으로 선택하는 검색 전략을 사용합니다. 
NAS의 결과는 단일 네트워크 인스턴스입니다. EfficientNet은 이 검색의 주목할 만한 결과입니다 :cite:`tan2019efficientnet`.

다음에서는 *단일 최고의 네트워크*를 찾는 탐구와는 상당히 다른 아이디어를 논의합니다. 
계산적으로 상대적으로 저렴하고, 도중에 과학적 통찰력으로 이어지며, 결과의 품질 측면에서 매우 효과적입니다. 
*네트워크 설계 공간을 설계*하기 위한 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 전략을 검토해 봅시다. 
이 전략은 수동 설계와 NAS의 강점을 결합합니다. 
*네트워크 분포*에 대해 작업하고 전체 네트워크 패밀리에 대해 좋은 성능을 얻는 방식으로 분포를 최적화함으로써 이를 달성합니다. 
그 결과는 *RegNets*, 구체적으로 RegNetX 및 RegNetY, 그리고 고성능 CNN 설계를 위한 다양한 지침 원칙입니다.

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
```

## AnyNet 설계 공간 (The AnyNet Design Space)
:label:`subsec_the-anynet-design-space`

아래 설명은 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 추론을 밀접하게 따르며 책의 범위에 맞게 일부 약어를 사용합니다. 
시작하려면 탐색할 네트워크 패밀리를 위한 템플릿이 필요합니다. 
이 장의 설계 중 공통점 중 하나는 네트워크가 *스템(stem)*, *바디(body)*, *헤드(head)*로 구성된다는 것입니다. 
스템은 종종 더 큰 윈도우 크기를 가진 합성곱을 통해 초기 이미지 처리를 수행합니다. 
바디는 원시 이미지에서 객체 표현으로 가는 데 필요한 변환의 대부분을 수행하는 다중 블록으로 구성됩니다. 
마지막으로 헤드는 이를 다중 클래스 분류를 위한 소프트맥스 회귀기와 같은 원하는 출력으로 변환합니다. 
바디는 차례로 감소하는 해상도에서 이미지에 대해 작업하는 다중 단계로 구성됩니다. 
사실 스템과 각 후속 단계는 공간 해상도를 4분의 1로 줄입니다. 
마지막으로 각 단계는 하나 이상의 블록으로 구성됩니다. 
이 패턴은 VGG에서 ResNeXt에 이르기까지 모든 네트워크에 공통적입니다. 
실제로 일반적인 AnyNet 네트워크 설계를 위해 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`는 :numref:`fig_resnext_block`의 ResNeXt 블록을 사용했습니다.


![AnyNet 설계 공간. 각 화살표를 따라 있는 숫자 $(\mathit{c}, \mathit{r})$은 해당 지점에서의 채널 수 $c$와 이미지의 해상도 $\mathit{r} \times \mathit{r}$를 나타냅니다. 왼쪽에서 오른쪽으로: 스템, 바디, 헤드로 구성된 일반적인 네트워크 구조; 4단계로 구성된 바디; 단계의 상세 구조; 다운샘플링이 없는 블록과 각 차원에서 해상도를 절반으로 줄이는 블록의 두 가지 대안 구조. 설계 선택에는 깊이 $\mathit{d_i}$, 출력 채널 수 $\mathit{c_i}$, 그룹 수 $\mathit{g_i}$, 그리고 모든 단계 $\mathit{i}$에 대한 병목 비율 $\mathit{k_i}$가 포함됩니다.](../img/anynet.svg)
:label:`fig_anynet_full`

:numref:`fig_anynet_full`에 설명된 구조를 자세히 검토해 봅시다. 언급했듯이 AnyNet은 스템, 바디, 헤드로 구성됩니다. 스템은 RGB 이미지(3채널)를 입력으로 받아 스트라이드가 $2$인 $3 \times 3$ 합성곱을 사용하고 배치 정규화가 뒤따라 해상도를 $r \times r$에서 $r/2 \times r/2$로 절반으로 줄입니다. 또한 바디에 입력으로 사용될 $c_0$ 채널을 생성합니다.

네트워크는 $224 \times 224 \times 3$ 모양의 ImageNet 이미지와 잘 작동하도록 설계되었으므로, 바디는 4단계(상기하자면 $224 / 2^{1+4} = 7$)를 통해 이를 $7 \times 7 \times c_4$로 줄이는 역할을 하며, 각 단계는 결국 스트라이드가 $2$입니다. 
마지막으로 헤드는 NiN(:numref:`sec_nin`)과 유사한 전역 평균 풀링을 통해 완전히 표준적인 설계를 채택하고, $n$-클래스 분류를 위한 $n$차원 벡터를 방출하기 위해 완전 연결 레이어가 뒤따릅니다.

대부분의 관련 설계 결정은 네트워크 바디에 내재되어 있습니다. 
바디는 단계적으로 진행되며, 각 단계는 :numref:`subsec_resnext`에서 논의한 것과 동일한 유형의 ResNeXt 블록으로 구성됩니다. 
거기서의 설계는 다시 완전히 일반적입니다: 스트라이드 $2$를 사용하여 해상도를 절반으로 줄이는 블록으로 시작합니다(:numref:`fig_anynet_full`의 맨 오른쪽). 
이에 맞추기 위해 ResNeXt 블록의 잔차 분기는 $1 \times 1$ 합성곱을 통과해야 합니다. 
이 블록 뒤에는 해상도와 채널 수를 변경하지 않는 가변 수의 추가 ResNeXt 블록이 잇따릅니다. 
일반적인 설계 관행은 합성곱 블록 설계에 약간의 병목 현상을 추가하는 것입니다. 
따라서 병목 비율 $k_i \geq 1$로 단계 $i$의 각 블록 내에 $c_i/k_i$ 채널을 제공합니다(실험에서 알 수 있듯이 이는 실제로 효과적이지 않으므로 건너뛰어야 합니다). 
마지막으로 ResNeXt 블록을 다루고 있으므로 단계 $i$에서 그룹화된 합성곱에 대한 그룹 수 $g_i$도 선택해야 합니다.

이 겉보기에 일반적인 설계 공간은 그럼에도 불구하고 우리에게 많은 파라미터를 제공합니다: 
블록 너비(채널 수) $c_0, \ldots c_4$, 단계별 깊이(블록 수) $d_1, \ldots d_4$, 병목 비율 $k_1, \ldots k_4$, 그룹 너비(그룹 수) $g_1, \ldots g_4$를 설정할 수 있습니다. 
총 17개의 파라미터가 추가되어 탐색을 정당화할 수 없을 만큼 많은 수의 구성이 생성됩니다. 
이 거대한 설계 공간을 효과적으로 줄이기 위한 도구가 필요합니다. 
이곳이 설계 공간의 개념적 아름다움이 들어오는 곳입니다. 그렇게 하기 전에 일반적인 설계를 먼저 구현해 봅시다.

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

각 단계는 `depth`개의 ResNeXt 블록으로 구성되며, 
`num_channels`는 블록 너비를 지정합니다. 
첫 번째 블록은 입력 이미지의 높이와 너비를 절반으로 줄인다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

네트워크 스템, 바디, 헤드를 합쳐 AnyNet 구현을 완료합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## 설계 공간의 분포 및 파라미터 (Distributions and Parameters of Design Spaces)

:numref:`subsec_the-anynet-design-space`에서 논의했듯이 설계 공간의 파라미터는 해당 설계 공간에 있는 네트워크의 하이퍼파라미터입니다. 
AnyNet 설계 공간에서 좋은 파라미터를 식별하는 문제를 고려해 보십시오. 
우리는 주어진 계산량(예: FLOPs 및 컴퓨팅 시간)에 대해 *단일 최고의* 파라미터 선택을 찾으려고 시도할 수 있습니다. 
각 파라미터에 대해 *두 가지* 가능한 선택만 허용하더라도, 최고의 솔루션을 찾기 위해 $2^{17} = 131072$ 조합을 탐색해야 합니다. 
이는 엄청난 비용 때문에 명백히 실행 불가능합니다. 
설상가상으로 우리는 네트워크를 어떻게 설계해야 하는지에 대해 이 연습에서 실제로 아무것도 배우지 못합니다. 
다음에 예를 들어 X-스테이지, 시프트 연산 또는 이와 유사한 것을 추가하면 처음부터 다시 시작해야 합니다. 
더 나쁜 것은 훈련의 확률성(반올림, 셔플링, 비트 오류) 때문에 두 번의 실행이 정확히 동일한 결과를 생성할 가능성이 낮다는 것입니다. 
더 나은 전략은 파라미터 선택이 어떻게 관련되어야 하는지에 대한 일반적인 지침을 결정하려고 노력하는 것입니다. 
예를 들어 병목 비율, 채널 수, 블록, 그룹 또는 레이어 간의 변경은 이상적으로는 일련의 간단한 규칙에 의해 관리되어야 합니다. 
:citet:`radosavovic2019network`의 접근 방식은 다음 네 가지 가정에 의존합니다:

1. 우리는 일반적인 설계 원칙이 실제로 존재한다고 가정하므로 이러한 요구 사항을 충족하는 많은 네트워크가 좋은 성능을 제공해야 합니다. 결과적으로 네트워크에 대한 *분포*를 식별하는 것은 합리적인 전략이 될 수 있습니다. 즉, 건초더미에 좋은 바늘이 많이 있다고 가정합니다.
2. 네트워크가 좋은지 평가하기 위해 수렴될 때까지 네트워크를 훈련할 필요는 없습니다. 대신 중간 결과를 최종 정확도에 대한 신뢰할 수 있는 지침으로 사용하는 것으로 충분합니다. 목적 함수를 최적화하기 위해 (근사) 프록시를 사용하는 것을 다중 충실도 최적화(multi-fidelity optimization)라고 합니다 :cite:`forrester2007multi`. 결과적으로 데이터셋을 몇 번 통과한 후 달성한 정확도를 기반으로 설계 최적화가 수행되어 비용을 크게 줄입니다.
3. 더 작은 규모(더 작은 네트워크)에서 얻은 결과는 더 큰 규모로 일반화됩니다. 결과적으로 최적화는 구조적으로 유사하지만 블록 수가 적고 채널이 적은 네트워크에 대해 수행됩니다. 결국에만 이렇게 찾은 네트워크가 대규모에서도 좋은 성능을 제공하는지 확인해야 합니다.
4. 설계의 측면은 대략적으로 인수 분해될 수 있으므로 결과의 품질에 미치는 영향을 다소 독립적으로 추론할 수 있습니다. 즉, 최적화 문제는 적당히 쉽습니다.

이러한 가정을 통해 우리는 많은 네트워크를 저렴하게 테스트할 수 있습니다. 특히 구성 공간에서 균일하게 *샘플링*하고 성능을 평가할 수 있습니다. 
그 후, 해당 네트워크로 달성할 수 있는 오류/정확도의 *분포*를 검토하여 파라미터 선택의 품질을 평가할 수 있습니다. 
$F(e)$를 확률 분포 $p$를 사용하여 추출된 주어진 설계 공간의 네트워크가 범한 오류에 대한 누적 분포 함수(CDF)로 표시합니다. 즉, 

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$ 

우리의 목표는 이제 대부분의 네트워크가 매우 낮은 오류율을 갖고 $p$의 지원(support)이 간결한 *네트워크*에 대한 분포 $p$를 찾는 것입니다. 
물론 이것을 정확하게 수행하는 것은 계산적으로 실행 불가능합니다. 
우리는 $p$에서 네트워크 샘플 $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ (각각 오류 $e_1, \ldots, e_n$ 포함)에 의존하고 대신 경험적 CDF $\hat{F}(e, \mathcal{Z})$를 사용합니다:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$ 

한 선택 세트에 대한 CDF가 다른 CDF를 지배(majorizes)(또는 일치)할 때마다 파라미터 선택이 우월(또는 무관)하다는 결론이 나옵니다. 
이에 따라 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`는 네트워크의 모든 단계 $i$에 대해 공유 네트워크 병목 비율 $k_i = k$를 실험했습니다. 
이것은 병목 비율을 지배하는 4개의 파라미터 중 3개를 제거합니다. 
이것이 성능에 (부정적인) 영향을 미치는지 평가하기 위해 제한된 분포와 제한되지 않은 분포에서 네트워크를 추출하고 해당 CDF를 비교할 수 있습니다. 
:numref:`fig_regnet-fig`의 첫 번째 패널에서 볼 수 있듯이 이 제약 조건은 네트워크 분포의 정확도에 전혀 영향을 미치지 않는 것으로 나타났습니다. 
마찬가지로 네트워크의 다양한 단계에서 발생하는 동일한 그룹 너비 $g_i = g$를 선택할 수 있습니다. 
다시 말하지만 :numref:`fig_regnet-fig`의 두 번째 패널에서 볼 수 있듯이 성능에는 영향을 미치지 않습니다. 
두 단계를 합치면 자유 파라미터 수가 6개 줄어듭니다.

![설계 공간의 오류 경험적 분포 함수 비교. $\textrm{AnyNet}_\mathit{A}$는 원래 설계 공간입니다. $\textrm{AnyNet}_\mathit{B}$는 병목 비율을 묶고, $\textrm{AnyNet}_\mathit{C}$는 그룹 너비도 묶으며, $\textrm{AnyNet}_\mathit{D}$는 단계 전반에 걸쳐 네트워크 깊이를 늘립니다. 왼쪽에서 오른쪽으로: (i) 병목 비율을 묶는 것은 성능에 영향을 미치지 않습니다; (ii) 그룹 너비를 묶는 것은 성능에 영향을 미치지 않습니다; (iii) 단계 전반에 걸쳐 네트워크 너비(채널)를 늘리면 성능이 향상됩니다; (iv) 단계 전반에 걸쳐 네트워크 깊이를 늘리면 성능이 향상됩니다. 그림 제공: :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

다음으로 우리는 단계의 너비와 깊이에 대한 수많은 잠재적 선택을 줄이는 방법을 찾습니다. 
더 깊이 들어갈수록 채널 수가 증가해야 한다는 것은 합리적인 가정입니다. 즉, $c_i \geq c_{i-1}$ (:numref:`fig_regnet-fig`의 표기법에 따르면 $w_{i+1} \geq w_i$), 이는 $\textrm{AnyNetX}_D$를 산출합니다. 
마찬가지로 단계가 진행됨에 따라 더 깊어져야 한다고 가정하는 것도 똑같이 합리적입니다. 즉, $d_i \geq d_{i-1}$, 이는 $\textrm{AnyNetX}_E$를 산출합니다. 
이것은 각각 :numref:`fig_regnet-fig`의 세 번째와 네 번째 패널에서 실험적으로 검증할 수 있습니다.

## RegNet

결과 $\textrm{AnyNetX}_E$ 설계 공간은 해석하기 쉬운 설계 원칙을 따르는 단순한 네트워크로 구성됩니다:

* 모든 단계 $i$에 대해 병목 비율 $k_i = k$를 공유합니다.
* 모든 단계 $i$에 대해 그룹 너비 $g_i = g$를 공유합니다.
* 단계 전반에 걸쳐 네트워크 너비를 늘립니다: $c_{i} \leq c_{i+1}$.
* 단계 전반에 걸쳐 네트워크 깊이를 늘립니다: $d_{i} \leq d_{i+1}$.

이것은 우리에게 마지막 선택 세트를 남깁니다: 최종 $\textrm{AnyNetX}_E$ 설계 공간의 위 파라미터에 대한 특정 값을 선택하는 방법입니다. 
$	extrm{AnyNetX}_E$의 분포에서 가장 성능이 좋은 네트워크를 연구함으로써 다음을 관찰할 수 있습니다: 네트워크의 너비는 이상적으로 네트워크 전반에 걸쳐 블록 인덱스와 함께 선형적으로 증가합니다. 즉, $c_j \approx c_0 + c_a j$, 여기서 $j$는 블록 인덱스이고 기울기 $c_a > 0$입니다. 
단계별로만 다른 블록 너비를 선택할 수 있으므로, 이 의존성과 일치하도록 설계된 조각별 상수 함수에 도달합니다. 
또한 실험은 병목 비율 $k = 1$이 가장 잘 수행됨을 보여줍니다. 즉, 병목 현상을 전혀 사용하지 않는 것이 좋습니다. 

관심 있는 독자는 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`를 정독하여 다양한 계산량에 대한 특정 네트워크 설계의 추가 세부 사항을 검토할 것을 권장합니다. 
예를 들어 효과적인 32레이어 RegNetX 변형은 $k = 1$(병목 없음), $g = 16$(그룹 너비 16), 첫 번째 및 두 번째 단계에 대해 각각 $c_1 = 32$ 및 $c_2 = 80$ 채널, 깊이는 $d_1=4$ 및 $d_2=6$ 블록으로 선택됩니다. 
설계에서 얻은 놀라운 통찰력은 더 큰 규모의 네트워크를 조사할 때에도 여전히 적용된다는 것입니다. 
더 좋은 점은 전역 채널 활성화를 가진 Squeeze-and-Excitation(SE) 네트워크 설계(RegNetY)에도 적용된다는 것입니다 :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

각 RegNetX 단계가 점진적으로 해상도를 줄이고 출력 채널을 늘리는 것을 볼 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## 훈련 (Training)

Fashion-MNIST 데이터셋에서 32레이어 RegNetX를 훈련하는 것은 이전과 같습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## 토론 (Discussion)

비전에 대한 지역성 및 평행 이동 불변성(:numref:`sec_why-conv`)과 같은 바람직한 귀납적 편향(가정 또는 선호도)으로 인해 CNN은 이 분야에서 지배적인 아키텍처였습니다. 
이것은 LeNet부터 Transformer(:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training`가 정확도 측면에서 CNN을 능가하기 시작할 때까지 유지되었습니다. 
비전 Transformer 측면의 최근 진전 중 많은 부분이 CNN으로 백포트(backported) *될 수* 있지만 :cite:`liu2022convnet`, 더 높은 계산 비용으로만 가능합니다. 
마찬가지로 중요한 것은 최근의 하드웨어 최적화(NVIDIA Ampere 및 Hopper)가 Transformer에 유리한 격차를 넓혔다는 것입니다.

Transformer는 CNN보다 지역성 및 평행 이동 불변성에 대한 귀납적 편향 정도가 상당히 낮다는 점에 주목할 가치가 있습니다. 
학습된 구조가 우세했던 것은 무엇보다도 최대 50억 개의 이미지가 있는 LAION-400m 및 LAION-5B :cite:`schuhmann2022laion`와 같은 대규모 이미지 컬렉션의 가용성 때문입니다. 
놀랍게도 이 맥락에서 더 관련성 있는 작업 중 일부는 MLP를 포함하기도 합니다 :cite:`tolstikhin2021mlp`.

요약하자면, 비전 Transformer(:numref:`sec_vision-transformer`)는 이제 대규모 이미지 분류에서 최첨단 성능을 주도하며 *확장성이 귀납적 편향을 이긴다*는 것을 보여줍니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. 
여기에는 다중 헤드 자체 주의(:numref:`sec_multihead-attention`)가 있는 대규모 Transformer 사전 훈련(:numref:`sec_large-pretraining-transformers`)이 포함됩니다. 
우리는 독자들이 훨씬 더 자세한 토론을 위해 이 장들에 뛰어들기를 권합니다.

## 연습 문제 (Exercises)

1. 단계 수를 4개로 늘리십시오. 더 잘 수행되는 더 깊은 RegNetX를 설계할 수 있습니까?
2. ResNeXt 블록을 ResNet 블록으로 교체하여 RegNet을 De-ResNeXt-ify하십시오. 새 모델은 어떻게 수행됩니까?
3. RegNetX의 설계 원칙을 *위반*하여 "VioNet" 패밀리의 여러 인스턴스를 구현하십시오. 그들은 어떻게 수행됩니까? ($d_i$, $c_i$, $g_i$, $b_i$) 중 가장 중요한 요소는 무엇입니까?
4. 당신의 목표는 "완벽한" MLP를 설계하는 것입니다. 위에서 소개한 설계 원칙을 사용하여 좋은 아키텍처를 찾을 수 있습니까? 작은 네트워크에서 큰 네트워크로 외삽하는 것이 가능합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18009)
:end_tab: