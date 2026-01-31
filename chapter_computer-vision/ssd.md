# 단일 샷 멀티박스 감지 (Single Shot Multibox Detection)
:label:`sec_ssd`

:numref:`sec_bbox`--:numref:`sec_object-detection-dataset`에서,
우리는 바운딩 박스, 앵커 박스,
다중 스케일 객체 감지 및 객체 감지를 위한 데이터셋을 소개했습니다.
이제 우리는 이러한 배경 지식을 사용하여 객체 감지 모델을 설계할 준비가 되었습니다:
바로 단일 샷 멀티박스 감지
(SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`입니다.
이 모델은 간단하고 빠르며 널리 사용됩니다.
이것은 방대한 양의 객체 감지 모델 중 하나일 뿐이지만,
이 섹션의 설계 원칙과 구현 세부 사항 중 일부는
다른 모델에도 적용될 수 있습니다.


## 모델 (Model)

:numref:`fig_ssd`는 단일 샷 멀티박스 감지의 설계를 개략적으로 보여줍니다.
이 모델은 주로 기본 네트워크(base network)와 그 뒤를 따르는
여러 다중 스케일 특징 맵 블록으로 구성됩니다.
기본 네트워크는 입력 이미지에서 특징을 추출하기 위한 것이므로
심층 CNN을 사용할 수 있습니다.
예를 들어,
원래의 단일 샷 멀티박스 감지 논문은
분류 레이어 이전에 잘린 VGG 네트워크를 채택했지만 :cite:`Liu.Anguelov.Erhan.ea.2016`,
ResNet도 일반적으로 사용되었습니다.
우리의 설계를 통해
우리는 기본 네트워크가 더 큰 특징 맵을 출력하도록 하여
더 작은 객체를 감지하기 위해 더 많은 앵커 박스를 생성할 수 있습니다.
그 후,
각 다중 스케일 특징 맵 블록은
이전 블록의 특징 맵의 높이와 너비를 줄이고(예: 절반으로),
특징 맵의 각 단위가 입력 이미지에서 수용 영역을 늘릴 수 있도록 합니다.


:numref:`sec_multiscale-object-detection`의
심층 신경망에 의한 이미지의 계층별 표현을 통한
다중 스케일 객체 감지의 설계를 상기하십시오.
:numref:`fig_ssd`의 상단에 더 가까운 다중 스케일 특징 맵은
더 작지만 더 큰 수용 영역을 가지므로,
더 적지만 더 큰 객체를 감지하는 데 적합합니다.

한마디로,
기본 네트워크와 여러 다중 스케일 특징 맵 블록을 통해,
단일 샷 멀티박스 감지는
다양한 크기의 앵커 박스를 생성하고,
이러한 앵커 박스의 클래스와 오프셋(따라서 바운딩 박스)을 예측하여
다양한 크기의 객체를 감지합니다;
따라서 이것은 다중 스케일 객체 감지 모델입니다.


![다중 스케일 객체 감지 모델로서, 단일 샷 멀티박스 감지는 주로 기본 네트워크와 그 뒤를 따르는 여러 다중 스케일 특징 맵 블록으로 구성됩니다.](../img/ssd.svg)
:label:`fig_ssd`


다음에서,
우리는 :numref:`fig_ssd`의 다른 블록들의 구현 세부 사항을 설명할 것입니다. 우선, 클래스 및 바운딩 박스 예측을 구현하는 방법에 대해 논의합니다.



### [**클래스 예측 레이어 (Class Prediction Layer)**]

객체 클래스의 수를 $q$라고 합시다.
그러면 앵커 박스에는 $q+1$개의 클래스가 있으며,
여기서 클래스 0은 배경입니다.
어떤 스케일에서,
특징 맵의 높이와 너비가 각각 $h$와 $w$라고 가정합니다.
이러한 특징 맵의 각 공간 위치를 중심으로 $a$개의 앵커 박스가 생성되면,
총 $hwa$개의 앵커 박스를 분류해야 합니다.
이것은 종종 무거운 파라미터화 비용으로 인해 완전 연결 레이어로 분류하는 것을 불가능하게 만듭니다.
:numref:`sec_nin`에서 클래스를 예측하기 위해 합성곱 레이어의 채널을 사용한 방법을 상기하십시오.
단일 샷 멀티박스 감지는 모델 복잡성을 줄이기 위해 동일한 기술을 사용합니다.

구체적으로,
클래스 예측 레이어는 특징 맵의 너비나 높이를 변경하지 않고 합성곱 레이어를 사용합니다.
이런 식으로,
특징 맵의 동일한 공간 차원(너비와 높이)에서
출력과 입력 사이에 일대일 대응이 있을 수 있습니다.
더 구체적으로,
어떤 공간 위치 ($x$, $y$)에서 출력 특징 맵의 채널은
입력 특징 맵의 ($x$, $y$)를 중심으로 하는
모든 앵커 박스에 대한 클래스 예측을 나타냅니다.
유효한 예측을 생성하려면 $a(q+1)$개의 출력 채널이 있어야 합니다.
여기서 동일한 공간 위치에 대해
인덱스 $i(q+1) + j$를 가진 출력 채널은
앵커 박스 $i$ ($0 \leq i < a$)에 대한
클래스 $j$ ($0 \leq j \leq q$)의 예측을 나타냅니다.

아래에서 우리는 `num_anchors` 및 `num_classes` 인수를 통해 각각 $a$와 $q$를 지정하여 이러한 클래스 예측 레이어를 정의합니다.
이 레이어는 패딩이 1인 $3\times3$ 합성곱 레이어를 사용합니다.
이 합성곱 레이어의 입력과 출력의 너비와 높이는 변경되지 않습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**바운딩 박스 예측 레이어 (Bounding Box Prediction Layer)**)

바운딩 박스 예측 레이어의 설계는 클래스 예측 레이어와 유사합니다.
유일한 차이점은 각 앵커 박스에 대한 출력 수에 있습니다:
여기서는 $q+1$개의 클래스가 아니라 4개의 오프셋을 예측해야 합니다.

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**다중 스케일 예측 연결 (Concatenating Predictions for Multiple Scales)**]

우리가 언급했듯이, 단일 샷 멀티박스 감지는
다중 스케일 특징 맵을 사용하여 앵커 박스를 생성하고 클래스와 오프셋을 예측합니다.
서로 다른 스케일에서,
특징 맵의 모양이나
동일한 단위를 중심으로 하는 앵커 박스의 수가 다를 수 있습니다.
따라서,
서로 다른 스케일에서 예측 출력의 모양이 다를 수 있습니다.

다음 예제에서,
우리는 동일한 미니배치에 대해 두 가지 다른 스케일의 특징 맵 `Y1`과 `Y2`를 구성합니다.
여기서 `Y2`의 높이와 너비는 `Y1`의 절반입니다.
클래스 예측을 예로 들어 보겠습니다.
`Y1`과 `Y2`의 각 단위에 대해
각각 5개와 3개의 앵커 박스가 생성된다고 가정합니다.
또한 객체 클래스의 수가 10개라고 가정합니다.
특징 맵 `Y1`과 `Y2`에 대해
클래스 예측 출력의 채널 수는 각각 $5\times(10+1)=55$와 $3\times(10+1)=33$이며,
두 출력 모양은 (배치 크기, 채널 수, 높이, 너비)입니다.

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

보시다시피, 배치 크기 차원을 제외하고
나머지 세 차원은 모두 크기가 다릅니다.
더 효율적인 계산을 위해 이 두 예측 출력을 연결하려면,
이 텐서들을 더 일관된 형식으로 변환해야 합니다.

채널 차원은 동일한 중심을 가진 앵커 박스에 대한 예측을 보유합니다.
우리는 먼저 이 차원을 가장 안쪽으로 이동합니다.
배치 크기는 다른 스케일에서도 동일하게 유지되므로,
우리는 예측 출력을
(배치 크기, 높이 $\times$ 너비 $\times$ 채널 수) 모양의
2차원 텐서로 변환할 수 있습니다.
그런 다음 차원 1을 따라
서로 다른 스케일의 출력을 연결할 수 있습니다.

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

이런 식으로,
`Y1`과 `Y2`의 채널, 높이, 너비가 다르더라도,
우리는 여전히 동일한 미니배치에 대해 두 가지 다른 스케일의 예측 출력을 연결할 수 있습니다.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**다운샘플링 블록 (Downsampling Block)**]

다중 스케일에서 객체를 감지하기 위해,
우리는 입력 특징 맵의 높이와 너비를 반으로 줄이는
다운샘플링 블록 `down_sample_blk`를 정의합니다.
사실,
이 블록은 :numref:`subsec_vgg-blocks`의 VGG 블록 설계를 적용합니다.
더 구체적으로,
각 다운샘플링 블록은
패딩이 1인 두 개의 $3\times3$ 합성곱 레이어와
그 뒤를 따르는 스트라이드가 2인 $2\times2$ 최대 풀링 레이어로 구성됩니다.
우리가 알다시피, 패딩이 1인 $3\times3$ 합성곱 레이어는 특징 맵의 모양을 변경하지 않습니다.
그러나 후속 $2\times2$ 최대 풀링은 입력 특징 맵의 높이와 너비를 반으로 줄입니다.
이 다운샘플링 블록의 입력 및 출력 특징 맵 모두에 대해,
$1\times 2+(3-1)+(3-1)=6$이기 때문에,
출력의 각 단위는
입력에 대해 $6\times6$ 수용 영역을 갖습니다.
따라서 다운샘플링 블록은 출력 특징 맵의 각 단위의 수용 영역을 확대합니다.

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

다음 예제에서, 우리가 구성한 다운샘플링 블록은 입력 채널 수를 변경하고 입력 특징 맵의 높이와 너비를 반으로 줄입니다.

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**기본 네트워크 블록 (Base Network Block)**]

기본 네트워크 블록은 입력 이미지에서 특징을 추출하는 데 사용됩니다.
간단하게 하기 위해,
우리는 각 블록에서 채널 수를 두 배로 늘리는
세 개의 다운샘플링 블록으로 구성된 작은 기본 네트워크를 구성합니다.
$256\times256$ 입력 이미지가 주어지면,
이 기본 네트워크 블록은 $32 \times 32$ 특징 맵을 출력합니다 ($256/2^3=32$).

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### 전체 모델 (The Complete Model)


[**완전한 단일 샷 멀티박스 감지 모델은 5개의 블록으로 구성됩니다.**] 
각 블록에서 생성된 특징 맵은
(i) 앵커 박스 생성
및 (ii) 이러한 앵커 박스의 클래스와 오프셋 예측 모두에 사용됩니다.
이 5개의 블록 중,
첫 번째 블록은 기본 네트워크 블록이고,
두 번째부터 네 번째 블록은 다운샘플링 블록이며,
마지막 블록은 글로벌 최대 풀링을 사용하여 높이와 너비를 모두 1로 줄입니다.
기술적으로,
두 번째부터 다섯 번째 블록은 모두
:numref:`fig_ssd`의
다중 스케일 특징 맵 블록입니다.

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

이제 각 블록에 대한 [**순방향 전파를 정의**]합니다.
이미지 분류 작업과 달리,
여기서의 출력에는
(i) CNN 특징 맵 `Y`,
(ii) 현재 스케일에서 `Y`를 사용하여 생성된 앵커 박스,
(iii) 이러한 앵커 박스에 대해 (`Y`를 기반으로) 예측된 클래스와 오프셋이 포함됩니다.

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

:numref:`fig_ssd`에서
상단에 더 가까운 다중 스케일 특징 맵 블록은
더 큰 객체를 감지하기 위한 것입니다;
따라서 더 큰 앵커 박스를 생성해야 합니다.
위의 순방향 전파에서,
각 다중 스케일 특징 맵 블록에서
우리는 호출된 `multibox_prior` 함수(:numref:`sec_anchor`에서 설명됨)의
`sizes` 인수를 통해 두 스케일 값 목록을 전달합니다.
다음에서,
0.2와 1.05 사이의 구간은
5개의 섹션으로 균등하게 분할되어
5개 블록에서 더 작은 스케일 값: 0.2, 0.37, 0.54, 0.71, 0.88을 결정합니다.
그런 다음 더 큰 스케일 값은
$\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$ 등으로 주어집니다.

[~~각 블록에 대한 하이퍼파라미터~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

이제 우리는 다음과 같이 [**전체 모델**] `TinySSD`를 정의할 수 있습니다.

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 할당 문 `self.blk_i = get_blk(i)`와 동일
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 여기서 `getattr(self, 'blk_%d' % i)`는 `self.blk_i`에 액세스합니다
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 할당 문 `self.blk_i = get_blk(i)`와 동일
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # 여기서 `getattr(self, 'blk_%d' % i)`는 `self.blk_i`에 액세스합니다
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

우리는 [**모델 인스턴스를 생성하고 이를 사용하여**]
$256 \times 256$ 이미지의 미니배치 `X`에 대해 [**순방향 전파를 수행합니다**].

이 섹션의 앞부분에서 본 것처럼,
첫 번째 블록은 $32 \times 32$ 특징 맵을 출력합니다.
두 번째부터 네 번째 다운샘플링 블록은
높이와 너비를 반으로 줄이고
다섯 번째 블록은 글로벌 풀링을 사용한다는 것을 상기하십시오.
공간 차원의 각 단위에 대해 4개의 앵커 박스가 생성되므로,
5개의 스케일 모두에서
각 이미지에 대해 총 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$개의 앵커 박스가 생성됩니다.

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 훈련 (Training)

이제 객체 감지를 위한 단일 샷 멀티박스 감지 모델을 훈련하는 방법을 설명하겠습니다.


### 데이터셋 읽기 및 모델 초기화 (Reading the Dataset and Initializing the Model)

우선,
:numref:`sec_object-detection-dataset`에서 설명한
[**바나나 감지 데이터셋을 읽어**] 봅시다.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

바나나 감지 데이터셋에는 클래스가 하나뿐입니다. 모델을 정의한 후,
(**파라미터를 초기화하고 최적화 알고리즘을 정의**)해야 합니다.

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**손실 및 평가 함수 정의 (Defining Loss and Evaluation Functions)**]

객체 감지에는 두 가지 유형의 손실이 있습니다.
첫 번째 손실은 앵커 박스의 클래스와 관련이 있습니다:
이 계산은
이미지 분류에 사용한 크로스 엔트로피 손실 함수를
간단히 재사용할 수 있습니다.
두 번째 손실은
양성(비배경) 앵커 박스의 오프셋과 관련이 있습니다:
이것은 회귀 문제입니다.
하지만 이 회귀 문제의 경우,
여기서는 :numref:`subsec_normal_distribution_and_squared_loss`에서 설명한 제곱 손실을 사용하지 않습니다.
대신,
우리는 예측과 실제 값 사이의 차이의 절대값인
$\ell_1$ 노름 손실을 사용합니다.
마스크 변수 `bbox_masks`는 손실 계산에서
음성 앵커 박스와 불법(패딩된) 앵커 박스를 필터링합니다.
결국, 우리는 앵커 박스 클래스 손실과 앵커 박스 오프셋 손실을 합산하여
모델의 손실 함수를 얻습니다.

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

우리는 분류 결과를 평가하기 위해 정확도를 사용할 수 있습니다.
오프셋에 사용된 $\ell_1$ 노름 손실로 인해,
예측된 바운딩 박스를 평가하기 위해 *평균 절대 오차(mean absolute error)*를 사용합니다.
이러한 예측 결과는
생성된 앵커 박스와 이에 대해 예측된 오프셋에서 얻어집니다.

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # 클래스 예측 결과가 마지막 차원에 있으므로 `argmax`는 이 차원을 지정해야 합니다.
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # 클래스 예측 결과가 마지막 차원에 있으므로 `argmax`는 이 차원을 지정해야 합니다.
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**모델 훈련 (Training the Model)**]

모델을 훈련할 때,
순방향 전파에서 다중 스케일 앵커 박스(`anchors`)를 생성하고
클래스(`cls_preds`)와 오프셋(`bbox_preds`)을 예측해야 합니다.
그런 다음 레이블 정보 `Y`를 기반으로 생성된 앵커 박스의 클래스(`cls_labels`)와 오프셋(`bbox_labels`)을 라벨링합니다.
마지막으로, 클래스와 오프셋의 예측된 값과 라벨링된 값을 사용하여
손실 함수를 계산합니다.
간결한 구현을 위해,
여기서는 테스트 데이터셋의 평가를 생략합니다.

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # 훈련 정확도의 합, 훈련 정확도 합의 예제 수,
    # 절대 오차의 합, 절대 오차 합의 예제 수
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # 다중 스케일 앵커 박스 생성 및 클래스와 오프셋 예측
            anchors, cls_preds, bbox_preds = net(X)
            # 이러한 앵커 박스의 클래스와 오프셋 라벨링
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # 클래스와 오프셋의 예측된 값과 라벨링된 값을 사용하여 손실 함수 계산
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 훈련 정확도의 합, 훈련 정확도 합의 예제 수,
    # 절대 오차의 합, 절대 오차 합의 예제 수
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 다중 스케일 앵커 박스 생성 및 클래스와 오프셋 예측
        anchors, cls_preds, bbox_preds = net(X)
        # 이러한 앵커 박스의 클래스와 오프셋 라벨링
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 클래스와 오프셋의 예측된 값과 라벨링된 값을 사용하여 손실 함수 계산
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**예측 (Prediction)**]

예측 중에,
목표는 이미지에서 관심 있는 모든 객체를 감지하는 것입니다. 아래
우리는 테스트 이미지를 읽고 크기를 조정하여,
합성곱 레이어에 필요한 4차원 텐서로 변환합니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

아래 `multibox_detection` 함수를 사용하여,
예측된 바운딩 박스는
앵커 박스와 예측된 오프셋에서 얻어집니다.
그런 다음 비최대 억제를 사용하여
유사한 예측된 바운딩 박스를 제거합니다.

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

마지막으로, 우리는 [**신뢰도가 0.9 이상인 모든 예측된 바운딩 박스를 출력으로 표시**]합니다.

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## 요약 (Summary)

* 단일 샷 멀티박스 감지는 다중 스케일 객체 감지 모델입니다. 기본 네트워크와 여러 다중 스케일 특징 맵 블록을 통해, 단일 샷 멀티박스 감지는 다양한 크기의 앵커 박스를 생성하고, 이러한 앵커 박스의 클래스와 오프셋(따라서 바운딩 박스)을 예측하여 다양한 크기의 객체를 감지합니다.
* 단일 샷 멀티박스 감지 모델을 훈련할 때, 손실 함수는 앵커 박스 클래스와 오프셋의 예측된 값과 라벨링된 값을 기반으로 계산됩니다.



## 연습 문제 (Exercises)

1. 손실 함수를 개선하여 단일 샷 멀티박스 감지를 개선할 수 있습니까? 예를 들어, 예측된 오프셋에 대해 $\ell_1$ 노름 손실을 부드러운(smooth) $\ell_1$ 노름 손실로 대체합니다. 이 손실 함수는 하이퍼파라미터 $\sigma$에 의해 제어되는 부드러움을 위해 0 주변에서 제곱 함수를 사용합니다:

$$ f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \textrm{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \textrm{otherwise}
    \end{cases}
$$

$\sigma$가 매우 클 때, 이 손실은 $\ell_1$ 노름 손실과 유사합니다. 값이 작을 때 손실 함수는 더 부드럽습니다.

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

또한 실험에서 우리는 클래스 예측을 위해 크로스 엔트로피 손실을 사용했습니다:
denoting by $p_j$ the predicted probability for the ground-truth class $j$, the cross-entropy loss is $-\log p_j$. We can also use the focal loss
:cite:`Lin.Goyal.Girshick.ea.2017`: given hyperparameters $\gamma > 0$
and $\alpha > 0$, this loss is defined as:

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$ 

As we can see, increasing $\gamma$ can effectively reduce the relative loss
for well-classified examples (e.g., $p_j > 0.5$)
so the training
can focus more on those difficult examples that are misclassified.

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 공간 제한으로 인해 이 섹션에서는 단일 샷 멀티박스 감지 모델의 일부 구현 세부 정보를 생략했습니다. 다음 측면에서 모델을 더 개선할 수 있습니까?
    1. 객체가 이미지에 비해 훨씬 작을 때, 모델은 입력 이미지의 크기를 더 크게 조정할 수 있습니다.
    1. 일반적으로 음성 앵커 박스의 수는 엄청나게 많습니다. 클래스 분포를 더 균형 있게 만들기 위해 음성 앵커 박스를 다운샘플링할 수 있습니다.
    1. 손실 함수에서 클래스 손실과 오프셋 손실에 서로 다른 가중치 하이퍼파라미터를 할당합니다.
    1. 단일 샷 멀티박스 감지 논문 :cite:`Liu.Anguelov.Erhan.ea.2016`에 있는 것과 같은 다른 방법을 사용하여 객체 감지 모델을 평가합니다.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:

```