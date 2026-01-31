# 다중 스케일 객체 감지 (Multiscale Object Detection)
:label:`sec_multiscale-object-detection`


:numref:`sec_anchor`에서,
우리는 입력 이미지의 각 픽셀을 중심으로 여러 앵커 박스를 생성했습니다.
본질적으로 이러한 앵커 박스는
이미지의 다른 영역 샘플을 나타냅니다.
그러나,
*모든* 픽셀에 대해 생성하면 계산할 앵커 박스가 너무 많아질 수 있습니다.
$561 \times 728$ 입력 이미지를 생각해 보십시오.
각 픽셀을 중심으로
다양한 모양의 앵커 박스 5개를 생성하면,
이미지에서 200만 개 이상의 앵커 박스($561 \times 728 \times 5$)를 라벨링하고 예측해야 합니다.

## 다중 스케일 앵커 박스 (Multiscale Anchor Boxes)
:label:`subsec_multiscale-anchor-boxes`

여러분은
이미지에서 앵커 박스를 줄이는 것이 어렵지 않다는 것을 깨달을 수 있습니다.
예를 들어,
입력 이미지에서 픽셀의 작은 부분만 균일하게 샘플링하여
이를 중심으로 앵커 박스를 생성할 수 있습니다.
또한,
다양한 스케일에서
다양한 크기의 앵커 박스를 생성할 수 있습니다.
직관적으로,
작은 객체는 큰 객체보다
이미지에 나타날 가능성이 더 높습니다.
예를 들어,
$1 \times 1$, $1 \times 2$, $2 \times 2$ 객체는
$2 \times 2$ 이미지에
각각 4, 2, 1가지 가능한 방식으로 나타날 수 있습니다.
따라서 작은 앵커 박스를 사용하여 작은 객체를 감지할 때는 더 많은 영역을 샘플링할 수 있고,
큰 객체의 경우 더 적은 영역을 샘플링할 수 있습니다.

다중 스케일에서 앵커 박스를 생성하는 방법을 보여주기 위해 이미지를 읽어보겠습니다.
높이와 너비는 각각 561과 728 픽셀입니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

:numref:`sec_conv_layer`에서
우리는 합성곱 레이어의 2차원 배열 출력을 특징 맵(feature map)이라고 불렀습니다.
특징 맵 모양을 정의함으로써,
모든 이미지에서 균일하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.


`display_anchors` 함수는 아래에 정의되어 있습니다.
[**우리는 각 단위(픽셀)를 앵커 박스 중심으로 하여 특징 맵(`fmap`)에 앵커 박스(`anchors`)를 생성합니다.**]
앵커 박스(`anchors`)의 $(x, y)$축 좌표 값은
특징 맵(`fmap`)의 너비와 높이로 나누어졌으므로,
이 값은 0과 1 사이이며,
이는 특징 맵에서 앵커 박스의 상대적 위치를 나타냅니다.

앵커 박스(`anchors`)의 중심은
특징 맵(`fmap`)의 모든 단위에 퍼져 있으므로,
이러한 중심은 상대적 공간 위치 측면에서
모든 입력 이미지에 *균일하게* 분포되어야 합니다.
더 구체적으로,
특징 맵의 너비와 높이가 각각 `fmap_w`와 `fmap_h`로 주어지면,
다음 함수는 모든 입력 이미지에서
`fmap_h` 행과 `fmap_w` 열의 픽셀을 *균일하게* 샘플링합니다.
이러한 균일하게 샘플링된 픽셀을 중심으로,
스케일 `s`(목록 `s`의 길이가 1이라고 가정)와 다양한 가로세로 비율(`ratios`)의 앵커 박스가
생성됩니다.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 처음 두 차원의 값은 출력에 영향을 미치지 않습니다
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 처음 두 차원의 값은 출력에 영향을 미치지 않습니다
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

먼저, [**작은 객체의 감지를 고려해 봅시다**].
표시할 때 구별하기 쉽도록 하기 위해, 여기서는 중심이 다른 앵커 박스가 겹치지 않습니다:
앵커 박스 스케일은 0.15로 설정되고
특징 맵의 높이와 너비는 4로 설정됩니다.
이미지의 4행 4열에 있는 앵커 박스의 중심이 균일하게 분포되어 있음을 알 수 있습니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

우리는 [**특징 맵의 높이와 너비를 절반으로 줄이고 더 큰 앵커 박스를 사용하여 더 큰 객체를 감지하는 것**]으로 이동합니다. 스케일을 0.4로 설정하면,
일부 앵커 박스가 서로 겹칩니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

마지막으로, [**특징 맵의 높이와 너비를 절반으로 더 줄이고 앵커 박스 스케일을 0.8로 늘립니다**]. 이제 앵커 박스의 중심이 이미지의 중심이 됩니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 다중 스케일 감지 (Multiscale Detection)


다중 스케일 앵커 박스를 생성했으므로,
이를 사용하여 다양한 스케일에서 다양한 크기의 객체를 감지합니다.
다음에서
우리는 :numref:`sec_ssd`에서 구현할
CNN 기반 다중 스케일 객체 감지 방법을 소개합니다.

어떤 스케일에서,
$h \times w$ 모양의 특징 맵이 $c$개 있다고 가정해 봅시다.
:numref:`subsec_multiscale-anchor-boxes`의 방법을 사용하여
$hw$ 세트의 앵커 박스를 생성합니다.
여기서 각 세트에는 중심이 같은 $a$개의 앵커 박스가 있습니다.
예를 들어,
:numref:`subsec_multiscale-anchor-boxes`의 실험 첫 번째 스케일에서,
10개(채널 수)의 $4 \times 4$ 특징 맵이 주어졌을 때,
우리는 16세트의 앵커 박스를 생성했으며,
각 세트에는 중심이 같은 3개의 앵커 박스가 포함되어 있습니다.
다음으로, 각 앵커 박스는 실제 바운딩 박스를 기반으로
클래스와 오프셋으로 라벨링됩니다. 현재 스케일에서 객체 감지 모델은 입력 이미지에 있는 $hw$ 세트의 앵커 박스(서로 다른 세트는 다른 중심을 가짐)의 클래스와 오프셋을 예측해야 합니다.


여기서 $c$개의 특징 맵이
입력 이미지를 기반으로 CNN 순방향 전파에 의해 얻은
중간 출력이라고 가정합니다. 각 특징 맵에는 $hw$개의 서로 다른 공간 위치가 있으므로,
동일한 공간 위치는 $c$개의 단위를 갖는 것으로 생각할 수 있습니다.
:numref:`sec_conv_layer`의 수용 영역(receptive field) 정의에 따르면,
특징 맵의 동일한 공간 위치에 있는
이 $c$개의 단위는
입력 이미지에서 동일한 수용 영역을 갖습니다:
그들은 동일한 수용 영역 내의 입력 이미지 정보를 나타냅니다.
따라서 우리는 동일한 공간 위치에 있는 특징 맵의 $c$개 단위를
이 공간 위치를 사용하여 생성된 $a$개 앵커 박스의
클래스와 오프셋으로 변환할 수 있습니다.
본질적으로,
우리는 입력 이미지의 특정 수용 영역에 있는 정보를 사용하여
입력 이미지의 해당 수용 영역에 가까운
앵커 박스의 클래스와 오프셋을 예측합니다.


서로 다른 레이어의 특징 맵이
입력 이미지에서 다양한 크기의 수용 영역을 가질 때, 이들은 서로 다른 크기의 객체를 감지하는 데 사용될 수 있습니다.
예를 들어, 우리는 출력 레이어에 더 가까운 특징 맵의 단위가
더 넓은 수용 영역을 갖도록 신경망을 설계하여,
입력 이미지에서 더 큰 객체를 감지할 수 있도록 할 수 있습니다.

한마디로, 우리는 다중 스케일 객체 감지를 위해
심층 신경망에 의한 여러 수준의 이미지 계층별 표현을 활용할 수 있습니다.
:numref:`sec_ssd`의 구체적인 예를 통해 이것이 어떻게 작동하는지 보여줄 것입니다.




## 요약 (Summary)

* 다중 스케일에서, 우리는 다양한 크기의 객체를 감지하기 위해 다양한 크기의 앵커 박스를 생성할 수 있습니다.
* 특징 맵의 모양을 정의함으로써, 우리는 모든 이미지에서 균일하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.
* 우리는 입력 이미지의 특정 수용 영역에 있는 정보를 사용하여 입력 이미지의 해당 수용 영역에 가까운 앵커 박스의 클래스와 오프셋을 예측합니다.
* 딥러닝을 통해, 우리는 다중 스케일 객체 감지를 위해 여러 수준의 이미지 계층별 표현을 활용할 수 있습니다.


## 연습 문제 (Exercises)

1. :numref:`sec_alexnet`에서의 논의에 따르면, 심층 신경망은 이미지에 대해 추상화 수준이 증가하는 계층적 특징을 학습합니다. 다중 스케일 객체 감지에서, 서로 다른 스케일의 특징 맵은 서로 다른 추상화 수준에 해당합니까? 그 이유는 무엇입니까?
1. :numref:`subsec_multiscale-anchor-boxes`의 실험 첫 번째 스케일(`fmap_w=4, fmap_h=4`)에서, 겹칠 수 있는 균일하게 분포된 앵커 박스를 생성하십시오.
1. 모양이 $1 \times c \times h \times w$인 특징 맵 변수가 주어졌을 때(여기서 $c, h, w$는 각각 특징 맵의 채널 수, 높이, 너비임), 이 변수를 앵커 박스의 클래스와 오프셋으로 어떻게 변환할 수 있습니까? 출력의 모양은 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab: