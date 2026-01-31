# 객체 감지 및 바운딩 박스 (Object Detection and Bounding Boxes)
:label:`sec_bbox`


이전 섹션(예: :numref:`sec_alexnet`--:numref:`sec_googlenet`)에서,
우리는 이미지 분류를 위한 다양한 모델을 소개했습니다.
이미지 분류 작업에서,
우리는 이미지에 *하나의* 주요 객체만 있다고 가정하고
그 범주를 인식하는 방법에만 초점을 맞춥니다.
그러나 관심 있는 이미지에는 종종 *여러* 객체가 있습니다.
우리는 범주뿐만 아니라 이미지 내의 구체적인 위치도 알고 싶어 합니다.
컴퓨터 비전에서는 이러한 작업을 *객체 감지(object detection)* (또는 *객체 인식(object recognition)*)라고 합니다.

객체 감지는 많은 분야에서 널리 적용되었습니다.
예를 들어, 자율 주행은 캡처된 비디오 이미지에서 차량, 보행자, 도로 및 장애물의 위치를 감지하여 주행 경로를 계획해야 합니다.
또한 로봇은 환경을 탐색하는 동안 관심 객체를 감지하고 위치를 파악하기 위해 이 기술을 사용할 수 있습니다.
게다가 보안 시스템은 침입자나 폭탄과 같은 비정상적인 물체를 감지해야 할 수도 있습니다.

다음 몇 섹션에서는 객체 감지를 위한 몇 가지 딥러닝 방법을 소개합니다.
객체의 *위치(positions)* (또는 *장소(locations)*)에 대한 소개로 시작하겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

이 섹션에서 사용할 샘플 이미지를 로드합니다. 이미지 왼쪽에 개가 있고 오른쪽에 고양이가 있는 것을 볼 수 있습니다.
이들은 이 이미지의 두 가지 주요 객체입니다.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 바운딩 박스 (Bounding Boxes)


객체 감지에서,
우리는 일반적으로 객체의 공간적 위치를 설명하기 위해 *바운딩 박스(bounding box)*를 사용합니다.
바운딩 박스는 직사각형이며, 직사각형의 왼쪽 상단 모서리의 $x$ 및 $y$ 좌표와 오른쪽 하단 모서리의 좌표에 의해 결정됩니다.
일반적으로 사용되는 또 다른 바운딩 박스 표현은 바운딩 박스 중심의 $(x, y)$축 좌표와 박스의 너비 및 높이입니다.

[**여기서 우리는 이 두 가지 표현 사이를 변환하는 함수를 정의합니다**]:
`box_corner_to_center`는 두 모서리 표현에서 중심-너비-높이 표현으로 변환하고,
`box_center_to_corner`는 그 반대로 변환합니다.
입력 인수 `boxes`는 ($n$, 4) 모양의 2차원 텐서여야 합니다. 여기서 $n$은 바운딩 박스의 수입니다.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """(왼쪽 상단, 오른쪽 하단)에서 (중심, 너비, 높이)로 변환합니다."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """(중심, 너비, 높이)에서 (왼쪽 상단, 오른쪽 하단)으로 변환합니다."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

좌표 정보를 기반으로 [**이미지에 있는 개와 고양이의 바운딩 박스를 정의**]합니다.
이미지 좌표의 원점은 이미지의 왼쪽 상단 모서리이며, 오른쪽과 아래쪽이 각각 $x$축과 $y$축의 양의 방향입니다.

```{.python .input}
#@tab all
# 여기서 `bbox`는 bounding box의 약어입니다
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

두 번 변환하여 두 바운딩 박스 변환 함수의 정확성을 확인할 수 있습니다.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

정확한지 확인하기 위해 [**이미지에 바운딩 박스를 그려봅시다**].
그리기 전에, `matplotlib` 패키지의 바운딩 박스 형식으로 바운딩 박스를 나타내는 도우미 함수 `bbox_to_rect`를 정의합니다.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """바운딩 박스를 matplotlib 형식으로 변환합니다."""
    # 바운딩 박스 (왼쪽 상단 x, 왼쪽 상단 y, 오른쪽 하단 x, 오른쪽 하단 y) 형식을
    # matplotlib 형식 ((왼쪽 상단 x, 왼쪽 상단 y), 너비, 높이)로 변환합니다
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

이미지에 바운딩 박스를 추가한 후,
두 객체의 주요 윤곽이 기본적으로 두 박스 안에 있는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 요약 (Summary)

* 객체 감지는 이미지에서 관심 있는 모든 객체뿐만 아니라 그 위치도 인식합니다. 위치는 일반적으로 직사각형 바운딩 박스로 표현됩니다.
* 우리는 일반적으로 사용되는 두 가지 바운딩 박스 표현 사이를 변환할 수 있습니다.

## 연습 문제 (Exercises)

1. 다른 이미지를 찾아 객체를 포함하는 바운딩 박스에 레이블을 지정해 보십시오. 바운딩 박스 레이블링과 범주 레이블링을 비교해 보십시오: 어느 것이 일반적으로 더 오래 걸립니까?
2. `box_corner_to_center`와 `box_center_to_corner`의 입력 인수 `boxes`의 가장 안쪽 차원이 항상 4인 이유는 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab: