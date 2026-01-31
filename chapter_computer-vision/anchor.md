# 앵커 박스 (Anchor Boxes)
:label:`sec_anchor`


객체 감지 알고리즘은 일반적으로
입력 이미지에서 수많은 영역을 샘플링하고, 이 영역에 관심 있는 객체가 포함되어 있는지 확인하며,
객체의 *실제 바운딩 박스(ground-truth bounding boxes)*를
더 정확하게 예측하도록 영역의 경계를 조정합니다.
모델마다 다른 영역 샘플링 방식을 채택할 수 있습니다.
여기서는 그러한 방법 중 하나를 소개합니다.
각 픽셀을 중심으로 다양한 스케일과 가로세로 비율(aspect ratio)을 가진 여러 바운딩 박스를 생성하는 것입니다.
이러한 바운딩 박스를 *앵커 박스(anchor boxes)*라고 합니다.
:numref:`sec_ssd`에서 앵커 박스를 기반으로 한 객체 감지 모델을 설계할 것입니다.

먼저 더 간결한 출력을 위해 인쇄 정확도를 수정해 보겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # 인쇄 정확도 단순화
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # 인쇄 정확도 단순화
```

## 여러 앵커 박스 생성 (Generating Multiple Anchor Boxes)

입력 이미지의 높이가 $h$, 너비가 $w$라고 가정해 보겠습니다.
우리는 이미지의 각 픽셀을 중심으로 다양한 모양의 앵커 박스를 생성합니다.
*스케일(scale)*을 $s\in (0, 1]$로,
*가로세로 비율(aspect ratio)* (높이 대비 너비의 비율)을 $r > 0$이라고 합시다.
그러면 [**앵커 박스의 너비와 높이는 각각 $ws\sqrt{r}$와 $hs/\sqrt{r}$입니다.**] 
중심 위치가 주어지면 너비와 높이가 알려진 앵커 박스가 결정됩니다.

다양한 모양의 여러 앵커 박스를 생성하기 위해,
일련의 스케일 $s_1,\ldots, s_n$과
일련의 가로세로 비율 $r_1,\ldots, r_m$을 설정해 봅시다.
각 픽셀을 중심으로 이러한 스케일과 가로세로 비율의 모든 조합을 사용할 때,
입력 이미지에는 총 $whnm$개의 앵커 박스가 생깁니다. 이러한 앵커 박스가 모든 실제 바운딩 박스를 커버할 수 있지만, 계산 복잡도가 너무 높아질 수 있습니다.
실제로,
우리는 (**$s_1$ 또는 $r_1$을 포함하는 조합만 고려**)할 수 있습니다:

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**)

즉, 동일한 픽셀을 중심으로 하는 앵커 박스의 수는 $n+m-1$입니다. 전체 입력 이미지에 대해 총 $wh(n+m-1)$개의 앵커 박스를 생성하게 됩니다.

위의 앵커 박스 생성 방법은 다음 `multibox_prior` 함수에 구현되어 있습니다. 입력 이미지, 스케일 목록, 가로세로 비율 목록을 지정하면 이 함수가 모든 앵커 박스를 반환합니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """서로 다른 모양의 앵커 박스를 픽셀 단위로 생성합니다."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # 앵커를 픽셀 중심으로 이동하려면 오프셋이 필요합니다. 
    # 픽셀의 높이가 1이고 너비가 1이므로 중심을 0.5만큼 오프셋하기로 선택합니다.
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # y축의 스케일링된 단계
    steps_w = 1.0 / in_width  # x축의 스케일링된 단계

    # 앵커 박스의 모든 중심점 생성
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # 나중에 앵커 박스 모서리 좌표(xmin, xmax, ymin, ymax)를 만드는 데 사용되는
    # `boxes_per_pixel` 수의 높이와 너비를 생성합니다.
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # 직사각형 입력 처리
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # 반 높이와 반 너비를 얻으려면 2로 나눕니다.
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # 각 중심점에는 `boxes_per_pixel` 수의 앵커 박스가 있으므로
    # `boxes_per_pixel` 반복으로 모든 앵커 박스 중심의 그리드를 생성합니다.
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """서로 다른 모양의 앵커 박스를 픽셀 단위로 생성합니다."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # 앵커를 픽셀 중심으로 이동하려면 오프셋이 필요합니다. 
    # 픽셀의 높이가 1이고 너비가 1이므로 중심을 0.5만큼 오프셋하기로 선택합니다.
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # y축의 스케일링된 단계
    steps_w = 1.0 / in_width  # x축의 스케일링된 단계

    # 앵커 박스의 모든 중심점 생성
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 나중에 앵커 박스 모서리 좌표(xmin, xmax, ymin, ymax)를 만드는 데 사용되는
    # `boxes_per_pixel` 수의 높이와 너비를 생성합니다.
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 직사각형 입력 처리
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 반 높이와 반 너비를 얻으려면 2로 나눕니다.
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 각 중심점에는 `boxes_per_pixel` 수의 앵커 박스가 있으므로
    # `boxes_per_pixel` 반복으로 모든 앵커 박스 중심의 그리드를 생성합니다.
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

[**반환된 앵커 박스 변수 `Y`의 모양**]은 (배치 크기, 앵커 박스 수, 4)임을 알 수 있습니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # 입력 데이터 구성
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # 입력 데이터 구성
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

앵커 박스 변수 `Y`의 모양을 (이미지 높이, 이미지 너비, 동일한 픽셀을 중심으로 하는 앵커 박스 수, 4)로 변경하면,
지정된 픽셀 위치를 중심으로 하는 모든 앵커 박스를 얻을 수 있습니다.
다음에서,
우리는 [**(250, 250)을 중심으로 하는 첫 번째 앵커 박스에 액세스합니다**]. 여기에는 앵커 박스의 왼쪽 상단 모서리의 $(x, y)$축 좌표와 오른쪽 하단 모서리의 $(x, y)$축 좌표인 네 가지 요소가 있습니다.
두 축의 좌표 값은 각각 이미지의 너비와 높이로 나누어집니다.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

[**이미지의 한 픽셀을 중심으로 하는 모든 앵커 박스를 표시**]하기 위해,
이미지에 여러 바운딩 박스를 그리는 `show_bboxes` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """바운딩 박스를 표시합니다."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

방금 본 것처럼 변수 `boxes`의 $x$ 및 $y$ 축 좌표 값은 각각 이미지의 너비와 높이로 나누어졌습니다.
앵커 박스를 그릴 때,
원래 좌표 값을 복원해야 하므로,
아래에서 `bbox_scale` 변수를 정의합니다.
이제 이미지의 (250, 250)을 중심으로 하는 모든 앵커 박스를 그릴 수 있습니다.
보시다시피 스케일이 0.75이고 가로세로 비율이 1인 파란색 앵커 박스가 이미지의 개를 잘 둘러싸고 있습니다.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**IoU (Intersection over Union)**]

방금 앵커 박스가 이미지의 개를 "잘" 둘러싸고 있다고 언급했습니다.
객체의 실제 바운딩 박스가 알려진 경우, 여기서 "잘"을 어떻게 정량화할 수 있습니까?
직관적으로, 우리는 앵커 박스와 실제 바운딩 박스 사이의 유사성을 측정할 수 있습니다.
우리는 *자카드 지수(Jaccard index)*가 두 집합 간의 유사성을 측정할 수 있음을 알고 있습니다. 집합 $\mathcal{A}$와 $\mathcal{B}$가 주어지면, 자카드 지수는 교집합의 크기를 합집합의 크기로 나눈 것입니다:

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$ 


사실, 우리는 모든 바운딩 박스의 픽셀 영역을 픽셀 집합으로 간주할 수 있습니다.
이런 식으로 픽셀 집합의 자카드 지수를 통해 두 바운딩 박스의 유사성을 측정할 수 있습니다. 두 바운딩 박스의 경우, 우리는 일반적으로 이 자카드 지수를 *IoU(Intersection over Union)*라고 부르며, 이는 :numref:`fig_iou`와 같이 교차 영역 대 합집합 영역의 비율입니다.
IoU의 범위는 0에서 1 사이입니다:
0은 두 바운딩 박스가 전혀 겹치지 않음을 의미하고,
1은 두 바운딩 박스가 동일함을 나타냅니다.

![IoU는 두 바운딩 박스의 교차 영역 대 합집합 영역의 비율입니다.](../img/iou.svg)
:label:`fig_iou`

이 섹션의 나머지 부분에서는 IoU를 사용하여 앵커 박스와 실제 바운딩 박스 간의 유사성, 그리고 서로 다른 앵커 박스 간의 유사성을 측정합니다.
두 개의 앵커 또는 바운딩 박스 목록이 주어지면,
다음 `box_iou`는 이 두 목록에 걸쳐 쌍별 IoU를 계산합니다.

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """두 앵커 또는 바운딩 박스 목록에 걸쳐 쌍별 IoU를 계산합니다."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`의 모양: (boxes1 수, 4),
    # (boxes2 수, 4), (boxes1 수,), (boxes2 수,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # `inter_upperlefts`, `inter_lowerrights`, `inters`의 모양: (boxes1 수,
    # boxes2 수, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # `inter_areas` 및 `union_areas`의 모양: (boxes1 수, boxes2 수)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """두 앵커 또는 바운딩 박스 목록에 걸쳐 쌍별 IoU를 계산합니다."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`의 모양: (boxes1 수, 4),
    # (boxes2 수, 4), (boxes1 수,), (boxes2 수,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # `inter_upperlefts`, `inter_lowerrights`, `inters`의 모양: (boxes1 수,
    # boxes2 수, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # `inter_areas` 및 `union_areas`의 모양: (boxes1 수, boxes2 수)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## 훈련 데이터에서 앵커 박스 라벨링 (Labeling Anchor Boxes in Training Data)
:label:`subsec_labeling-anchor-boxes`


훈련 데이터셋에서,
우리는 각 앵커 박스를 훈련 예제로 간주합니다.
객체 감지 모델을 훈련하려면,
각 앵커 박스에 대한 *클래스(class)* 및 *오프셋(offset)* 레이블이 필요합니다.
전자는 앵커 박스와 관련된 객체의 클래스이고,
후자는 앵커 박스에 대한 실제 바운딩 박스의 오프셋입니다.
예측 중에,
각 이미지에 대해
여러 앵커 박스를 생성하고,
모든 앵커 박스에 대한 클래스와 오프셋을 예측하고,
예측된 오프셋에 따라 위치를 조정하여 예측된 바운딩 박스를 얻고,
마지막으로 특정 기준을 충족하는 예측된 바운딩 박스만 출력합니다.


알다시피, 객체 감지 훈련 세트에는
*실제 바운딩 박스*의 위치와
둘러싸인 객체의 클래스에 대한 레이블이 함께 제공됩니다.
생성된 *앵커 박스*에 레이블을 지정하기 위해,
앵커 박스에 가장 가까운 *할당된* 실제 바운딩 박스의 레이블 위치와 클래스를 참조합니다.
다음에서,
가장 가까운 실제 바운딩 박스를 앵커 박스에 할당하는 알고리즘을 설명합니다.

### [**실제 바운딩 박스를 앵커 박스에 할당하기**]

이미지가 주어졌을 때,
앵커 박스가 $A_1, A_2, \ldots, A_{n_a}$이고 실제 바운딩 박스가 $B_1, B_2, \ldots, B_{n_b}$라고 가정합니다. 여기서 $n_a \geq n_b$입니다.
행렬 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$를 정의합시다. 여기서 $i$번째 행과 $j$번째 열의 요소 $x_{ij}$는 앵커 박스 $A_i$와 실제 바운딩 박스 $B_j$의 IoU입니다. 알고리즘은 다음 단계로 구성됩니다:

1. 행렬 $\mathbf{X}$에서 가장 큰 요소를 찾아 행과 열 인덱스를 각각 $i_1$과 $j_1$로 표시합니다. 그러면 실제 바운딩 박스 $B_{j_1}$이 앵커 박스 $A_{i_1}$에 할당됩니다. 이것은 꽤 직관적입니다. $A_{i_1}$과 $B_{j_1}$이 모든 앵커 박스와 실제 바운딩 박스 쌍 중에서 가장 가깝기 때문입니다. 첫 번째 할당 후, 행렬 $\mathbf{X}$의 ${i_1}$번째 행과 ${j_1}$번째 열의 모든 요소를 버립니다.
1. 행렬 $\mathbf{X}$의 나머지 요소 중에서 가장 큰 요소를 찾아 행과 열 인덱스를 각각 $i_2$와 $j_2$로 표시합니다. 실제 바운딩 박스 $B_{j_2}$를 앵커 박스 $A_{i_2}$에 할당하고 행렬 $\mathbf{X}$의 ${i_2}$번째 행과 ${j_2}$번째 열의 모든 요소를 버립니다.
1. 이 시점에서 행렬 $\mathbf{X}$의 두 행과 두 열의 요소가 버려졌습니다. 행렬 $\mathbf{X}$의 $n_b$ 열에 있는 모든 요소가 버려질 때까지 진행합니다. 이때, 우리는 $n_b$개의 앵커 박스 각각에 실제 바운딩 박스를 할당했습니다.
1. 나머지 $n_a - n_b$ 앵커 박스만 순회합니다. 예를 들어, 앵커 박스 $A_i$가 주어지면 행렬 $\mathbf{X}$의 $i$번째 행 전체에서 $A_i$와 가장 큰 IoU를 가진 실제 바운딩 박스 $B_j$를 찾고, 이 IoU가 미리 정의된 임계값보다 큰 경우에만 $B_j$를 $A_i$에 할당합니다.

구체적인 예를 사용하여 위 알고리즘을 설명해 보겠습니다.
:numref:`fig_anchor_label` (왼쪽)과 같이, 행렬 $\mathbf{X}$의 최대값이 $x_{23}$이라고 가정하면 실제 바운딩 박스 $B_3$을 앵커 박스 $A_2$에 할당합니다.
그런 다음 행렬의 2행 3열의 모든 요소를 버리고, 나머지 요소(음영 처리된 영역)에서 가장 큰 $x_{71}$을 찾아 실제 바운딩 박스 $B_1$을 앵커 박스 $A_7$에 할당합니다.
다음으로, :numref:`fig_anchor_label` (가운데)와 같이 행렬의 7행 1열의 모든 요소를 버리고, 나머지 요소(음영 처리된 영역)에서 가장 큰 $x_{54}$를 찾아 실제 바운딩 박스 $B_4$를 앵커 박스 $A_5$에 할당합니다.
마지막으로, :numref:`fig_anchor_label` (오른쪽)과 같이 행렬의 5행 4열의 모든 요소를 버리고, 나머지 요소(음영 처리된 영역)에서 가장 큰 $x_{92}$를 찾아 실제 바운딩 박스 $B_2$를 앵커 박스 $A_9$에 할당합니다.
그 후에는 나머지 앵커 박스 $A_1, A_3, A_4, A_6, A_8$을 순회하고 임계값에 따라 실제 바운딩 박스를 할당할지 여부를 결정하기만 하면 됩니다.

![실제 바운딩 박스를 앵커 박스에 할당하기.](../img/anchor-label.svg)
:label:`fig_anchor_label`

이 알고리즘은 다음 `assign_anchor_to_bbox` 함수에 구현되어 있습니다.

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """가장 가까운 실제 바운딩 박스를 앵커 박스에 할당합니다."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # i번째 행과 j번째 열의 요소 x_ij는 앵커 박스 i와 실제 바운딩 박스 j의 IoU입니다.
    jaccard = box_iou(anchors, ground_truth)
    # 각 앵커에 대해 할당된 실제 바운딩 박스를 유지할 텐서를 초기화합니다.
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # 임계값에 따라 실제 바운딩 박스를 할당합니다.
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # 가장 큰 IoU 찾기
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """가장 가까운 실제 바운딩 박스를 앵커 박스에 할당합니다."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # i번째 행과 j번째 열의 요소 x_ij는 앵커 박스 i와 실제 바운딩 박스 j의 IoU입니다.
    jaccard = box_iou(anchors, ground_truth)
    # 각 앵커에 대해 할당된 실제 바운딩 박스를 유지할 텐서를 초기화합니다.
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 임계값에 따라 실제 바운딩 박스를 할당합니다.
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # 가장 큰 IoU 찾기
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### 클래스 및 오프셋 라벨링 (Labeling Classes and Offsets)

이제 각 앵커 박스에 대한 클래스와 오프셋을 라벨링할 수 있습니다. 앵커 박스 $A$가 실제 바운딩 박스 $B$에 할당되었다고 가정합니다.
한편으로,
앵커 박스 $A$의 클래스는 $B$의 클래스로 라벨링됩니다.
다른 한편으로, 앵커 박스 $A$의 오프셋은 $B$와 $A$의 중심 좌표 사이의 상대적 위치와
이 두 박스 사이의 상대적 크기에 따라 라벨링됩니다.
데이터셋에 있는 다양한 상자의 위치와 크기가 주어지면,
우리는 더 균일하게 분포된 오프셋으로 이어질 수 있는 변환을
해당 상대적 위치와 크기에 적용할 수 있습니다.
이러한 오프셋은 맞추기(fit) 더 쉽습니다.
여기서는 일반적인 변환을 설명합니다.
[**$A$와 $B$의 중심 좌표가 각각 $(x_a, y_a)$와 $(x_b, y_b)$이고,
너비가 $w_a$와 $w_b$,
높이가 $h_a$와 $h_b$라고 주어졌을 때. 우리는 $A$의 오프셋을 다음과 같이 라벨링할 수 있습니다.

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),
**]
여기서 상수의 기본값은 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, $\sigma_w=\sigma_h=0.2$입니다.
이 변환은 아래 `offset_boxes` 함수에 구현되어 있습니다.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """앵커 박스 오프셋을 위한 변환."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

앵커 박스에 할당된 실제 바운딩 박스가 없는 경우, 앵커 박스의 클래스를 "배경(background)"으로 라벨링합니다.
클래스가 배경인 앵커 박스를 종종 *음성(negative)* 앵커 박스라고 하고,
나머지를 *양성(positive)* 앵커 박스라고 합니다.
우리는 다음 `multibox_target` 함수를 구현하여
실제 바운딩 박스(`labels` 인수)를 사용하여 [**앵커 박스(`anchors` 인수)에 대한 클래스와 오프셋을 라벨링**]합니다.
이 함수는 배경 클래스를 0으로 설정하고 새 클래스의 정수 인덱스를 1씩 증가시킵니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """실제 바운딩 박스를 사용하여 앵커 박스에 라벨을 지정합니다."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0), 
                                            axis=-1)), (1, 4)).astype('int32')
        # 클래스 레이블 및 할당된 바운딩 박스 좌표를 0으로 초기화
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # 할당된 실제 바운딩 박스를 사용하여 앵커 박스의 클래스에 라벨을 지정합니다.
        # 앵커 박스에 할당된 것이 없으면 클래스를 배경으로 라벨링합니다(값은 0으로 유지됨).
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 오프셋 변환
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """실제 바운딩 박스를 사용하여 앵커 박스에 라벨을 지정합니다."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 클래스 레이블 및 할당된 바운딩 박스 좌표를 0으로 초기화
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 할당된 실제 바운딩 박스를 사용하여 앵커 박스의 클래스에 라벨을 지정합니다.
        # 앵커 박스에 할당된 것이 없으면 클래스를 배경으로 라벨링합니다(값은 0으로 유지됨).
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 오프셋 변환
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 예제 (An Example)

구체적인 예를 통해 앵커 박스 라벨링을 설명해 보겠습니다.
로드된 이미지의 개와 고양이에 대한 실제 바운딩 박스를 정의합니다.
첫 번째 요소는 클래스(개는 0, 고양이는 1)이고 나머지 4개 요소는
왼쪽 상단 모서리와 오른쪽 하단 모서리의 $(x, y)$축 좌표입니다(범위는 0과 1 사이).
또한 왼쪽 상단 모서리와 오른쪽 하단 모서리의 좌표를 사용하여
라벨링할 5개의 앵커 박스 $A_0, \ldots, A_4$를 구성합니다(인덱스는 0부터 시작).
그런 다음 [**이 실제 바운딩 박스와 앵커 박스를 이미지에 그립니다.**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

위에서 정의한 `multibox_target` 함수를 사용하여,
개와 고양이에 대한 [**실제 바운딩 박스를 기반으로
이러한 앵커 박스의 클래스와 오프셋을 라벨링**]할 수 있습니다.
이 예에서 배경, 개, 고양이 클래스의 인덱스는 각각 0, 1, 2입니다.
아래에서 앵커 박스 및 실제 바운딩 박스의 예제에 대한 차원을 추가합니다.

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

반환된 결과에는 세 가지 항목이 있으며 모두 텐서 형식입니다.
세 번째 항목에는 입력 앵커 박스의 라벨링된 클래스가 포함됩니다.

이미지의 앵커 박스 및 실제 바운딩 박스 위치를 기반으로 반환된 클래스 레이블을 분석해 보겠습니다.
먼저, 모든 앵커 박스 및 실제 바운딩 박스 쌍 중에서
앵커 박스 $A_4$와 고양이의 실제 바운딩 박스의 IoU가 가장 큽니다.
따라서 $A_4$의 클래스는 고양이로 라벨링됩니다.
$A_4$ 또는 고양이의 실제 바운딩 박스를 포함하는 쌍을 제외하고, 나머지 중에서
앵커 박스 $A_1$과 개의 실제 바운딩 박스 쌍이 가장 큰 IoU를 가집니다.
따라서 $A_1$의 클래스는 개로 라벨링됩니다.
다음으로, 나머지 세 개의 라벨이 지정되지 않은 앵커 박스 $A_0, A_2, A_3$을 순회해야 합니다.
$A_0$의 경우,
IoU가 가장 큰 실제 바운딩 박스의 클래스는 개이지만,
IoU가 미리 정의된 임계값(0.5) 미만이므로 클래스는 배경으로 라벨링됩니다.
$A_2$의 경우,
IoU가 가장 큰 실제 바운딩 박스의 클래스는 고양이이고 IoU가 임계값을 초과하므로 클래스는 고양이로 라벨링됩니다.
$A_3$의 경우,
IoU가 가장 큰 실제 바운딩 박스의 클래스는 고양이이지만 값이 임계값 미만이므로 클래스는 배경으로 라벨링됩니다.

```{.python .input}
#@tab all
labels[2]
```

두 번째 반환된 항목은 (배치 크기, 앵커 박스 수의 4배) 모양의 마스크 변수입니다.
마스크 변수의 4개 요소마다 각 앵커 박스의 4개 오프셋 값에 해당합니다.
배경 감지에는 신경 쓰지 않으므로,
이 음성 클래스의 오프셋은 목적 함수에 영향을 주지 않아야 합니다.
요소별 곱셈을 통해 마스크 변수의 0은 목적 함수를 계산하기 전에 음성 클래스 오프셋을 필터링합니다.

```{.python .input}
#@tab all
labels[1]
```

첫 번째 반환된 항목에는 각 앵커 박스에 대해 라벨링된 4개의 오프셋 값이 포함됩니다.
음성 클래스 앵커 박스의 오프셋은 0으로 라벨링된다는 점에 유의하십시오.

```{.python .input}
#@tab all
labels[0]
```

## 비최대 억제로 바운딩 박스 예측 (Predicting Bounding Boxes with Non-Maximum Suppression)
:label:`subsec_predicting-bounding-boxes-nms`

예측 중에,
우리는 이미지에 대해 여러 앵커 박스를 생성하고 각각에 대한 클래스와 오프셋을 예측합니다.
따라서 *예측된 바운딩 박스*는 예측된 오프셋이 있는 앵커 박스에 따라 얻어집니다. 아래에서 앵커와 오프셋 예측을 입력으로 받아 [**역 오프셋 변환을 적용하여 예측된 바운딩 박스 좌표를 반환**]하는 `offset_inverse` 함수를 구현합니다.

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """예측된 오프셋이 있는 앵커 박스를 기반으로 바운딩 박스를 예측합니다."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

앵커 박스가 많은 경우,
동일한 객체를 둘러싸기 위해 유사한(상당한 겹침이 있는) 예측된 바운딩 박스가 많이 출력될 수 있습니다.
출력을 단순화하기 위해, *비최대 억제(non-maximum suppression, NMS)*를 사용하여
동일한 객체에 속하는 유사한 예측된 바운딩 박스를 병합할 수 있습니다.

비최대 억제 작동 방식은 다음과 같습니다.
예측된 바운딩 박스 $B$에 대해,
객체 감지 모델은 각 클래스에 대한 예측 가능성을 계산합니다.
가장 큰 예측 가능성을 $p$라고 하면, 이 확률에 해당하는 클래스가 $B$의 예측 클래스입니다.
구체적으로, 우리는 $p$를 예측된 바운딩 박스 $B$의 *신뢰도(confidence)* (점수)라고 합니다.
동일한 이미지에서,
예측된 모든 비배경 바운딩 박스는 신뢰도에 따라 내림차순으로 정렬되어
목록 $L$을 생성합니다.
그런 다음 다음 단계에서 정렬된 목록 $L$을 조작합니다.

1. $L$에서 가장 높은 신뢰도를 가진 예측된 바운딩 박스 $B_1$을 기준으로 선택하고, $B_1$과의 IoU가 미리 정의된 임계값 $\epsilon$을 초과하는 모든 비기준 예측된 바운딩 박스를 $L$에서 제거합니다. 이 시점에서 $L$은 가장 높은 신뢰도를 가진 예측된 바운딩 박스를 유지하지만 너무 유사한 다른 바운딩 박스는 삭제합니다. 한마디로, *비최대* 신뢰도 점수를 가진 것들은 *억제*됩니다.
1. $L$에서 두 번째로 높은 신뢰도를 가진 예측된 바운딩 박스 $B_2$를 다른 기준으로 선택하고, $B_2$과의 IoU가 $\epsilon$을 초과하는 모든 비기준 예측된 바운딩 박스를 $L$에서 제거합니다.
1. $L$의 모든 예측된 바운딩 박스가 기준으로 사용될 때까지 위의 과정을 반복합니다. 이때 $L$에 있는 예측된 바운딩 박스 쌍의 IoU는 임계값 $\epsilon$ 미만이므로 서로 너무 유사한 쌍은 없습니다.
1. 목록 $L$에 있는 모든 예측된 바운딩 박스를 출력합니다.

[**다음 `nms` 함수는 신뢰도 점수를 내림차순으로 정렬하고 인덱스를 반환합니다.**]

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """예측된 바운딩 박스의 신뢰도 점수를 정렬합니다."""
    B = scores.argsort()[::-1]
    keep = []  # 유지될 예측된 바운딩 박스의 인덱스
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """예측된 바운딩 박스의 신뢰도 점수를 정렬합니다."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 유지될 예측된 바운딩 박스의 인덱스
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

우리는 다음 `multibox_detection`을 정의하여
[**비최대 억제를 적용하여
바운딩 박스를 예측**]합니다.
구현이 조금 복잡하더라도 걱정하지 마십시오. 구현 직후 구체적인 예제를 통해 작동 방식을 보여드리겠습니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """비최대 억제를 사용하여 바운딩 박스를 예측합니다."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 모든 비 `keep` 인덱스를 찾아 클래스를 배경으로 설정
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # 여기서 `pos_threshold`는 양성(비배경) 예측을 위한 임계값입니다
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """비최대 억제를 사용하여 바운딩 박스를 예측합니다."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 모든 비 `keep` 인덱스를 찾아 클래스를 배경으로 설정
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # 여기서 `pos_threshold`는 양성(비배경) 예측을 위한 임계값입니다
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

이제 [**위의 구현을 4개의 앵커 박스가 있는 구체적인 예제에 적용**]해 보겠습니다.
간단하게 하기 위해, 예측된 오프셋이 모두 0이라고 가정합니다.
이는 예측된 바운딩 박스가 앵커 박스임을 의미합니다.
배경, 개, 고양이 중 각 클래스에 대해
예측된 가능성도 정의합니다.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # 예측된 배경 가능성 
                      [0.9, 0.8, 0.7, 0.1],  # 예측된 개 가능성 
                      [0.1, 0.2, 0.3, 0.9]])  # 예측된 고양이 가능성
```

우리는 [**이미지에 신뢰도와 함께 예측된 바운딩 박스를 그릴 수 있습니다.**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

이제 `multibox_detection` 함수를 호출하여
비최대 억제를 수행할 수 있습니다.
여기서 임계값은 0.5로 설정됩니다.
텐서 입력에 예제에 대한 차원을 추가한다는 점에 유의하십시오.

[**반환된 결과의 모양**]은
(배치 크기, 앵커 박스 수, 6)임을 알 수 있습니다.
가장 안쪽 차원의 6개 요소는
동일한 예측된 바운딩 박스에 대한 출력 정보를 제공합니다.
첫 번째 요소는 예측된 클래스 인덱스로, 0부터 시작합니다(0은 개, 1은 고양이). 값 -1은 비최대 억제에서의 배경 또는 제거를 나타냅니다.
두 번째 요소는 예측된 바운딩 박스의 신뢰도입니다.
나머지 4개 요소는 예측된 바운딩 박스의 왼쪽 상단 모서리와
오른쪽 하단 모서리의 $(x, y)$축 좌표입니다(범위는 0과 1 사이).

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

클래스 -1인 예측된 바운딩 박스를 제거한 후,
[**비최대 억제에 의해 유지된 최종 예측된 바운딩 박스를 출력**]할 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

실제로, 비최대 억제를 수행하기 전에도 낮은 신뢰도를 가진 예측된 바운딩 박스를 제거하여 이 알고리즘의 계산을 줄일 수 있습니다.
또한 비최대 억제의 출력을 후처리할 수도 있습니다. 예를 들어, 더 높은 신뢰도를 가진 결과만
최종 출력에 유지하는 것입니다.


## 요약 (Summary)

* 우리는 이미지의 각 픽셀을 중심으로 다양한 모양의 앵커 박스를 생성합니다.
* 자카드 지수라고도 하는 IoU(Intersection over Union)는 두 바운딩 박스의 유사성을 측정합니다. 교집합 영역 대 합집합 영역의 비율입니다.
* 훈련 세트에서, 각 앵커 박스에 대해 두 가지 유형의 레이블이 필요합니다. 하나는 앵커 박스와 관련된 객체의 클래스이고 다른 하나는 앵커 박스에 대한 실제 바운딩 박스의 오프셋입니다.
* 예측 중에, 비최대 억제(NMS)를 사용하여 유사한 예측된 바운딩 박스를 제거하여 출력을 단순화할 수 있습니다.


## 연습 문제 (Exercises)

1. `multibox_prior` 함수에서 `sizes`와 `ratios` 값을 변경해 보십시오. 생성된 앵커 박스에 어떤 변화가 있습니까?
1. IoU가 0.5인 두 바운딩 박스를 구성하고 시각화해 보십시오. 서로 어떻게 겹칩니까?
1. :numref:`subsec_labeling-anchor-boxes` 및 :numref:`subsec_predicting-bounding-boxes-nms`에서 변수 `anchors`를 수정해 보십시오. 결과가 어떻게 변합니까?
1. 비최대 억제는 *제거*함으로써 예측된 바운딩 박스를 억제하는 탐욕 알고리즘입니다. 제거된 것 중 일부가 실제로 유용할 수 있습니까? 이 알고리즘을 *부드럽게(softly)* 억제하도록 어떻게 수정할 수 있습니까? Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`를 참조할 수 있습니다.
1. 수작업이 아닌, 비최대 억제를 학습할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:

```