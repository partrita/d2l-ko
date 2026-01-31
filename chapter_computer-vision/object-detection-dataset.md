# 객체 감지 데이터셋 (The Object Detection Dataset)
:label:`sec_object-detection-dataset`

객체 감지 분야에는 MNIST나 Fashion-MNIST와 같은 작은 데이터셋이 없습니다.
객체 감지 모델을 빠르게 시연하기 위해,
[**우리는 작은 데이터셋을 수집하고 라벨링했습니다**].
먼저, 사무실에서 무료 바나나 사진을 찍어
회전과 크기가 다른 1000개의 바나나 이미지를 생성했습니다.
그런 다음 각 바나나 이미지를
일부 배경 이미지의 임의 위치에 배치했습니다.
마지막으로, 이미지에 있는 바나나에 대한 바운딩 박스를 라벨링했습니다.


## [**데이터셋 다운로드**]

모든 이미지 및 csv 라벨 파일이 포함된 바나나 감지 데이터셋은 인터넷에서 직접 다운로드할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 데이터셋 읽기 (Reading the Dataset)

우리는 아래의 `read_data_bananas` 함수에서 [**바나나 감지 데이터셋을 읽을 것입니다**].
데이터셋에는 객체 클래스 레이블과
왼쪽 상단 및 오른쪽 하단 모서리의 실제 바운딩 박스 좌표에 대한 csv 파일이 포함되어 있습니다.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """바나나 감지 데이터셋 이미지와 라벨을 읽습니다."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 여기서 `target`은 (클래스, 왼쪽 상단 x, 왼쪽 상단 y, 오른쪽 하단 x, 오른쪽 하단 y)를 포함합니다.
        # 모든 이미지는 동일한 바나나 클래스(인덱스 0)를 갖습니다.
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """바나나 감지 데이터셋 이미지와 라벨을 읽습니다."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # 여기서 `target`은 (클래스, 왼쪽 상단 x, 왼쪽 상단 y, 오른쪽 하단 x, 오른쪽 하단 y)를 포함합니다.
        # 모든 이미지는 동일한 바나나 클래스(인덱스 0)를 갖습니다.
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

`read_data_bananas` 함수를 사용하여 이미지와 라벨을 읽음으로써,
다음 `BananasDataset` 클래스는 바나나 감지 데이터셋을 로드하기 위한
[**사용자 정의 `Dataset` 인스턴스를 생성**]할 수 있게 해줍니다.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """바나나 감지 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """바나나 감지 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

마지막으로, [**훈련 및 테스트 세트 모두에 대해 두 개의 데이터 반복자 인스턴스를 반환**]하는 `load_data_bananas` 함수를 정의합니다.
테스트 데이터셋의 경우, 무작위 순서로 읽을 필요가 없습니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """바나나 감지 데이터셋을 로드합니다."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """바나나 감지 데이터셋을 로드합니다."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

[**미니배치를 읽고 이 미니배치에 있는 이미지와 라벨의 모양을 인쇄**]해 봅시다.
이미지 미니배치의 모양 (배치 크기, 채널 수, 높이, 너비)은 익숙해 보입니다. 이전 이미지 분류 작업과 동일합니다.
라벨 미니배치의 모양은 (배치 크기, $m$, 5)입니다. 여기서 $m$은 데이터셋의 이미지에 있을 수 있는 가장 큰 바운딩 박스 수입니다.

미니배치에서의 계산이 더 효율적이지만, 연결을 통해 미니배치를 형성하려면 모든 이미지 예제에 동일한 수의 바운딩 박스가 포함되어야 합니다.
일반적으로 이미지는 다양한 수의 바운딩 박스를 가질 수 있습니다. 따라서 $m$개 미만의 바운딩 박스를 가진 이미지는 $m$개에 도달할 때까지 불법 바운딩 박스로 채워집니다.
그런 다음 각 바운딩 박스의 라벨은 길이 5의 배열로 표현됩니다.
배열의 첫 번째 요소는 바운딩 박스에 있는 객체의 클래스이며, -1은 패딩을 위한 불법 바운딩 박스를 나타냅니다.
배열의 나머지 4개 요소는 바운딩 박스의 왼쪽 상단 모서리와 오른쪽 하단 모서리의 ($x$, $y$)-좌표 값입니다(범위는 0과 1 사이).
바나나 데이터셋의 경우, 각 이미지에 하나의 바운딩 박스만 있으므로 $m=1$입니다.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**시연 (Demonstration)**]

라벨링된 실제 바운딩 박스와 함께 10개의 이미지를 시연해 봅시다.
이 모든 이미지에서 바나나의 회전, 크기, 위치가 다양하다는 것을 알 수 있습니다.
물론 이것은 단순한 인공 데이터셋일 뿐입니다.
실제로 실제 데이터셋은 일반적으로 훨씬 더 복잡합니다.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 요약 (Summary)

* 우리가 수집한 바나나 감지 데이터셋은 객체 감지 모델을 시연하는 데 사용할 수 있습니다.
* 객체 감지를 위한 데이터 로딩은 이미지 분류와 유사합니다. 그러나 객체 감지에서 라벨에는 이미지 분류에는 없는 실제 바운딩 박스 정보도 포함됩니다.


## 연습 문제 (Exercises)

1. 바나나 감지 데이터셋에서 실제 바운딩 박스가 있는 다른 이미지를 시연하십시오. 바운딩 박스와 객체 측면에서 어떻게 다릅니까?
1. 무작위 자르기와 같은 데이터 증강을 객체 감지에 적용하고 싶다고 가정해 봅시다. 이미지 분류와 어떻게 다를 수 있습니까? 힌트: 잘린 이미지에 객체의 일부만 포함된 경우 어떻게 됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab: