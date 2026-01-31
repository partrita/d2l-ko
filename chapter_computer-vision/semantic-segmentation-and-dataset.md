# 시맨틱 분할과 데이터셋 (Semantic Segmentation and the Dataset)
:label:`sec_semantic_segmentation`

:numref:`sec_bbox`--:numref:`sec_rcnn`에서 객체 감지 작업을 논의할 때,
직사각형 바운딩 박스를 사용하여 이미지의 객체를 라벨링하고 예측했습니다. 
이 섹션에서는 이미지를 서로 다른 의미 클래스에 속하는 영역으로 나누는 방법에 초점을 맞춘 *시맨틱 분할(semantic segmentation)* 문제에 대해 논의할 것입니다.
다른 객체 감지와 달리,
시맨틱 분할은
이미지에 무엇이 있는지 픽셀 수준에서 인식하고 이해합니다: 
그 의미 영역의 라벨링 및 예측은 픽셀 수준입니다.
:numref:`fig_segmentation`은 시맨틱 분할에서 이미지의 개, 고양이, 배경에 대한 라벨을 보여줍니다.
객체 감지와 비교할 때,
시맨틱 분할에서 라벨링된 픽셀 수준 경계는 분명히 더 세분화되어 있습니다.


![시맨틱 분할에서 이미지의 개, 고양이, 배경의 라벨.](../img/segmentation.svg)
:label:`fig_segmentation`


## 이미지 분할 및 인스턴스 분할 (Image Segmentation and Instance Segmentation)

컴퓨터 비전 분야에는 시맨틱 분할과 유사한 두 가지 중요한 작업,
즉 이미지 분할과 인스턴스 분할도 있습니다.
우리는 다음과 같이 시맨틱 분할과 구별하여 간략하게 설명할 것입니다.

* *이미지 분할(Image segmentation)*은 이미지를 여러 구성 영역으로 나눕니다. 이러한 유형의 문제에 대한 방법은 일반적으로 이미지 픽셀 간의 상관 관계를 사용합니다. 훈련 중 이미지 픽셀에 대한 라벨 정보가 필요하지 않으며, 예측 중에 분할된 영역이 우리가 얻고자 하는 의미를 갖는다고 보장할 수 없습니다. :numref:`fig_segmentation`의 이미지를 입력으로 사용하면, 이미지 분할은 개를 두 영역으로 나눌 수 있습니다. 하나는 주로 검은색인 입과 눈을 덮고, 다른 하나는 주로 노란색인 몸의 나머지 부분을 덮습니다.
* *인스턴스 분할(Instance segmentation)*은 *동시 감지 및 분할*이라고도 합니다. 이미지 내의 각 객체 인스턴스의 픽셀 수준 영역을 인식하는 방법을 연구합니다. 시맨틱 분할과 달리 인스턴스 분할은 의미뿐만 아니라 다른 객체 인스턴스도 구별해야 합니다. 예를 들어 이미지에 두 마리의 개가 있는 경우, 인스턴스 분할은 픽셀이 두 마리의 개 중 어느 것에 속하는지 구별해야 합니다.



## Pascal VOC2012 시맨틱 분할 데이터셋 (The Pascal VOC2012 Semantic Segmentation Dataset)

[**가장 중요한 시맨틱 분할 데이터셋 중 하나는 [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)입니다.**]
다음에서,
우리는 이 데이터셋을 살펴볼 것입니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

데이터셋의 tar 파일은 약 2GB이므로,
파일을 다운로드하는 데 시간이 좀 걸릴 수 있습니다. 
추출된 데이터셋은 `../data/VOCdevkit/VOC2012`에 있습니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

`../data/VOCdevkit/VOC2012` 경로로 들어가면,
데이터셋의 다양한 구성 요소를 볼 수 있습니다.
`ImageSets/Segmentation` 경로에는 훈련 및 테스트 샘플을 지정하는 텍스트 파일이 포함되어 있고,
`JPEGImages` 및 `SegmentationClass` 경로에는
각 예제에 대한 입력 이미지와 라벨이 각각 저장되어 있습니다.
여기서 라벨도 이미지 형식이며,
라벨링된 입력 이미지와 크기가 같습니다.
또한,
모든 라벨 이미지에서 동일한 색상을 가진 픽셀은 동일한 의미 클래스에 속합니다.
다음은 `read_voc_images` 함수를 정의하여 [**모든 입력 이미지와 라벨을 메모리로 읽어옵니다**].

```{.python .input}
#@tab mxnet
#@save
def read_voc_images(voc_dir, is_train=True):
    """모든 VOC 특징 및 라벨 이미지를 읽습니다."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """모든 VOC 특징 및 라벨 이미지를 읽습니다."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

우리는 [**처음 5개의 입력 이미지와 해당 라벨을 그립니다**].
라벨 이미지에서 흰색과 검은색은 각각 경계와 배경을 나타내며, 다른 색상은 서로 다른 클래스에 해당합니다.

```{.python .input}
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

다음으로, 우리는 이 데이터셋의 모든 라벨에 대해 [**RGB 색상 값과 클래스 이름을 열거합니다**].

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

위에서 정의한 두 가지 상수를 사용하여,
우리는 [**라벨의 각 픽셀에 대한 클래스 인덱스를 편리하게 찾을 수 있습니다**].
우리는 위 RGB 색상 값에서 클래스 인덱스로의 매핑을 구축하기 위해 `voc_colormap2label` 함수를 정의하고,
이 Pascal VOC2012 데이터셋의 모든 RGB 값을 해당 클래스 인덱스로 매핑하기 위해 `voc_label_indices` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def voc_colormap2label():
    """VOC 라벨을 위해 RGB에서 클래스 인덱스로 매핑을 구축합니다."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """VOC 라벨의 모든 RGB 값을 클래스 인덱스로 매핑합니다."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """VOC 라벨을 위해 RGB에서 클래스 인덱스로 매핑을 구축합니다."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """VOC 라벨의 모든 RGB 값을 클래스 인덱스로 매핑합니다."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**예를 들어**], 첫 번째 예제 이미지에서,
비행기 앞부분의 클래스 인덱스는 1이고,
배경 인덱스는 0입니다.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### 데이터 전처리 (Data Preprocessing)

:numref:`sec_alexnet`--:numref:`sec_googlenet`과 같은 이전 실험에서,
이미지는 모델의 필수 입력 모양에 맞게 크기가 조정(rescaling)되었습니다.
그러나 시맨틱 분할에서는
그렇게 하면 예측된 픽셀 클래스를
원래 입력 이미지 모양으로 다시 크기 조정해야 합니다.
이러한 크기 조정은 특히 클래스가 다른 분할된 영역의 경우 부정확할 수 있습니다. 이 문제를 피하기 위해,
우리는 크기 조정 대신 이미지를 *고정된* 모양으로 자릅니다. 구체적으로, [**이미지 증강의 무작위 자르기를 사용하여 입력 이미지와 라벨의 동일한 영역을 자릅니다**].

```{.python .input}
#@tab mxnet
#@save
def voc_rand_crop(feature, label, height, width):
    """특징 및 라벨 이미지를 모두 무작위로 자릅니다."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """특징 및 라벨 이미지를 모두 무작위로 자릅니다."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**사용자 정의 시맨틱 분할 데이터셋 클래스 (Custom Semantic Segmentation Dataset Class)**]

우리는 고수준 API에서 제공하는 `Dataset` 클래스를 상속하여 사용자 정의 시맨틱 분할 데이터셋 클래스 `VOCSegDataset`을 정의합니다.
`__getitem__` 함수를 구현함으로써,
우리는 데이터셋에서 `idx`로 인덱싱된 입력 이미지와 이 이미지의 각 픽셀의 클래스 인덱스에 임의로 액세스할 수 있습니다.
데이터셋의 일부 이미지는
무작위 자르기의 출력 크기보다 작기 때문에,
이러한 예제는 사용자 정의 `filter` 함수에 의해 필터링됩니다.
또한, 입력 이미지의 세 가지 RGB 채널 값을 표준화하기 위해
`normalize_image` 함수도 정의합니다.

```{.python .input}
#@tab mxnet
#@save
class VOCSegDataset(gluon.data.Dataset):
    """VOC 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (
                feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """VOC 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**데이터셋 읽기 (Reading the Dataset)**]

우리는 `VOCSegDatase`t 클래스를 사용하여
훈련 세트와 테스트 세트의 인스턴스를 각각 생성합니다.
무작위로 자른 이미지의 출력 모양을 $320	imes 480$으로 지정한다고 가정합니다.
아래에서 훈련 세트와 테스트 세트에 유지되는 예제의 수를 확인할 수 있습니다.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

배치 크기를 64로 설정하고,
훈련 세트에 대한 데이터 반복자를 정의합니다.
첫 번째 미니배치의 모양을 인쇄해 봅시다.
이미지 분류나 객체 감지와 달리, 여기서 라벨은 3차원 텐서입니다.

```{.python .input}
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**종합하기 (Putting It All Together)**]

마지막으로, Pascal VOC2012 시맨틱 분할 데이터셋을 다운로드하고 읽기 위해
다음 `load_data_voc` 함수를 정의합니다.
이 함수는 훈련 및 테스트 데이터셋 모두에 대한 데이터 반복자를 반환합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_voc(batch_size, crop_size):
    """VOC 시맨틱 분할 데이터셋을 로드합니다."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """VOC 시맨틱 분할 데이터셋을 로드합니다."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## 요약 (Summary)

* 시맨틱 분할은 이미지를 다른 의미 클래스에 속하는 영역으로 나누어 픽셀 수준에서 이미지에 무엇이 있는지 인식하고 이해합니다.
* 가장 중요한 시맨틱 분할 데이터셋 중 하나는 Pascal VOC2012입니다.
* 시맨틱 분할에서는 입력 이미지와 라벨이 픽셀에서 일대일로 대응하므로, 입력 이미지는 크기가 조정되는 것이 아니라 고정된 모양으로 무작위로 잘립니다.


## 연습 문제 (Exercises)

1. 자율 주행 차량 및 의료 영상 진단에 시맨틱 분할을 어떻게 적용할 수 있습니까? 다른 응용 분야를 생각할 수 있습니까?
1. :numref:`sec_image_augmentation`의 데이터 증강 설명을 상기하십시오. 이미지 분류에 사용되는 이미지 증강 방법 중 시맨틱 분할에 적용할 수 없는 것은 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab: