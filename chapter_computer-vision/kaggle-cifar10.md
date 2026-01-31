# Kaggle의 이미지 분류 (CIFAR-10)
:label:`sec_kaggle_cifar10`

지금까지 우리는 딥러닝 프레임워크의 고수준 API를 사용하여 텐서 형식의 이미지 데이터셋을 직접 얻었습니다.
그러나 사용자 정의 이미지 데이터셋은
종종 이미지 파일 형태로 제공됩니다.
이 섹션에서는 원시 이미지 파일에서 시작하여
단계별로 정리하고, 읽고, 텐서 형식으로 변환합니다.

우리는 :numref:`sec_image_augmentation`에서 CIFAR-10 데이터셋을 실험했는데,
이는 컴퓨터 비전에서 중요한 데이터셋입니다.
이 섹션에서는
이전 섹션에서 배운 지식을 적용하여
CIFAR-10 이미지 분류의 Kaggle 대회를 연습할 것입니다.
(**대회의 웹 주소는 https://www.kaggle.com/c/cifar-10입니다**)

:numref:`fig_kaggle_cifar10`은 대회 웹페이지의 정보를 보여줍니다.
결과를 제출하려면 Kaggle 계정을 등록해야 합니다.

![CIFAR-10 이미지 분류 대회 웹페이지 정보. "Data" 탭을 클릭하여 대회 데이터셋을 얻을 수 있습니다.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`


```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## 데이터셋 획득 및 정리 (Obtaining and Organizing the Dataset)

대회 데이터셋은 훈련 세트와 테스트 세트로 나뉘며,
각각 50,000개와 300,000개의 이미지를 포함합니다.
테스트 세트에서,
10,000개 이미지가 평가에 사용되며,
나머지 290,000개 이미지는 평가되지 않습니다:
이들은 단지 *수동으로* 라벨링된 테스트 세트 결과로 부정행위를 하기 어렵게 만들기 위해 포함되었습니다.
이 데이터셋의 이미지는
모두 png 컬러(RGB 채널) 이미지 파일이며,
높이와 너비는 모두 32 픽셀입니다.
이미지는 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭의 총 10개 범주를 다룹니다.
:numref:`fig_kaggle_cifar10`의 왼쪽 상단 모서리에는 데이터셋의 비행기, 자동차, 새 이미지 일부가 표시되어 있습니다.


### 데이터셋 다운로드 (Downloading the Dataset)

Kaggle에 로그인한 후, :numref:`fig_kaggle_cifar10`에 표시된 CIFAR-10 이미지 분류 대회 웹페이지에서 "Data" 탭을 클릭하고 "Download All" 버튼을 클릭하여 데이터셋을 다운로드할 수 있습니다.
다운로드한 파일을 `../data`에 압축 해제하고 그 안의 `train.7z`와 `test.7z` 압축을 해제하면 다음 경로에서 전체 데이터셋을 찾을 수 있습니다.

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

여기서 `train` 및 `test` 디렉터리에는 각각 훈련 및 테스트 이미지가 포함되어 있고, `trainLabels.csv`는 훈련 이미지에 대한 라벨을 제공하며, `sample_submission.csv`는 샘플 제출 파일입니다.

더 쉽게 시작할 수 있도록, [**처음 1000개의 훈련 이미지와 5개의 무작위 테스트 이미지가 포함된 소규모 샘플 데이터셋을 제공합니다.**] 
Kaggle 대회의 전체 데이터셋을 사용하려면 다음 `demo` 변수를 `False`로 설정해야 합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# Kaggle 대회용으로 다운로드한 전체 데이터셋을 사용하는 경우 `demo`를 False로 설정하십시오
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**데이터셋 정리 (Organizing the Dataset)**]

모델 훈련 및 테스트를 용이하게 하기 위해 데이터셋을 정리해야 합니다.
먼저 csv 파일에서 라벨을 읽어봅시다.
다음 함수는 파일 이름의 확장자가 없는 부분을 라벨에 매핑하는 사전을 반환합니다.

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """`fname`을 읽어 파일 이름 대 라벨 사전을 반환합니다."""
    with open(fname, 'r') as f:
        # 파일 헤더 라인(열 이름) 건너뛰기
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 훈련 예제:', len(labels))
print('# 클래스:', len(set(labels.values())))
```

다음으로, [**원래 훈련 세트에서 검증 세트를 분리**]하는 `reorg_train_valid` 함수를 정의합니다.
이 함수의 `valid_ratio` 인수는 원래 훈련 세트의 예제 수에 대한 검증 세트의 예제 수 비율입니다.
더 구체적으로,
$n$을 예제가 가장 적은 클래스의 이미지 수라고 하고 $r$을 비율이라고 합시다.
검증 세트는 각 클래스에 대해 $\max(\lfloor nr\rfloor,1)$개의 이미지를 분리합니다.
`valid_ratio=0.1`을 예로 들어보겠습니다. 원래 훈련 세트에 50,000개의 이미지가 있으므로,
`train_valid_test/train` 경로에 45,000개의 이미지가 훈련에 사용되고,
나머지 5,000개의 이미지는 `train_valid_test/valid` 경로에 검증 세트로 분리됩니다. 데이터셋을 정리한 후, 동일한 클래스의 이미지는 동일한 폴더 아래에 배치됩니다.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """파일을 대상 디렉터리로 복사합니다."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """원래 훈련 세트에서 검증 세트를 분리합니다."""
    # 훈련 데이터셋에서 예제 수가 가장 적은 클래스의 예제 수
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 검증 세트의 클래스당 예제 수
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

아래의 `reorg_test` 함수는 [**예측 중 데이터 로딩을 위해 테스트 세트를 정리합니다.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """예측 중 데이터 로딩을 위해 테스트 세트를 정리합니다."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

마지막으로, 함수를 사용하여 (**위에서 정의한**) `read_csv_labels`, `reorg_train_valid`, `reorg_test` [**함수를 호출합니다.**]

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

여기서는 데이터셋의 소규모 샘플에 대해 배치 크기를 32로만 설정합니다.
Kaggle 대회의 전체 데이터셋을 훈련하고 테스트할 때,
`batch_size`는 128과 같이 더 큰 정수로 설정해야 합니다.
하이퍼파라미터 튜닝을 위해 훈련 예제의 10%를 검증 세트로 분리합니다.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**이미지 증강 (Image Augmentation)**]

우리는 과대적합을 해결하기 위해 이미지 증강을 사용합니다.
예를 들어, 훈련 중에 이미지를 무작위로 수평 뒤집을 수 있습니다.
또한 컬러 이미지의 세 가지 RGB 채널에 대해 표준화를 수행할 수 있습니다. 아래에는 조정할 수 있는 몇 가지 작업이 나열되어 있습니다.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # 이미지를 높이와 너비 모두 40 픽셀의 정사각형으로 확대
    gluon.data.vision.transforms.Resize(40),
    # 높이와 너비 모두 40 픽셀인 정사각형 이미지를 무작위로 자르기하여
    # 원본 이미지 면적의 0.64에서 1배인 작은 정사각형을 생성한 다음,
    # 높이와 너비 모두 32 픽셀인 정사각형으로 축소
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # 이미지의 각 채널 표준화
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # 이미지를 높이와 너비 모두 40 픽셀의 정사각형으로 확대
    torchvision.transforms.Resize(40),
    # 높이와 너비 모두 40 픽셀인 정사각형 이미지를 무작위로 자르기하여
    # 원본 이미지 면적의 0.64에서 1배인 작은 정사각형을 생성한 다음,
    # 높이와 너비 모두 32 픽셀인 정사각형으로 축소
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 이미지의 각 채널 표준화
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

테스트 중에는
평가 결과의 무작위성을 제거하기 위해
이미지에 대해 표준화만 수행합니다.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## 데이터셋 읽기 (Reading the Dataset)

다음으로, [**원시 이미지 파일로 구성된 정리된 데이터셋을 읽습니다**]. 각 예제에는 이미지와 라벨이 포함됩니다.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

훈련 중에,
우리는 [**위에서 정의한 모든 이미지 증강 작업을 지정**]해야 합니다.
하이퍼파라미터 튜닝 중 모델 평가에 검증 세트가 사용될 때는
이미지 증강으로 인한 무작위성을 도입해서는 안 됩니다.
최종 예측 전에, 라벨이 지정된 모든 데이터를 최대한 활용하기 위해 훈련 세트와 검증 세트를 합친 데이터로 모델을 훈련합니다.

```{.python .input}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**모델**] 정의 (Defining the Model)

:begin_tab:`mxnet`
여기서는 `HybridBlock` 클래스를 기반으로 잔차 블록을 구축하는데, 이는
:numref:`sec_resnet`에서 설명한 구현과 약간 다릅니다.
이는 계산 효율성을 향상시키기 위함입니다.
:end_tab:

```{.python .input}
#@tab mxnet
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
다음으로 ResNet-18 모델을 정의합니다.
:end_tab:

```{.python .input}
#@tab mxnet
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
훈련이 시작되기 전에 :numref:`subsec_xavier`에 설명된 Xavier 초기화를 사용합니다.
:end_tab:

:begin_tab:`pytorch`
:numref:`sec_resnet`에 설명된 ResNet-18 모델을 정의합니다.
:end_tab:

```{.python .input}
#@tab mxnet
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## [**훈련 함수**] 정의 (Defining the Training Function)

검증 세트에서의 모델 성능에 따라 모델을 선택하고 하이퍼파라미터를 조정할 것입니다.
다음에서 모델 훈련 함수 `train`을 정의합니다.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**모델 훈련 및 검증 (Training and Validating the Model)**]

이제 모델을 훈련하고 검증할 수 있습니다.
다음의 모든 하이퍼파라미터는 조정 가능합니다.
예를 들어 에포크 수를 늘릴 수 있습니다.
`lr_period`와 `lr_decay`가 각각 4와 0.9로 설정되면 최적화 알고리즘의 학습률은 4 에포크마다 0.9배가 됩니다. 시연의 편의를 위해,
여기서는 20 에포크만 훈련합니다.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
net(next(iter(train_iter))[0])
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**테스트 세트 분류 (Classifying the Testing Set)**] 및 Kaggle에 결과 제출

유망한 모델과 하이퍼파라미터를 얻은 후,
모든 라벨이 지정된 데이터(검증 세트 포함)를 사용하여 모델을 다시 훈련하고 테스트 세트를 분류합니다.

```{.python .input}
#@tab mxnet
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
net(next(iter(train_valid_iter))[0])
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

위의 코드는
`submission.csv` 파일을 생성하며,
그 형식은
Kaggle 대회의 요구 사항을 충족합니다.
Kaggle에 결과를 제출하는 방법은
:numref:`sec_kaggle_house`와 유사합니다.

## 요약 (Summary)

* 우리는 원시 이미지 파일을 포함하는 데이터셋을 필요한 형식으로 정리한 후 읽을 수 있습니다.

:begin_tab:`mxnet`
* 우리는 이미지 분류 대회에서 합성곱 신경망, 이미지 증강 및 하이브리드 프로그래밍을 사용할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
* 우리는 이미지 분류 대회에서 합성곱 신경망과 이미지 증강을 사용할 수 있습니다.
:end_tab:

## 연습 문제 (Exercises)

1. 이 Kaggle 대회에 전체 CIFAR-10 데이터셋을 사용하십시오. 하이퍼파라미터를 `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50`, `lr_decay = 0.1`로 설정하십시오.  이 대회에서 어떤 정확도와 순위를 얻을 수 있는지 확인하십시오. 더 향상시킬 수 있습니까?
1. 이미지 증강을 사용하지 않을 때 어떤 정확도를 얻을 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:
