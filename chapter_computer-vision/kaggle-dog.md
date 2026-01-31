# Kaggle의 개 품종 식별 (Dog Breed Identification on Kaggle)

이 섹션에서는 Kaggle에서
개 품종 식별 문제를 연습할 것입니다. (**이 대회의 웹 주소는 https://www.kaggle.com/c/dog-breed-identification입니다**)

이 대회에서는
120가지 다른 품종의 개를 인식해야 합니다.
사실,
이 대회의 데이터셋은
ImageNet 데이터셋의 하위 집합입니다.
:numref:`sec_kaggle_cifar10`의 CIFAR-10 데이터셋 이미지와 달리,
ImageNet 데이터셋의 이미지는 높이와 너비가 다양하고 더 큽니다.
:numref:`fig_kaggle_dog`는 대회 웹페이지의 정보를 보여줍니다. 결과를 제출하려면 Kaggle 계정이 필요합니다.


![개 품종 식별 대회 웹사이트. "Data" 탭을 클릭하여 대회 데이터셋을 얻을 수 있습니다.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## 데이터셋 획득 및 정리 (Obtaining and Organizing the Dataset)

대회 데이터셋은 훈련 세트와 테스트 세트로 나뉘며, 각각 10,222개와 10,357개의 세 RGB(컬러) 채널 JPEG 이미지를 포함합니다.
훈련 데이터셋에는
래브라도, 푸들, 닥스훈트, 사모예드, 허스키, 치와와, 요크셔 테리어 등 120종의 개가 있습니다.


### 데이터셋 다운로드 (Downloading the Dataset)

Kaggle에 로그인한 후,
:numref:`fig_kaggle_dog`에 표시된 대회 웹페이지에서 "Data" 탭을 클릭하고 "Download All" 버튼을 클릭하여 데이터셋을 다운로드할 수 있습니다.
다운로드한 파일을 `../data`에 압축 해제하면 다음 경로에서 전체 데이터셋을 찾을 수 있습니다.

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

위의 구조는
:numref:`sec_kaggle_cifar10`의 CIFAR-10 대회와 유사하다는 것을 알 수 있습니다. 여기서 `train/` 및 `test/` 폴더에는 각각 훈련 및 테스트 개 이미지가 포함되어 있고, `labels.csv`에는
훈련 이미지에 대한 라벨이 포함되어 있습니다.
마찬가지로, 더 쉽게 시작할 수 있도록, 위에서 언급한 [**데이터셋의 작은 샘플을 제공합니다**]: `train_valid_test_tiny.zip`.
Kaggle 대회의 전체 데이터셋을 사용하려면 아래 `demo` 변수를 `False`로 변경해야 합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# Kaggle 대회용으로 다운로드한 전체 데이터셋을 사용하는 경우,
# 아래 변수를 `False`로 변경하십시오.
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**데이터셋 정리 (Organizing the Dataset)**]

:numref:`sec_kaggle_cifar10`에서 했던 것과 유사하게 데이터셋을 정리할 수 있습니다. 즉, 원래 훈련 세트에서 검증 세트를 분리하고 이미지를 라벨별로 그룹화된 하위 폴더로 이동합니다.

아래의 `reorg_dog_data` 함수는
훈련 데이터 라벨을 읽고, 검증 세트를 분리하고, 훈련 세트를 정리합니다.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**이미지 증강 (Image Augmentation)**]

이 개 품종 데이터셋은
ImageNet 데이터셋의 하위 집합이며,
이미지는 :numref:`sec_kaggle_cifar10`의 CIFAR-10 데이터셋보다 큽니다.
다음은 상대적으로 큰 이미지에 유용할 수 있는 몇 가지 이미지 증강 작업을 나열합니다.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # 이미지를 무작위로 잘라 원래 면적의 0.08에서 1배이고 높이 대 너비 비율이 3/4에서 4/3 사이인 이미지를 얻습니다.
    # 그런 다음 이미지를 스케일링하여 새로운 224 x 224 이미지를 만듭니다.
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # 밝기, 대비 및 채도를 무작위로 변경
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # 무작위 노이즈 추가
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # 이미지의 각 채널 표준화
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # 이미지를 무작위로 잘라 원래 면적의 0.08에서 1배이고 높이 대 너비 비율이 3/4에서 4/3 사이인 이미지를 얻습니다.
    # 그런 다음 이미지를 스케일링하여 새로운 224 x 224 이미지를 만듭니다.
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 밝기, 대비 및 채도를 무작위로 변경
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 무작위 노이즈 추가
    torchvision.transforms.ToTensor(),
    # 이미지의 각 채널 표준화
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

예측 중에는
무작위성이 없는 이미지 전처리 작업만 사용합니다.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # 이미지 중심에서 224 x 224 정사각형 영역 자르기
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 이미지 중심에서 224 x 224 정사각형 영역 자르기
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**데이터셋 읽기 (Reading the Dataset)**]

:numref:`sec_kaggle_cifar10`에서와 같이,
우리는 원시 이미지 파일로 구성된 정리된 데이터셋을 읽을 수 있습니다.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

아래에서 :numref:`sec_kaggle_cifar10`과 동일한 방식으로 데이터 반복자 인스턴스를 생성합니다.

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

## [**사전 훈련된 모델 미세 조정 (Fine-Tuning a Pretrained Model)**]

다시 말하지만,
이 대회의 데이터셋은 ImageNet 데이터셋의 하위 집합입니다.
따라서 우리는 :numref:`sec_fine_tuning`에서 논의된 접근 방식을 사용하여
전체 ImageNet 데이터셋에서 사전 훈련된 모델을 선택하고 이를 사용하여 이미지 특징을 추출하여
사용자 정의 소규모 출력 네트워크에 공급할 수 있습니다.
딥러닝 프레임워크의 고수준 API는
ImageNet 데이터셋에서 사전 훈련된 다양한 모델을 제공합니다.
여기서는 사전 훈련된 ResNet-34 모델을 선택합니다.
여기서 우리는 단순히
이 모델의 출력 레이어의 입력(즉, 추출된 특징)을 재사용합니다.
그런 다음 원래 출력 레이어를 훈련 가능한 작은 사용자 정의 출력 네트워크로 대체할 수 있습니다.
예를 들어 두 개의 완전 연결 레이어를 쌓는 것입니다.
:numref:`sec_fine_tuning`의 실험과 달리,
다음은 특징 추출에 사용되는 사전 훈련된 모델을 다시 훈련하지 않습니다.
이렇게 하면 훈련 시간과 기울기를 저장하기 위한 메모리가 줄어듭니다.

전체 ImageNet 데이터셋에 대해 세 가지 RGB 채널의 평균과 표준 편차를 사용하여 이미지를 표준화했다는 점을 상기하십시오.
사실,
이는 ImageNet에서 사전 훈련된 모델에 의한 표준화 작업과도 일치합니다.

```{.python .input}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # 새로운 출력 네트워크 정의
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120개의 출력 범주가 있습니다.
    finetune_net.output_new.add(nn.Dense(120))
    # 출력 네트워크 초기화
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # 계산에 사용되는 CPU 또는 GPU에 모델 파라미터 배포
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 새로운 출력 네트워크 정의 (120개의 출력 범주가 있습니다)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 모델을 장치로 이동
    finetune_net = finetune_net.to(devices[0])
    # 특징 레이어의 파라미터 고정
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

[**손실을 계산**]하기 전에,
우리는 먼저 사전 훈련된 모델의 출력 레이어 입력, 즉 추출된 특징을 얻습니다.
그런 다음 이 특징을 작은 사용자 정의 출력 네트워크의 입력으로 사용하여 손실을 계산합니다.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## [**훈련 함수**] 정의 (Defining the Training Function)

우리는 검증 세트에서의 모델 성능에 따라 모델을 선택하고 하이퍼파라미터를 조정할 것입니다. 모델 훈련 함수 `train`은 작은 사용자 정의 출력 네트워크의 파라미터만 반복합니다.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 작은 사용자 정의 출력 네트워크만 훈련
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # 작은 사용자 정의 출력 네트워크만 훈련
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**모델 훈련 및 검증 (Training and Validating the Model)**]

이제 모델을 훈련하고 검증할 수 있습니다.
다음의 모든 하이퍼파라미터는 조정 가능합니다.
예를 들어 에포크 수를 늘릴 수 있습니다. `lr_period`와 `lr_decay`가 각각 2와 0.9로 설정되어 있으므로 최적화 알고리즘의 학습률은 2 에포크마다 0.9배가 됩니다.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**테스트 세트 분류 (Classifying the Testing Set)**] 및 Kaggle에 결과 제출


:numref:`sec_kaggle_cifar10`의 마지막 단계와 유사하게,
결국 모든 라벨이 지정된 데이터(검증 세트 포함)는 모델을 훈련하고 테스트 세트를 분류하는 데 사용됩니다.
우리는 훈련된 사용자 정의 출력 네트워크를 사용하여 분류를 수행합니다.

```{.python .input}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

위의 코드는
:numref:`sec_kaggle_house`에서 설명한 것과 동일한 방식으로
Kaggle에 제출할 `submission.csv` 파일을 생성합니다.


## 요약 (Summary)


* ImageNet 데이터셋의 이미지는 CIFAR-10 이미지보다 큽니다(다양한 크기). 다른 데이터셋의 작업에 대해 이미지 증강 작업을 수정할 수 있습니다.
* ImageNet 데이터셋의 하위 집합을 분류하기 위해, 우리는 전체 ImageNet 데이터셋에서 사전 훈련된 모델을 활용하여 특징을 추출하고 작은 사용자 정의 출력 네트워크만 훈련할 수 있습니다. 이렇게 하면 계산 시간과 메모리 비용이 줄어듭니다.


## 연습 문제 (Exercises)

1. 전체 Kaggle 대회 데이터셋을 사용할 때, `batch_size`(배치 크기)와 `num_epochs`(에포크 수)를 늘리고 다른 하이퍼파라미터를 `lr = 0.01`, `lr_period = 10`, `lr_decay = 0.1`로 설정하면 어떤 결과를 얻을 수 있습니까?
1. 더 깊은 사전 훈련된 모델을 사용하면 더 나은 결과를 얻습니까? 하이퍼파라미터를 어떻게 조정합니까? 결과를 더 향상시킬 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:
