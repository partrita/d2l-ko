# 완전 합성곱 네트워크 (Fully Convolutional Networks)
:label:`sec_fcn`

:numref:`sec_semantic_segmentation`에서 논의한 바와 같이,
시맨틱 분할은
이미지를 픽셀 수준에서 분류합니다.
완전 합성곱 네트워크(FCN)는
이미지 픽셀을 픽셀 클래스로 변환하기 위해 합성곱 신경망을 사용합니다 :cite:`Long.Shelhamer.Darrell.2015`.
이미지 분류 또는 객체 감지를 위해
이전에 접했던 CNN과 달리,
완전 합성곱 네트워크는
중간 특징 맵의 높이와 너비를
입력 이미지의 높이와 너비로 다시 변환합니다.
이것은 :numref:`sec_transposed_conv`에서 소개한
전치 합성곱 레이어에 의해 달성됩니다.
결과적으로,
분류 출력과 입력 이미지는
픽셀 수준에서 일대일 대응을 가집니다.
어떤 출력 픽셀의 채널 차원은
동일한 공간 위치에 있는 입력 픽셀에 대한 분류 결과를 보유합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## 모델 (The Model)

여기서는 완전 합성곱 네트워크 모델의 기본 설계를 설명합니다.
:numref:`fig_fcn`에 표시된 것처럼,
이 모델은 먼저 CNN을 사용하여 이미지 특징을 추출하고,
그 다음 $1\times 1$ 합성곱 레이어를 통해 채널 수를 클래스 수로 변환하며,
마지막으로 :numref:`sec_transposed_conv`에서 소개한 전치 합성곱을 통해
특징 맵의 높이와 너비를
입력 이미지의 높이와 너비로 변환합니다.
결과적으로,
모델 출력은 입력 이미지와 동일한 높이와 너비를 가지며,
출력 채널에는 동일한 공간 위치에 있는 입력 픽셀에 대한 예측 클래스가 포함됩니다.


![완전 합성곱 네트워크.](../img/fcn.svg)
:label:`fig_fcn`

아래에서, 우리는 [**ImageNet 데이터셋에서 사전 훈련된 ResNet-18 모델을 사용하여 이미지 특징을 추출**]하고
모델 인스턴스를 `pretrained_net`으로 표시합니다.
이 모델의 마지막 몇 개 레이어에는
글로벌 평균 풀링 레이어와 완전 연결 레이어가 포함되어 있습니다.
이들은 완전 합성곱 네트워크에서 필요하지 않습니다.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

다음으로, 우리는 [**완전 합성곱 네트워크 인스턴스 `net`을 생성**]합니다.
이것은 ResNet-18의 모든 사전 훈련된 레이어를 복사합니다.
단, 출력에 가장 가까운 최종 글로벌 평균 풀링 레이어와 완전 연결 레이어는 제외합니다.

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

높이와 너비가 각각 320과 480인 입력이 주어지면,
`net`의 순방향 전파는
입력 높이와 너비를 원래의 1/32, 즉 10과 15로 줄입니다.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

다음으로, 우리는 [**$1\times 1$ 합성곱 레이어를 사용하여 출력 채널 수를 Pascal VOC2012 데이터셋의 클래스 수(21)로 변환합니다.**]
마지막으로, 특징 맵을 입력 이미지의 높이와 너비로 다시 변경하기 위해 (**특징 맵의 높이와 너비를 32배로 늘려야 합니다**).
:numref:`sec_padding`에서 합성곱 레이어의 출력 모양을 계산하는 방법을 상기하십시오.
$(320-64+16\times2+32)/32=10$이고 $(480-64+16\times2+32)/32=15$이므로, 우리는 스트라이드 $32$인 전치 합성곱 레이어를 구성하고,
커널의 높이와 너비를 $64$로, 패딩을 $16$으로 설정합니다. 
일반적으로,
스트라이드 $s$,
패딩 $s/2$($s/2$가 정수라고 가정),
커널의 높이와 너비 $2s$에 대해,
전치 합성곱은 입력의 높이와 너비를 $s$배로 늘립니다.

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**전치 합성곱 레이어 초기화 (Initializing Transposed Convolutional Layers)**]


우리는 이미 전치 합성곱 레이어가
특징 맵의 높이와 너비를 늘릴 수 있다는 것을 알고 있습니다.
이미지 처리에서, 우리는 이미지를 확대해야 할 수도 있습니다. 즉, *업샘플링(upsampling)*입니다.
*이중 선형 보간법(Bilinear interpolation)*은
일반적으로 사용되는 업샘플링 기술 중 하나입니다.
이것은 또한 전치 합성곱 레이어를 초기화하는 데 자주 사용됩니다.

이중 선형 보간법을 설명하기 위해, 
입력 이미지가 주어졌을 때
업샘플링된 출력 이미지의 각 픽셀을
계산하고 싶다고 가정해 봅시다.
좌표 $(x, y)$에 있는 출력 이미지의 픽셀을 계산하기 위해,
먼저 입력 크기 대 출력 크기의 비율에 따라 $(x, y)$를 입력 이미지의 좌표 $(x', y')$로 매핑합니다.
매핑된 $x'$와 $y'$는 실수입니다.
그런 다음, 입력 이미지에서 좌표 $(x', y')$에 가장 가까운 4개의 픽셀을 찾습니다.
마지막으로, 좌표 $(x, y)$에 있는 출력 이미지의 픽셀은 입력 이미지의 이 4개의 가장 가까운 픽셀과
$(x', y')$로부터의 상대적 거리를 기반으로 계산됩니다.

이중 선형 보간법의 업샘플링은
다음 `bilinear_kernel` 함수에 의해 구성된 커널을 가진 전치 합성곱 레이어로 구현될 수 있습니다.
공간 제한으로 인해, 알고리즘 설계에 대한 논의 없이 아래에 `bilinear_kernel` 함수의 구현만 제공합니다.

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

전치 합성곱 레이어에 의해 구현된 [**이중 선형 보간법의 업샘플링을 실험**]해 봅시다.
우리는 높이와 너비를 두 배로 늘리는 전치 합성곱 레이어를 구성하고,
`bilinear_kernel` 함수로 커널을 초기화합니다.

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

이미지 `X`를 읽고 업샘플링 출력을 `Y`에 할당합니다. 이미지를 인쇄하려면 채널 차원의 위치를 조정해야 합니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

보시다시피, 전치 합성곱 레이어는 이미지의 높이와 너비를 모두 2배로 늘립니다.
좌표의 스케일이 다른 것을 제외하고,
이중 선형 보간법으로 확대된 이미지와 :numref:`sec_bbox`에서 인쇄된 원본 이미지는 동일하게 보입니다.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

완전 합성곱 네트워크에서, 우리는 [**이중 선형 보간법의 업샘플링으로 전치 합성곱 레이어를 초기화합니다. $1\times 1$ 합성곱 레이어의 경우 Xavier 초기화를 사용합니다.**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**데이터셋 읽기 (Reading the Dataset)**]

우리는 :numref:`sec_semantic_segmentation`에서 소개한
시맨틱 분할 데이터셋을 읽습니다.
무작위 자르기의 출력 이미지 모양은 $320\times 480$으로 지정됩니다. 높이와 너비 모두 32로 나누어떨어집니다.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**훈련 (Training)**]


이제 우리가 구성한
완전 합성곱 네트워크를 훈련할 수 있습니다. 
여기서의 손실 함수와 정확도 계산은
이전 장의 이미지 분류와 본질적으로 다르지 않습니다. 
우리는 전치 합성곱 레이어의 출력 채널을 사용하여
각 픽셀에 대한 클래스를 예측하기 때문에,
손실 계산에서 채널 차원이 지정됩니다. 
또한, 정확도는 모든 픽셀에 대해 예측된 클래스의 정확성을 기반으로 계산됩니다.

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**예측 (Prediction)**]


예측할 때, 우리는 각 채널에서 입력 이미지를 표준화하고
이미지를 CNN에 필요한 4차원 입력 형식으로 변환해야 합니다.

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

각 픽셀의 [**예측된 클래스를 시각화**]하기 위해, 예측된 클래스를 데이터셋의 라벨 색상으로 다시 매핑합니다.

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

테스트 데이터셋의 이미지는 크기와 모양이 다양합니다.
모델은 스트라이드가 32인 전치 합성곱 레이어를 사용하므로,
입력 이미지의 높이나 너비가 32로 나누어떨어지지 않으면
전치 합성곱 레이어의 출력 높이나 너비가 입력 이미지의 모양에서 벗어날 것입니다.
이 문제를 해결하기 위해,
우리는 이미지에서 높이와 너비가 32의 정수 배수인 여러 직사각형 영역을 자르고,
이 영역의 픽셀에 대해 개별적으로 순방향 전파를 수행할 수 있습니다. 
이러한 직사각형 영역의 합집합은 입력 이미지를 완전히 덮어야 합니다. 
픽셀이 여러 직사각형 영역에 의해 덮일 때, 
이 동일한 픽셀에 대한 별도 영역의 전치 합성곱 출력 평균을
소프트맥스 연산에 입력하여 클래스를 예측할 수 있습니다.


간단하게 하기 위해, 우리는 몇 개의 더 큰 테스트 이미지만 읽고, 
이미지의 왼쪽 상단 모서리에서 시작하여 예측을 위해 $320\times 480$ 영역을 자릅니다. 
이 테스트 이미지들에 대해, 우리는
자른 영역,
예측 결과,
그리고 실제(ground-truth)를 줄별로 인쇄합니다.

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## 요약 (Summary)

* 완전 합성곱 네트워크는 먼저 CNN을 사용하여 이미지 특징을 추출하고, 그 다음 $1\times 1$ 합성곱 레이어를 통해 채널 수를 클래스 수로 변환하며, 마지막으로 전치 합성곱을 통해 특징 맵의 높이와 너비를 입력 이미지의 높이와 너비로 변환합니다.
* 완전 합성곱 네트워크에서, 우리는 전치 합성곱 레이어를 초기화하기 위해 이중 선형 보간법의 업샘플링을 사용할 수 있습니다.


## 연습 문제 (Exercises)

1. 실험에서 전치 합성곱 레이어에 Xavier 초기화를 사용하면 결과가 어떻게 변합니까?
1. 하이퍼파라미터를 조정하여 모델의 정확도를 더 향상시킬 수 있습니까?
1. 테스트 이미지의 모든 픽셀 클래스를 예측하십시오.
1. 원래의 완전 합성곱 네트워크 논문은 또한 일부 중간 CNN 레이어의 출력을 사용합니다 :cite:`Long.Shelhamer.Darrell.2015`. 이 아이디어를 구현해 보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
