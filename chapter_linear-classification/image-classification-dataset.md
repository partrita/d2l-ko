```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 이미지 분류 데이터셋 (The Image Classification Dataset)
:label:`sec_fashion_mnist`

이미지 분류를 위해 널리 사용되는 데이터셋 중 하나는 손글씨 숫자로 구성된 [MNIST 데이터셋](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`입니다. 1990년대 출시 당시에는 60,000개의 $28 \times 28$ 픽셀 해상도 이미지(와 10,000개의 테스트 데이터셋)로 구성되어 대부분의 머신러닝 알고리즘에 만만치 않은 도전을 안겨주었습니다. 당시 상황을 돌이켜보면, 1995년에 무려 64MB의 RAM과 5 MFLOPs의 성능을 가진 Sun SPARCStation 5가 AT&T Bell 연구소에서 머신러닝을 위한 최첨단 장비로 간주되었습니다. 숫자 인식에서 높은 정확도를 달성하는 것은 1990년대 USPS의 우편물 분류 자동화의 핵심 구성 요소였습니다. LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`와 같은 심층 네트워크, 불변성을 가진 서포트 벡터 머신(SVM) :cite:`Scholkopf.Burges.Vapnik.1996`, 탄젠트 거리 분류기 :cite:`Simard.LeCun.Denker.ea.1998` 등은 모두 1% 미만의 오차율에 도달할 수 있었습니다. 

10년 넘게 MNIST는 머신러닝 알고리즘을 비교하는 *기준점* 역할을 했습니다. 하지만 벤치마크 데이터셋으로서의 명성이 무색하게도 오늘날의 기준으로는 단순한 모델조차 95% 이상의 분류 정확도를 달성하여, 강력한 모델과 약한 모델을 구분하기에 부적절해졌습니다. 더욱이 이 데이터셋은 많은 분류 문제에서 일반적으로 볼 수 없는 *매우* 높은 수준의 정확도를 허용합니다. 이는 active set methods나 boundary-seeking active set algorithms와 같이 깨끗한 데이터셋을 활용할 수 있는 특정 알고리즘 제품군 위주로 알고리즘 개발을 편향시켰습니다. 오늘날 MNIST는 벤치마크라기보다는 정상성 확인(sanity check) 용도에 가깝습니다. ImageNet :cite:`Deng.Dong.Socher.ea.2009`이 훨씬 더 유의미한 도전을 제시합니다. 불행히도 ImageNet은 이 책의 많은 예제와 그림에 사용하기에는 너무 커서, 예제를 대화식으로 만들기 위해 훈련하는 데 너무 오랜 시간이 걸립니다. 대용으로 우리는 다음 섹션에서 질적으로는 유사하지만 훨씬 작은 Fashion-MNIST 데이터셋 :cite:`Xiao.Rasul.Vollgraf.2017`에 초점을 맞출 것입니다. 2017년에 출시된 이 데이터셋은 $28 \times 28$ 픽셀 해상도의 10가지 범주의 의류 이미지를 포함하고 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## 데이터셋 로드하기 (Loading the Dataset)

Fashion-MNIST 데이터셋은 매우 유용하기 때문에 모든 주요 프레임워크에서 전처리된 버전을 제공합니다. 우리는 [**프레임워크의 내장 유틸리티를 사용하여 이를 다운로드하고 메모리로 읽어올 수 있습니다.**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """Fashion-MNIST 데이터셋입니다."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """Fashion-MNIST 데이터셋입니다."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """Fashion-MNIST 데이터셋입니다."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST는 10개의 카테고리로 구성되며, 각 카테고리는 훈련 데이터셋에 6000개, 테스트 데이터셋에 1000개의 이미지로 표현됩니다. *테스트 데이터셋*은 모델 성능을 평가하는 데 사용됩니다(훈련에 사용되어서는 안 됩니다). 결과적으로 훈련 세트와 테스트 세트는 각각 60,000개와 10,000개의 이미지를 포함합니다.

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

이미지는 그레이스케일이며 위에서 $32 \times 32$ 픽셀 해상도로 업스케일되었습니다. 이는 (이진) 흑백 이미지로 구성된 원래의 MNIST 데이터셋과 유사합니다. 그러나 대부분의 현대적인 이미지 데이터는 3개의 채널(빨강, 초록, 파랑)을 가지며 초분광(hyperspectral) 이미지는 100개 이상의 채널을 가질 수 있습니다(HyMap 센서는 126개 채널을 가짐). 관례에 따라 이미지는 $c \times h \times w$ 텐서로 저장됩니다. 여기서 $c$는 색상 채널 수, $h$는 높이, $w$는 너비입니다.

```{.python .input}
%%tab all
data.train[0][0].shape
```

Fashion-MNIST의 카테고리는 사람이 이해할 수 있는 이름을 가지고 있습니다. 다음 편의 메서드는 숫자 레이블과 그 이름 사이를 변환합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """텍스트 레이블을 반환합니다."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## 미니배치 읽기 (Reading a Minibatch)

훈련 세트와 테스트 세트에서 읽을 때 편의를 위해, 처음부터 만드는 대신 내장 데이터 반복기를 사용합니다. 매 반복마다 데이터 반복기는 [**`batch_size` 크기의 데이터 미니배치를 읽습니다.**] 또한 훈련 데이터 반복기에 대해 예제를 무작위로 셔플링합니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

어떻게 작동하는지 확인하기 위해 `train_dataloader` 메서드를 호출하여 이미지 미니배치를 로드해 보겠습니다. 64개의 이미지를 포함하고 있습니다.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

이미지를 읽는 데 걸리는 시간을 살펴봅시다. 내장 로더임에도 불구하고 엄청나게 빠르지는 않습니다. 그럼에도 불구하고 심층 네트워크로 이미지를 처리하는 데는 상당한 시간이 더 걸리기 때문에 이는 충분합니다. 따라서 네트워크 훈련이 I/O에 의해 제한되지 않을 정도로 충분히 빠릅니다.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} 초'
```

## 시각화 (Visualization)

우리는 종종 Fashion-MNIST 데이터셋을 사용할 것입니다. 편의 함수 `show_images`를 사용하여 이미지와 관련 레이블을 시각화할 수 있습니다. 구현 세부 사항은 생략하고 아래 인터페이스만 보여드립니다. 이러한 유틸리티 함수에 대해서는 작동 방식보다는 `d2l.show_images`를 호출하는 방법만 알면 됩니다.

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """이미지 리스트를 플롯합니다."""
    raise NotImplementedError
```

유용하게 활용해 봅시다. 일반적으로 여러분이 훈련하고 있는 데이터를 시각화하고 검사하는 것이 좋은 아이디어입니다. 인간은 이상한 점을 발견하는 데 매우 능숙하며, 그 덕분에 시각화는 실험 설계의 실수와 오차에 대한 추가적인 안전장치 역할을 합니다. 여기 훈련 데이터셋의 처음 몇 가지 예제에 대한 [**이미지와 그에 대응하는 (텍스트 형태의) 레이블**]이 있습니다.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

이제 다음 섹션에서 Fashion-MNIST 데이터셋을 사용하여 작업할 준비가 되었습니다.

## 요약 (Summary)

이제 분류에 사용할 수 있는 약간 더 현실적인 데이터셋을 확보했습니다. Fashion-MNIST는 10가지 범주를 나타내는 이미지로 구성된 의류 분류 데이터셋입니다. 우리는 이후 섹션과 장에서 단순한 선형 모델부터 고급 잔차 네트워크(residual networks)까지 다양한 네트워크 설계를 평가하기 위해 이 데이터셋을 사용할 것입니다. 이미지에서 흔히 하듯이, 이미지를 (배치 크기, 채널 수, 높이, 너비) 모양의 텐서로 읽어옵니다. 현재 이미지는 그레이스케일이므로 채널이 하나뿐입니다(위의 시각화는 가독성을 높이기 위해 가상 컬러 팔레트를 사용했습니다). 

마지막으로, 데이터 반복기는 효율적인 성능을 위한 핵심 구성 요소입니다. 예를 들어, 효율적인 이미지 압축 해제, 비디오 트랜스코딩 또는 기타 전처리를 위해 GPU를 사용할 수 있습니다. 가능할 때마다 훈련 루프의 속도가 느려지지 않도록 고성능 컴퓨팅을 활용하는 잘 구현된 데이터 반복기에 의존해야 합니다.


## 연습 문제 (Exercises)

1. `batch_size`를 줄이면(예를 들어 1로) 읽기 성능에 영향을 미칩니까?
2. 데이터 반복기 성능은 중요합니다. 현재 구현이 충분히 빠르다고 생각하십니까? 성능을 개선하기 위한 다양한 옵션을 탐색해 보십시오. 시스템 프로파일러를 사용하여 병목 지점이 어디인지 찾아보십시오.
3. 프레임워크의 온라인 API 문서를 확인해 보십시오. 다른 어떤 데이터셋들을 사용할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17980)
:end_tab: