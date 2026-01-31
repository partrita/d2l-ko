```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 심층 합성곱 신경망 (AlexNet) (Deep Convolutional Neural Networks (AlexNet))
:label:`sec_alexnet`


LeNet :cite:`LeCun.Jackel.Bottou.ea.1995`이 소개된 후 컴퓨터 비전 및 머신러닝 커뮤니티에 CNN이 잘 알려지기는 했지만, 
이 분야를 즉시 지배하지는 못했습니다. 
LeNet은 초기 소규모 데이터셋에서 좋은 결과를 얻었지만, 더 크고 현실적인 데이터셋에 대한 CNN의 성능과 훈련 가능성은 아직 확립되지 않았습니다. 
실제로 1990년대 초반과 2012년의 분수령이 되는 결과 :cite:`Krizhevsky.Sutskever.Hinton.2012` 사이의 시간 동안, 
신경망은 커널 방법 :cite:`Scholkopf.Smola.2002`, 앙상블 방법 :cite:`Freund.Schapire.ea.1996`, 구조적 추정 :cite:`Taskar.Guestrin.Koller.2004`과 같은 다른 머신러닝 방법에 의해 종종 추월당했습니다.

컴퓨터 비전의 경우 이 비교가 전적으로 정확하지 않을 수 있습니다. 
즉, 합성곱 네트워크에 대한 입력은 원시 또는 가볍게 처리된(예: 중심 맞추기) 픽셀 값으로 구성되지만, 실무자들은 전통적인 모델에 원시 픽셀을 공급하지 않았습니다. 
대신 일반적인 컴퓨터 비전 파이프라인은 SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, 시각적 단어 가방(bags of visual words) :cite:`Sivic.Zisserman.2003`과 같은 수동으로 엔지니어링된 특성 추출 파이프라인으로 구성되었습니다. 
특성을 *학습*하기보다는 특성을 *만들었습니다*. 
대부분의 진전은 한편으로는 특성 추출에 대한 더 기발한 아이디어와 다른 한편으로는 기하학에 대한 깊은 통찰력 :cite:`Hartley.Zisserman.2000`에서 비롯되었습니다. 학습 알고리즘은 종종 나중에 생각하는 것으로 간주되었습니다.

1990년대에 일부 신경망 가속기를 사용할 수 있었지만, 
많은 수의 파라미터를 가진 깊은 다중 채널, 다층 CNN을 만들기에는 아직 충분히 강력하지 않았습니다. 
예를 들어 1999년 NVIDIA의 GeForce 256은 게임 이외의 작업에 대한 의미 있는 프로그래밍 프레임워크 없이 초당 최대 4억 8천만 개의 부동 소수점 연산(덧셈 및 곱셈 등, MFLOPS)만 처리할 수 있었습니다. 
오늘날의 가속기는 장치당 1000 TFLOPs를 초과하는 성능을 발휘할 수 있습니다. 
게다가 데이터셋은 여전히 상대적으로 작았습니다: 60,000개의 저해상도 $28 \times 28$ 픽셀 이미지에 대한 OCR은 매우 어려운 작업으로 간주되었습니다. 
이러한 장애물에 더해 파라미터 초기화 휴리스틱 :cite:`Glorot.Bengio.2010`, 확률적 경사 하강법의 영리한 변형 :cite:`Kingma.Ba.2014`, 비스쿼싱(non-squashing) 활성화 함수 :cite:`Nair.Hinton.2010`, 효과적인 정규화 기술 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`을 포함한 신경망 훈련을 위한 핵심 트릭이 여전히 누락되었습니다.

따라서 *엔드 투 엔드*(픽셀에서 분류까지) 시스템을 훈련하는 대신 고전적인 파이프라인은 다음과 같았습니다:

1. 흥미로운 데이터셋을 얻습니다. 초기에는 이러한 데이터셋에 값비싼 센서가 필요했습니다. 예를 들어 1994년 [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake)은 무려 0.3메가픽셀(VGA) 해상도를 자랑했으며 최대 8개의 이미지를 저장할 수 있었고 가격은 1000달러였습니다.
2. 광학, 기하학, 기타 분석 도구에 대한 지식, 그리고 때로는 운 좋은 대학원생의 우연한 발견을 바탕으로 수작업으로 만든 특성으로 데이터셋을 전처리합니다.
3. SIFT(scale-invariant feature transform) :cite:`Lowe.2004`, SURF(speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006` 또는 기타 수동으로 튜닝된 파이프라인과 같은 표준 특성 추출기 세트를 통해 데이터를 공급합니다. OpenCV는 오늘날에도 여전히 SIFT 추출기를 제공합니다!
4. 결과 표현을 선형 모델이나 커널 방법과 같은 선호하는 분류기에 덤프하여 분류기를 훈련합니다.

머신러닝 연구자들과 이야기했다면, 그들은 머신러닝이 중요하고 아름답다고 대답했을 것입니다. 
우아한 이론은 다양한 분류기의 속성을 증명했고 :cite:`boucheron2005theory` 볼록 최적화 :cite:`Boyd.Vandenberghe.2004`는 이를 얻기 위한 주류가 되었습니다. 
머신러닝 분야는 번성하고 엄격하며 매우 유용했습니다. 하지만, 
컴퓨터 비전 연구자와 이야기했다면 매우 다른 이야기를 들었을 것입니다. 
이미지 인식의 더러운 진실은 새로운 학습 알고리즘이 아니라 특성, 기하학 :cite:`Hartley.Zisserman.2000,hartley2009global` 및 엔지니어링이 발전을 주도했다는 것이라고 그들은 말할 것입니다. 
컴퓨터 비전 연구자들은 약간 더 크거나 깨끗한 데이터셋 또는 약간 개선된 특성 추출 파이프라인이 어떤 학습 알고리즘보다 최종 정확도에 훨씬 더 중요하다고 정당하게 믿었습니다.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 표현 학습 (Representation Learning)

상황을 설명하는 또 다른 방법은 파이프라인의 가장 중요한 부분이 표현이었다는 것입니다. 
그리고 2012년까지 표현은 대부분 기계적으로 계산되었습니다. 
사실 새로운 특성 함수 세트를 엔지니어링하고 결과를 개선하고 방법을 작성하는 것이 모두 논문에서 두드러지게 다루어졌습니다. 
SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG(histograms of oriented gradient) :cite:`Dalal.Triggs.2005`, 시각적 단어 가방 :cite:`Sivic.Zisserman.2003` 및 유사한 특성 추출기들이 주도권을 잡았습니다.

Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari, Juergen Schmidhuber를 포함한 또 다른 연구자 그룹은 다른 계획을 가지고 있었습니다. 
그들은 특성 자체가 학습되어야 한다고 믿었습니다. 
더욱이 그들은 합리적으로 복잡해지기 위해서는 특성이 학습 가능한 파라미터를 가진 여러 공동 학습 레이어로 계층적으로 구성되어야 한다고 믿었습니다. 
이미지의 경우, 동물의 시각 시스템이 입력을 처리하는 방식과 유사하게 가장 낮은 레이어는 가장자리, 색상 및 텍스처를 감지할 수 있습니다. 특히 희소 코딩 :cite:`olshausen1996emergence`으로 얻은 것과 같은 시각적 특성의 자동 설계는 현대 CNN이 도래할 때까지 열린 과제로 남아 있었습니다. 
:citet:`Dean.Corrado.Monga.ea.2012,le2013building`에 이르러서야 이미지 데이터에서 자동으로 특성을 생성한다는 아이디어가 상당한 견인력을 얻었습니다.

최초의 현대식 CNN :cite:`Krizhevsky.Sutskever.Hinton.2012`은 발명가 중 한 명인 Alex Krizhevsky의 이름을 따서 *AlexNet*이라고 명명되었으며, 대체로 LeNet의 진화적 개선입니다. 2012년 ImageNet 챌린지에서 뛰어난 성능을 달성했습니다.

![AlexNet의 첫 번째 레이어에서 학습한 이미지 필터. 복제 허가: :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

흥미롭게도 네트워크의 가장 낮은 레이어에서 모델은 일부 전통적인 필터와 유사한 특성 추출기를 학습했습니다. 
:numref:`fig_filters`는 하위 수준 이미지 기술자를 보여줍니다. 
네트워크의 상위 레이어는 이러한 표현을 기반으로 눈, 코, 풀잎 등과 같은 더 큰 구조를 나타낼 수 있습니다. 
더 높은 레이어는 사람, 비행기, 개 또는 프리스비와 같은 전체 객체를 나타낼 수 있습니다. 
궁극적으로 최종 은닉 상태는 다른 범주에 속하는 데이터를 쉽게 분리할 수 있도록 콘텐츠를 요약하는 이미지의 압축된 표현을 학습합니다.

AlexNet(2012)과 그 전신인 LeNet(1995)은 많은 아키텍처 요소를 공유합니다. 그렇다면 왜 그렇게 오래 걸렸을까요? 
주요 차이점은 지난 20년 동안 데이터 양과 사용 가능한 컴퓨팅 성능이 크게 증가했다는 것입니다. 따라서 AlexNet은 훨씬 더 컸습니다: 1995년에 사용할 수 있었던 CPU에 비해 훨씬 더 빠른 GPU에서 훨씬 더 많은 데이터로 훈련되었습니다.

### 누락된 요소: 데이터 (Missing Ingredient: Data)

많은 레이어가 있는 심층 모델은 볼록 최적화에 기반한 전통적인 방법(예: 선형 및 커널 방법)을 훨씬 능가하는 영역에 진입하기 위해 많은 양의 데이터를 필요로 합니다. 
그러나 컴퓨터의 제한된 저장 용량, (이미징) 센서의 상대적인 비용, 1990년대의 비교적 부족한 연구 예산으로 인해 대부분의 연구는 작은 데이터셋에 의존했습니다. 
수많은 논문이 UCI 데이터셋 모음에 의존했는데, 그중 다수는 저해상도와 종종 인위적으로 깨끗한 배경으로 캡처된 수백 또는 (몇) 수천 개의 이미지만 포함하고 있었습니다.

2009년에 ImageNet 데이터셋이 공개되어 :cite:`Deng.Dong.Socher.ea.2009` 연구자들에게 1,000개의 고유한 객체 범주에서 각각 1,000개씩 100만 개의 예제로 모델을 학습하도록 도전했습니다. 범주 자체는 WordNet :cite:`Miller.1995`의 가장 인기 있는 명사 노드를 기반으로 했습니다. 
ImageNet 팀은 Google 이미지 검색을 사용하여 각 범주에 대한 대규모 후보 세트를 사전 필터링하고 Amazon Mechanical Turk 크라우드소싱 파이프라인을 사용하여 각 이미지가 관련 범주에 속하는지 확인했습니다. 
이 규모는 전례가 없었으며 다른 것들을 한 자리수 이상 초과했습니다(예: CIFAR-100은 60,000개의 이미지를 가짐). 또 다른 측면은 이미지가 $32 \times 32$ 픽셀 썸네일로 구성된 8천만 크기의 TinyImages 데이터셋 :cite:`Torralba.Fergus.Freeman.2008`과 달리 $224 \times 224$ 픽셀의 비교적 고해상도라는 점이었습니다. 
이를 통해 더 높은 수준의 특성을 형성할 수 있었습니다. 
ImageNet 대규모 시각 인식 챌린지(ImageNet Large Scale Visual Recognition Challenge) :cite:`russakovsky2015imagenet`라고 불리는 관련 대회는 컴퓨터 비전 및 머신러닝 연구를 발전시켜 연구자들이 이전에 학계에서 고려했던 것보다 더 큰 규모에서 어떤 모델이 가장 잘 수행되는지 식별하도록 도전했습니다. LAION-5B :cite:`schuhmann2022laion`와 같은 가장 큰 비전 데이터셋에는 추가 메타데이터가 있는 수십억 개의 이미지가 포함되어 있습니다.

### 누락된 요소: 하드웨어 (Missing Ingredient: Hardware)

딥러닝 모델은 컴퓨팅 사이클의 엄청난 소비자입니다. 
훈련에는 수백 에폭이 걸릴 수 있으며, 각 반복은 계산 비용이 많이 드는 선형 대수 연산의 많은 레이어를 통해 데이터를 전달해야 합니다. 
이것이 1990년대와 2000년대 초반에 보다 효율적으로 최적화된 볼록 목적 함수에 기반한 간단한 알고리즘이 선호되었던 주된 이유 중 하나입니다.

*그래픽 처리 장치*(GPU)는 딥러닝을 실현 가능하게 만드는 게임 체임저로 입증되었습니다. 
이 칩들은 이전에 컴퓨터 게임에 도움이 되도록 그래픽 처리를 가속화하기 위해 개발되었습니다. 
특히 많은 컴퓨터 그래픽 작업에 필요한 높은 처리량의 $4 \times 4$ 행렬-벡터 곱에 최적화되었습니다. 
다행히도 이 수학은 합성곱 레이어를 계산하는 데 필요한 것과 놀랍도록 유사합니다. 
그 무렵 NVIDIA와 ATI는 범용 컴퓨팅 작업을 위해 GPU를 최적화하기 시작했으며 :cite:`Fernando.2004`, *범용 GPU*(GPGPU)로 마케팅하기까지 했습니다.

직관을 제공하기 위해 최신 마이크로프로세서(CPU)의 코어를 고려해 보십시오. 
각 코어는 높은 클록 주파수에서 실행되고 대형 캐시(최대 수 메가바이트의 L3)를 자랑하는 상당히 강력한 성능을 제공합니다. 
각 코어는 분기 예측기, 깊은 파이프라인, 특수 실행 유닛, 투기적 실행 및 정교한 제어 흐름을 가진 다양한 프로그램을 실행할 수 있는 기타 많은 부가 기능을 통해 광범위한 명령을 실행하는 데 적합합니다. 
그러나 이 명백한 강점은 아킬레스건이기도 합니다. 범용 코어는 구축하는 데 비용이 많이 듭니다. 그들은 제어 흐름이 많은 범용 코드에 탁월합니다. 
이를 위해서는 계산이 일어나는 실제 ALU(산술 논리 장치)뿐만 아니라 앞서 언급한 모든 부가 기능, 메모리 인터페이스, 코어 간 캐싱 로직, 고속 상호 연결 등을 위한 많은 칩 면적이 필요합니다. CPU는 전용 하드웨어와 비교할 때 단일 작업에 상대적으로 나쁩니다. 
최신 노트북에는 4~8개의 코어가 있으며 하이엔드 서버조차도 소켓당 64개 코어를 거의 초과하지 않습니다. 단순히 비용 효율적이지 않기 때문입니다.

이에 비해 GPU는 수천 개의 작은 처리 요소(NIVIDA의 최신 Ampere 칩에는 최대 6912개의 CUDA 코어가 있음)로 구성될 수 있으며 종종 더 큰 그룹(NVIDIA는 워프라고 부름)으로 그룹화됩니다. 
세부 사항은 NVIDIA, AMD, ARM 및 기타 칩 공급업체마다 다소 다릅니다. 각 코어는 약 1GHz 클록 주파수에서 실행되어 상대적으로 약하지만, GPU를 CPU보다 몇 자릿수 더 빠르게 만드는 것은 그러한 코어의 총수입니다. 
예를 들어 NVIDIA의 최근 Ampere A100 GPU는 특수 16비트 정밀도(BFLOAT16) 행렬-행렬 곱셈에 대해 칩당 300 TFLOPs 이상을 제공하고 보다 범용적인 부동 소수점 연산(FP32)에 대해 최대 20 TFLOPs를 제공합니다. 
동시에 CPU의 부동 소수점 성능은 1 TFLOPs를 거의 초과하지 않습니다. 예를 들어 Amazon의 Graviton 3은 16비트 정밀도 연산에 대해 2 TFLOPs 피크 성능에 도달하는데, 이는 Apple M1 프로세서의 GPU 성능과 비슷한 수치입니다.

FLOPs 측면에서 GPU가 CPU보다 훨씬 빠른 데는 여러 가지 이유가 있습니다. 
첫째, 전력 소비는 클록 주파수에 따라 *이차적으로* 증가하는 경향이 있습니다. 
따라서 4배 더 빠르게 실행되는 CPU 코어의 전력 예산(일반적인 수치)으로 $\frac{1}{4}$ 속도의 GPU 코어 16개를 사용할 수 있으며, 이는 $16 \times \frac{1}{4} = 4$배의 성능을 산출합니다. 
둘째, GPU 코어는 훨씬 단순하여(사실 오랫동안 범용 코드를 실행할 수조차 *없었음*) 에너지 효율이 더 높습니다. 예를 들어 (i) 투기적 평가를 지원하지 않는 경향이 있고, (ii) 일반적으로 각 처리 요소를 개별적으로 프로그래밍할 수 없으며, (iii) 코어당 캐시가 훨씬 작습니다. 
마지막으로, 딥러닝의 많은 작업에는 높은 메모리 대역폭이 필요합니다. 
다시 말하지만, GPU는 많은 CPU보다 적어도 10배 더 넓은 버스로 여기서 빛납니다.

2012년으로 돌아가 봅시다. Alex Krizhevsky와 Ilya Sutskever가 GPU에서 실행할 수 있는 심층 CNN을 구현했을 때 주요 돌파구가 마련되었습니다. 
그들은 CNN의 계산 병목 현상인 합성곱과 행렬 곱셈이 모두 하드웨어에서 병렬화할 수 있는 연산이라는 것을 깨달았습니다. 
각각 1.5 TFLOPs(10년 후 대부분의 CPU에게도 여전히 도전적인 수치)의 성능을 가진 3GB 메모리의 NVIDIA GTX 580 2개를 사용하여 빠른 합성곱을 구현했습니다. 
[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) 코드는 몇 년 동안 업계 표준이 되어 딥러닝 붐의 첫 몇 년을 이끌 만큼 훌륭했습니다.

## AlexNet

8레이어 CNN을 채용한 AlexNet은 2012 ImageNet 대규모 시각 인식 챌린지에서 큰 격차로 우승했습니다 :cite:`Russakovsky.Deng.Huang.ea.2013`. 
이 네트워크는 학습을 통해 얻은 특성이 수동으로 설계된 특성을 초월할 수 있음을 처음으로 보여주며 컴퓨터 비전의 이전 패러다임을 깨뜨렸습니다.

![LeNet(왼쪽)에서 AlexNet(오른쪽)으로.](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet과 LeNet 사이에는 중요한 차이점도 있습니다. 
첫째, AlexNet은 비교적 작은 LeNet-5보다 훨씬 깊습니다. 
AlexNet은 8개 레이어로 구성됩니다: 5개의 합성곱 레이어, 2개의 완전 연결 은닉 레이어, 1개의 완전 연결 출력 레이어입니다. 
둘째, AlexNet은 활성화 함수로 시그모이드 대신 ReLU를 사용했습니다. 아래에서 세부 사항을 살펴봅시다.

### 아키텍처 (Architecture)

AlexNet의 첫 번째 레이어에서 합성곱 윈도우 모양은 $11\times11$입니다. 
ImageNet의 이미지는 MNIST 이미지보다 높이와 너비가 8배 더 크기 때문에, ImageNet 데이터의 객체는 더 많은 시각적 세부 정보와 함께 더 많은 픽셀을 차지하는 경향이 있습니다. 
결과적으로 객체를 포착하려면 더 큰 합성곱 윈도우가 필요합니다. 
두 번째 레이어의 합성곱 윈도우 모양은 $5\times5$로 줄어들고, 그다음에는 $3\times3$이 이어집니다. 
또한 첫 번째, 두 번째, 다섯 번째 합성곱 레이어 뒤에 네트워크는 윈도우 모양이 $3\times3$이고 스트라이드가 2인 최대 풀링 레이어를 추가합니다. 
게다가 AlexNet은 LeNet보다 10배 더 많은 합성곱 채널을 가지고 있습니다.

마지막 합성곱 레이어 뒤에는 4096개의 출력을 가진 두 개의 거대한 완전 연결 레이어가 있습니다. 
이 레이어들은 거의 1GB의 모델 파라미터를 필요로 합니다. 
초기 GPU의 메모리 제한 때문에 원래 AlexNet은 이중 데이터 스트림 설계를 사용하여 두 개의 GPU 각각이 모델의 절반만 저장하고 계산하는 것을 담당하도록 했습니다. 
다행히 지금은 GPU 메모리가 비교적 풍부하므로 요즘에는 모델을 GPU에 걸쳐 분할해야 하는 경우가 드뭅니다(우리의 AlexNet 모델 버전은 이 측면에서 원래 논문과 다릅니다).

### 활성화 함수 (Activation Functions)

또한 AlexNet은 시그모이드 활성화 함수를 더 간단한 ReLU 활성화 함수로 변경했습니다. 한편으로 ReLU 활성화 함수의 계산은 더 간단합니다. 예를 들어 시그모이드 활성화 함수에서 볼 수 있는 지수 연산이 없습니다. 
다른 한편으로 ReLU 활성화 함수는 다른 파라미터 초기화 방법을 사용할 때 모델 훈련을 더 쉽게 만듭니다. 이는 시그모이드 활성화 함수의 출력이 0 또는 1에 매우 가까울 때 이 영역의 기울기가 거의 0이 되어 역전파가 일부 모델 파라미터를 계속 업데이트할 수 없기 때문입니다. 대조적으로 양의 구간에서 ReLU 활성화 함수의 기울기는 항상 1입니다(:numref:`subsec_activation-functions`). 따라서 모델 파라미터가 적절하게 초기화되지 않으면 시그모이드 함수는 양의 구간에서 거의 0인 기울기를 얻어 모델을 효과적으로 훈련할 수 없게 될 수 있습니다.

### 용량 제어 및 전처리 (Capacity Control and Preprocessing)

AlexNet은 드롭아웃(:numref:`sec_dropout`)으로 완전 연결 레이어의 모델 복잡도를 제어하는 반면, LeNet은 가중치 감쇠만 사용합니다. 
데이터를 더욱 증강하기 위해 AlexNet의 훈련 루프는 뒤집기(flipping), 자르기(clipping), 색상 변경과 같은 많은 이미지 증강을 추가했습니다. 
이는 모델을 더 견고하게 만들고 더 큰 샘플 크기는 과대적합을 효과적으로 줄입니다. 
이러한 전처리 단계에 대한 심층적인 검토는 :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020`를 참조하십시오.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

우리는 높이와 너비가 224인 [**단일 채널 데이터 예제를 구성**]하여 (**각 레이어의 출력 모양을 관찰**)합니다. 이는 :numref:`fig_alexnet`의 AlexNet 아키텍처와 일치합니다.

```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## 훈련 (Training)

AlexNet은 :citet:`Krizhevsky.Sutskever.Hinton.2012`에서 ImageNet으로 훈련되었지만, 여기서는 Fashion-MNIST를 사용합니다. 
ImageNet 모델을 수렴할 때까지 훈련하는 것은 최신 GPU에서도 몇 시간 또는 며칠이 걸릴 수 있기 때문입니다. 
[**Fashion-MNIST**]에 AlexNet을 직접 적용할 때의 문제 중 하나는 (**이미지가 ImageNet 이미지보다**) (**해상도가 낮다는 것**)(28 $\times$ 28 픽셀)입니다. 
제대로 작동하게 하기 위해, (**우리는 이를 $224 \times 224$로 업샘플링합니다**). 
이는 단순히 정보를 추가하지 않고 계산 복잡성을 증가시키기 때문에 일반적으로 현명한 관행은 아닙니다. 그럼에도 불구하고 우리는 AlexNet 아키텍처에 충실하기 위해 여기서 이렇게 합니다. 
`d2l.FashionMNIST` 생성자의 `resize` 인수를 사용하여 이 크기 조정을 수행합니다.

이제 [**AlexNet 훈련을 시작할 수 있습니다.**] 
:numref:`sec_lenet`의 LeNet과 비교할 때, 주요 변경 사항은 더 깊고 넓은 네트워크, 더 높은 이미지 해상도, 더 많은 비용이 드는 합성곱으로 인해 더 작은 학습률을 사용하고 훈련이 훨씬 느리다는 것입니다.

```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## 토론 (Discussion)

AlexNet의 구조는 정확도(드롭아웃)와 훈련 용이성(ReLU) 모두에 대한 여러 가지 중요한 개선 사항과 함께 LeNet과 놀랍도록 유사합니다. 마찬가지로 놀라운 것은 딥러닝 툴링 측면에서 이루어진 진전의 양입니다. 2012년에 몇 달이 걸렸던 작업은 이제 모든 최신 프레임워크를 사용하여 12줄의 코드로 수행할 수 있습니다.

아키텍처를 검토해 보면 AlexNet은 효율성 측면에서 아킬레스건이 있습니다: 마지막 두 은닉층에는 각각 $6400 \times 4096$ 및 $4096 \times 4096$ 크기의 행렬이 필요합니다. 이는 164MB의 메모리와 81 MFLOPs의 계산에 해당하며, 둘 다 특히 휴대전화와 같은 소형 장치에서는 사소하지 않은 지출입니다. 이것이 AlexNet이 다음 섹션에서 다룰 훨씬 더 효과적인 아키텍처에 의해 추월당한 이유 중 하나입니다. 그럼에도 불구하고 이는 오늘날 사용되는 얕은 네트워크에서 깊은 네트워크로 가는 핵심 단계입니다. 실험에서 파라미터 수가 훈련 데이터 양을 훨씬 초과하더라도(마지막 두 레이어에는 6만 개의 이미지 데이터셋에서 훈련된 4천만 개 이상의 파라미터가 있음) 과대적합이 거의 없습니다: 훈련 및 검증 손실은 훈련 내내 거의 동일합니다. 이는 현대 심층 네트워크 설계에 내재된 드롭아웃과 같은 개선된 정규화 때문입니다.

AlexNet 구현이 LeNet보다 몇 줄 더 많은 것처럼 보이지만, 학계가 이 개념적 변화를 받아들이고 뛰어난 실험 결과를 활용하는 데는 수년이 걸렸습니다. 이는 또한 효율적인 계산 도구의 부족 때문이기도 했습니다. 당시에는 DistBelief :cite:`Dean.Corrado.Monga.ea.2012`나 Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`가 존재하지 않았고 Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`는 여전히 많은 특징적인 기능이 부족했습니다. 상황을 극적으로 바꾼 것은 TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`의 가용성이었습니다.

## 연습 문제 (Exercises)

1. 위 논의에 이어 AlexNet의 계산적 특성을 분석하십시오.
    1. 합성곱과 완전 연결 레이어의 메모리 사용량을 각각 계산하십시오. 어느 것이 지배적입니까?
    1. 합성곱과 완전 연결 레이어의 계산 비용을 계산하십시오.
    1. 메모리(읽기 및 쓰기 대역폭, 대기 시간, 크기)가 계산에 어떤 영향을 미칩니까? 훈련과 추론에 미치는 영향에 차이가 있습니까?
2. 당신은 칩 설계자이고 계산과 메모리 대역폭을 절충해야 합니다. 예를 들어 더 빠른 칩은 더 많은 전력과 아마도 더 큰 칩 면적을 필요로 합니다. 더 많은 메모리 대역폭은 더 많은 핀과 제어 로직을 필요로 하므로 더 많은 면적을 필요로 합니다. 어떻게 최적화합니까?
3. 엔지니어들이 더 이상 AlexNet에 대한 성능 벤치마크를 보고하지 않는 이유는 무엇입니까?
4. AlexNet을 훈련할 때 에폭 수를 늘려 보십시오. LeNet과 비교할 때 결과가 어떻게 다릅니까? 그 이유는 무엇입니까?
5. AlexNet은 Fashion-MNIST 데이터셋, 특히 초기 이미지의 저해상도 때문에 너무 복잡할 수 있습니다.
    1. 정확도가 크게 떨어지지 않도록 하면서 훈련 속도를 높이도록 모델을 단순화해 보십시오.
    1. $28 \times 28$ 이미지에서 직접 작동하는 더 나은 모델을 설계하십시오.
6. 배치 크기를 수정하고 처리량(이미지/초), 정확도 및 GPU 메모리의 변화를 관찰하십시오.
7. LeNet-5에 드롭아웃과 ReLU를 적용하십시오. 개선됩니까? 이미지에 내재된 불변성을 활용하기 위해 전처리를 통해 상황을 더 개선할 수 있습니까?
8. AlexNet을 과대적합하게 만들 수 있습니까? 훈련을 중단시키려면 어떤 특성을 제거하거나 변경해야 합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/276)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18001)
:end_tab:

```