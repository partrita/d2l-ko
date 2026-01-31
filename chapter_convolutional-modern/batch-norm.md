```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 배치 정규화 (Batch Normalization)
:label:`sec_batch_norm`

심층 신경망을 훈련하는 것은 어렵습니다. 
합리적인 시간 내에 수렴하도록 만드는 것은 까다로울 수 있습니다. 
이 섹션에서는 심층 네트워크의 수렴을 일관되게 가속화하는 인기 있고 효과적인 기술인 *배치 정규화(batch normalization)*를 설명합니다 :cite:`Ioffe.Szegedy.2015`. 
나중에 :numref:`sec_resnet`에서 다룰 잔차 블록과 함께 배치 정규화는 실무자들이 100개 이상의 레이어를 가진 네트워크를 일상적으로 훈련할 수 있게 만들었습니다. 
배치 정규화의 부차적인(우연한) 이점은 고유한 정규화(regularization)에 있습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## 심층 네트워크 훈련 (Training Deep Networks)

데이터로 작업할 때 우리는 종종 훈련 전에 전처리를 합니다. 
데이터 전처리에 관한 선택은 종종 최종 결과에 엄청난 차이를 만듭니다. 
주택 가격 예측(:numref:`sec_kaggle_house`)에 MLP를 적용했던 것을 상기해 보십시오. 
실제 데이터로 작업할 때 첫 번째 단계는 여러 관찰에 걸쳐 입력 특성을 평균이 0($\boldsymbol{\mu} = 0$)이고 분산이 1($\boldsymbol{\Sigma} = \boldsymbol{1}$)이 되도록 표준화하는 것이었습니다 :cite:`friedman1987exploratory`. 대각선이 1이 되도록, 즉 $\Sigma_{ii} = 1$이 되도록 후자를 재조정하는 경우가 많습니다. 
또 다른 전략은 벡터를 단위 길이로, 가능하면 *관찰당* 평균이 0이 되도록 재조정하는 것입니다. 
이것은 공간 센서 데이터 등에 잘 작동할 수 있습니다. 이러한 전처리 기술과 다른 많은 기술들은 추정 문제를 잘 통제하는 데 유익합니다. 
특성 선택 및 추출에 대한 검토는 예를 들어 :citet:`guyon2008feature`의 기사를 참조하십시오. 
벡터를 표준화하는 것은 그것에 작용하는 함수의 함수 복잡도를 제한하는 좋은 부작용도 있습니다. 예를 들어 서포트 벡터 머신의 유명한 반지름-마진 경계(radius-margin bound) :cite:`Vapnik95`와 퍼셉트론 수렴 정리(Perceptron Convergence Theorem) :cite:`Novikoff62`는 경계가 있는 노름의 입력에 의존합니다.

직관적으로 이 표준화는 파라미터를 *사전적으로* 비슷한 스케일에 놓기 때문에 최적화기와 잘 어울립니다. 
따라서 심층 네트워크 *내부*에 해당하는 정규화 단계가 유익하지 않을까 묻는 것은 자연스러운 일입니다. 이것이 배치 정규화 :cite:`Ioffe.Szegedy.2015`의 발명으로 이어진 추론은 아니지만, 통일된 프레임워크 내에서 그것과 사촌 격인 레이어 정규화 :cite:`Ba.Kiros.Hinton.2016`를 이해하는 유용한 방법입니다.

둘째, 일반적인 MLP나 CNN의 경우 훈련함에 따라 중간 레이어의 변수(예: MLP의 아핀 변환 출력)는 
입력에서 출력까지의 레이어를 따라, 동일한 레이어의 유닛 전체에 걸쳐, 그리고 모델 파라미터 업데이트로 인해 시간이 지남에 따라 크기가 매우 다양할 수 있습니다. 
배치 정규화의 발명가들은 이러한 변수 분포의 이동이 네트워크의 수렴을 방해할 수 있다고 비공식적으로 가정했습니다. 
직관적으로 한 레이어의 변수 활성화가 다른 레이어의 100배라면 학습률을 보상적으로 조정해야 할 수 있다고 추측할 수 있습니다. AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` 또는 Distributed Shampoo :cite:`anil2020scalable`와 같은 적응형 솔버는 최적화 관점에서 이를 해결하려고 합니다(예: 2차 방법의 측면을 추가하여). 
대안은 단순히 적응형 정규화를 통해 문제가 발생하는 것을 방지하는 것입니다.

셋째, 더 깊은 네트워크는 복잡하고 과대적합되기 쉬운 경향이 있습니다. 
이는 정규화(regularization)가 더 중요해진다는 것을 의미합니다. 정규화를 위한 일반적인 기술은 노이즈 주입입니다. 이것은 오랫동안 알려져 왔습니다(예: 입력에 대한 노이즈 주입 :cite:`Bishop.1995`). 또한 :numref:`sec_dropout`의 드롭아웃의 기초를 형성합니다. 아주 우연히도 배치 정규화는 전처리, 수치적 안정성, 정규화라는 세 가지 이점을 모두 전달합니다.

배치 정규화는 개별 레이어 또는 선택적으로 모든 레이어에 적용됩니다: 
각 훈련 반복에서 우리는 먼저 (배치 정규화의) 입력의 평균을 빼고 표준 편차로 나누어 정규화합니다. 
여기서 둘 다 현재 미니배치의 통계를 기반으로 추정됩니다. 
다음으로 손실된 자유도를 복구하기 위해 스케일 계수와 오프셋을 적용합니다. *배치* 통계를 기반으로 한 이 *정규화* 때문에 *배치 정규화*라는 이름이 유래되었습니다.

크기 1의 미니배치로 배치 정규화를 적용하려고 하면 아무것도 배울 수 없다는 점에 유의하십시오. 
평균을 뺀 후 각 은닉 유닛이 값 0을 취하기 때문입니다. 
짐작하시겠지만, 우리가 배치 정규화에 전체 섹션을 할애하고 있으므로 충분히 큰 미니배치에서는 이 접근 방식이 효과적이고 안정적인 것으로 입증되었습니다. 
여기서 얻을 수 있는 한 가지 교훈은 배치 정규화를 적용할 때 배치 크기의 선택이 배치 정규화가 없을 때보다 훨씬 더 중요하다는 것입니다. 또는 적어도 배치 크기를 조정할 때 적절한 보정이 필요합니다.

$\\mathcal{B}$를 미니배치라고 하고 $\\mathbf{x} \\in \\mathcal{B}$를 배치 정규화($\\textrm{BN}$)에 대한 입력이라고 합시다. 이 경우 배치 정규화는 다음과 같이 정의됩니다:

$$\\textrm{BN}(\\mathbf{x}) = \\boldsymbol{\\gamma} \\odot \\frac{\\mathbf{x} - \\hat{\\boldsymbol{\\mu}}_\\mathcal{B}}{\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}} + \\boldsymbol{\\beta}.$$:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm`에서 $\\hat{\\boldsymbol{\\mu}}_\\mathcal{B}$는 표본 평균이고 $\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}$는 미니배치 $\\mathcal{B}$의 표본 표준 편차입니다. 
표준화를 적용한 후 결과 미니배치는 평균 0과 단위 분산을 갖습니다. 
단위 분산(다른 마법의 숫자 대신)의 선택은 임의적입니다. 우리는 $\\mathbf{x}$와 동일한 모양을 가진 요소별 *스케일 파라미터* $\\boldsymbol{\\gamma}$와 *시프트 파라미터* $\\boldsymbol{\\beta}$를 포함하여 이 자유도를 복구합니다. 둘 다 모델 훈련의 일부로 학습해야 하는 파라미터입니다.

중간 레이어의 변수 크기는 훈련 중에 발산할 수 없습니다. 배치 정규화가 주어진 평균과 크기로 적극적으로 중심을 맞추고 재조정하기 때문입니다($\\hat{\\boldsymbol{\\mu}}_\\mathcal{B}$ 및 ${\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}}$를 통해). 
실제 경험에 따르면 특성 재조정에 대해 논의할 때 암시했듯이 배치 정규화는 더 공격적인 학습률을 허용하는 것 같습니다. 
우리는 :eqref:`eq_batchnorm`에서 $\\hat{\\boldsymbol{\\mu}}_\\mathcal{B}$와 ${\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}}$를 다음과 같이 계산합니다:

$$\\hat{\\boldsymbol{\\mu}}_\\mathcal{B} = \\frac{1}{|\\mathcal{B}|} \\sum_{\\mathbf{x} \\in \\mathcal{B}} \\mathbf{x}
\\textrm{ 그리고 }
\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}^2 = \\frac{1}{|\\mathcal{B}|} \\sum_{\\mathbf{x} \\in \\mathcal{B}} (\\mathbf{x} - \\hat{\\boldsymbol{\\mu}}_\\{\\mathcal{B}})^2 + \\epsilon.$$

경험적 분산 추정치가 매우 작거나 사라질 수 있는 경우에도 0으로 나누는 것을 방지하기 위해 분산 추정치에 작은 상수 $\\epsilon > 0$을 더한다는 점에 유의하십시오. 
추정치 $\\hat{\\boldsymbol{\\mu}}_\\mathcal{B}$와 ${\\hat{\\boldsymbol{\\sigma}}_\\mathcal{B}}$는 평균과 분산의 잡음이 있는 추정치를 사용하여 스케일링 문제에 대응합니다. 
이 잡음이 문제라고 생각할 수 있습니다. 
반대로 실제로는 유익합니다.

이는 딥러닝에서 반복되는 주제인 것으로 밝혀졌습니다. 
아직 이론적으로 잘 특성화되지 않은 이유로, 최적화에서 다양한 노이즈 소스는 종종 더 빠른 훈련과 더 적은 과대적합으로 이어집니다: 
이 변동은 일종의 정규화 역할을 하는 것으로 보입니다. 
:citet:`Teye.Azizpour.Smith.2018`와 :citet:`Luo.Wang.Shao.ea.2018`는 배치 정규화의 속성을 각각 베이지안 사전 확률 및 페널티와 관련시켰습니다. 
특히 이것은 배치 정규화가 50-100 범위의 적당한 미니배치 크기에서 가장 잘 작동하는 이유에 대한 수수께끼를 어느 정도 밝혀줍니다. 
이 특정 크기의 미니배치는 $\\hat{\\boldsymbol{\\sigma}}$를 통한 스케일 측면과 $\\hat{\\boldsymbol{\\mu}}$를 통한 오프셋 측면 모두에서 레이어당 "적절한 양"의 노이즈를 주입하는 것으로 보입니다: 더 큰 미니배치는 더 안정적인 추정치로 인해 덜 정규화하는 반면, 아주 작은 미니배치는 높은 분산으로 인해 유용한 신호를 파괴합니다. 이 방향을 더 탐구하여 대안적인 유형의 전처리 및 필터링을 고려하면 다른 효과적인 유형의 정규화로 이어질 수 있습니다.

훈련된 모델을 고정하면 전체 데이터셋을 사용하여 평균과 분산을 추정하는 것을 선호할 것이라고 생각할 수 있습니다. 
훈련이 완료되면 동일한 이미지가 속해 있는 배치에 따라 다르게 분류되는 것을 원치 않기 때문입니다. 
훈련 중에는 모든 데이터 예제에 대한 중간 변수가 모델을 업데이트할 때마다 변경되기 때문에 이러한 정확한 계산은 불가능합니다. 
그러나 모델이 훈련되면 전체 데이터셋을 기반으로 각 레이어 변수의 평균과 분산을 계산할 수 있습니다. 
실제로 이것은 배치 정규화를 사용하는 모델의 표준 관행입니다; 
따라서 배치 정규화 레이어는 *훈련 모드*(미니배치 통계로 정규화)와 *예측 모드*(데이터셋 통계로 정규화)에서 다르게 작동합니다. 
이 형태에서는 노이즈가 훈련 중에만 주입되는 :numref:`sec_dropout`의 드롭아웃 정규화 동작과 매우 유사합니다.


## 배치 정규화 레이어 (Batch Normalization Layers)

완전 연결 레이어와 합성곱 레이어에 대한 배치 정규화 구현은 약간 다릅니다. 
배치 정규화와 다른 레이어 간의 한 가지 주요 차이점은 전자가 한 번에 전체 미니배치에서 작동하기 때문에 다른 레이어를 소개할 때처럼 배치 차원을 무시할 수 없다는 것입니다.

### 완전 연결 레이어 (Fully Connected Layers)

완전 연결 레이어에 배치 정규화를 적용할 때 :citet:`Ioffe.Szegedy.2015`는 원래 논문에서 아핀 변환 후 비선형 활성화 함수 *전*에 배치 정규화를 삽입했습니다. 나중의 응용 프로그램에서는 활성화 함수 바로 *뒤*에 배치 정규화를 삽입하는 실험을 했습니다. 
완전 연결 레이어에 대한 입력을 $\\mathbf{x}$, 아핀 변환을 $\\mathbf{W}\\mathbf{x} + \\mathbf{b}$(가중치 파라미터 $\\mathbf{W}$ 및 편향 파라미터 $\\mathbf{b}$ 포함), 활성화 함수를 $\\phi$로 나타내면, 배치 정규화가 활성화된 완전 연결 레이어 출력 $\\mathbf{h}$의 계산을 다음과 같이 표현할 수 있습니다:

$$\\mathbf{h} = \\phi(\\textrm{BN}(\\mathbf{W}\\mathbf{x} + \\mathbf{b}) ).$$

평균과 분산은 변환이 적용되는 *동일한* 미니배치에서 계산된다는 것을 기억하십시오.

### 합성곱 레이어 (Convolutional Layers)

마찬가지로 합성곱 레이어에서도 합성곱 후 비선형 활성화 함수 전에 배치 정규화를 적용할 수 있습니다. 완전 연결 레이어의 배치 정규화와의 주요 차이점은 *모든 위치에 걸쳐* 채널별로 연산을 적용한다는 것입니다. 이것은 합성곱으로 이어진 평행 이동 불변성 가정과 호환됩니다: 우리는 이미지 내 패턴의 특정 위치가 이해 목적에 중요하지 않다고 가정했습니다.

미니배치에 $m$개의 예제가 있고 각 채널에 대해 합성곱의 출력이 높이 $p$와 너비 $q$를 갖는다고 가정합니다. 
합성곱 레이어의 경우 출력 채널당 $m \\cdot p \\cdot q$ 요소에 대해 동시에 각 배치 정규화를 수행합니다. 
따라서 평균과 분산을 계산할 때 모든 공간 위치에 걸쳐 값을 수집하고, 결과적으로 주어진 채널 내에서 동일한 평균과 분산을 적용하여 각 공간 위치의 값을 정규화합니다. 
각 채널에는 고유한 스케일 및 시프트 파라미터가 있으며 둘 다 스칼라입니다.

### 레이어 정규화 (Layer Normalization)
:label:`subsec_layer-normalization-in-bn`

합성곱의 맥락에서 배치 정규화는 크기 1의 미니배치에 대해서도 잘 정의되어 있다는 점에 유의하십시오: 결국 평균을 낼 이미지 전체의 모든 위치가 있기 때문입니다. 결과적으로 단일 관찰 내일지라도 평균과 분산이 잘 정의됩니다. 
이러한 고려 사항은 :citet:`Ba.Kiros.Hinton.2016`가 *레이어 정규화* 개념을 도입하게 했습니다. 이것은 배치 정규화처럼 작동하지만 한 번에 하나의 관찰에 적용된다는 점만 다릅니다. 결과적으로 오프셋과 스케일링 계수 모두 스칼라입니다. $n$차원 벡터 $\\mathbf{x}$에 대해 레이어 정규화는 다음과 같이 주어집니다.

$$\\mathbf{x} \\rightarrow \\textrm{LN}(\\mathbf{x}) =  \\frac{\\mathbf{x} - \\hat{\\mu}}{\\hat{\\sigma}},$$

여기서 스케일링과 오프셋은 계수별로 적용되며 다음과 같이 주어집니다.

$$\\hat{\\mu} \\stackrel{\\textrm{def}}{=} \\frac{1}{n} \\sum_{i=1}^n x_i \\textrm{ 그리고 }\\n\\hat{\\sigma}^2 \\stackrel{\\textrm{def}}{=} \\frac{1}{n} \\sum_{i=1}^n (x_i - \\hat{\\mu})^2 + \\epsilon.$$

이전과 마찬가지로 0으로 나누는 것을 방지하기 위해 작은 오프셋 $\\epsilon > 0$을 더합니다. 레이어 정규화를 사용하는 주요 이점 중 하나는 발산을 방지한다는 것입니다. 결국 $\\epsilon$을 무시하면 레이어 정규화의 출력은 스케일에 독립적입니다. 즉, $\\alpha \\neq 0$인 어떤 선택에 대해서도 $\\textrm{LN}(\\mathbf{x}) \\approx \\textrm{LN}(\\alpha \\mathbf{x})$를 갖습니다. 이것은 $|\\alpha| \\to \\infty$일 때 등식이 됩니다(근사 등식은 분산에 대한 오프셋 $\\epsilon$ 때문입니다). 

레이어 정규화의 또 다른 장점은 미니배치 크기에 의존하지 않는다는 것입니다. 또한 훈련 중인지 테스트 중인지 여부와도 무관합니다. 즉, 단순히 활성화를 주어진 스케일로 표준화하는 결정론적 변환입니다. 이는 최적화에서 발산을 방지하는 데 매우 유익할 수 있습니다. 자세한 내용은 생략하고 관심 있는 독자는 원본 논문을 참조할 것을 권장합니다.

### 예측 중 배치 정규화 (Batch Normalization During Prediction)

앞서 언급했듯이 배치 정규화는 일반적으로 훈련 모드와 예측 모드에서 다르게 작동합니다. 
첫째, 미니배치에서 각각을 추정하는 데서 발생하는 표본 평균과 표본 분산의 노이즈는 모델을 훈련한 후에는 더 이상 바람직하지 않습니다. 
둘째, 우리는 배치별 정규화 통계를 계산할 여유가 없을 수 있습니다. 
예를 들어 한 번에 하나의 예측을 수행하기 위해 모델을 적용해야 할 수도 있습니다.

일반적으로 훈련 후에는 전체 데이터셋을 사용하여 변수 통계의 안정적인 추정치를 계산한 다음 예측 시에 고정합니다. 
따라서 배치 정규화는 훈련 중과 테스트 시에 다르게 동작합니다. 
드롭아웃도 이 특성을 보인다는 것을 기억하십시오.


## (**밑바닥부터 구현하기 (Implementation from Scratch)**)

배치 정규화가 실제로 어떻게 작동하는지 보기 위해 아래에서 밑바닥부터 구현합니다.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # autograd를 사용하여 훈련 모드인지 확인
    if not autograd.is_training():
        # 예측 모드에서는 이동 평균으로 얻은 평균과 분산 사용
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 완전 연결 레이어를 사용할 때 특성 차원에서 평균과 분산 계산
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 2차원 합성곱 레이어를 사용할 때 채널 차원(axis=1)에서 평균과 분산 계산.
            # 나중에 브로드캐스팅 연산을 수행할 수 있도록 X의 모양을 유지해야 합니다
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 훈련 모드에서는 현재 평균과 분산 사용
        X_hat = (X - mean) / np.sqrt(var + eps)
        # 이동 평균을 사용하여 평균과 분산 업데이트
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # 스케일 및 시프트
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # is_grad_enabled를 사용하여 훈련 모드인지 확인
    if not torch.is_grad_enabled():
        # 예측 모드에서는 이동 평균으로 얻은 평균과 분산 사용
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 완전 연결 레이어를 사용할 때 특성 차원에서 평균과 분산 계산
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 2차원 합성곱 레이어를 사용할 때 채널 차원(axis=1)에서 평균과 분산 계산.
            # 나중에 브로드캐스팅 연산을 수행할 수 있도록 X의 모양을 유지해야 합니다
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 훈련 모드에서는 현재 평균과 분산 사용
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 이동 평균을 사용하여 평균과 분산 업데이트
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # 스케일 및 시프트
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # 이동 분산의 제곱근 역수를 요소별로 계산
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 스케일 및 시프트
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # `deterministic`을 사용하여 현재 모드가 훈련 모드인지 예측 모드인지 확인
    if deterministic:
        # 예측 모드에서는 이동 평균으로 얻은 평균과 분산 사용
        # `linen.Module.variables`는 배열을 포함하는 `value` 속성을 가집니다
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 완전 연결 레이어를 사용할 때 특성 차원에서 평균과 분산 계산
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 2차원 합성곱 레이어를 사용할 때 채널 차원(axis=1)에서 평균과 분산 계산.
            # 나중에 브로드캐스팅 연산을 수행할 수 있도록 `X`의 모양을 유지해야 합니다
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 훈련 모드에서는 현재 평균과 분산 사용
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # 이동 평균을 사용하여 평균과 분산 업데이트
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 스케일 및 시프트
    return Y
```

이제 [**적절한 `BatchNorm` 레이어를 생성**]할 수 있습니다. 
우리 레이어는 스케일 `gamma`와 시프트 `beta`에 대한 적절한 파라미터를 유지하며, 둘 다 훈련 과정에서 업데이트됩니다. 
또한 우리 레이어는 이후 모델 예측 시 사용하기 위해 평균과 분산의 이동 평균을 유지합니다.

알고리즘 세부 사항은 제쳐두고 우리 레이어 구현의 기저에 있는 디자인 패턴에 주목하십시오. 
일반적으로 우리는 `batch_norm`과 같은 별도의 함수에 수학을 정의합니다. 
그런 다음 이 기능을 사용자 정의 레이어에 통합합니다. 이 레이어의 코드는 주로 올바른 장치 컨텍스트로 데이터 이동, 필요한 변수 할당 및 초기화, 이동 평균 추적(여기서는 평균 및 분산) 등과 같은 부기(bookkeeping) 문제를 처리합니다. 
이 패턴은 수학과 상용구 코드의 깔끔한 분리를 가능하게 합니다. 
또한 편의를 위해 여기서는 입력 모양을 자동으로 추론하는 것에 대해 걱정하지 않았습니다. 
따라서 전체적으로 특성 수를 지정해야 합니다. 
지금까지 모든 최신 딥러닝 프레임워크는 고수준 배치 정규화 API에서 크기 및 모양의 자동 감지를 제공합니다(실제로는 대신 이것을 사용할 것입니다).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: 완전 연결 레이어의 출력 수 또는 합성곱 레이어의 출력 채널 수.
    # `num_dims`: 완전 연결 레이어의 경우 2, 합성곱 레이어의 경우 4
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 스케일 파라미터와 시프트 파라미터(모델 파라미터)는 각각 1과 0으로 초기화됩니다
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 모델 파라미터가 아닌 변수는 0과 1로 초기화됩니다
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # `X`가 메인 메모리에 없으면 `moving_mean`과 `moving_var`를 `X`가 있는 장치로 복사합니다
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # 업데이트된 `moving_mean`과 `moving_var` 저장
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: 완전 연결 레이어의 출력 수 또는 합성곱 레이어의 출력 채널 수.
    # num_dims: 완전 연결 레이어의 경우 2, 합성곱 레이어의 경우 4
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 스케일 파라미터와 시프트 파라미터(모델 파라미터)는 각각 1과 0으로 초기화됩니다
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 모델 파라미터가 아닌 변수는 0과 1로 초기화됩니다
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # X가 메인 메모리에 없으면 moving_mean과 moving_var를 X가 있는 장치로 복사합니다
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 업데이트된 moving_mean과 moving_var 저장
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # 스케일 파라미터와 시프트 파라미터(모델 파라미터)는 각각 1과 0으로 초기화됩니다
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # 모델 파라미터가 아닌 변수는 0으로 초기화됩니다
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: 완전 연결 레이어의 출력 수 또는 합성곱 레이어의 출력 채널 수.
    # `num_dims`: 완전 연결 레이어의 경우 2, 합성곱 레이어의 경우 4
    # `deterministic`을 사용하여 현재 모드가 훈련 모드인지 예측 모드인지 결정
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # 스케일 파라미터와 시프트 파라미터(모델 파라미터)는 각각 1과 0으로 초기화됩니다
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # 모델 파라미터가 아닌 변수는 0과 1로 초기화됩니다. 'batch_stats' 컬렉션에 저장합니다
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

과거 평균과 분산 추정치에 대한 집계를 제어하기 위해 `momentum`을 사용했습니다. 이는 최적화의 *모멘텀* 항과 전혀 관련이 없으므로 약간 잘못된 이름입니다. 그럼에도 불구하고 이 항에 대해 일반적으로 채택된 이름이며 API 명명 관례에 따라 코드에서 동일한 변수 이름을 사용합니다.

## [**배치 정규화가 있는 LeNet**]

문맥에서 `BatchNorm`을 적용하는 방법을 보기 위해, 아래에서는 이를 기존 LeNet 모델(:numref:`sec_lenet`)에 적용합니다. 
배치 정규화는 합성곱 레이어 또는 완전 연결 레이어 다음, 해당 활성화 함수 이전에 적용됨을 기억하십시오.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
`BatchNorm` 레이어는 배치 통계(평균 및 분산)를 계산해야 하므로, Flax는 `batch_stats` 딕셔너리를 추적하고 모든 미니배치마다 업데이트합니다. `batch_stats`와 같은 컬렉션은 `TrainState` 객체(:numref:`oo-design-training`에 정의된 `d2l.Trainer` 클래스에 있음)에 속성으로 저장될 수 있으며 모델의 순전파 동안 `mutable` 인수에 전달되어야 Flax가 변경된 변수를 반환합니다.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

이전과 마찬가지로 [**Fashion-MNIST 데이터셋에서 네트워크를 훈련**]합니다. 
이 코드는 LeNet을 처음 훈련했을 때와 거의 동일합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

첫 번째 배치 정규화 레이어에서 학습된 [**스케일 파라미터 `gamma`와 시프트 파라미터 `beta`를 살펴봅시다**].

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**간결한 구현**]

방금 직접 정의한 `BatchNorm` 클래스와 비교하여, 딥러닝 프레임워크의 고수준 API에 정의된 `BatchNorm` 클래스를 직접 사용할 수 있습니다. 
코드는 위의 구현과 거의 동일해 보이지만 차원을 올바르게 맞추기 위해 추가 인수를 제공할 필요가 없다는 점이 다릅니다.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

아래에서는 [**동일한 하이퍼파라미터를 사용하여 모델을 훈련**]합니다. 
평소와 같이 고수준 API 변형이 훨씬 더 빠르게 실행된다는 점에 유의하십시오. 
코드가 C++ 또는 CUDA로 컴파일된 반면 사용자 정의 구현은 Python으로 해석되어야 하기 때문입니다.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## 토론 (Discussion)

직관적으로 배치 정규화는 최적화 지형을 더 매끄럽게 만든다고 생각됩니다. 
하지만 심층 모델을 훈련할 때 관찰하는 현상에 대한 투기적 직관과 실제 설명을 구별하도록 주의해야 합니다. 
우리는 더 단순한 심층 신경망(MLP 및 기존 CNN)이 애초에 왜 잘 일반화되는지조차 모른다는 것을 상기하십시오. 
드롭아웃과 가중치 감쇠를 사용하더라도, 보지 못한 데이터로 일반화할 수 있는 능력이 너무 유연해서 훨씬 더 정제된 학습 이론적 일반화 보장이 필요할 가능성이 높습니다.

배치 정규화를 제안한 원본 논문 :cite:`Ioffe.Szegedy.2015`은 강력하고 유용한 도구를 도입했을 뿐만 아니라 *내부 공변량 이동(internal covariate shift)*을 줄임으로써 작동한다고 설명했습니다. 
아마도 *내부 공변량 이동*이란 훈련 과정에서 변수 값의 분포가 변경된다는 개념과 같은 직관을 의미했을 것입니다. 
그러나 이 설명에는 두 가지 문제가 있었습니다: 
i) 이 드리프트는 *공변량 이동*과 매우 다르므로 이름이 잘못되었습니다. 굳이 말하자면 개념 드리프트(concept drift)에 더 가깝습니다. 
ii) 설명은 명시되지 않은 직관을 제공하지만 *이 기술이 정확히 왜 작동하는지*에 대한 질문은 엄격한 설명을 기다리는 열린 문제로 남겨 둡니다. 
이 책 전반에 걸쳐 우리는 실무자들이 심층 신경망 개발을 안내하는 데 사용하는 직관을 전달하는 것을 목표로 합니다. 
그러나 우리는 이러한 지침이 되는 직관을 확립된 과학적 사실과 분리하는 것이 중요하다고 믿습니다. 
결국 이 자료를 마스터하고 자신의 연구 논문을 작성하기 시작할 때 기술적 주장과 예감을 명확히 구분하고 싶을 것입니다.

배치 정규화의 성공에 이어, *내부 공변량 이동* 측면에서의 설명은 기술 문헌의 토론과 머신러닝 연구를 발표하는 방법에 대한 광범위한 담론에서 반복적으로 표면화되었습니다. 
2017 NeurIPS 컨퍼런스에서 Test of Time Award를 수상하면서 한 기억에 남는 연설에서 Ali Rahimi는 현대 딥러닝 관행을 연금술에 비유하는 주장의 초점으로 *내부 공변량 이동*을 사용했습니다. 
그 후 이 예제는 머신러닝의 골치 아픈 추세를 개괄하는 입장 논문에서 자세히 재검토되었습니다 :cite:`Lipton.Steinhardt.2018`. 
다른 저자들은 배치 정규화의 성공에 대한 대안적 설명을 제안했으며, 일부 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`는 배치 정규화의 성공이 원본 논문에서 주장한 것과 어떤 면에서는 반대되는 행동을 보임에도 불구하고 온다고 주장했습니다.


우리는 *내부 공변량 이동*이 매년 기술 머신러닝 문헌에서 만들어지는 수천 개의 유사하게 모호한 주장보다 더 비판받을 가치가 없다는 점에 주목합니다. 
이 논쟁의 초점으로서의 공명은 대상 청중에게 널리 인식되기 때문일 것입니다. 
배치 정규화는 거의 모든 배포된 이미지 분류기에 적용되는 필수 불가결한 방법임이 입증되었으며, 이 기술을 소개한 논문은 수만 건의 인용을 얻었습니다. 하지만 노이즈 주입을 통한 정규화, 재조정을 통한 가속화, 마지막으로 전처리라는 기본 원칙이 미래에 레이어와 기술의 추가 발명으로 이어질 수 있다고 추측합니다.

좀 더 실용적인 측면에서 배치 정규화에 대해 기억해야 할 몇 가지 사항이 있습니다:

* 모델 훈련 중 배치 정규화는 미니배치의 평균과 표준 편차를 활용하여 네트워크의 중간 출력을 지속적으로 조정하므로 신경망 전체의 각 레이어에서 중간 출력 값이 더 안정적입니다.
* 배치 정규화는 완전 연결 레이어와 합성곱 레이어에 대해 약간 다릅니다. 사실 합성곱 레이어의 경우 레이어 정규화가 때때로 대안으로 사용될 수 있습니다.
* 드롭아웃 레이어와 마찬가지로 배치 정규화 레이어는 훈련 모드와 예측 모드에서 다르게 동작합니다.
* 배치 정규화는 정규화 및 최적화 수렴 개선에 유용합니다. 대조적으로 내부 공변량 이동을 줄인다는 원래 동기는 타당한 설명이 아닌 것으로 보입니다.
* 입력 섭동에 덜 민감한 더 강력한 모델을 위해 배치 정규화를 제거하는 것을 고려하십시오 :cite:`wang2022removing`.

## 연습 문제 (Exercises)

1. 배치 정규화 전에 완전 연결 레이어 또는 합성곱 레이어에서 편향 파라미터를 제거해야 합니까? 그 이유는 무엇입니까?
2. 배치 정규화가 있는 경우와 없는 경우 LeNet의 학습률을 비교하십시오.
    1. 검증 정확도 증가를 플롯하십시오.
    2. 두 경우 모두 최적화가 실패하기 전까지 학습률을 얼마나 크게 만들 수 있습니까?
3. 모든 레이어에 배치 정규화가 필요합니까? 실험해 보십시오.
4. 평균만 제거하거나 대안으로 분산만 제거하는 배치 정규화의 "라이트" 버전을 구현하십시오. 어떻게 동작합니까?
5. 파라미터 `beta`와 `gamma`를 고정하십시오. 결과를 관찰하고 분석하십시오.
6. 드롭아웃을 배치 정규화로 대체할 수 있습니까? 동작이 어떻게 변경됩니까?
7. 연구 아이디어: 적용할 수 있는 다른 정규화 변환을 생각해 보십시오:
    1. 확률 적분 변환(probability integral transform)을 적용할 수 있습니까?
    2. 전체 순위 공분산 추정치를 사용할 수 있습니까? 아마도 그렇게 하지 않아야 하는 이유는 무엇입니까? 
    3. 다른 압축 행렬 변형(블록 대각선, 저변위 순위, Monarch 등)을 사용할 수 있습니까?
    4. 희소화 압축이 정규화기 역할을 합니까? 
    5. 사용할 수 있는 다른 투영(예: 볼록 원뿔, 대칭 그룹별 변환)이 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18005)
:end_tab: