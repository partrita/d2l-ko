```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다층 퍼셉트론 (Multilayer Perceptrons)
:label:`sec_mlp`

:numref:`sec_softmax`에서 우리는 소프트맥스 회귀를 소개하고, 알고리즘을 밑바닥부터 구현(:numref:`sec_softmax_scratch`)하거나 고수준 API를 사용(:numref:`sec_softmax_concise`)하여 구현했습니다. 이를 통해 저해상도 이미지에서 10가지 의류 카테고리를 인식할 수 있는 분류기를 훈련했습니다. 
그 과정에서 데이터를 다루고, 출력을 유효한 확률 분포로 강제 변환하고, 적절한 손실 함수를 적용하고, 모델의 파라미터에 대해 이를 최소화하는 방법을 배웠습니다. 
이제 단순한 선형 모델의 맥락에서 이러한 메커니즘을 마스터했으므로, 이 책의 주된 관심사이며 비교적 풍부한 모델 클래스인 심층 신경망에 대한 탐구를 시작할 수 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## 은닉층 (Hidden Layers)

우리는 :numref:`subsec_linear_model`에서 아핀 변환을 편향이 추가된 선형 변환으로 설명했습니다. 
시작하기 위해, :numref:`fig_softmaxreg`에 설명된 소프트맥스 회귀 예제에 해당하는 모델 아키텍처를 상기해 보십시오. 
이 모델은 단일 아핀 변환과 그 뒤를 잇는 소프트맥스 연산을 통해 입력을 출력에 직접 매핑합니다. 
만약 우리의 레이블이 정말로 단순한 아핀 변환에 의해 입력 데이터와 관련이 있다면 이 접근 방식만으로 충분할 것입니다. 
그러나 (아핀 변환에서의) 선형성은 *강력한* 가정입니다.

### 선형 모델의 한계 (Limitations of Linear Models)

예를 들어, 선형성은 *단조성(monotonicity)*이라는 *더 약한* 가정을 내포합니다. 즉, 특성의 증가가 항상 모델 출력의 증가를 유발하거나(해당 가중치가 양수인 경우), 항상 모델 출력의 감소를 유발해야 합니다(해당 가중치가 음수인 경우). 
때로는 이것이 타당합니다. 
예를 들어, 개인이 대출을 갚을지 예측하려고 할 때, 다른 모든 조건이 동일하다면 소득이 높은 신청자가 낮은 신청자보다 항상 갚을 가능성이 더 높다고 합리적으로 가정할 수 있습니다. 
단조적이기는 하지만, 이 관계가 상환 확률과 선형적으로 관련이 있을 가능성은 낮습니다. 소득이 0달러에서 5만 달러로 증가하는 것은 100만 달러에서 105만 달러로 증가하는 것보다 상환 가능성 증가에 더 큰 영향을 미칠 가능성이 높습니다. 
이를 처리하는 한 가지 방법은 로지스틱 맵(따라서 결과 확률의 로그)을 사용하여 선형성이 더 그럴듯해지도록 결과를 후처리하는 것일 수 있습니다.

단조성을 위반하는 예제를 쉽게 생각해 낼 수 있다는 점에 유의하십시오. 
예를 들어 체온의 함수로 건강을 예측하고 싶다고 가정해 봅시다. 
체온이 37°C(98.6°F) 이상인 정상 체온을 가진 개인의 경우, 체온이 높을수록 위험이 큽니다. 
그러나 체온이 37°C 아래로 떨어지면 체온이 낮을수록 위험이 큽니다! 
다시 말하지만, 37°C로부터의 거리를 특성으로 사용하는 것과 같은 기발한 전처리를 통해 문제를 해결할 수도 있습니다.


하지만 고양이와 개 이미지를 분류하는 것은 어떨까요? 
위치 (13, 17)에 있는 픽셀의 강도를 높이는 것이 이미지가 개를 묘사할 가능성을 항상 증가(또는 항상 감소)시켜야 할까요? 
선형 모델에 의존하는 것은 고양이와 개를 구별하는 유일한 요구 사항이 개별 픽셀의 밝기를 평가하는 것이라는 암묵적인 가정에 해당합니다. 
이 접근 방식은 이미지를 반전시켜도 범주가 보존되는 세상에서는 실패할 운명입니다.

그럼에도 불구하고 이전 예제들과 비교할 때 여기에서의 선형성이 명백히 터무니없음에도 불구하고, 간단한 전처리 수정으로 문제를 해결할 수 있을지는 덜 명확합니다. 
그 이유는 픽셀의 중요성이 맥락(주변 픽셀의 값)에 복잡한 방식으로 의존하기 때문입니다. 
데이터에 대한 표현이 존재하여 특성 간의 관련 상호 작용을 고려하고 그 위에 선형 모델이 적합할 수도 있겠지만, 우리는 단순히 그것을 손으로 계산하는 방법을 모릅니다. 
심층 신경망을 사용하여, 우리는 관찰 데이터를 통해 은닉층을 통한 표현과 그 표현에 작용하는 선형 예측기를 공동으로 학습합니다.

이 비선형성 문제는 적어도 한 세기 동안 연구되어 왔습니다 :cite:`Fisher.1928`. 예를 들어, 가장 기본적인 형태의 결정 트리는 클래스 멤버십을 결정하기 위해 일련의 이진 결정을 사용합니다 :cite:`quinlan2014c4`. 마찬가지로 커널 방법은 수십 년 동안 비선형 의존성을 모델링하는 데 사용되었습니다 :cite:`Aronszajn.1950`. 이것은 비모수 스플라인 모델 :cite:`Wahba.1990`과 커널 방법 :cite:`Scholkopf.Smola.2002`으로 이어졌습니다. 이것은 또한 뇌가 아주 자연스럽게 해결하는 문제이기도 합니다. 결국 뉴런은 다른 뉴런으로 공급되고, 이는 다시 다른 뉴런으로 공급됩니다 :cite:`Cajal.Azoulay.1894`. 
결과적으로 우리는 상대적으로 단순한 변환의 시퀀스를 갖게 됩니다.

### 은닉층 통합하기 (Incorporating Hidden Layers)

우리는 하나 이상의 은닉층을 통합하여 선형 모델의 한계를 극복할 수 있습니다. 
가장 쉬운 방법은 많은 완전 연결 레이어를 서로 쌓는 것입니다. 
각 레이어는 출력을 생성할 때까지 그 위의 레이어로 공급됩니다. 
처음 $L-1$개의 레이어를 표현으로 생각하고 마지막 레이어를 선형 예측기로 생각할 수 있습니다. 
이 아키텍처는 일반적으로 *다층 퍼셉트론(multilayer perceptron)*이라고 불리며 종종 *MLP*로 축약됩니다 (:numref:`fig_mlp`).

![5개의 은닉 유닛이 있는 은닉층을 가진 MLP.](../img/mlp.svg)
:label:`fig_mlp`

이 MLP는 4개의 입력, 3개의 출력을 가지며, 은닉층에는 5개의 은닉 유닛이 있습니다. 
입력 레이어는 어떠한 계산도 포함하지 않으므로, 이 네트워크로 출력을 생성하려면 은닉층과 출력 레이어 모두에 대한 계산을 구현해야 합니다. 따라서 이 MLP의 레이어 수는 2개입니다. 
두 레이어 모두 완전 연결되어 있다는 점에 유의하십시오. 
모든 입력은 은닉층의 모든 뉴런에 영향을 미치고, 이들 각각은 다시 출력 레이어의 모든 뉴런에 영향을 미칩니다. 아쉽게도 아직 끝난 것은 아닙니다.

### 선형에서 비선형으로 (From Linear to Nonlinear)

이전과 마찬가지로, 행렬 $\mathbf{X} \in \mathbb{R}^{n \times d}$를 각 예제가 $d$개의 입력(특성)을 가진 $n$개 예제의 미니배치라고 표시합니다. 
$h$개의 은닉 유닛을 가진 은닉층이 있는 1-은닉층 MLP의 경우, $\mathbf{H} \in \mathbb{R}^{n \times h}$를 은닉층의 출력으로 표시하며, 이는 *은닉 표현(hidden representations)*입니다. 
은닉층과 출력 레이어 모두 완전 연결되어 있으므로, 은닉층 가중치 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$와 편향 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$, 그리고 출력층 가중치 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$와 편향 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$를 갖습니다. 
이를 통해 1-은닉층 MLP의 출력 $\mathbf{O} \in \mathbb{R}^{n \times q}$를 다음과 같이 계산할 수 있습니다:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\
\end{aligned}
$$ 

은닉층을 추가한 후, 이제 우리 모델은 추가적인 파라미터 세트를 추적하고 업데이트해야 합니다. 
그렇다면 그 대가로 무엇을 얻었을까요? 
위에서 정의된 모델에서는 *수고에 대해 아무것도 얻지 못했다*는 사실을 알게 되어 놀랄 수도 있습니다! 
이유는 명백합니다. 
위의 은닉 유닛은 입력의 아핀 함수로 주어지고, 출력(소프트맥스 전)은 단지 은닉 유닛의 아핀 함수일 뿐입니다. 
아핀 함수의 아핀 함수는 그 자체로 아핀 함수입니다. 
게다가 우리 선형 모델은 이미 모든 아핀 함수를 표현할 수 있었습니다.

이를 공식적으로 확인하기 위해 위 정의에서 은닉층을 붕괴시켜 파라미터 $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$와 $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$를 가진 동등한 단일 레이어 모델을 얻을 수 있습니다:

$$ 
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}. 
$$ 

다층 아키텍처의 잠재력을 실현하기 위해, 우리는 한 가지 핵심 요소가 더 필요합니다: 아핀 변환 후 각 은닉 유닛에 적용될 비선형 *활성화 함수(activation function)* $\sigma$입니다. 예를 들어 인기 있는 선택은 인수에 요소별로 작동하는 ReLU(rectified linear unit) 활성화 함수 :cite:`Nair.Hinton.2010` $\sigma(x) = \mathrm{max}(0, x)$입니다. 
활성화 함수 $\sigma(\cdot)$의 출력을 *활성화(activations)*라고 합니다. 
일반적으로 활성화 함수가 있으면 더 이상 MLP를 선형 모델로 붕괴시킬 수 없습니다:

$$ 
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\
\end{aligned}
$$ 

$\\mathbf{X}$의 각 행은 미니배치의 예제에 해당하므로, 약간의 표기법 남용과 함께 비선형성 $\sigma$가 입력에 행별 방식으로, 즉 한 번에 한 예제씩 적용되도록 정의합니다. 
:numref:`subsec_softmax_vectorization`에서 행별 연산을 나타낼 때 소프트맥스에 대해 동일한 표기법을 사용했다는 점에 유의하십시오. 
우리가 사용하는 활성화 함수는 단순히 행별이 아니라 요소별로 적용되는 경우가 꽤 많습니다. 이는 레이어의 선형 부분을 계산한 후, 다른 은닉 유닛이 취한 값을 보지 않고 각 활성화를 계산할 수 있음을 의미합니다.

더 일반적인 MLP를 구축하기 위해, 우리는 이러한 은닉층을 계속 쌓을 수 있습니다. 
예를 들어 $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ 및 $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$와 같이 차곡차곡 쌓아 훨씬 더 표현력이 풍부한 모델을 만들 수 있습니다.

### 보편적 근사자 (Universal Approximators)

우리는 뇌가 매우 정교한 통계 분석을 할 수 있다는 것을 알고 있습니다. 따라서 심층 네트워크가 *얼마나 강력할 수 있는지* 물어볼 가치가 있습니다. 이 질문은 MLP의 맥락에서 :citet:`Cybenko.1989`에 의해, 그리고 단일 은닉층을 가진 방사형 기저 함수(RBF) 네트워크로 볼 수 있는 방식으로 재생 커널 힐베르트 공간의 맥락에서 :citet:`micchelli1984interpolation`에 의해 여러 번 대답되었습니다. 
이러한 (및 관련 결과)는 단일 은닉층 네트워크라도 충분한 노드(어쩌면 터무니없이 많은)와 올바른 가중치 세트가 주어지면 어떤 함수든 모델링할 수 있음을 시사합니다. 
하지만 실제로 그 함수를 학습하는 것은 어려운 부분입니다. 
신경망을 C 프로그래밍 언어와 비슷하다고 생각할 수 있습니다. 
다른 모든 현대 언어와 마찬가지로 이 언어는 모든 계산 가능한 프로그램을 표현할 수 있습니다. 
하지만 사양을 충족하는 프로그램을 실제로 생각해 내는 것이 어려운 부분입니다.

게다가 단일 은닉층 네트워크가 어떤 함수든 학습할 *수* 있다고 해서 모든 문제를 하나로 해결하려고 해서는 안 됩니다. 사실 이 경우 커널 방법은 무한 차원 공간에서도 *정확하게* 문제를 해결할 수 있으므로 훨씬 더 효과적입니다 :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`. 
실제로 우리는 (넓은 것보다는) 더 깊은 네트워크를 사용하여 많은 함수를 훨씬 더 간결하게 근사할 수 있습니다 :cite:`Simonyan.Zisserman.2014`. 
우리는 후속 장에서 더 엄격한 주장을 다룰 것입니다.


## 활성화 함수 (Activation Functions)
:label:`subsec_activation-functions`

활성화 함수는 가중 합을 계산하고 여기에 편향을 더함으로써 뉴런이 활성화되어야 하는지 여부를 결정합니다. 
이들은 입력 신호를 출력으로 변환하기 위한 미분 가능한 연산자이며, 그들 대부분은 비선형성을 추가합니다. 
활성화 함수는 딥러닝의 기본이므로, (**몇 가지 일반적인 함수를 간략하게 살펴봅시다**).

### ReLU 함수

구현의 단순성과 다양한 예측 작업에서의 우수한 성능으로 인해 가장 인기 있는 선택은 *ReLU(rectified linear unit)*입니다 :cite:`Nair.Hinton.2010`. 
[**ReLU는 매우 간단한 비선형 변환을 제공합니다**]. 
요소 $x$가 주어지면 함수는 해당 요소와 $0$ 중 최댓값으로 정의됩니다:

$$\operatorname{ReLU}(x) = \max(x, 0).$$ 

비공식적으로 ReLU 함수는 양의 요소만 유지하고 해당 활성화를 0으로 설정하여 모든 음의 요소를 버립니다. 
직관을 얻기 위해 함수를 플롯할 수 있습니다. 
보시다시피 활성화 함수는 부분적으로 선형입니다.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

입력이 음수일 때 ReLU 함수의 도함수는 0이고, 입력이 양수일 때 ReLU 함수의 도함수는 1입니다. 
입력이 정확히 0인 값을 취할 때 ReLU 함수는 미분 불가능하다는 점에 유의하십시오. 
이러한 경우, 우리는 기본적으로 좌측 도함수를 사용하여 입력이 0일 때 도함수가 0이라고 말합니다. 
입력이 실제로 0이 되는 경우는 없을 수 있기 때문에(수학자들은 측도가 0인 집합에서 미분 불가능하다고 말할 것입니다) 우리는 이를 넘어갈 수 있습니다. 
미묘한 경계 조건이 중요하다면 우리는 아마도 공학이 아니라 (*진짜*) 수학을 하고 있다는 옛 격언이 있습니다. 
그 통념이 여기에 적용될 수 있거나, 적어도 우리가 제약 최적화 :cite:`Mangasarian.1965,Rockafellar.1970`를 수행하지 않는다는 사실이 적용될 수 있습니다. 
아래에 ReLU 함수의 도함수를 플롯합니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

ReLU를 사용하는 이유는 도함수가 특히 잘 작동하기 때문입니다: 사라지거나 인수를 그대로 통과시킵니다. 
이는 최적화가 더 잘 작동하게 만들고 신경망의 이전 버전을 괴롭혔던 잘 문서화된 기울기 소실(vanishing gradients) 문제를 완화했습니다(나중에 자세히 설명).

*파라미터화된 ReLU* (*pReLU*) 함수 :cite:`He.Zhang.Ren.ea.2015`를 포함하여 ReLU 함수에는 많은 변형이 있습니다. 
이 변형은 ReLU에 선형 항을 추가하여 인수가 음수일 때도 일부 정보가 여전히 통과되도록 합니다:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$ 

### 시그모이드 함수 (Sigmoid Function)

[** *시그모이드 함수(sigmoid function)*는 값이 도메인 $\mathbb{R}$에 있는 입력을**] (**구간 (0, 1)에 있는 출력으로 변환합니다.**) 
그렇기 때문에 시그모이드는 종종 *스쿼싱 함수(squashing function)*라고 불립니다: (-inf, inf) 범위의 모든 입력을 (0, 1) 범위의 어떤 값으로 찌그러뜨립니다:

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$ 

초기 신경망에서 과학자들은 *발화하거나* *발화하지 않는* 생물학적 뉴런을 모델링하는 데 관심이 있었습니다. 
따라서 인공 뉴런의 발명가인 맥컬록과 피츠로 거슬러 올라가는 이 분야의 개척자들은 임계값 유닛에 집중했습니다 :cite:`McCulloch.Pitts.1943`. 
임계값 활성화는 입력이 어떤 임계값보다 낮을 때 0 값을 갖고 입력이 임계값을 초과할 때 1 값을 갖습니다.

관심이 경사 기반 학습으로 옮겨갔을 때, 시그모이드 함수는 임계값 유닛에 대한 부드럽고 미분 가능한 근사치이기 때문에 자연스러운 선택이었습니다. 
시그모이드는 이진 분류 문제에 대해 출력을 확률로 해석하고 싶을 때 출력 유닛의 활성화 함수로 여전히 널리 사용됩니다: 시그모이드를 소프트맥스의 특수한 경우로 생각할 수 있습니다. 
그러나 시그모이드는 은닉층의 대부분의 용도에서 더 간단하고 훈련하기 쉬운 ReLU로 대체되었습니다. 
이는 시그모이드가 큰 양수 *및* 음수 인수에 대해 기울기가 사라지기 때문에 최적화에 어려움을 준다는 사실과 관련이 있습니다 :cite:`LeCun.Bottou.Orr.ea.1998`. 
이로 인해 탈출하기 어려운 고원(plateaus)이 생길 수 있습니다. 
그럼에도 불구하고 시그모이드는 중요합니다. 순환 신경망에 대한 나중 장(예: :numref:`sec_lstm`)에서 우리는 시그모이드 유닛을 활용하여 시간에 따른 정보의 흐름을 제어하는 아키텍처를 설명할 것입니다.

아래에 시그모이드 함수를 플롯합니다. 
입력이 0에 가까울 때 시그모이드 함수는 선형 변환에 접근한다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

시그모이드 함수의 도함수는 다음 방정식으로 주어집니다:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\(1-\\operatorname{sigmoid}(x)\).$$ 

시그모이드 함수의 도함수는 아래에 플롯되어 있습니다. 
입력이 0일 때 시그모이드 함수의 도함수는 최대 0.25에 도달합니다. 
입력이 어느 방향으로든 0에서 멀어질수록 도함수는 0에 접근합니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# 이전 기울기 지우기
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Tanh 함수 (Tanh Function)
:label:`subsec_tanh`

시그모이드 함수처럼, [**tanh(하이퍼볼릭 탄젠트) 함수도 입력을 찌그러뜨려**] (**$-1$과 $1$ 사이의**) 구간의 요소로 변환합니다:

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$ 

아래에 tanh 함수를 플롯합니다. 입력이 0에 가까워지면 tanh 함수는 선형 변환에 접근합니다. 함수의 모양은 시그모이드 함수의 모양과 유사하지만, tanh 함수는 좌표계 원점에 대해 점 대칭을 보입니다 :cite:`Kalman.Kwasny.1992`.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh 함수의 도함수는 다음과 같습니다:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$ 

아래에 플롯되어 있습니다. 
입력이 0에 가까워지면 tanh 함수의 도함수는 최대 1에 접근합니다. 
그리고 시그모이드 함수에서 보았듯이 입력이 어느 방향으로든 0에서 멀어지면 tanh 함수의 도함수는 0에 접근합니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# 이전 기울기 지우기
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## 요약 및 토론 (Summary and Discussion)

우리는 이제 비선형성을 통합하여 표현력 있는 다층 신경망 아키텍처를 구축하는 방법을 알게 되었습니다. 
참고로 여러분의 지식은 이미 여러분을 1990년경의 실무자와 비슷한 도구 세트를 다룰 수 있는 위치에 올려놓았습니다. 
어떤 면에서 여러분은 그 당시 일하던 누구보다 유리한 위치에 있습니다. 강력한 오픈 소스 딥러닝 프레임워크를 활용하여 단 몇 줄의 코드로 모델을 빠르게 구축할 수 있기 때문입니다. 
이전에는 이러한 네트워크를 훈련하기 위해 연구자들은 레이어와 도함수를 C, Fortran, 심지어 Lisp(LeNet의 경우)로 명시적으로 코딩해야 했습니다.

부차적인 이점은 ReLU가 시그모이드나 tanh 함수보다 최적화에 훨씬 더 적합하다는 것입니다. 
이것이 지난 10년 동안 딥러닝의 부활을 도운 주요 혁신 중 하나라고 주장할 수 있습니다. 
하지만 활성화 함수에 대한 연구는 멈추지 않았다는 점에 유의하십시오. 
예를 들어, :citet:`Hendrycks.Gimpel.2016`의 GELU(Gaussian error linear unit) 활성화 함수 $x \Phi(x)$ ($\\Phi(x)$는 표준 정규 누적 분포 함수)와 :citet:`Ramachandran.Zoph.Le.2017`에서 제안한 Swish 활성화 함수 $\sigma(x) = x \operatorname{sigmoid}(\beta x)$는 많은 경우에 더 나은 정확도를 산출할 수 있습니다.

## 연습 문제 (Exercises)

1. *선형* 심층 네트워크, 즉 비선형성 $\sigma$가 없는 네트워크에 레이어를 추가해도 네트워크의 표현력이 결코 증가하지 않음을 보이십시오. 
   적극적으로 줄이는 예를 드십시오.
2. pReLU 활성화 함수의 도함수를 계산하십시오.
3. Swish 활성화 함수 $x \operatorname{sigmoid}(\beta x)$의 도함수를 계산하십시오.
4. ReLU(또는 pReLU)만 사용하는 MLP가 연속적인 조각별 선형 함수를 구성함을 보이십시오.
5. 시그모이드와 tanh는 매우 유사합니다.
    1. $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$임을 보이십시오.
    2. 두 비선형성에 의해 파라미터화된 함수 클래스가 동일함을 증명하십시오. 힌트: 아핀 레이어에는 편향 항도 있습니다.
6. 배치 정규화 :cite:`Ioffe.Szegedy.2015`와 같이 한 번에 하나의 미니배치에 적용되는 비선형성이 있다고 가정해 봅시다. 이것이 어떤 종류의 문제를 일으킬 것으로 예상하십니까?
7. 시그모이드 활성화 함수에 대해 기울기가 사라지는 예를 제시하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/226)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17984)
:end_tab:

```