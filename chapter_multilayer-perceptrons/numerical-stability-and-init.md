```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 수치적 안정성과 초기화 (Numerical Stability and Initialization)
:label:`sec_numerical_stability`


지금까지 우리가 구현한 모든 모델은 미리 지정된 분포에 따라 파라미터를 초기화해야 했습니다. 
지금까지는 이러한 선택이 어떻게 이루어지는지에 대한 세부 사항을 얼버무리며 초기화 방식을 당연하게 여겼습니다. 
이러한 선택이 특별히 중요하지 않다는 인상을 받았을 수도 있습니다. 
반대로, 초기화 방식의 선택은 신경망 학습에서 중요한 역할을 하며, 수치적 안정성을 유지하는 데 결정적일 수 있습니다. 
게다가 이러한 선택은 비선형 활성화 함수의 선택과 흥미로운 방식으로 연결될 수 있습니다. 
우리가 선택한 함수와 파라미터 초기화 방식은 최적화 알고리즘이 얼마나 빨리 수렴하는지를 결정할 수 있습니다. 
여기서 잘못된 선택을 하면 훈련 중에 기울기 폭발이나 소실을 겪을 수 있습니다. 
이 섹션에서는 이러한 주제를 더 자세히 살펴보고 딥러닝 경력 내내 유용하게 사용할 수 있는 몇 가지 유용한 휴리스틱에 대해 논의합니다.

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

## 기울기 소실 및 폭발 (Vanishing and Exploding Gradients)

$L$개의 레이어, 입력 $\mathbf{x}$, 출력 $\mathbf{o}$를 가진 심층 네트워크를 고려해 봅시다. 
각 레이어 $l$이 가중치 $\mathbf{W}^{(l)}$로 파라미터화된 변환 $f_l$에 의해 정의되고, 은닉층 출력이 $\mathbf{h}^{(l)}$ ($\mathbf{h}^{(0)} = \mathbf{x}$라고 가정)일 때, 네트워크는 다음과 같이 표현될 수 있습니다:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ 따라서 } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$ 

모든 은닉층 출력과 입력이 벡터인 경우, 임의의 파라미터 세트 $\mathbf{W}^{(l)}$에 대한 $\mathbf{o}$의 기울기를 다음과 같이 쓸 수 있습니다:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$ 

즉, 이 기울기는 $L-l$개의 행렬 $\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$과 기울기 벡터 $\mathbf{v}^{(l)}$의 곱입니다. 
따라서 너무 많은 확률을 함께 곱할 때 종종 발생하는 수치적 언더플로 문제에 취약합니다. 
확률을 다룰 때 일반적인 트릭은 로그 공간으로 전환하는 것입니다. 즉, 수치 표현의 압력을 가수(mantissa)에서 지수(exponent)로 옮기는 것입니다. 
불행히도 위의 문제는 더 심각합니다. 처음에 행렬 $\mathbf{M}^{(l)}$은 다양한 고유값을 가질 수 있습니다. 
그것들은 작거나 클 수 있으며, 그들의 곱은 *매우 크거나* *매우 작을* 수 있습니다.

불안정한 기울기로 인한 위험은 수치 표현을 넘어섭니다. 
예측할 수 없는 크기의 기울기는 최적화 알고리즘의 안정성도 위협합니다. 
우리는 (i) 지나치게 커서 모델을 파괴하는 파라미터 업데이트(*기울기 폭발* 문제); 
또는 (ii) 지나치게 작아서 파라미터가 각 업데이트에서 거의 움직이지 않아 학습을 불가능하게 만드는 파라미터 업데이트(*기울기 소실* 문제)에 직면할 수 있습니다.


### (**기울기 소실 (Vanishing Gradients)**)

기울기 소실 문제를 일으키는 한 가지 빈번한 범인은 각 레이어의 선형 연산 뒤에 추가되는 활성화 함수 $\sigma$의 선택입니다. 
역사적으로 시그모이드 함수 $1/(1 + \exp(-x))$ (:numref:`sec_mlp`에서 소개됨)는 임계값 함수와 유사하기 때문에 인기가 있었습니다. 
초기 인공 신경망은 생물학적 신경망에서 영감을 얻었기 때문에, *완전히* 발화하거나 *전혀* 발화하지 않는 뉴런(생물학적 뉴런처럼)이라는 아이디어가 매력적으로 보였습니다. 
시그모이드가 왜 기울기 소실을 일으킬 수 있는지 자세히 살펴봅시다.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

보시다시피, (**시그모이드의 기울기는 입력이 크거나 작을 때 모두 사라집니다**). 
게다가 많은 레이어를 통해 역전파할 때, 많은 시그모이드의 입력이 0에 가까운 골디락스 존(Goldilocks zone)에 있지 않는 한 전체 곱의 기울기가 사라질 수 있습니다. 
네트워크가 많은 레이어를 자랑할 때, 주의하지 않으면 기울기가 어떤 레이어에서 잘릴 가능성이 높습니다. 
실제로 이 문제는 심층 네트워크 훈련을 괴롭히곤 했습니다. 
결과적으로 더 안정적인(하지만 신경적으로는 덜 그럴듯한) ReLU가 실무자들의 기본 선택으로 부상했습니다.


### [**기울기 폭발 (Exploding Gradients)**]

반대 문제인 기울기 폭발도 마찬가지로 골치 아플 수 있습니다. 
이를 좀 더 잘 설명하기 위해 100개의 가우스 랜덤 행렬을 그려 초기 행렬과 곱합니다. 
우리가 선택한 스케일(분산 $\sigma^2=1$ 선택)에 대해 행렬 곱이 폭발합니다. 
심층 네트워크의 초기화로 인해 이런 일이 발생하면 경사 하강법 최적화기를 수렴시킬 기회가 없습니다.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('단일 행렬', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('100개의 행렬을 곱한 후', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('단일 행렬 \n ',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('100개의 행렬을 곱한 후\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('단일 행렬 \n ', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('100개의 행렬을 곱한 후\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # PRNG 키 생성
M = jax.random.normal(get_key(), (4, 4))
print('단일 행렬 \n ', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('100개의 행렬을 곱한 후\n', M)
```

### 대칭 깨기 (Breaking the Symmetry)

신경망 설계의 또 다른 문제는 파라미터화에 내재된 대칭성입니다. 
하나의 은닉층과 두 개의 유닛이 있는 간단한 MLP가 있다고 가정해 봅시다. 
이 경우 첫 번째 레이어의 가중치 $\mathbf{W}^{(1)}$을 치환하고 출력 레이어의 가중치를 마찬가지로 치환하여 동일한 함수를 얻을 수 있습니다. 
첫 번째 은닉 유닛과 두 번째 은닉 유닛을 구별하는 특별한 것은 없습니다. 
즉, 각 레이어의 은닉 유닛 간에는 순열 대칭(permutation symmetry)이 있습니다.

이것은 단순한 이론적 성가심 이상입니다. 
앞서 언급한 두 개의 은닉 유닛이 있는 1-은닉층 MLP를 고려해 보십시오. 
설명을 위해 출력 레이어가 두 은닉 유닛을 단 하나의 출력 유닛으로 변환한다고 가정해 보십시오. 
은닉층의 모든 파라미터를 어떤 상수 $c$에 대해 $\mathbf{W}^{(1)} = c$로 초기화하면 어떤 일이 일어날지 상상해 보십시오. 
이 경우 순전파 동안 두 은닉 유닛은 동일한 입력과 파라미터를 받아 동일한 활성화를 생성하고 출력 유닛으로 공급됩니다. 
역전파 동안 출력 유닛을 파라미터 $\mathbf{W}^{(1)}$로 미분하면 모든 요소가 동일한 값을 갖는 기울기를 얻습니다. 
따라서 경사 기반 반복(예: 미니배치 확률적 경사 하강법) 후에도 $\mathbf{W}^{(1)}$의 모든 요소는 여전히 동일한 값을 갖습니다. 
이러한 반복은 스스로 *대칭을 깨뜨릴* 수 없으며 우리는 네트워크의 표현력을 결코 실현할 수 없을지도 모릅니다. 
은닉층은 마치 단 하나의 유닛만 있는 것처럼 동작할 것입니다. 
미니배치 확률적 경사 하강법은 이 대칭을 깨뜨리지 못하지만, 드롭아웃 정규화(나중에 소개됨)는 깨뜨릴 수 있다는 점에 유의하십시오!


## 파라미터 초기화 (Parameter Initialization)

위에서 제기된 문제를 해결하거나 적어도 완화하는 한 가지 방법은 신중한 초기화를 통하는 것입니다. 
나중에 보겠지만 최적화 중의 추가적인 주의와 적절한 정규화가 안정성을 더욱 향상시킬 수 있습니다.


### 기본 초기화 (Default Initialization)

이전 섹션들, 예: :numref:`sec_linear_concise`에서 우리는 가중치 값을 초기화하기 위해 정규 분포를 사용했습니다. 
초기화 방법을 지정하지 않으면 프레임워크는 기본 무작위 초기화 방법을 사용하며, 이는 중간 규모의 문제 크기에 대해 실제로 잘 작동하는 경우가 많습니다.






### Xavier 초기화 (Xavier Initialization)
:label:`subsec_xavier`

*비선형성 없는* 완전 연결 레이어에 대한 출력 $o_{i}$의 스케일 분포를 살펴봅시다. 
이 레이어에 대해 $n_\textrm{in}$개의 입력 $x_j$와 관련 가중치 $w_{ij}$가 있을 때, 출력은 다음과 같이 주어집니다.

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$ 

가중치 $w_{ij}$는 모두 동일한 분포에서 독립적으로 추출됩니다. 
또한 이 분포가 평균 0과 분산 $\sigma^2$을 갖는다고 가정해 봅시다. 
이것은 분포가 가우시안이어야 한다는 것을 의미하는 것이 아니라, 평균과 분산이 존재해야 함을 의미합니다. 
지금은 레이어에 대한 입력 $x_j$도 평균 0과 분산 $\gamma^2$을 가지며 $w_{ij}$와 독립적이고 서로 독립적이라고 가정해 봅시다. 
이 경우 $o_i$의 평균을 계산할 수 있습니다:

$$ 
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, 
\end{aligned}
$$ 

그리고 분산:

$$ 
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$ 

분산을 고정하는 한 가지 방법은 $n_\textrm{in} \sigma^2 = 1$로 설정하는 것입니다. 
이제 역전파를 고려해 봅시다. 
거기서 우리는 출력에 더 가까운 레이어에서 전파되는 기울기와 함께 유사한 문제에 직면합니다. 
순전파와 동일한 추론을 사용하여, $n_\textrm{out} \sigma^2 = 1$이 아닌 한 기울기의 분산이 폭발할 수 있음을 알 수 있습니다. 여기서 $n_\textrm{out}$은 이 레이어의 출력 수입니다. 
이는 우리를 딜레마에 빠뜨립니다: 두 조건을 동시에 만족시키는 것은 불가능합니다. 
대신 우리는 단순히 다음을 만족시키려고 노력합니다:

$$ 
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ 또는 동등하게 } 
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$ 

이것이 창시자 중 제1 저자의 이름을 딴 현재 표준이자 실용적으로 유익한 *Xavier 초기화(Xavier initialization)*의 기본 추론입니다 :cite:`Glorot.Bengio.2010`. 
일반적으로 Xavier 초기화는 평균 0과 분산 $\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$인 가우스 분포에서 가중치를 샘플링합니다. 
우리는 또한 균등 분포에서 가중치를 샘플링할 때 분산을 선택하도록 이를 조정할 수 있습니다. 
균등 분포 $U(-a, a)$는 분산 $\frac{a^2}{3}$을 가짐을 상기하십시오. 
$\sigma^2$에 대한 조건에 $\frac{a^2}{3}$을 대입하면 다음에 따라 초기화하도록 유도됩니다.

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$ 

비록 위의 수학적 추론에서 비선형성이 존재하지 않는다는 가정이 신경망에서 쉽게 위반될 수 있지만, Xavier 초기화 방법은 실제로 잘 작동하는 것으로 밝혀졌습니다.


### 그 너머 (Beyond)

위의 추론은 파라미터 초기화에 대한 현대적 접근 방식의 겉핥기에 불과합니다. 
딥러닝 프레임워크는 종종 12개 이상의 서로 다른 휴리스틱을 구현합니다. 
게다가 파라미터 초기화는 딥러닝의 근본적인 연구 분야로 계속 남아 있습니다. 
그중에는 묶인(공유된) 파라미터, 초해상도, 시퀀스 모델 및 기타 상황에 특화된 휴리스틱이 있습니다. 
예를 들어 :citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018`는 신중하게 설계된 초기화 방법을 사용하여 아키텍처 트릭 없이 10,000개 레이어 신경망을 훈련할 가능성을 보여주었습니다.

이 주제에 관심이 있다면 이 모듈의 제공 사항을 깊이 파고들고, 각 휴리스틱을 제안하고 분석한 논문을 읽은 다음, 해당 주제에 대한 최신 간행물을 탐색할 것을 제안합니다. 
어쩌면 기발한 아이디어를 우연히 발견하거나 발명하여 딥러닝 프레임워크에 구현을 기여할 수도 있을 것입니다.


## 요약 (Summary)

기울기 소실 및 폭발은 심층 네트워크에서 흔한 문제들입니다. 기울기와 파라미터가 잘 제어되도록 하려면 파라미터 초기화에 각별한 주의가 필요합니다.
초기 기울기가 너무 크거나 작지 않도록 초기화 휴리스틱이 필요합니다.
무작위 초기화는 최적화 전에 대칭이 깨지도록 하는 데 핵심적입니다.
Xavier 초기화는 각 레이어에 대해 출력의 분산이 입력 수에 영향을 받지 않고, 기울기의 분산이 출력 수에 영향을 받지 않을 것을 제안합니다.
ReLU 활성화 함수는 기울기 소실 문제를 완화합니다. 이는 수렴을 가속화할 수 있습니다.

## 연습 문제 (Exercises)

1. MLP 레이어의 순열 대칭 외에 깨져야 할 대칭을 보일 수 있는 신경망의 다른 사례를 설계할 수 있습니까?
1. 선형 회귀나 소프트맥스 회귀의 모든 가중치 파라미터를 동일한 값으로 초기화할 수 있습니까?
1. 두 행렬 곱의 고유값에 대한 해석적 경계를 찾아보십시오. 이것은 기울기가 잘 조건화되도록 보장하는 것에 대해 무엇을 알려줍니까?
1. 일부 항이 발산한다는 것을 안다면 사후에 이를 고칠 수 있습니까? 영감을 얻으려면 계층별 적응형 속도 스케일링(layerwise adaptive rate scaling)에 대한 논문을 살펴보십시오 :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17986)
:end_tab: