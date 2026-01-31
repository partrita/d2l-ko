# 일변수 미적분학 (Single Variable Calculus)
:label:`sec_single_variable_calculus`

:numref:`sec_calculus`에서 우리는 미분학의 기본 요소들을 보았습니다. 이 섹션에서는 미적분학의 기초를 더 깊이 파고들어 머신러닝의 맥락에서 이를 어떻게 이해하고 적용할 수 있는지 살펴봅니다.

## 미분학 (Differential Calculus)
미분학은 근본적으로 함수가 작은 변화 아래에서 어떻게 행동하는지에 대한 연구입니다. 이것이 왜 딥러닝의 핵심인지 알아보기 위해 예제를 하나 고려해 봅시다.

편의를 위해 가중치들이 단일 벡터 $\mathbf{w} = (w_1, \ldots, w_n)$로 연결된 심층 신경망이 있다고 가정합시다. 훈련 데이터셋이 주어졌을 때, 이 데이터셋에 대한 우리 신경망의 손실을 고려하며 이를 $\mathcal{L}(\mathbf{w})$라고 쓰겠습니다.

이 함수는 주어진 아키텍처의 가능한 모든 모델의 이 데이터셋에서의 성능을 인코딩하므로 매우 복잡하며, 어떤 가중치 세트 $\mathbf{w}$가 손실을 최소화할지 알기는 거의 불가능합니다. 따라서 실전에서는 종종 가중치를 *무작위로* 초기화하는 것부터 시작하여, 손실을 가능한 한 빨리 줄이는 방향으로 작은 단계들을 반복적으로 밟아 나갑니다.

그러면 질문은 표면상으로는 전혀 쉬워지지 않은 어떤 것이 됩니다: 가중치를 가능한 한 빨리 줄이는 방향을 어떻게 찾을까요? 이를 파헤치기 위해, 먼저 가중치가 단 하나뿐인 경우인 단일 실수 값 $x$에 대한 $L(\mathbf{w}) = L(x)$ 사례를 살펴봅시다.

$x$를 취하여 이를 $x + \epsilon$으로 아주 조금 바꾸었을 때 어떤 일이 일어나는지 이해해 보려고 노력합시다. 구체적으로 생각하고 싶다면 $\epsilon = 0.0000001$과 같은 숫자를 떠올려 보십시오. 무슨 일이 일어나는지 시각화하는 데 도움을 주기 위해, $[0, 3]$ 구간에서 예제 함수 $f(x) = \sin(x^x)$를 그래프로 그려봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# 일반적인 범위에서 함수 플롯
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # torch에서 pi 정의

# 일반적인 범위에서 함수 플롯
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # TensorFlow에서 pi 정의

# 일반적인 범위에서 함수 플롯
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

이 큰 스케일에서 함수의 동작은 단순하지 않습니다. 그러나 범위를 $[1.75, 2.25]$와 같이 더 작게 줄여보면 그래프가 훨씬 더 단순해지는 것을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
# 아주 작은 범위에서 동일한 함수 플롯
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# 아주 작은 범위에서 동일한 함수 플롯
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# 아주 작은 범위에서 동일한 함수 플롯
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

이를 극단으로 가져가서 아주 작은 세그먼트로 확대해 보면, 동작은 훨씬 더 단순해집니다: 그것은 그냥 직선입니다.

```{.python .input}
#@tab mxnet
# 아주 작은 범위에서 동일한 함수 플롯
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# 아주 작은 범위에서 동일한 함수 플롯
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# 아주 작은 범위에서 동일한 함수 플롯
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

이것이 일변수 미적분학의 핵심 관찰입니다: 친숙한 함수들의 동작은 충분히 작은 범위에서 직선으로 모델링될 수 있습니다. 이는 대부분의 함수에 대해, 함수의 $x$ 값을 아주 조금 옮길 때 출력 $f(x)$ 또한 아주 조금 옮겨질 것이라고 기대하는 것이 합리적임을 의미합니다. 우리가 답해야 할 유일한 질문은 "입력의 변화에 비해 출력의 변화가 얼마나 큰가? 절반만큼 큰가? 두 배만큼 큰가?" 하는 것입니다.

따라서 우리는 함수의 입력에서의 작은 변화에 대한 함수의 출력 변화의 비율을 고려할 수 있습니다. 이를 다음과 같이 공식적으로 쓸 수 있습니다.

$$ 
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}. 
$$ 

이것은 이미 코드에서 가지고 놀기 시작하기에 충분합니다. 예를 들어 $L(x) = x^{2} + 1701(x-4)^3$임을 안다고 가정합시다. 그러면 우리는 $x = 4$ 지점에서 이 값이 얼마나 큰지 다음과 같이 볼 수 있습니다.

```{.python .input}
#@tab all
# 함수 정의
def L(x):
    return x**2 + 1701*(x-4)**3

# 여러 에필론(epsilon)에 대해 차이를 에필론으로 나눈 값 인쇄
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

이제 우리가 주의 깊게 본다면, 이 숫자의 출력이 의심스러울 정도로 $8$에 가깝다는 것을 알아차릴 것입니다. 실제로 $\epsilon$을 줄이면 값이 점진적으로 $8$에 더 가까워지는 것을 볼 수 있습니다. 따라서 우리는 우리가 구하는 값(입력의 변화가 출력을 변화시키는 정도)이 $x=4$ 지점에서 $8$이어야 한다고 올바르게 결론지을 수 있습니다. 수학자가 이 사실을 인코딩하는 방식은 다음과 같습니다.

$$ 
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8. 
$$ 

역사적인 여담으로서: 신경망 연구의 처음 몇십 년 동안, 과학자들은 작은 섭동 하에서 손실 함수가 어떻게 변하는지 평가하기 위해 이 알고리즘(*유한 차분법*)을 사용했습니다: 그냥 가중치를 바꾸고 손실이 어떻게 변하는지 보는 것이었습니다. 이는 계산적으로 비효율적이며, 하나의 변수의 단일 변화가 손실에 어떤 영향을 미치는지 보기 위해 손실 함수를 두 번 평가해야 합니다. 만약 우리가 단 몇 천 개의 파라미터로라도 이를 시도했다면, 전체 데이터셋에 대해 네트워크를 수천 번 평가해야 했을 것입니다! :citet:`Rumelhart.Hinton.Williams.ea.1988`에서 도입된 *역전파 알고리즘*이 데이터셋에 대한 네트워크의 단일 예측과 동일한 계산 시간 내에 가중치의 *임의의* 변화가 손실을 어떻게 변화시킬지 계산하는 방법을 제공한 1986년에 이르러서야 이 문제가 해결되었습니다.

우리 예제로 돌아가서, 이 값 $8$은 $x$의 값마다 다르므로 이를 $x$의 함수로 정의하는 것이 타당합니다. 더 공식적으로, 이 값 의존적인 변화율을 *도함수(derivative)*라고 부르며 다음과 같이 씁니다.

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$ 
:eqlabel:`eq_der_def`

교재마다 도함수에 대해 다른 표기법을 사용할 것입니다. 예를 들어 아래의 표기법들은 모두 동일한 것을 나타냅니다.

$$ 
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x. 
$$ 

대부분의 저자들은 하나의 표기법을 골라 그것을 고수하겠지만, 그것조차 보장되지는 않습니다. 이러한 모든 것들에 익숙해지는 것이 가장 좋습니다. 우리는 이 텍스트 전체에서 복잡한 식의 도함수를 취하고 싶을 때를 제외하고는 $\frac{df}{dx}$ 표기법을 사용할 것이며, 그 경우에는 다음과 같은 식을 쓰기 위해 $\frac{d}{dx}f$를 사용할 것입니다.
$$ 
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right]. 
$$ 

종종 우리가 $x$를 아주 조금 바꿀 때 함수가 어떻게 변하는지 보기 위해 도함수의 정의 :eqref:`eq_der_def`를 다시 풀어보는 것이 직관적으로 유용합니다.

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). 
\end{aligned}$$ 
:eqlabel:`eq_small_change`

마지막 방정식은 명시적으로 불러낼 가치가 있습니다. 이는 당신이 임의의 함수를 취해 입력을 아주 작은 양만큼 바꾼다면, 출력은 도함수에 의해 스케일링된 그 작은 양만큼 바뀔 것임을 알려줍니다.

이런 식으로 우리는 도함수를 입력의 변화로부터 우리가 얻는 출력의 변화가 얼마나 큰지 알려주는 스케일링 인자로 이해할 수 있습니다.

## 미적분 규칙 (Rules of Calculus)
:label:`sec_derivative_table`

이제 우리는 명시적인 함수의 도함수를 계산하는 방법을 이해하는 과제로 넘어갑니다. 미적분학의 완전한 정식 처리는 제1원리로부터 모든 것을 유도할 것입니다. 우리는 여기서 이러한 유혹에 빠지지 않고, 흔히 마주치는 일반적인 규칙들에 대한 이해를 제공할 것입니다.

### 일반적인 도함수 (Common Derivatives)
:numref:`sec_calculus`에서 보았듯이, 도함수를 계산할 때 종종 일련의 규칙을 사용하여 계산을 몇 가지 핵심 함수로 축소할 수 있습니다. 참조의 편의를 위해 여기에서 반복합니다.

* **상수 함수의 미분.** $\frac{d}{dx}c = 0$.
* **선형 함수의 미분.** $\frac{d}{dx}(ax) = a$.
* **거듭제곱 법칙.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **지수 함수의 미분.** $\frac{d}{dx}e^x = e^x$.
* **로그 함수의 미분.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### 미분 규칙 (Derivative Rules)
모든 도함수를 별도로 계산하여 표에 저장해야 한다면 미분학은 거의 불가능할 것입니다. 위의 도함수들을 일반화하고 $f(x) = \log\left(1+(x-1)^{10}\right)$의 도함수를 찾는 것과 같은 더 복잡한 도함수를 계산할 수 있다는 것은 수학의 선물입니다. :numref:`sec_calculus`에서 언급했듯이, 그렇게 하기 위한 핵심은 우리가 함수들을 가져와 다양한 방식으로 결합할 때 어떤 일이 일어나는지를 성문화하는 것이며, 가장 중요한 것은 합, 곱, 그리고 합성입니다.

* **합의 법칙.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **곱의 미분법.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **연쇄 법칙.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

이러한 규칙들을 이해하기 위해 :eqref:`eq_small_change`를 어떻게 사용할 수 있는지 살펴봅시다. 합의 법칙의 경우 다음과 같은 추론 체인을 고려해 보십시오.

$$ 
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\ & \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\ & = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\ & = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right). 
\end{aligned} 
$$ 

이 결과를 $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$라는 사실과 비교함으로써, 우리가 원하던 대로 $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$임을 알 수 있습니다. 여기서의 직관은 다음과 같습니다: 우리가 입력 $x$를 바꿀 때, $g$와 $h$는 출력의 변화에 $\frac{dg}{dx}(x)$와 $\frac{dh}{dx}(x)$만큼 공동으로 기여합니다.


곱의 미분법은 더 미묘하며, 이러한 식들로 작업하는 방법에 대한 새로운 관찰을 필요로 할 것입니다. 우리는 이전과 같이 :eqref:`eq_small_change`를 사용하여 시작하겠습니다.

$$ 
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\ & \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\ & = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\ & = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\ 
\end{aligned} 
$$ 


이것은 위에서 수행된 계산과 닮았으며, 실제로 우리는 $\epsilon$ 옆에 앉아 있는 우리의 답($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$)을 봅니다. 하지만 크기 $\epsilon^{2}$인 그 항의 문제가 있습니다. 우리는 $\epsilon^2$의 차수가 $\epsilon^1$의 차수보다 높기 때문에 이를 *고차 항(higher-order term)*이라고 부를 것입니다. 우리는 나중 섹션에서 가끔 이것들을 추적하고 싶어 할 것임을 보게 되겠지만, 지금은 $\epsilon = 0.0000001$이라면 $\epsilon^{2}= 0.0000000000001$로 훨씬 더 작다는 점에 주목하십시오. 우리가 $\epsilon \rightarrow 0$으로 보냄에 따라 우리는 안전하게 고차 항들을 무시할 수 있습니다. 이 부록에서의 일반적인 관례로서, 우리는 "$\\%$"를 두 항이 고차 항들까지 같음을 나타내는 데 사용할 것입니다. 하지만 좀 더 정식으로 하고 싶다면 차분 몫(difference quotient)을 조사하여

$$ 
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x), 
$$ 

$\epsilon \rightarrow 0$으로 보냄에 따라 우변의 항도 0으로 가는 것을 볼 수 있습니다.

마지막으로 연쇄 법칙의 경우, 우리는 다시 이전과 같이 :eqref:`eq_small_change`를 사용하여 진행하여 다음을 볼 수 있습니다.

$$ 
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\ & \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\ & \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\ & = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x), 
\end{aligned} 
$$ 

여기서 두 번째 줄에서 우리는 함수 $g$가 아주 작은 양 $\epsilon \frac{dh}{dx}(x)$만큼 시프트된 입력($h(x)$)을 갖는 것으로 간주합니다.

이러한 규칙들은 본질적으로 원하는 어떤 식이라도 계산할 수 있는 유연한 도구 세트를 제공합니다. 예를 들어 다음과 같습니다.

$$ 
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\ & = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\ & = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\ & = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\ & = \frac{10(x-1)^9}{1+(x-1)^{10}}. 
\end{aligned} 
$$ 

여기서 각 줄은 다음 규칙들을 사용했습니다.

1. 연쇄 법칙과 로그의 도함수.
2. 합의 법칙.
3. 상수의 도함수, 연쇄 법칙, 그리고 거듭제곱 법칙.
4. 합의 법칙, 선형 함수의 도함수, 상수의 도함수.

이 예제를 하고 나서 두 가지가 분명해져야 합니다.

1. 합, 곱, 상수, 거듭제곱, 지수, 로그를 사용하여 우리가 쓸 수 있는 임의의 함수는 이러한 규칙들을 따름으로써 기계적으로 그 도함수가 계산될 수 있습니다.
2. 인간이 이러한 규칙들을 따르는 것은 지루하고 오류가 발생하기 쉬울 수 있습니다!

다행히도 이러한 두 가지 사실은 함께 앞으로 나아갈 길을 암시합니다: 이것은 기계화하기에 완벽한 후보입니다! 실제로 우리가 나중에 다시 살펴볼 역전파(backpropagation)가 바로 그것입니다.

### 선형 근사 (Linear Approximation)
도함수로 작업할 때, 위에서 사용된 근사를 기하학적으로 해석하는 것이 종종 유용합니다. 특히 다음 방정식은

$$ 
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x), 
$$ 

$(x, f(x))$ 점을 지나고 기울기가 $\frac{df}{dx}(x)$인 직선으로 $f$의 값을 근사합니다. 이런 식으로 우리는 도함수가 아래 그림과 같이 함수 $f$에 대한 선형 근사를 제공한다고 말합니다.

```{.python .input}
#@tab mxnet
# sin 계산
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# 몇 가지 선형 근사 계산. d(sin(x)) / dx = cos(x) 사용
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# sin 계산
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# 몇 가지 선형 근사 계산. d(sin(x))/dx = cos(x) 사용
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# sin 계산
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# 몇 가지 선형 근사 계산. d(sin(x))/dx = cos(x) 사용
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### 고계 도함수 (Higher Order Derivatives)

이제 겉보기에는 이상해 보일 수 있는 일을 해봅시다. 함수 $f$를 취해 도함수 $\frac{df}{dx}$를 계산합니다. 이는 우리에게 임의의 점에서의 $f$의 변화율을 제공합니다.

그러나 도함수 $\frac{df}{dx}$ 자체도 하나의 함수로 볼 수 있으므로, $\frac{df}{dx}$의 도함수를 다시 구하여 $\frac{d^2f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right)$를 얻는 것을 막는 것은 아무것도 없습니다. 우리는 이것을 $f$의 2계 도함수(second derivative)라고 부를 것입니다. 이 함수는 $f$의 변화율의 변화율, 즉 변화율이 어떻게 변하고 있는지를 나타냅니다. 우리는 도함수를 몇 번이라도 적용하여 이른바 $n$계 도함수를 얻을 수 있습니다. 표기법을 깔끔하게 유지하기 위해 $n$계 도함수를 다음과 같이 나타낼 것입니다.

$$ 
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f. 
$$ 

이것이 *왜* 유용한 개념인지 이해해 봅시다. 아래에서 $f^{(2)}(x)$, $f^{(1)}(x)$, 그리고 $f(x)$를 시각화합니다.

먼저, 2계 도함수 $f^{(2)}(x)$가 양의 상수인 경우를 고려해 보십시오. 이는 1계 도함수의 기울기가 양수임을 의미합니다. 결과적으로 1계 도함수 $f^{(1)}(x)$는 음수에서 시작하여 한 지점에서 0이 되고 마지막에 양수가 될 수 있습니다. 이는 우리 원래 함수 $f$의 기울기를 알려주며, 따라서 함수 $f$ 자체는 감소하다가 평평해졌다가 다시 증가합니다. 즉, 그림 :numref:`fig_positive-second`에 표시된 것처럼 함수 $f$는 위로 휘어져 있으며 단일 최소값을 갖습니다.

![우리가 2계 도함수가 양의 상수라고 가정한다면, 1계 도함수는 증가하고 있으며, 이는 함수 자체가 최소값을 가짐을 의미합니다.](../img/posSecDer.svg)
:label:`fig_positive-second`


둘째, 만약 2계 도함수가 음의 상수라면, 이는 1계 도함수가 감소하고 있음을 의미합니다. 이는 1계 도함수가 양수에서 시작하여 한 지점에서 0이 되고 음수가 될 수 있음을 시사합니다. 따라서 함수 $f$ 자체는 증가하다가 평평해졌다가 감소합니다. 즉, 그림 :numref:`fig_negative-second`에 표시된 것처럼 함수 $f$는 아래로 휘어져 있으며 단일 최대값을 갖습니다.

![우리가 2계 도함수가 음의 상수라고 가정한다면, 1계 도함수는 감소하고 있으며, 이는 함수 자체가 최대값을 가짐을 의미합니다.](../img/negSecDer.svg)
:label:`fig_negative-second`


셋째, 2계 도함수가 항상 0이라면, 1계 도함수는 결코 변하지 않을 것입니다 - 즉 상수입니다! 이는 $f$가 고정된 비율로 증가(또는 감소)함을 의미하며, $f$는 그림 :numref:`fig_zero-second`에 표시된 것처럼 그 자체가 직선입니다.

![우리가 2계 도함수가 0이라고 가정한다면, 1계 도함수는 상수이며, 이는 함수 자체가 직선임을 의미합니다.](../img/zeroSecDer.svg)
:label:`fig_zero-second`


요약하자면, 2계 도함수는 함수 $f$가 휘어지는 방식을 설명하는 것으로 해석될 수 있습니다. 양의 2계 도함수는 위쪽 곡선으로 이어지는 반면, 음의 2계 도함수는 함수가 아래쪽으로 휘어짐을 의미하고, 0인 2계 도함수는 함수가 전혀 휘어지지 않음을 의미합니다.

이를 한 단계 더 진행해 봅시다. 함수 $g(x) = ax^{2}+ bx + c$를 고려해 보십시오. 그러면 다음과 같이 계산할 수 있습니다.

$$ 
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\ 
\frac{d^2g}{dx^2}(x) & = 2a. 
\end{aligned} 
$$ 

만약 우리가 어떤 원래 함수 $f(x)$를 염두에 두고 있다면, 우리는 처음 두 개의 도함수를 계산하고 이 계산과 일치하도록 $a, b, c$ 값을 찾을 수 있습니다. 1계 도함수가 직선으로 최상의 근사를 제공했던 이전 섹션과 유사하게, 이 구조는 이차식에 의한 최상의 근사를 제공합니다. $f(x) = \sin(x)$에 대해 이를 시각화해 봅시다.

```{.python .input}
#@tab mxnet
# sin 계산
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# 몇 가지 이차 근사 계산. d(sin(x)) / dx = cos(x) 사용
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# sin 계산
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# 몇 가지 이차 근사 계산. d(sin(x)) / dx = cos(x) 사용
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# sin 계산
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# 몇 가지 이차 근사 계산. d(sin(x)) / dx = cos(x) 사용
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

우리는 다음 섹션에서 이 아이디어를 *테일러 급수*의 개념으로 확장할 것입니다.

### 테일러 급수 (Taylor Series)


*테일러 급수(Taylor series)*는 한 점 $x_0$에서의 처음 $n$개 도함수 값, 즉 $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$이 주어졌을 때 함수 $f(x)$를 근사하는 방법을 제공합니다. 아이디어는 $x_0$에서 주어진 모든 도함수와 일치하는 $n$차 다항식을 찾는 것입니다.

우리는 이전 섹션에서 $n=2$인 경우를 보았고 약간의 대수는 이것이 다음과 같음을 보여줍니다.

$$ 
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0). 
$$ 

위에서 볼 수 있듯이, 분모의 $2$는 우리가 $x^2$의 두 도함수를 취할 때 얻는 $2$를 상쇄하기 위해 있는 반면, 다른 항들은 모두 0입니다. 동일한 로직이 1계 도함수와 값 자체에도 적용됩니다.

로직을 $n=3$까지 밀어붙인다면 다음과 같이 결론지을 것입니다.

$$ 
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0). 
$$ 

여기서 $6 = 3 \times 2 = 3!$은 $x^3$의 세 도함수를 취할 때 앞에 붙는 상수로부터 옵니다.


더 나아가, 우리는 다음과 같이 $n$차 다항식을 얻을 수 있습니다.

$$ 
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}. 
$$ 

여기서 표기법은 다음과 같습니다.

$$ 
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f. 
$$ 


실제로 $P_n(x)$는 우리 함수 $f(x)$에 대한 최상의 $n$차 다항식 근사로 간주될 수 있습니다.

위 근사의 오차를 끝까지 파고들지는 않겠지만, 무한 극한에 대해 언급할 가치가 있습니다. 이 경우 $\cos(x)$나 $e^{x}$와 같이 잘 행동하는 함수(실해석적 함수로 알려짐)에 대해, 우리는 무한한 수의 항을 써서 정확히 동일한 함수를 근사할 수 있습니다.

$$ 
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}. 
$$ 

$f(x) = e^{x}$를 예로 들어봅시다. $e^{x}$는 자기 자신이 도함수이므로 $f^{(n)}(x) = e^{x}$임을 압니다. 따라서 $e^{x}$는 $x_0 = 0$에서 테일러 급수를 취함으로써 재구성될 수 있습니다. 즉 다음과 같습니다.

$$ 
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots. 
$$ 

이것이 코드에서 어떻게 작동하는지 살펴보고, 테일러 근사의 차수를 높이는 것이 어떻게 우리가 원하는 함수 $e^x$에 더 가까워지게 하는지 관찰해 봅시다.

```{.python .input}
#@tab mxnet
# 지수 함수 계산
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# 몇 가지 테일러 급수 근사 계산
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# 지수 함수 계산
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# 몇 가지 테일러 급수 근사 계산
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# 지수 함수 계산
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# 몇 가지 테일러 급수 근사 계산
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

테일러 급수에는 두 가지 주요 응용 분야가 있습니다.

1. *이론적 응용*: 우리가 너무 복잡한 함수를 이해하려고 할 때, 테일러 급수를 사용하면 우리가 직접 다룰 수 있는 다항식으로 변환할 수 있는 경우가 많습니다.

2. *수치적 응용*: $e^{x}$나 $\cos(x)$와 같은 일부 함수는 기계가 계산하기 어렵습니다. 기계는 고정된 정밀도로 값의 테이블을 저장할 수 있고(종종 그렇게 합니다), 하지만 "$\\cos(1)$의 1000번째 자리는 무엇인가?"와 같은 미결 질문이 여전히 남습니다. 테일러 급수는 종종 그러한 질문에 답하는 데 도움이 됩니다.


## 요약 (Summary)

* 도함수는 우리가 입력을 아주 작은 양만큼 바꿀 때 함수가 어떻게 변하는지 표현하는 데 사용될 수 있습니다.
* 기초적인 도함수들은 미분 규칙을 사용하여 결합되어 임의로 복잡한 도함수를 생성할 수 있습니다.
* 도함수를 반복하여 2계 이상의 고계 도함수를 얻을 수 있습니다. 차수가 증가할 때마다 함수의 행동에 대해 더 세밀한 정보를 제공합니다.
* 단일 데이터 예제의 도함수에 있는 정보를 사용하여, 우리는 테일러 급수로부터 얻은 다항식으로 잘 행동하는 함수를 근사할 수 있습니다.


## 연습 문제 (Exercises)

1. $x^3-4x+1$의 도함수는 무엇입니까?
2. $\log(\frac{1}{x})$의 도함수는 무엇입니까?
3. 맞음 또는 틀림: 만약 $f'(x) = 0$이라면 $f$는 $x$에서 최대값이나 최소값을 갖습니까?
4. $x\ge0$에 대해 $f(x) = x\log(x)$의 최소값은 어디입니까 (여기서 $f(0)$에서 $f$는 극한값 0을 취한다고 가정합시다)?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab: