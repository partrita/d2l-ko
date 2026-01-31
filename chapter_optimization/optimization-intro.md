# 최적화와 딥러닝 (Optimization and Deep Learning)
:label:`sec_optimization-intro`

이 섹션에서는 최적화와 딥러닝 간의 관계뿐만 아니라 딥러닝에서 최적화를 사용할 때의 과제에 대해 논의할 것입니다.
딥러닝 문제의 경우, 일반적으로 먼저 *손실 함수(loss function)*를 정의합니다. 손실 함수가 있으면 손실을 최소화하기 위해 최적화 알고리즘을 사용할 수 있습니다.
최적화에서 손실 함수는 종종 최적화 문제의 *목적 함수(objective function)*라고 불립니다. 전통과 관례에 따라 대부분의 최적화 알고리즘은 *최소화*와 관련이 있습니다. 목적 함수를 최대화해야 하는 경우 간단한 해결책이 있습니다: 목적 함수의 부호를 바꾸기만 하면 됩니다.

## 최적화의 목표 (Goal of Optimization)

최적화가 딥러닝을 위한 손실 함수를 최소화하는 방법을 제공하지만, 본질적으로 최적화와 딥러닝의 목표는 근본적으로 다릅니다.
전자는 주로 목적 함수를 최소화하는 것과 관련이 있는 반면, 후자는 유한한 데이터가 주어졌을 때 적절한 모델을 찾는 것과 관련이 있습니다.
:numref:`sec_generalization_basics`에서 우리는 이 두 목표의 차이점에 대해 자세히 논의했습니다.
예를 들어,
훈련 오차와 일반화 오차는 일반적으로 다릅니다: 최적화 알고리즘의 목적 함수는 보통 훈련 데이터셋에 기반한 손실 함수이므로 최적화의 목표는 훈련 오차를 줄이는 것입니다.
그러나 딥러닝(또는 더 넓게는 통계적 추론)의 목표는 일반화 오차를 줄이는 것입니다.
후자를 달성하기 위해 우리는 훈련 오차를 줄이기 위해 최적화 알고리즘을 사용하는 것 외에도 과대적합(overfitting)에 주의를 기울여야 합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

앞서 언급한 서로 다른 목표를 설명하기 위해,
경험적 위험(empirical risk)과 위험(risk)을 고려해 봅시다.
:numref:`subsec_empirical-risk-and-risk`에서 설명한 대로,
경험적 위험은 훈련 데이터셋에서의 평균 손실인 반면,
위험은 전체 데이터 모집단에서의 기대 손실입니다.
아래에서 우리는 두 함수를 정의합니다:
위험 함수 `f`와 경험적 위험 함수 `g`입니다.
우리가 유한한 양의 훈련 데이터만 가지고 있다고 가정해 봅시다.
결과적으로, 여기서 `g`는 `f`보다 덜 매끄럽습니다.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

아래 그래프는 훈련 데이터셋에서의 경험적 위험의 최소값이 위험(일반화 오차)의 최소값과 다른 위치에 있을 수 있음을 보여줍니다.

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## 딥러닝에서의 최적화 과제 (Optimization Challenges in Deep Learning)

이 장에서는 모델의 일반화 오차보다는 목적 함수를 최소화하는 최적화 알고리즘의 성능에 구체적으로 초점을 맞출 것입니다.
:numref:`sec_linear_regression`에서 우리는 최적화 문제에서 해석적 해(analytical solutions)와 수치적 해(numerical solutions)를 구분했습니다.
딥러닝에서 대부분의 목적 함수는 복잡하며 해석적 해가 없습니다. 대신 우리는 수치적 최적화 알고리즘을 사용해야 합니다.
이 장의 최적화 알고리즘은 모두 이 범주에 속합니다.

딥러닝 최적화에는 많은 과제가 있습니다. 가장 골치 아픈 것 중 일부는 국소 최소값(local minima), 안장점(saddle points), 그리고 기울기 소실(vanishing gradients)입니다.
이들을 살펴봅시다.


### 국소 최소값 (Local Minima)

임의의 목적 함수 $f(x)$에 대해,
$x$에서의 $f(x)$ 값이 $x$ 근처의 다른 어떤 점들에서의 $f(x)$ 값보다 작으면, $f(x)$는 국소 최소값이 될 수 있습니다.
$x$에서의 $f(x)$ 값이 전체 도메인에 걸쳐 목적 함수의 최소값이면,
$f(x)$는 전역 최소값(global minimum)입니다.

예를 들어, 다음과 같은 함수가 주어졌을 때

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

우리는 이 함수의 국소 최소값과 전역 최소값을 근사할 수 있습니다.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

딥러닝 모델의 목적 함수는 보통 많은 국소 최적점(local optima)을 가집니다.
최적화 문제의 수치적 해가 국소 최적점 근처에 있을 때, 목적 함수 해의 기울기가 0에 접근하거나 0이 됨에 따라 최종 반복에 의해 얻어진 수치적 해는 목적 함수를 전역적으로가 아니라 *국소적으로*만 최소화할 수 있습니다.
어느 정도의 노이즈만이 파라미터를 국소 최소값에서 끄집어낼 수 있습니다. 사실, 이것은 미니배치에 대한 기울기의 자연스러운 변동이 파라미터를 국소 최소값에서 떼어낼 수 있는 미니배치 확률적 경사 하강법의 유익한 속성 중 하나입니다.


### 안장점 (Saddle Points)

국소 최소값 외에도 안장점은 기울기가 사라지는 또 다른 이유입니다. *안장점(saddle point)*은 함수의 모든 기울기가 사라지지만 전역 최소값도 국소 최소값도 아닌 위치입니다.
함수 $f(x) = x^3$을 고려해 보십시오. 그 1계 및 2계 도함수는 $x=0$에서 사라집니다. 최적화는 이 지점에서 정체될 수 있으며, 비록 그곳이 최소값은 아닐지라도 말입니다.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

고차원에서의 안장점은 아래 예제가 보여주는 것처럼 훨씬 더 교활합니다. 함수 $f(x, y) = x^2 - y^2$를 고려해 보십시오. 이 함수는 $(0, 0)$에서 안장점을 가집니다. 이는 $y$에 대해서는 최대값이고 $x$에 대해서는 최소값입니다. 더욱이, 그것은 말의 안장처럼 *보이는데*, 이것이 이 수학적 속성이 이름을 얻은 이유입니다.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

우리는 함수의 입력이 $k$차원 벡터이고 그 출력이 스칼라라고 가정하므로, 그 헤시안(Hessian) 행렬은 $k$개의 고유값을 가질 것입니다. 함수의 해는 함수 기울기가 0인 위치에서 국소 최소값, 국소 최대값 또는 안장점이 될 수 있습니다:

* 기울기가 0인 위치에서 함수의 헤시안 행렬의 고유값이 모두 양수일 때, 우리는 함수의 국소 최소값을 갖습니다.
* 기울기가 0인 위치에서 함수의 헤시안 행렬의 고유값이 모두 음수일 때, 우리는 함수의 국소 최대값을 갖습니다.
* 기울기가 0인 위치에서 함수의 헤시안 행렬의 고유값이 음수와 양수가 섞여 있을 때, 우리는 함수의 안장점을 갖습니다.

고차원 문제의 경우 적어도 *일부* 고유값이 음수일 가능성은 상당히 높습니다. 이는 안장점을 국소 최소값보다 더 가능성 있게 만듭니다. 우리는 다음 섹션에서 볼록성(convexity)을 도입할 때 이 상황에 대한 몇 가지 예외를 논의할 것입니다. 요컨대, 볼록 함수는 헤시안의 고유값이 결코 음수가 아닌 함수들입니다. 하지만 슬프게도 대부분의 딥러닝 문제는 이 범주에 속하지 않습니다. 그럼에도 불구하고 이는 최적화 알고리즘을 연구하기 위한 훌륭한 도구입니다.

### 기울기 소실 (Vanishing Gradients)

아마도 마주칠 수 있는 가장 교활한 문제는 기울기 소실입니다.
:numref:`subsec_activation-functions`에서 우리가 흔히 사용하는 활성화 함수와 그 도함수를 상기해 보십시오.
예를 들어, 우리가 $f(x) = \tanh(x)$ 함수를 최소화하고 싶고 마침 $x = 4$에서 시작하게 되었다고 가정해 봅시다. 보시다시피 $f$의 기울기는 거의 0에 가깝습니다.
더 구체적으로, $f'(x) = 1 - \tanh^2(x)$이므로 $f'(4) = 0.0013$입니다.
결과적으로 최적화는 진전을 이루기 전에 오랜 시간 동안 정체될 것입니다. 이것이 ReLU 활성화 함수가 도입되기 전에 딥러닝 모델을 훈련하는 것이 꽤 까다로웠던 이유 중 하나임이 밝혀졌습니다.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

우리가 보았듯이, 딥러닝을 위한 최적화는 도전 과제로 가득 차 있습니다. 다행히도 잘 작동하며 초보자도 사용하기 쉬운 견고한 알고리즘들이 존재합니다. 더욱이, 반드시 *최상의* 솔루션을 찾을 필요는 없습니다. 국소 최적점이나 그에 대한 근사해만으로도 여전히 매우 유용합니다.

## 요약 (Summary)

* 훈련 오차를 최소화하는 것이 일반화 오차를 최소화하기 위한 최상의 파라미터 세트를 찾는 것을 보장하지는 *않습니다*.
* 최적화 문제는 많은 국소 최소값을 가질 수 있습니다.
* 일반적으로 문제는 볼록하지 않으므로 훨씬 더 많은 안장점을 가질 수 있습니다.
* 기울기 소실은 최적화를 정체시킬 수 있습니다. 종종 문제의 재파라미터화(reparametrization)가 도움이 됩니다. 파라미터의 좋은 초기화도 유익할 수 있습니다.


## 연습 문제 (Exercises)

1. 은닉층이 하나뿐이고 은닉층 차원이 $d$이며 단일 출력을 갖는 간단한 MLP를 고려해 보십시오. 임의의 국소 최소값에 대해 동일하게 행동하는 적어도 $d!$개의 등가 솔루션이 있음을 보여주십시오.
2. 항목 $M_{ij} = M_{ji}$가 각각 어떤 확률 분포 $p_{ij}$에서 추출되는 대칭 무작위 행렬 $\mathbf{M}$이 있다고 가정합니다. 더욱이 $p_{ij}(x) = p_{ij}(-x)$, 즉 분포가 대칭이라고 가정합니다(자세한 내용은 예: :citet:`Wigner.1958` 참조).
    1. 고유값에 대한 분포도 대칭임을 증명하십시오. 즉, 임의의 고유벡터 $\mathbf{v}$에 대해 관련 고유값 $\lambda$가 $P(\lambda > 0) = P(\lambda < 0)$을 만족할 확률입니다.
    2. 왜 위의 내용이 $P(\lambda > 0) = 0.5$를 의미하지는 *않을까요*?
3. 딥러닝 최적화와 관련된 또 다른 과제는 무엇이 있을까요?
4. (실제) 안장 위에 (실제) 공의 균형을 맞추고 싶다고 가정해 봅시다.
    1. 왜 이것이 어려울까요?
    2. 이 효과를 최적화 알고리즘에도 활용할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab: