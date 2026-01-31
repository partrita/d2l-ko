# 볼록성 (Convexity)
:label:`sec_convexity`

볼록성은 최적화 알고리즘 설계에서 중요한 역할을 합니다. 
이는 주로 그러한 맥락에서 알고리즘을 분석하고 테스트하기가 훨씬 쉽기 때문입니다. 
즉, 알고리즘이 볼록 설정에서조차 성능이 좋지 않다면 일반적으로 다른 상황에서도 좋은 결과를 기대해서는 안 됩니다. 
더욱이, 딥러닝의 최적화 문제는 일반적으로 비볼록(nonconvex)이지만, 국소 최소값 근처에서 볼록한 것의 몇 가지 특성을 나타내는 경우가 많습니다. 이는 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`과 같은 흥미로운 새로운 최적화 변형으로 이어질 수 있습니다.

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

## 정의 (Definitions)

볼록 분석 전에 *볼록 집합(convex sets)*과 *볼록 함수(convex functions)*를 정의해야 합니다. 
이들은 머신러닝에 흔히 적용되는 수학적 도구로 이어집니다.


### 볼록 집합 (Convex Sets)

집합은 볼록성의 기초입니다. 간단히 말해서, 벡터 공간의 집합 $\mathcal{X}$는 임의의 $a, b \in \mathcal{X}$에 대해 $a$와 $b$를 잇는 선분도 $\mathcal{X}$에 있으면 *볼록*합니다. 수학적 용어로 이는 모든 $\lambda  [0, 1]$에 대해 다음을 의미합니다.

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \textrm{ whenever } a, b \in \mathcal{X}.$$ 

이것은 약간 추상적으로 들릴 수 있습니다. :numref:`fig_pacman`을 고려해 보십시오. 첫 번째 집합은 그 안에 포함되지 않는 선분이 존재하므로 볼록하지 않습니다. 다른 두 집합은 그런 문제가 없습니다.

![첫 번째 집합은 비볼록하고 다른 두 개는 볼록합니다.](../img/pacman.svg)
:label:`fig_pacman`

정의 자체는 그것으로 무언가를 할 수 없다면 특별히 유용하지 않습니다. 이 경우 :numref:`fig_convex_intersect`에 표시된 교집합을 살펴볼 수 있습니다. $\mathcal{X}$와 $\mathcal{Y}$가 볼록 집합이라고 가정합시다. 그러면 $\mathcal{X} \cap \mathcal{Y}$도 볼록합니다. 이를 확인하려면 임의의 $a, b \in \mathcal{X} \cap \mathcal{Y}$를 고려하십시오. $\mathcal{X}$와 $\mathcal{Y}$가 볼록하므로 $a$와 $b$를 잇는 선분은 $\mathcal{X}$와 $\mathcal{Y}$ 모두에 포함됩니다. 따라서 그들은 $\mathcal{X} \cap \mathcal{Y}$에도 포함되어야 하며, 이로써 정리가 증명됩니다.

![두 볼록 집합의 교집합은 볼록합니다.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

우리는 큰 노력 없이 이 결과를 강화할 수 있습니다: 볼록 집합 $\mathcal{X}_i$가 주어졌을 때, 그들의 교집합 $\cap_{i} \mathcal{X}_i$는 볼록합니다. 역이 참이 아님을 보려면 두 개의 분리된 집합 $\mathcal{X} \cap \mathcal{Y} = \emptyset$을 고려하십시오. 이제 $a \in \mathcal{X}$ 및 $b \in \mathcal{Y}$를 선택하십시오. $\mathcal{X} \cap \mathcal{Y} = \emptyset$이라고 가정했으므로 :numref:`fig_nonconvex`에서 $a$와 $b$를 잇는 선분은 $\mathcal{X}$에도 $\mathcal{Y}$에도 속하지 않는 일부 부분을 포함해야 합니다. 따라서 선분은 $\mathcal{X} \cup \mathcal{Y}$에도 있지 않으며, 이는 일반적으로 볼록 집합의 합집합이 볼록할 필요는 없음을 증명합니다.

![두 볼록 집합의 합집합은 볼록할 필요가 없습니다.](../img/nonconvex.svg)
:label:`fig_nonconvex`

일반적으로 딥러닝의 문제는 볼록 집합에서 정의됩니다. 예를 들어 실수들의 $d$차원 벡터 집합인 $\mathbb{R}^d$는 볼록 집합입니다(결국 $\mathbb{R}^d$의 임의의 두 점 사이의 직선은 $\mathbb{R}^d$에 남아 있습니다). 어떤 경우에는 ${\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\| \leq r}$로 정의된 반지름 $r$의 공과 같이 길이가 제한된 변수들로 작업합니다.

### 볼록 함수 (Convex Functions)

이제 볼록 집합이 있으므로 *볼록 함수* $f$를 도입할 수 있습니다. 
볼록 집합 $\mathcal{X}$가 주어졌을 때, 함수 $f: \mathcal{X} \to \mathbb{R}$은 모든 $x, x' \in \mathcal{X}$와 모든 $\lambda  [0, 1]$에 대해 다음을 만족하면 *볼록*합니다.

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$ 

이를 설명하기 위해 몇 가지 함수를 플롯하고 어떤 함수가 요구 사항을 충족하는지 확인해 봅시다. 아래에서 볼록 함수와 비볼록 함수를 모두 정의합니다.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # 볼록
g = lambda x: d2l.cos(np.pi * x)  # 비볼록
h = lambda x: d2l.exp(0.5 * x)  # 볼록

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

예상대로 코사인 함수는 *비볼록*인 반면, 포물선과 지수 함수는 볼록합니다. 조건이 의미가 있으려면 $\mathcal{X}$가 볼록 집합이어야 한다는 요구 사항이 필요하다는 점에 유의하십시오. 그렇지 않으면 $f(\lambda x + (1-\lambda) x')$의 결과가 잘 정의되지 않을 수 있습니다.


### 젠센 부등식 (Jensen's Inequality)

볼록 함수 $f$가 주어졌을 때, 
가장 유용한 수학적 도구 중 하나는 *젠센 부등식(Jensen's inequality)*입니다. 
이는 볼록성 정의의 일반화에 해당합니다.

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ 및 }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$ 
:eqlabel:`eq_jensens-inequality`

여기서 $\alpha_i$는 $\sum_i \alpha_i = 1$을 만족하는 비음수 실수이고 $X$는 확률 변수입니다. 
즉, 볼록 함수의 기대값은 기대값의 볼록 함수보다 작지 않으며, 후자는 일반적으로 더 단순한 식입니다. 
첫 번째 부등식을 증명하기 위해 우리는 합계의 한 항씩 볼록성 정의를 반복적으로 적용합니다.


젠센 부등식의 일반적인 응용 중 하나는 
더 복잡한 식을 더 단순한 식으로 경계 짓는 것입니다. 
예를 들어, 
그 응용은 부분적으로 관찰된 확률 변수의 로그 우도에 관한 것일 수 있습니다. 즉, 우리는 다음을 사용합니다.

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$ 

$\int P(Y) P(X \mid Y) dY = P(X)$이기 때문입니다. 
이는 변분 방법(variational methods)에서 사용될 수 있습니다. 여기서 $Y$는 일반적으로 관찰되지 않은 확률 변수이고, $P(Y)$는 그것이 어떻게 분포되어 있을지에 대한 최선의 추측이며, $P(X)$는 $Y$가 통합되어 제거된 분포입니다. 예를 들어, 클러스터링에서 $Y$는 클러스터 레이블일 수 있고 $P(X \mid Y)$는 클러스터 레이블을 적용할 때의 생성 모델입니다.



## 속성 (Properties)

볼록 함수는 유용한 속성을 많이 가지고 있습니다. 아래에서 흔히 사용되는 몇 가지를 설명합니다.


### 국소 최소값은 전역 최소값이다 (Local Minima Are Global Minima)

무엇보다도, 볼록 함수의 국소 최소값은 전역 최소값이기도 합니다. 
우리는 이를 다음과 같이 귀류법으로 증명할 수 있습니다.

볼록 집합 $\mathcal{X}$에서 정의된 볼록 함수 $f$를 고려하십시오. 
$x^{\ast} \in \mathcal{X}$가 국소 최소값이라고 가정합시다: 
$0 < |x - x^{\ast}| \leq p$를 만족하는 $x \in \mathcal{X}$에 대해 $f(x^{\ast}) < f(x)$가 성립하도록 하는 작은 양수 값 $p$가 존재합니다.

국소 최소값 $x^{\ast}$가 $f$의 전역 최소값이 아니라고 가정합시다: 
$f(x') < f(x^{\ast})$인 $x' \in \mathcal{X}$가 존재합니다. 
또한 $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$가 되도록 
$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$와 같은 
$\lambda \in [0, 1)$가 존재합니다. 

그러나 볼록 함수의 정의에 따라 우리는 다음을 갖습니다.

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$ 

이는 $x^{\ast}$가 국소 최소값이라는 우리의 진술과 모순됩니다. 
따라서 $f(x') < f(x^{\ast})$인 $x' \in \mathcal{X}$는 존재하지 않습니다. 국소 최소값 $x^{\ast}$는 전역 최소값이기도 합니다.

예를 들어, 볼록 함수 $f(x) = (x-1)^2$는 $x=1$에서 국소 최소값을 가지며, 이는 전역 최소값이기도 합니다.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

볼록 함수의 국소 최소값이 전역 최소값이기도 하다는 사실은 매우 편리합니다. 
이는 우리가 함수를 최소화할 때 "갇힐" 수 없음을 의미합니다. 
하지만 이것이 전역 최소값이 하나 이상 존재할 수 없다는 것을 의미하거나 전역 최소값이 반드시 존재한다는 것을 의미하지는 않는다는 점에 유의하십시오. 예를 들어, 함수 $f(x) = \mathrm{max}(|x|-1, 0)$은 구간 $[-1, 1]$에서 최소값을 달성합니다. 반대로 함수 $f(x) = \exp(x)$는 $\mathbb{R}$에서 최소값을 달성하지 않습니다: $x \to -\infty$에 대해 $0$으로 점근하지만, $f(x) = 0$이 되는 $x$는 없습니다.

### 볼록 함수의 하위 집합은 볼록하다 (Below Sets of Convex Functions Are Convex)

우리는 볼록 함수의 *하위 집합(below sets)*을 통해 
볼록 집합을 편리하게 정의할 수 있습니다. 
구체적으로, 
볼록 집합 $\mathcal{X}$에서 정의된 볼록 함수 $f$가 주어졌을 때, 임의의 하위 집합

$$\mathcal{S}_b \stackrel{\textrm{def}}{=} \{x | x \in \mathcal{X} \textrm{ 및 } f(x) \leq b\}$$ 

은 볼록합니다. 

이를 빠르게 증명해 봅시다. 임의의 $x, x' \in \mathcal{S}_b$에 대해 $\lambda \in [0, 1]$인 한 $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$임을 보여야 합니다. 
$f(x) \leq b$ 및 $f(x') \leq b$이므로, 볼록성의 정의에 의해 우리는 다음을 갖습니다.

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$ 



### 볼록성과 2계 도함수 (Convexity and Second Derivatives)

함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 2계 도함수가 존재할 때마다 $f$가 볼록한지 확인하는 것은 매우 쉽습니다. 
우리가 해야 할 일은 $f$의 헤시안(Hessian)이 양의 반정부호(positive semidefinite)인지 확인하는 것입니다: $\nabla^2f \succeq 0$, 즉 
헤시안 행렬 $\nabla^2f$를 $\mathbf{H}$로 표시할 때, 
모든 $\mathbf{x} \in \mathbb{R}^n$에 대해 
$\mathbf{x}^\top \mathbf{H}\mathbf{x} \geq 0$입니다. 
예를 들어, 함수 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$은 $\nabla^2 f = \mathbf{1}$, 즉 그 헤시안이 단위 행렬이므로 볼록합니다.


공식적으로, 두 번 미분 가능한 1차원 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$은 
그 2계 도함수 $f'' \geq 0$일 때만 볼록합니다. 임의의 두 번 미분 가능한 다차원 함수 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$에 대해, 
그 헤시안 $\nabla^2f \succeq 0$일 때만 볼록합니다.

먼저 1차원 사례를 증명해야 합니다. 
$f$의 볼록성이 $f'' \geq 0$을 의미한다는 것을 보기 위해 우리는 다음 사실을 사용합니다.

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$ 

2계 도함수가 유한 차분에 대한 극한으로 주어지므로 다음이 따릅니다.

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$ 

$f'' \geq 0$이 $f$가 볼록함을 의미한다는 것을 보기 위해 $f'' \geq 0$이 $f'$이 단조 비감소 함수임을 의미한다는 사실을 사용합니다. $a < x < b$를 $\mathbb{R}$의 세 점이라고 합시다. 
여기서 $x = (1-\lambda)a + \lambda b$이고 $\lambda \in (0, 1)$입니다. 
평균값 정리에 따라, 다음을 만족하는 $\alpha \in [a, x]$ 및 $\beta \in [x, b]$가 존재합니다.

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \textrm{ 및 } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$ 


단조성에 의해 $f'(\beta) \geq f'(\alpha)$이므로,

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$ 

$x = (1-\lambda)a + \lambda b$이므로, 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$ 

이로써 볼록성이 증명됩니다.

둘째, 다차원 사례를 증명하기 전에 보조 정리가 필요합니다: 
$f: \mathbb{R}^n \rightarrow \mathbb{R}$은 
모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해 

$$g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \textrm{ where } z \in [0,1]$$ 

이 볼록할 때만 볼록합니다.

$f$의 볼록성이 $g$가 볼록함을 의미한다는 것을 증명하기 위해, 모든 $a, b, \lambda \in [0, 1]$(따라서 $0 \leq \lambda a + (1-\lambda) b \leq 1$)에 대해 다음을 보일 수 있습니다.

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\
=&f\left(\((\lambda a + (1-\lambda) b)\)\mathbf{x} + \(1-\lambda a - (1-\lambda) b)\)\mathbf{y} \right)\
=&f\left(\lambda \(a \mathbf{x} + (1-a)  \mathbf{y}\)  + (1-\lambda) \(b \mathbf{x} + (1-b)  \mathbf{y}\) \right)\
\leq& \lambda f\(a \mathbf{x} + (1-a)  \mathbf{y}\)  + (1-\lambda) f\(b \mathbf{x} + (1-b)  \mathbf{y}\) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$ 

역을 증명하기 위해, 모든 $\lambda \in [0, 1]$에 대해 다음을 보일 수 있습니다.

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$ 


마지막으로, 위의 보조 정리와 1차원 사례의 결과를 사용하여 다차원 사례를 다음과 같이 증명할 수 있습니다. 
다차원 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$은 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해 $z \in [0,1]$인 $g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$가 볼록할 때만 볼록합니다. 
1차원 사례에 따르면, 이는 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해 $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$($\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2f$)일 때만 성립하며, 이는 양의 반정부호 행렬의 정의에 따라 $\mathbf{H} \succeq 0$과 동등합니다.


## 제약 조건 (Constraints)

볼록 최적화의 좋은 속성 중 하나는 제약 조건을 효율적으로 처리할 수 있게 해 준다는 것입니다. 즉, 다음과 같은 형태의 *제약 조건이 있는 최적화(constrained optimization)* 문제를 풀 수 있게 해 줍니다.

$$\begin{aligned} \mathop{\textrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \textrm{ subject to } & c_i(\mathbf{x}) \leq 0 \textrm{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$ 

여기서 $f$는 목적 함수이고 함수 $c_i$는 제약 함수입니다. 이것이 무엇을 하는지 보려면 $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$인 경우를 고려해 보십시오. 이 경우 파라미터 $\mathbf{x}$는 단위 공(unit ball)으로 제약됩니다. 두 번째 제약 조건이 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$라면, 이는 반공간(half-space)에 놓인 모든 $\mathbf{x}$에 대응합니다. 두 제약 조건을 동시에 만족하는 것은 공의 한 조각을 선택하는 것과 같습니다.

### 라그랑지안 (Lagrangian)

일반적으로 제약 조건이 있는 최적화 문제를 푸는 것은 어렵습니다. 이를 다루는 한 가지 방법은 다소 간단한 직관을 가진 물리학에서 비롯되었습니다. 상자 안에 공이 있다고 상상해 보십시오. 공은 가장 낮은 곳으로 굴러갈 것이고 중력의 힘은 상자의 측면이 공에 가할 수 있는 힘과 균형을 이룰 것입니다. 요컨대, 목적 함수의 기울기(즉, 중력)는 제약 함수의 기울기(벽이 "밀어내는" 힘에 의해 공이 상자 안에 머물러야 함)에 의해 상쇄될 것입니다. 
일부 제약 조건은 활성화되지 않을 수 있음에 유의하십시오: 
공에 닿지 않는 벽은 공에 어떤 힘도 가할 수 없습니다.


라그랑지안 $L$의 유도를 건너뛰고, 위의 추론은 
다음과 같은 안장점 최적화 문제로 표현될 수 있습니다.

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \textrm{ where } \alpha_i \geq 0.$$ 

여기서 변수 $\alpha_i$ ($i=1,\ldots,n$)는 제약 조건이 적절하게 적용되도록 보장하는 이른바 *라그랑주 승수(Lagrange multipliers)*입니다. 이들은 $c_i(\mathbf{x}) \leq 0$이 모든 $i$에 대해 보장되도록 충분히 크게 선택됩니다. 예를 들어, 자연스럽게 $c_i(\mathbf{x}) < 0$인 임의의 $\mathbf{x}$에 대해 우리는 결국 $\alpha_i = 0$을 선택하게 될 것입니다. 더욱이, 이것은 모든 $\alpha_i$에 대해 $L$을 *최대화*하고 동시에 $\mathbf{x}$에 대해 $L$을 *최소화*하려는 안장점 최적화 문제입니다. 함수 $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$에 어떻게 도달하는지 설명하는 풍부한 문헌이 있습니다. 우리의 목적을 위해서는 $L$의 안장점이 원래의 제약 조건이 있는 최적화 문제가 최적으로 해결되는 지점이라는 것을 아는 것으로 충분합니다.

### 페널티 (Penalties)

제약 조건이 있는 최적화 문제를 적어도 *근사적으로* 만족시키는 한 가지 방법은 라그랑지안 $L$을 조정하는 것입니다. 
$c_i(\mathbf{x}) \leq 0$을 만족시키는 대신 단순히 목적 함수 $f(x)$에 $\alpha_i c_i(\mathbf{x})$를 더합니다. 이는 제약 조건이 너무 심하게 위반되지 않도록 보장합니다.

사실, 우리는 줄곧 이 트릭을 사용해 왔습니다. :numref:`sec_weight_decay`의 가중치 감쇠(weight decay)를 고려해 보십시오. 여기서 우리는 $\mathbf{w}$가 너무 크게 자라지 않도록 보장하기 위해 목적 함수에 $\frac{\lambda}{2} \|\mathbf{w}\|^2$를 더합니다. 제약 조건이 있는 최적화 관점에서 보면, 이것이 어떤 반지름 $r$에 대해 $\|\mathbf{w}\|^2 - r^2 \leq 0$을 보장할 것임을 알 수 있습니다. $\lambda$ 값을 조정하면 $\mathbf{w}$의 크기를 변경할 수 있습니다.

일반적으로 페널티를 추가하는 것은 근사적인 제약 조건 만족을 보장하는 좋은 방법입니다. 실전에서 이것은 정확한 만족보다 훨씬 더 견고한 것으로 밝혀졌습니다. 더욱이, 비볼록 문제의 경우 볼록 사례에서 정확한 접근 방식을 매력적으로 만들었던 많은 속성(예: 최적성)이 더 이상 유지되지 않습니다.

### 투영 (Projections)

제약 조건을 만족시키기 위한 대안 전략은 투영(projections)입니다. 다시 말하지만, 우리는 이전에 이를 접했습니다. 예를 들어 :numref:`sec_rnn-scratch`에서 기울기 클리핑(gradient clipping)을 다룰 때입니다. 거기서 우리는 다음을 통해 기울기의 길이가 $\theta$로 제한되도록 보장했습니다.

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$ 

이것은 기울기 $\mathbf{g}$를 반지름 $\theta$의 공 위로 *투영*한 것으로 밝혀졌습니다. 더 일반적으로, 볼록 집합 $\mathcal{X}$에 대한 투영은 다음과 같이 정의됩니다.

$$\textrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$ 

이는 $\mathcal{X}$에서 $\mathbf{x}$와 가장 가까운 점입니다. 

![볼록 투영.](../img/projections.svg)
:label:`fig_projections`

투영의 수학적 정의는 약간 추상적으로 들릴 수 있습니다. :numref:`fig_projections`은 이를 좀 더 명확하게 설명합니다. 여기에는 원과 다이아몬드라는 두 개의 볼록 집합이 있습니다. 
두 집합 내부의 점(노란색)은 투영 중에 변경되지 않고 그대로 유지됩니다. 
두 집합 외부의 점(검은색)은 원래 점(검은색)과 가장 가까운 집합 내부의 점(빨간색)으로 투영됩니다. 
$\\ell_2$ 공의 경우 이것이 방향을 바꾸지 않고 유지하지만, 다이아몬드의 사례에서 볼 수 있듯이 일반적으로는 그렇지 않을 수도 있습니다.


볼록 투영의 용도 중 하나는 희소 가중치 벡터(sparse weight vectors)를 계산하는 것입니다. 이 경우 가중치 벡터를 $\\ell_1$ 공 위로 투영하는데, 
이는 :numref:`fig_projections`의 다이아몬드 사례의 일반화된 버전입니다.


## 요약 (Summary)

딥러닝의 맥락에서 볼록 함수의 주요 목적은 최적화 알고리즘에 동기를 부여하고 이를 자세히 이해하도록 돕는 것입니다. 다음에서는 경사 하강법과 확률적 경사 하강법이 그에 따라 어떻게 유도될 수 있는지 볼 것입니다.


* 볼록 집합의 교집합은 볼록합니다. 합집합은 그렇지 않습니다.
* 볼록 함수의 기대값은 기대값의 볼록 함수보다 작지 않습니다(젠센 부등식).
* 두 번 미분 가능한 함수는 그 헤시안(2계 도함수 행렬)이 양의 반정부호일 때만 볼록합니다.
* 볼록 제약 조건은 라그랑지안을 통해 추가될 수 있습니다. 실전에서는 단순히 목적 함수에 페널티와 함께 추가할 수 있습니다.
* 투영은 볼록 집합에서 원래 점과 가장 가까운 점으로 매핑합니다.

## 연습 문제 (Exercises)

1. 집합 내의 점들 사이의 모든 선을 긋고 그 선들이 포함되는지 확인함으로써 집합의 볼록성을 확인하고 싶다고 가정합니다.
    1. 경계에 있는 점들만 확인하는 것으로 충분함을 증명하십시오.
    2. 집합의 정점(vertices)들만 확인하는 것으로 충분함을 증명하십시오.
2. $\mathcal{B}_p[r] \stackrel{\textrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ 및 } \|\mathbf{x}\|_p \leq r\}$를 $p$-노름을 사용한 반지름 $r$의 공이라고 합시다. 모든 $p \geq 1$에 대해 $\mathcal{B}_p[r]$이 볼록함을 증명하십시오.
3. 볼록 함수 $f$와 $g$가 주어졌을 때, $\mathrm{max}(f, g)$도 볼록함을 보이십시오. $\mathrm{min}(f, g)$는 볼록하지 않음을 증명하십시오.
4. 소프트맥스 함수의 정규화가 볼록함을 증명하십시오. 더 구체적으로 $f(x) = \log \sum_i \exp(x_i)$의 볼록성을 증명하십시오.
5. 선형 부분 공간, 즉 $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$가 볼록 집합임을 증명하십시오.
6. $\mathbf{b} = \mathbf{0}$인 선형 부분 공간의 경우 투영 $\textrm{Proj}_\mathcal{X}$가 어떤 행렬 $\mathbf{M}$에 대해 $\mathbf{M} \mathbf{x}$로 쓰일 수 있음을 증명하십시오.
7. 두 번 미분 가능한 볼록 함수 $f$에 대해 어떤 $\xi \in [0, \epsilon]$에 대해 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''((x + \xi)$로 쓸 수 있음을 보이십시오.
8. 볼록 집합 $\mathcal{X}$와 두 벡터 $\mathbf{x}$ 및 $\mathbf{y}$가 주어졌을 때, 투영이 거리를 결코 증가시키지 않음을 증명하십시오. 즉, $\|\mathbf{x} - \mathbf{y}\| \geq \|\textrm{Proj}_\mathcal{X}(\mathbf{x}) - \textrm{Proj}_\mathcal{X}(\mathbf{y})\|$입니다.


[Discussions](https://discuss.d2l.ai/t/350)