# 경사 하강법 (Gradient Descent)
:label:`sec_gd`

이 섹션에서는 *경사 하강법(gradient descent)*의 기초가 되는 기본 개념을 소개할 것입니다. 
딥러닝에서 직접 사용되는 경우는 드물지만, 경사 하강법을 이해하는 것은 확률적 경사 하강법 알고리즘을 이해하는 데 핵심입니다. 
예를 들어, 최적화 문제는 지나치게 큰 학습률로 인해 발산할 수 있습니다. 이 현상은 이미 경사 하강법에서 볼 수 있습니다. 마찬가지로, 프리컨디셔닝(preconditioning)은 경사 하강법의 일반적인 기술이며 더 발전된 알고리즘으로 이어집니다. 
간단한 특수 사례부터 시작해 봅시다.


## 1차원 경사 하강법 (One-Dimensional Gradient Descent)

1차원 경사 하강법은 왜 경사 하강법 알고리즘이 목적 함수의 값을 줄일 수 있는지 설명하는 훌륭한 예입니다. 어떤 연속 미분 가능한 실수 값 함수 $f: ℝ> ℝ$을 고려해 보십시오. 테일러 전개를 사용하면 다음을 얻습니다.

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$ 
:eqlabel:`gd-taylor`

즉, 1차 근사에서 $f(x+\epsilon)$은 $x$에서의 함수 값 $f(x)$와 1계 도함수 $f'(x)$에 의해 주어집니다. 작은 $\epsilon$에 대해 음의 기울기 방향으로 이동하면 $f$가 감소할 것이라고 가정하는 것은 무리가 아닙니다. 단순함을 위해 고정된 단계 크기 $\eta > 0$을 선택하고 $\epsilon = -\eta f'(x)$를 선택합니다. 이를 위의 테일러 전개에 대입하면 다음을 얻습니다.

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$ 
:eqlabel:`gd-taylor-2`

도함수 $f'(x) \neq 0$이 사라지지 않으면 $\eta f'^2(x)>0$이므로 진전을 이룹니다. 게다가, 우리는 항상 고차 항이 무관해질 만큼 충분히 작은 $\eta$를 선택할 수 있습니다. 따라서 다음과 같이 도달합니다.

$$f(x - \eta f'(x)) \lessapprox f(x).$$ 

이것은 우리가

$$x \leftarrow x - \eta f'(x)$$

를 사용하여 $x$를 반복하면 함수 $f(x)$의 값이 감소할 수 있음을 의미합니다. 따라서 경사 하강법에서는 먼저 초기 값 $x$와 상수 $\eta > 0$을 선택한 다음, 중지 조건(예: 기울기 $|f'(x)|$의 크기가 충분히 작아지거나 반복 횟수가 특정 값에 도달했을 때)에 도달할 때까지 이를 사용하여 $x$를 계속 반복합니다.

단순함을 위해 목적 함수 $f(x)=x^2$를 선택하여 경사 하강법을 구현하는 방법을 설명합니다. $x=0$이 $f(x)$를 최소화하는 해임을 알고 있지만, 여전히 이 단순한 함수를 사용하여 $x$가 어떻게 변하는지 관찰합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # 목적 함수
    return x ** 2

def f_grad(x):  # 목적 함수의 기울기(도함수)
    return 2 * x
```

다음으로, $x=10$을 초기 값으로 사용하고 $\eta=0.2$라고 가정합니다. 경사 하강법을 사용하여 $x$를 10번 반복하면 결국 $x$의 값이 최적의 해에 접근하는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$에 대한 최적화 과정은 다음과 같이 그릴 수 있습니다.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 학습률 (Learning Rate)
:label:`subsec_gd-learningrate`

학습률 $\eta$는 알고리즘 설계자가 설정할 수 있습니다. 너무 작은 학습률을 사용하면 $x$가 매우 느리게 업데이트되어 더 나은 해를 얻기 위해 더 많은 반복이 필요합니다. 그런 경우에 어떤 일이 일어나는지 보여주기 위해, $\eta = 0.05$일 때 동일한 최적화 문제에서의 과정을 고려해 보십시오. 보시다시피 10단계 후에도 우리는 여전히 최적의 해에서 매우 멀리 떨어져 있습니다.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

반대로 지나치게 높은 학습률을 사용하면 $\left|\eta f'(x)\right|$가 1차 테일러 전개 공식에 비해 너무 클 수 있습니다. 즉, :eqref:`gd-taylor-2`의 항 $\mathcal{O}(\eta^2 f'^2(x))$가 중요해질 수 있습니다. 이 경우 $x$의 반복이 $f(x)$의 값을 낮출 수 있다고 보장할 수 없습니다. 예를 들어 학습률을 $\eta=1.1$로 설정하면 $x$가 최적의 해 $x=0$을 지나치고 점차 발산합니다.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 국소 최소값 (Local Minima)

비볼록(nonconvex) 함수의 경우 어떤 일이 일어나는지 설명하기 위해 어떤 상수 $c$에 대해 $f(x) = x \cdot \cos(cx)$인 경우를 고려해 보십시오. 이 함수는 무한히 많은 국소 최소값을 가집니다. 학습률 선택과 문제의 조건이 얼마나 좋은지에 따라 우리는 많은 해 중 하나에 도달할 수 있습니다. 아래 예제는 (비현실적으로) 높은 학습률이 좋지 않은 국소 최소값으로 어떻게 이어지는지 보여줍니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 목적 함수
    return x * d2l.cos(c * x)

def f_grad(x):  # 목적 함수의 기울기
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 다변량 경사 하강법 (Multivariate Gradient Descent)

이제 일변량 사례에 대해 더 나은 직관을 얻었으므로 $\mathbf{x} = [x_1, x_2, \ldots, x_d]^⊥$인 상황을 고려해 봅시다. 즉, 목적 함수 $f: ℝ>^d 	o ℝ$은 벡터를 스칼라로 매핑합니다. 그에 대응하여 기울기도 다변량입니다. 이는 $d$개의 편미분으로 구성된 벡터입니다.

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^⊥.$$

기울기의 각 편미분 요소 $\partial f(\mathbf{x})/\partial x_i$는 입력 $x_i$에 대한 $\mathbf{x}$에서의 $f$의 변화율을 나타냅니다. 이전 일변량 사례와 마찬가지로 다변량 함수에 대한 해당 테일러 근사를 사용하여 무엇을 해야 할지 감을 잡을 수 있습니다. 특히 다음을 얻습니다.

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$ 
:eqlabel:`gd-multi-taylor`

즉, $\boldsymbol{\epsilon}$의 2차 항까지 가장 가파른 하강 방향은 음의 기울기 $-\nabla f(\mathbf{x})$에 의해 주어집니다. 적절한 학습률 $\eta > 0$을 선택하면 전형적인 경사 하강법 알고리즘이 생성됩니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

알고리즘이 실제로 어떻게 작동하는지 확인하기 위해 2차원 벡터 $\mathbf{x} = [x_1, x_2]^⊥$를 입력으로 하고 스칼라를 출력으로 하는 목적 함수 $f(\mathbf{x})=x_1^2+2x_2^2$를 구성해 봅시다. 기울기는 $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^⊥$로 주어집니다. 초기 위치 $[-5, -2]$에서 경사 하강법에 의한 $\mathbf{x}$의 궤적을 관찰할 것입니다.

우선 두 개의 보조 함수가 더 필요합니다. 첫 번째는 업데이트 함수를 사용하여 초기 값에 20번 적용합니다. 두 번째 보조 함수는 $\mathbf{x}$의 궤적을 시각화합니다.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """맞춤형 트레이너로 2D 목적 함수를 최적화합니다."""
    # `s1` 및 `s2`는 모멘텀, adagrad, RMSProp에서 사용될 내부 상태 변수입니다
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """최적화 중 2D 변수의 궤적을 보여줍니다."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """최적화 중 2D 변수의 궤적을 보여줍니다."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """최적화 중 2D 변수의 궤적을 보여줍니다."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

다음으로 학습률 $\eta = 0.1$에 대한 최적화 변수 $\mathbf{x}$의 궤적을 관찰합니다. 20단계 후에 $\mathbf{x}$의 값이 $[0, 0]$의 최소값에 접근하는 것을 볼 수 있습니다. 진행은 다소 느리지만 상당히 잘 작동합니다.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # 목적 함수
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # 목적 함수의 기울기
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 적응형 방법 (Adaptive Methods)

:numref:`subsec_gd-learningrate`에서 볼 수 있듯이 학습률 $\eta$를 "딱 맞게" 맞추는 것은 까다롭습니다. 너무 작게 잡으면 진전이 거의 없습니다. 너무 크게 잡으면 해가 진동하고 최악의 경우 발산할 수도 있습니다. $\eta$를 자동으로 결정하거나 학습률을 전혀 선택할 필요가 없게 할 수 있다면 어떨까요?
목적 함수의 값과 기울기뿐만 아니라 그 *곡률(curvature)*까지 살펴보는 2차 방법이 이 경우에 도움이 될 수 있습니다. 이러한 방법은 계산 비용 때문에 딥러닝에 직접 적용할 수는 없지만, 아래에 설명된 알고리즘의 바람직한 속성을 많이 모방하는 고급 최적화 알고리즘을 설계하는 방법에 대한 유용한 직관을 제공합니다.


### 뉴턴 방법 (Newton's Method)

어떤 함수 $f: ℝ>^d 	o ℝ$의 테일러 전개를 검토해 보면 첫 번째 항 이후에 멈출 필요가 없습니다. 사실 다음과 같이 쓸 수 있습니다.

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$ 
:eqlabel:`gd-hot-taylor`

번거로운 표기법을 피하기 위해 $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$를 $f$의 헤시안(Hessian)으로 정의합니다. 이는 $d \times d$ 행렬입니다. 작은 $d$와 단순한 문제의 경우 $\mathbf{H}$는 계산하기 쉽습니다. 반면 심층 신경망의 경우 $\mathcal{O}(d^2)$ 항목을 저장하는 비용 때문에 $\mathbf{H}$가 엄청나게 클 수 있습니다. 더욱이 역전파를 통해 계산하기에 너무 비쌀 수 있습니다. 지금은 그러한 고려 사항을 무시하고 어떤 알고리즘을 얻게 될지 살펴봅시다.

결국 $f$의 최소값은 $\nabla f = 0$을 만족합니다.
:numref:`subsec_calculus-grad`의 미적분 규칙에 따라, $\boldsymbol{\epsilon}$에 대해 :eqref:`gd-hot-taylor`의 도함수를 취하고 고차 항을 무시하면 다음과 같이 도달합니다.

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ 이고 따라서 } 
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

즉, 최적화 문제의 일부로 헤시안 $\mathbf{H}$를 반전시켜야 합니다.

간단한 예로 $f(x) = \frac{1}{2} x^2$의 경우 $\nabla f(x) = x$ 및 $\mathbf{H} = 1$입니다. 따라서 모든 $x$에 대해 $\epsilon = -x$를 얻습니다. 즉, 조정할 필요 없이 *단 한 번의* 단계로 완벽하게 수렴하기에 충분합니다! 아쉽게도 여기서는 운이 좋았습니다. $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$이므로 테일러 전개가 정확했기 때문입니다.

다른 문제에서는 어떻게 되는지 봅시다.
어떤 상수 $c$에 대해 볼록 쌍곡선 코사인 함수 $f(x) = \cosh(cx)$가 주어졌을 때, $x=0$에서의 전역 최소값이 몇 번의 반복 후에 도달하는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # 목적 함수
    return d2l.cosh(c * x)

def f_grad(x):  # 목적 함수의 기울기
    return c * d2l.sinh(c * x)

def f_hess(x):  # 목적 함수의 헤시안
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

이제 $f(x) = x \cos(c x)$와 같은 *비볼록* 함수를 고려해 봅시다. 결국 뉴턴 방법에서는 헤시안으로 나누게 됩니다. 이는 2계 도함수가 *음수*이면 $f$의 값이 *증가*하는 방향으로 걸어갈 수 있음을 의미합니다.
그것은 알고리즘의 치명적인 결함입니다.
실제로 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 목적 함수
    return x * d2l.cos(c * x)

def f_grad(x):  # 목적 함수의 기울기
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # 목적 함수의 헤시안
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

이것은 아주 잘못되었습니다. 어떻게 고칠 수 있을까요? 한 가지 방법은 헤시안의 절댓값을 취하여 헤시안을 "수정"하는 것입니다. 또 다른 전략은 학습률을 다시 도입하는 것입니다. 이것은 목적을 달성하지 못하는 것처럼 보일 수 있지만 꼭 그렇지는 않습니다. 2차 정보를 가지면 곡률이 클 때마다 주의하고 목적 함수가 더 평평할 때마다 더 긴 단계를 밟을 수 있습니다.
학습률을 약간 작게 하여(예: $\eta = 0.5$) 어떻게 작동하는지 봅시다. 보시다시피 상당히 효율적인 알고리즘을 갖게 됩니다.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 수렴 분석 (Convergence Analysis)

우리는 2계 도함수가 0이 아닌($f''>0$) 어떤 볼록하고 세 번 미분 가능한 목적 함수 $f$에 대해서만 뉴턴 방법의 수렴 속도를 분석합니다. 다변량 증명은 아래의 1차원 주장을 직접적으로 확장한 것이며 직관 측면에서 큰 도움이 되지 않으므로 생략합니다.

$k^\textrm{th}$ 반복에서의 $x$ 값을 $x^{(k)}$라고 하고, $k^\textrm{th}$ 반복에서의 최적성으로부터의 거리를 $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$라고 합시다. 테일러 전개에 의해 조건 $f'(x^*) = 0$은 다음과 같이 쓰일 수 있습니다.

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

이는 어떤 $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$에 대해 성립합니다. 위의 전개를 $f''(x^{(k)})$로 나누면 다음을 얻습니다.

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$ 

우리는 업데이트 $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$를 가짐을 상기하십시오.
이 업데이트 방정식을 대입하고 양변의 절댓값을 취하면 다음을 얻습니다.

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$ 

결과적으로, 유계인 $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$인 영역에 있을 때마다 우리는 이차적으로 감소하는 오차를 갖습니다.

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


참고로, 최적화 연구자들은 이를 *선형(linear)* 수렴이라고 부르는 반면, $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$와 같은 조건은 *상수(constant)* 수렴 속도라고 부릅니다.
이 분석에는 몇 가지 주의 사항이 있습니다.
첫째, 우리는 언제 빠른 수렴 영역에 도달할지 정말로 알 수 없습니다. 대신 그곳에 도달하면 수렴이 매우 빠를 것이라는 것만 압니다. 둘째, 이 분석은 $f$가 고차 도함수까지 잘 작동할 것을 요구합니다. 이는 $f$가 값이 어떻게 변할 수 있는지 측면에서 "놀라운" 속성을 갖지 않도록 보장하는 것으로 귀결됩니다.



### 프리컨디셔닝 (Preconditioning)

전체 헤시안을 계산하고 저장하는 것이 매우 비싸다는 것은 그리 놀라운 일이 아닙니다. 따라서 대안을 찾는 것이 바람직합니다. 상황을 개선하는 한 가지 방법은 *프리컨디셔닝(preconditioning)*입니다. 헤시안 전체를 계산하는 것을 피하고 *대각* 항목만 계산합니다. 이는 다음과 같은 업데이트 알고리즘으로 이어집니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


이것이 전체 뉴턴 방법만큼 좋지는 않지만, 사용하지 않는 것보다는 훨씬 낫습니다.
왜 이것이 좋은 아이디어일 수 있는지 이해하기 위해 한 변수는 밀리미터 단위의 높이를 나타내고 다른 변수는 킬로미터 단위의 높이를 나타내는 상황을 고려해 보십시오. 두 변수 모두 자연스러운 척도가 미터라고 가정하면 파라미터화에서 끔찍한 불일치가 발생합니다. 다행히 프리컨디셔닝을 사용하면 이를 제거할 수 있습니다. 경사 하강법과 함께 프리컨디셔닝을 효과적으로 수행하는 것은 각 변수(벡터 $\mathbf{x}$의 좌표)에 대해 다른 학습률을 선택하는 것과 같습니다.
나중에 보게 되겠지만, 프리컨디셔닝은 확률적 경사 하강법 최적화 알고리즘의 혁신 중 일부를 주도합니다.


### 라인 검색을 사용한 경사 하강법 (Gradient Descent with Line Search)

경사 하강법의 주요 문제 중 하나는 목표를 지나치거나 진전이 불충분할 수 있다는 것입니다. 이 문제에 대한 간단한 수정은 경사 하강법과 함께 라인 검색(line search)을 사용하는 것입니다. 즉, $\nabla f(\mathbf{x})$에 의해 주어진 방향을 사용한 다음 어떤 학습률 $\eta$가 $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$를 최소화하는지에 대해 이진 검색을 수행합니다.

이 알고리즘은 빠르게 수렴합니다(분석 및 증명은 예: :citet:`Boyd.Vandenberghe.2004` 참조). 그러나 딥러닝의 목적을 위해 이것은 그다지 실용적이지 않습니다. 라인 검색의 각 단계마다 전체 데이터셋에서 목적 함수를 평가해야 하기 때문입니다. 이를 수행하기에는 비용이 너무 많이 듭니다.

## 요약 (Summary)

* 학습률은 중요합니다. 너무 크면 발산하고 너무 작으면 진전이 없습니다.
* 경사 하강법은 국소 최소값에 갇힐 수 있습니다.
* 고차원에서는 학습률을 조정하는 것이 복잡합니다.
* 프리컨디셔닝은 스케일 조정에 도움이 될 수 있습니다.
* 뉴턴 방법은 볼록 문제에서 제대로 작동하기 시작하면 훨씬 더 빠릅니다.
* 비볼록 문제에 대해 아무런 조정 없이 뉴턴 방법을 사용하는 것을 주의하십시오.

## 연습 문제 (Exercises)

1. 경사 하강법에 대해 다양한 학습률과 목적 함수로 실험해 보십시오.
2. 구간 $[a, b]$에서 볼록 함수를 최소화하기 위해 라인 검색을 구현하십시오.
    1. 이진 검색을 위해, 즉 $[a, (a+b)/2]$를 선택할지 $[(a+b)/2, b]$를 선택할지 결정하기 위해 도함수가 필요합니까?
    2. 알고리즘의 수렴 속도는 얼마나 빠릅니까?
    3. 알고리즘을 구현하고 $\log (\exp(x) + \exp(-2x -3))$를 최소화하는 데 적용하십시오.
3. 경사 하강법이 매우 느린 $\mathbb{R}^2$에 정의된 목적 함수를 설계하십시오. 힌트: 다른 좌표의 스케일을 다르게 조정하십시오.
4. 프리컨디셔닝을 사용하여 뉴턴 방법의 경량 버전을 구현하십시오.
    1. 대각 헤시안을 프리컨디셔너로 사용하십시오.
    2. 실제(잠재적으로 부호가 있는) 값보다는 그것의 절댓값을 사용하십시오.
    3. 위의 문제에 이를 적용하십시오.
5. 위의 알고리즘을 여러 목적 함수(볼록하거나 그렇지 않거나)에 적용하십시오. 좌표를 $45$도 회전하면 어떻게 됩니까?

[Discussions](https://discuss.d2l.ai/t/351)