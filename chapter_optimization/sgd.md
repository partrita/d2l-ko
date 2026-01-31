# 확률적 경사 하강법 (Stochastic Gradient Descent)
:label:`sec_sgd`

이전 장에서 우리는 훈련 절차에서 확률적 경사 하강법을 계속 사용했지만, 이것이 왜 작동하는지 설명하지 않았습니다. 
이에 대해 설명하기 위해 
우리는 :numref:`sec_gd`에서 경사 하강법의 기본 원리를 설명했습니다. 
이 섹션에서는 계속해서 
*확률적 경사 하강법(stochastic gradient descent)*에 대해 더 자세히 논의합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 확률적 경사 업데이트 (Stochastic Gradient Updates)

딥러닝에서 목적 함수는 일반적으로 훈련 데이터셋의 각 예제에 대한 손실 함수의 평균입니다. 
$n$개의 예제로 구성된 훈련 데이터셋이 주어졌을 때, 
우리는 $f_i(\mathbf{x})$를 인덱스 $i$의 훈련 예제에 대한 손실 함수라고 가정합니다. 
여기서 $\mathbf{x}$는 파라미터 벡터입니다. 
그러면 우리는 다음과 같은 목적 함수에 도달합니다.

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$에서의 목적 함수의 기울기는 다음과 같이 계산됩니다.

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

경사 하강법을 사용하는 경우, 각 독립 변수 반복에 대한 계산 비용은 $\mathcal{O}(n)$이며, 이는 $n$에 따라 선형적으로 증가합니다. 따라서 훈련 데이터셋이 클수록 반복당 경사 하강법 비용이 더 높아집니다.

확률적 경사 하강법(SGD)은 각 반복에서의 계산 비용을 줄입니다. 확률적 경사 하강법의 각 반복에서 우리는 데이터 예제에 대해 인덱스 $i\in\{1,\ldots, n\}$를 무작위로 균일하게 샘플링하고 $\mathbf{x}$를 업데이트하기 위해 기울기 $\nabla f_i(\mathbf{x})$를 계산합니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

여기서 $\eta$는 학습률입니다. 각 반복에 대한 계산 비용이 경사 하강법의 $\mathcal{O}(n)$에서 상수 $\mathcal{O}(1)$로 떨어지는 것을 볼 수 있습니다. 또한 확률적 기울기 $\nabla f_i(\mathbf{x})$는 전체 기울기 $\nabla f(\mathbf{x})$의 편향되지 않은 추정치(unbiased estimate)임을 강조하고 싶습니다.

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

이는 평균적으로 확률적 기울기가 기울기의 좋은 추정치임을 의미합니다.

이제 확률적 경사 하강법을 시뮬레이션하기 위해 기울기에 평균이 0이고 분산이 1인 무작위 노이즈를 추가하여 경사 하강법과 비교할 것입니다.

```{.python .input}
#@tab all
def f(x1, x2):  # 목적 함수
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # 목적 함수의 기울기
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 노이즈가 있는 기울기 시뮬레이션
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 노이즈가 있는 기울기 시뮬레이션
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 노이즈가 있는 기울기 시뮬레이션
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # 일정한 학습률
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

보시다시피 확률적 경사 하강법에서 변수의 궤적은 :numref:`sec_gd`의 경사 하강법에서 관찰한 것보다 훨씬 더 시끄럽습니다. 이는 기울기의 확률적 특성 때문입니다. 즉, 최소값 근처에 도착하더라도 $\eta \nabla f_i(\mathbf{x})$를 통해 주입된 순간적인 기울기의 불확실성에 여전히 영향을 받습니다. 50단계 후에도 품질은 여전히 좋지 않습니다. 설상가상으로 추가 단계 후에도 개선되지 않습니다(이를 확인하기 위해 더 많은 단계로 실험해 보시기 바랍니다). 이것은 우리에게 유일한 대안을 남깁니다: 학습률 $\eta$를 변경하는 것입니다. 그러나 이것을 너무 작게 선택하면 초기에 의미 있는 진전을 이루지 못할 것입니다. 반면에 너무 크게 선택하면 위에서 본 것처럼 좋은 솔루션을 얻지 못할 것입니다. 이러한 상충되는 목표를 해결하는 유일한 방법은 최적화가 진행됨에 따라 학습률을 *동적으로* 줄이는 것입니다.

이것이 `sgd` 단계 함수에 학습률 함수 `lr`을 추가하는 이유이기도 합니다. 위의 예에서 `lr` 함수를 상수로 설정했기 때문에 학습률 스케줄링을 위한 기능은 휴면 상태입니다.

## 동적 학습률 (Dynamic Learning Rate)

$\eta$를 시간 의존적 학습률 $\eta(t)$로 대체하면 최적화 알고리즘의 수렴 제어 복잡성이 증가합니다. 특히 $\eta$가 얼마나 빨리 감소해야 하는지 파악해야 합니다. 너무 빠르면 조기에 최적화를 중단하게 됩니다. 너무 천천히 줄이면 최적화에 너무 많은 시간을 낭비하게 됩니다. 다음은 시간에 따라 $\eta$를 조정하는 데 사용되는 몇 가지 기본 전략입니다(나중에 고급 전략에 대해 논의할 것입니다).

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

첫 번째 *구간별 상수(piecewise constant)* 시나리오에서는 예를 들어 최적화 진전이 멈출 때마다 학습률을 줄입니다. 이는 딥 네트워크를 훈련하기 위한 일반적인 전략입니다. 대안으로 *지수 감쇠(exponential decay)*를 통해 훨씬 더 공격적으로 줄일 수 있습니다. 불행히도 이는 종종 알고리즘이 수렴하기 전에 조기 중지로 이어집니다. 인기 있는 선택은 $\alpha = 0.5$인 *다항식 감쇠(polynomial decay)*입니다. 볼록 최적화의 경우 이 속도가 잘 작동한다는 것을 보여주는 여러 증명이 있습니다.

지수 감쇠가 실제로 어떻게 보이는지 봅시다.

```{.python .input}
#@tab all
def exponential_lr():
    # 이 함수 외부에서 정의되고 내부에서 업데이트되는 전역 변수
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

예상대로 파라미터의 분산이 상당히 줄어듭니다. 그러나 이는 최적의 해 $\mathbf{x} = (0, 0)$으로 수렴하지 못하는 비용을 치르게 됩니다. 1000번의 반복 단계 후에도 우리는 여전히 최적의 해에서 매우 멀리 떨어져 있습니다. 실제로 알고리즘은 전혀 수렴하지 못합니다. 반면 학습률이 단계 수의 역제곱근으로 감소하는 다항식 감쇠를 사용하면 50단계 만에 수렴이 더 좋아집니다.

```{.python .input}
#@tab all
def polynomial_lr():
    # 이 함수 외부에서 정의되고 내부에서 업데이트되는 전역 변수
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

학습률을 설정하는 방법에 대한 더 많은 선택지가 존재합니다. 예를 들어, 작은 속도로 시작한 다음 급격히 높이고 다시 천천히 줄일 수 있습니다. 더 작은 학습률과 더 큰 학습률을 번갈아 사용할 수도 있습니다. 그러한 스케줄은 매우 다양합니다. 지금은 포괄적인 이론적 분석이 가능한 학습률 스케줄, 즉 볼록 설정에서의 학습률에 초점을 맞춰 보겠습니다. 일반적인 비볼록 문제의 경우 의미 있는 수렴 보장을 얻기가 매우 어렵습니다. 일반적으로 비선형 비볼록 문제를 최소화하는 것은 NP hard이기 때문입니다. 설문 조사는 예: Tibshirani 2015의 훌륭한 [강의 노트](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf)를 참조하십시오.



## 볼록 목적 함수에 대한 수렴 분석 (Convergence Analysis for Convex Objectives)

볼록 목적 함수에 대한 확률적 경사 하강법의 다음 수렴 분석은 선택 사항이며 주로 문제에 대한 더 많은 직관을 전달하는 역할을 합니다. 
우리는 가장 간단한 증명 중 하나로 제한합니다 :cite:`Nesterov.Vial.2000`. 
예를 들어 목적 함수가 특히 잘 작동할 때 훨씬 더 발전된 증명 기술이 존재합니다.


목적 함수 $f(\boldsymbol{\xi}, \mathbf{x})$가 모든 $\boldsymbol{\xi}$에 대해 $\mathbf{x}$에서 볼록하다고 가정합니다. 
더 구체적으로, 
우리는 확률적 경사 하강법 업데이트를 고려합니다:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

여기서 $f(\boldsymbol{\xi}_t, \mathbf{x})$는 
단계 $t$에서 어떤 분포로부터 추출된 훈련 예제 $\boldsymbol{\xi}_t$에 대한 목적 함수이고 
$\mathbf{x}$는 모델 파라미터입니다. 
다음을 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

기대 위험(expected risk)으로, $R^*$를 $\mathbf{x}$에 대한 최소값으로 표시합니다. 마지막으로 $\mathbf{x}^*$를 최소화자(minimizer)라고 합시다($\mathbf{x}$가 정의된 도메인 내에 존재한다고 가정). 이 경우 시간 $t$에서의 현재 파라미터 $\mathbf{x}_t$와 위험 최소화자 $\mathbf{x}^*$ 사이의 거리를 추적하고 시간이 지남에 따라 개선되는지 확인할 수 있습니다.

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

우리는 확률적 기울기 $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$의 $\ell_2$ 노름이 어떤 상수 $L$에 의해 제한된다고 가정하므로 다음을 갖습니다.

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


우리는 $\mathbf{x}_t$와 $\mathbf{x}^*$ 사이의 거리가 *기대치에서* 어떻게 변하는지에 가장 관심이 있습니다. 사실 어떤 특정 단계 시퀀스에 대해서는 우리가 마주치는 $\boldsymbol{\xi}_t$에 따라 거리가 증가할 수도 있습니다. 따라서 내적을 제한해야 합니다. 
어떤 볼록 함수 $f$에 대해서도 모든 $\mathbf{x}$와 $\mathbf{y}$에 대해 
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$이 성립하므로, 
볼록성에 의해 우리는 다음을 갖습니다.

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

부등식 :eqref:`eq_sgd-L`과 :eqref:`eq_sgd-f-xi-xstar`를 :eqref:`eq_sgd-xt+1-xstar`에 대입하면 다음과 같이 시간 $t+1$에서의 파라미터 간 거리에 대한 경계를 얻습니다.

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

이것은 현재 손실과 최적 손실 간의 차이가 $\eta_t L^2/2$보다 큰 한 진전을 이룬다는 것을 의미합니다. 이 차이는 0으로 수렴해야 하므로 학습률 $\eta_t$도 *사라져야* 합니다.

다음으로 :eqref:`eqref_sgd-xt-diff`에 대해 기대값을 취합니다. 이것은 다음을 산출합니다.

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

마지막 단계는 $t \in \{1, \ldots, T\}$에 대한 부등식을 합산하는 것을 포함합니다. 합이 텔레스코핑(telescoping)되고 낮은 항을 삭제함으로써 다음을 얻습니다.

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

우리는 $\mathbf{x}_1$이 주어졌으므로 기대값을 삭제할 수 있다는 점을 이용했습니다. 마지막으로 다음을 정의합니다.

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

다음이 성립하므로

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

젠센 부등식(:eqref:`eq_jensens-inequality`에서 $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$로 설정)과 $R$의 볼록성에 의해 $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$가 따르며, 따라서

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

이를 부등식 :eqref:`eq_sgd-x1-xstar`에 대입하면 경계를 얻습니다.

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

여기서 $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$는 초기 파라미터 선택과 최종 결과 간의 거리에 대한 경계입니다. 요컨대, 수렴 속도는 
확률적 기울기의 노름이 어떻게 제한되는지($L$)와 초기 파라미터 값이 최적성에서 얼마나 멀리 떨어져 있는지($r$)에 따라 달라집니다. 경계는 $\mathbf{x}_T$가 아니라 $\bar{\mathbf{x}}$에 대한 것임에 유의하십시오. $\bar{\mathbf{x}}$는 최적화 경로의 평활화된 버전이기 때문입니다. 
$r, L, T$를 알고 있다면 학습률 $\eta = r/(L \sqrt{T})$를 선택할 수 있습니다. 이것은 상한 $rL/\sqrt{T}$를 산출합니다. 즉, 우리는 $\mathcal{O}(1/\sqrt{T})$의 속도로 최적의 해에 수렴합니다.





## 확률적 기울기와 유한 샘플 (Stochastic Gradients and Finite Samples)

지금까지 우리는 확률적 경사 하강법에 대해 이야기할 때 약간 느슨하게 다루었습니다. 우리는 어떤 분포 $p(x, y)$에서 인스턴스 $x_i$, 일반적으로 레이블 $y_i$를 추출하고 이를 사용하여 모델 파라미터를 어떤 방식으로 업데이트한다고 가정했습니다. 특히 유한 샘플 크기에 대해 우리는 단순히 이산 분포 $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ 
(어떤 함수 $\delta_{x_i}$ 및 $\delta_{y_i}$에 대해)가 
그 위에서 확률적 경사 하강법을 수행할 수 있게 한다고 주장했습니다.

그러나 이것은 우리가 실제로 한 일이 아닙니다. 현재 섹션의 장난감 예제에서는 단순히 확률적이지 않은 기울기에 노이즈를 추가했습니다. 즉, 쌍 $(x_i, y_i)$가 있는 척했습니다. 여기서 이것이 정당하다는 것이 밝혀졌습니다(자세한 논의는 연습 문제 참조). 더 골치 아픈 것은 이전의 모든 논의에서 우리가 분명히 이렇게 하지 않았다는 것입니다. 대신 우리는 모든 인스턴스를 *정확히 한 번* 반복했습니다. 이것이 왜 바람직한지 보려면 반대의 경우, 즉 이산 분포에서 *복원 추출(replacement)*로 $n$개의 관찰을 샘플링한다고 생각해 보십시오. 무작위로 요소 $i$를 선택할 확률은 $1/n$입니다. 따라서 그것을 *적어도* 한 번 선택할 확률은 다음과 같습니다.

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

유사한 추론은 어떤 샘플(즉, 훈련 예제)을 *정확히 한 번* 선택할 확률이 다음과 같이 주어짐을 보여줍니다.

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

복원 추출로 샘플링하면 *비복원 추출(without replacement)*로 샘플링하는 것에 비해 분산이 증가하고 데이터 효율성이 감소합니다. 따라서 실제로는 후자를 수행합니다(그리고 이것이 이 책 전체의 기본 선택입니다). 마지막으로 훈련 데이터셋을 통한 반복적인 패스는 *다른* 무작위 순서로 순회한다는 점에 유의하십시오.


## 요약 (Summary)

* 볼록 문제의 경우 광범위한 학습률 선택에 대해 확률적 경사 하강법이 최적의 해로 수렴함을 증명할 수 있습니다.
* 딥러닝의 경우 일반적으로 그렇지 않습니다. 그러나 볼록 문제 분석은 최적화에 접근하는 방법, 즉 학습률을 점진적으로 줄이되 너무 빨리 줄이지 않는 방법에 대한 유용한 통찰력을 제공합니다.
* 학습률이 너무 작거나 너무 클 때 문제가 발생합니다. 실제로 적절한 학습률은 종종 여러 번의 실험 후에야 발견됩니다.
* 훈련 데이터셋에 더 많은 예제가 있는 경우 경사 하강법의 각 반복을 계산하는 데 더 많은 비용이 들기 때문에 이 경우 확률적 경사 하강법이 선호됩니다.
* 확률적 경사 하강법에 대한 최적성 보장은 비볼록 사례에서 일반적으로 사용할 수 없습니다. 확인해야 할 국소 최소값의 수가 기하급수적일 수 있기 때문입니다.




## 연습 문제 (Exercises)

1. 확률적 경사 하강법에 대해 다양한 학습률 스케줄과 다양한 반복 횟수로 실험해 보십시오. 특히 반복 횟수의 함수로 최적의 해 $(0, 0)$으로부터의 거리를 플롯하십시오.
2. 함수 $f(x_1, x_2) = x_1^2 + 2 x_2^2$에 대해 기울기에 정규 노이즈를 추가하는 것은 $\mathbf{x}$가 정규 분포에서 추출되는 손실 함수 $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$를 최소화하는 것과 동등함을 증명하십시오.
3. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$에서 복원 추출로 샘플링할 때와 비복원 추출로 샘플링할 때 확률적 경사 하강법의 수렴을 비교하십시오.
4. 어떤 기울기(또는 그와 관련된 어떤 좌표)가 다른 모든 기울기보다 일관되게 크다면 확률적 경사 하강법 솔버를 어떻게 변경하겠습니까?
5. $f(x) = x^2 (1 + \sin x)$라고 가정합니다. $f$는 몇 개의 국소 최소값을 가집니까? $f$를 최소화하기 위해 모든 국소 최소값을 평가해야 하도록 $f$를 변경할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab: