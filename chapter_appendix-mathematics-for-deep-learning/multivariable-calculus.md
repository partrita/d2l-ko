# 다변수 미적분학 (Multivariable Calculus)
:label:`sec_multivariable_calculus`

이제 단일 변수 함수의 도함수에 대해 꽤 강력한 이해를 얻었으므로, 잠재적으로 수십억 개의 가중치를 가진 손실 함수를 고려하고 있었던 원래 질문으로 돌아가 봅시다.

## 고차원 미분 (Higher-Dimensional Differentiation)
:numref:`sec_single_variable_calculus`가 우리에게 알려주는 것은, 이 수십억 개의 가중치 중 단 하나를 바꾸고 다른 모든 것들은 고정해 둔다면 어떤 일이 일어날지 안다는 것입니다! 이것은 단일 변수 함수에 지나지 않으므로 다음과 같이 쓸 수 있습니다.

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$ 
:eqlabel:`eq_part_der`

다른 변수들을 고정하면서 한 변수에 대해 미분하는 것을 *편미분(partial derivative)*이라고 부르며, :eqref:`eq_part_der`에서의 도함수에 대해 $\frac{\partial}{\partial w_1}$ 표기법을 사용할 것입니다.

이제 이것을 가져와 $w_2$를 $w_2 + \epsilon_2$로 조금 바꿔봅시다.

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$ 

우리는 이전 섹션에서 $\epsilon^{2}$을 버릴 수 있었던 것과 같은 방식으로 $\epsilon_1\epsilon_2$가 버릴 수 있는 고차 항이라는 아이디어를 :eqref:`eq_part_der`에서 본 것과 함께 다시 사용했습니다. 이런 방식으로 계속하면 다음과 같이 쓸 수 있습니다.

$$ 
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$ 

이것이 엉망으로 보일 수 있지만, 우변의 합계가 정확히 내적처럼 보인다는 점에 주목하여 이를 더 친숙하게 만들 수 있습니다. 따라서 다음과 같이 둔다면

$$ 
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \textrm{및} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top, 
$$ 

다음과 같습니다.

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$ 
:eqlabel:`eq_nabla_use`

우리는 벡터 $\nabla_{\mathbf{w}} L$을 $L$의 *기울기(gradient)*라고 부를 것입니다.

방정식 :eqref:`eq_nabla_use`는 잠시 감상할 가치가 있습니다. 이것은 우리가 1차원에서 마주쳤던 것과 정확히 동일한 형식을 가지고 있으며, 단지 모든 것을 벡터와 내적으로 변환했을 뿐입니다. 이는 입력에 대한 임의의 섭동이 주어졌을 때 함수 $L$이 대략 어떻게 변할지 알려줍니다. 다음 섹션에서 보겠지만, 이는 우리가 기울기에 포함된 정보를 사용하여 어떻게 학습할 수 있는지를 기하학적으로 이해하는 데 중요한 도구를 제공할 것입니다.

하지만 먼저, 이 근사가 작동하는 것을 예제로 살펴봅시다. 다음과 같은 함수로 작업한다고 가정합시다.

$$ 
f(x, y) = \log(e^x + e^y) \; \textrm{이며 기울기는} \; \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right] \; \textrm{입니다.} $$ 

$(0, \log(2))$와 같은 점을 보면 다음을 알 수 있습니다.

$$ 
f(x, y) = \log(3) \; \textrm{이며 기울기는} \; \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right] \; \textrm{입니다.} $$ 

따라서 $(\epsilon_1, \log(2) + \epsilon_2)$에서 $f$를 근사하고 싶다면, :eqref:`eq_nabla_use`의 구체적인 사례를 가져야 함을 알 수 있습니다.

$$ 
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2. $$ 

우리는 근사가 얼마나 좋은지 확인하기 위해 코드에서 이를 테스트할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'근사값: {grad_approx}, 참값: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'근사값: {grad_approx}, 참값: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'근사값: {grad_approx}, 참값: {true_value}'
```

## 기울기와 경사 하강법의 기하학 (Geometry of Gradients and Gradient Descent)
:eqref:`eq_nabla_use` 식을 다시 고려해 보십시오.

$$ 
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$ 

내가 이것을 사용하여 우리의 손실 $L$을 최소화하는 데 도움을 주고 싶다고 가정합시다. :numref:`sec_autograd`에서 처음 설명한 경사 하강법 알고리즘을 기하학적으로 이해해 봅시다. 우리가 할 일은 다음과 같습니다.

1. 초기 파라미터 $\mathbf{w}$에 대해 무작위 선택으로 시작합니다.
2. $\mathbf{w}$에서 $L$을 가장 빠르게 감소시키는 방향 $\mathbf{v}$를 찾습니다.
3. 해당 방향으로 작은 단계를 밟습니다: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. 반복합니다.

우리가 정확히 어떻게 해야 할지 모르는 유일한 일은 두 번째 단계에서 벡터 $\mathbf{v}$를 계산하는 것입니다. 우리는 그러한 방향을 *가장 가파른 하강 방향(direction of steepest descent)*이라고 부를 것입니다. :numref:`sec_geometry-linear-algebraic-ops`의 내적에 대한 기하학적 이해를 사용하여, 우리는 :eqref:`eq_nabla_use`를 다음과 같이 다시 쓸 수 있음을 알 수 있습니다.

$$ 
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$ 

편의상 우리의 방향이 길이 1을 갖도록 취했고, $\mathbf{v}$와 $\nabla_{\mathbf{w}} L(\mathbf{w})$ 사이의 각도에 대해 $\theta$를 사용했습니다. 만약 $L$을 가능한 한 빨리 감소시키는 방향을 찾고 싶다면, 우리는 이 식을 가능한 한 음수가 되게 만들고 싶습니다. 우리가 선택한 방향이 이 방정식에 들어가는 유일한 방법은 $\cos(\theta)$를 통해서이므로, 우리는 이 코사인을 가능한 한 음수가 되게 만들고 싶습니다. 이제 코사인의 모양을 상기하면, $\cos(\theta) = -1$이 되게 하거나 동등하게 기울기와 우리가 선택한 방향 사이의 각도를 $\pi$ 라디안, 즉 $180$도가 되게 함으로써 이를 가능한 한 음수가 되게 만들 수 있습니다. 이를 달성하는 유일한 방법은 정확히 반대 방향으로 향하는 것입니다: $\nabla_{\mathbf{w}} L(\mathbf{w})$와 정확히 반대 방향을 가리키도록 $\mathbf{v}$를 선택하십시오!

이것은 우리를 머신러닝에서 가장 중요한 수학적 개념 중 하나로 이끕니다: 가장 가파른 하강 방향은 $-\nabla_{\mathbf{w}}L(\mathbf{w})$ 방향을 가리킵니다. 따라서 우리의 비공식적인 알고리즘은 다음과 같이 다시 쓰일 수 있습니다.

1. 초기 파라미터 $\mathbf{w}$에 대해 무작위 선택으로 시작합니다.
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$를 계산합니다.
3. 해당 방향의 반대 방향으로 작은 단계를 밟습니다: $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. 반복합니다.


이 기본 알고리즘은 많은 연구자에 의해 여러 방식으로 수정되고 조정되었지만 핵심 개념은 그들 모두에서 동일하게 유지됩니다. 기울기를 사용하여 손실을 가능한 한 빨리 줄이는 방향을 찾고, 해당 방향으로 단계를 밟도록 파라미터를 업데이트하십시오.

## 수학적 최적화에 관한 노트 (A Note on Mathematical Optimization)
이 책 전체에서 우리는 딥러닝 설정에서 마주치는 모든 함수가 명시적으로 최소화하기에 너무 복잡하다는 실질적인 이유 때문에 수치적 최적화 기술에 정면으로 집중합니다.

하지만 우리가 위에서 얻은 기하학적 이해가 함수를 직접 최적화하는 것에 대해 무엇을 말해주는지 고려하는 것은 유용한 연습입니다.

어떤 함수 $L(\mathbf{x})$를 최소화하는 $\mathbf{x}_0$ 값을 찾고 싶다고 가정합시다. 더욱이 누군가가 우리에게 값을 주면서 그것이 $L$을 최소화하는 값이라고 말한다고 가정합시다. 그들의 답이 그럴듯한지 확인하기 위해 우리가 확인할 수 있는 것이 있을까요?

다시 :eqref:`eq_nabla_use`를 고려하십시오.
$$ 
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$ 

기울기가 0이 아니라면, 우리는 더 작은 $L$ 값을 찾기 위해 $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ 방향으로 단계를 밟을 수 있음을 압니다. 따라서 우리가 진정으로 최소값에 있다면, 이것은 불가능해야 합니다! 우리는 $\mathbf{x}_0$가 최소값이라면 $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$이라고 결론지을 수 있습니다. 우리는 $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$인 점들을 *임계점(critical points)*이라고 부릅니다.

이것은 좋은데, 왜냐하면 드문 설정에서 우리는 기울기가 0인 모든 점을 명시적으로 찾고 가장 작은 값을 가진 점을 찾을 수 있기 때문입니다.

구체적인 예로 다음 함수를 고려해 보십시오.
$$ 
f(x) = 3x^4 - 4x^3 -12x^2.
$$ 

이 함수는 다음과 같은 도함수를 갖습니다.
$$ 
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$ 

최소값의 가능한 위치는 $x = -1, 0, 2$뿐이며, 여기서 함수는 각각 $-5, 0, -32$ 값을 취하므로, 우리는 $x = 2$일 때 함수를 최소화한다고 결론지을 수 있습니다. 빠른 플롯이 이를 확인해 줍니다.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

이는 이론적으로나 수치적으로 작업할 때 알아야 할 중요한 사실을 강조합니다: 함수를 최소화(또는 최대화)할 수 있는 유일하게 가능한 점들은 기울기가 0인 점들이지만, 기울기가 0인 모든 점이 진정한 *전역* 최소값(또는 최대값)은 아닙니다.

## 다변수 연쇄 법칙 (Multivariate Chain Rule)
여러 항을 합성하여 만들 수 있는 네 가지 변수($w, x, y, z$)의 함수가 있다고 가정해 봅시다.

$$\begin{aligned}f(u, v) & = (u+v)^{2} \u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$ 
:eqlabel:`eq_multi_func_def`

그러한 방정식 체인은 신경망으로 작업할 때 흔하므로, 그러한 함수의 기울기를 계산하는 방법을 이해하는 것이 핵심입니다. 어떤 변수들이 서로 직접적으로 관련되어 있는지 살펴본다면 그림 :numref:`fig_chain-1`에서 이러한 연결의 시각적 힌트를 보기 시작할 수 있습니다.

![위의 함수 관계. 노드는 값을 나타내고 엣지는 함수적 의존성을 보여줍니다.](../img/chain-net1.svg)
:label:`fig_chain-1`

:eqref:`eq_multi_func_def`의 모든 것을 합성하여 다음과 같이 쓰는 것을 막는 것은 아무것도 없습니다.

$$ 
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2. $$ 

그런 다음 단일 변수 도함수만 사용하여 도함수를 취할 수도 있겠지만, 그렇게 한다면 금방 수많은 항에 휩싸이게 될 것이고, 그중 다수는 반복될 것입니다! 실제로 예를 들어 다음을 알 수 있습니다.

$$ 
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$ 

우리가 $\frac{\partial f}{\partial x}$도 계산하고 싶다면, 다시 많은 반복되는 항과 두 도함수 사이의 많은 *공유되는* 반복 항이 있는 유사한 방정식을 얻게 될 것입니다. 이는 엄청난 양의 낭비되는 작업을 나타내며, 만약 우리가 이런 방식으로 도함수를 계산해야 했다면 전체 딥러닝 혁명은 시작되기도 전에 멈췄을 것입니다!


문제를 쪼개 봅시다. 우리는 본질적으로 $w, x, y, z$가 모두 존재하지 않는다고 가정하고 $a$를 바꿀 때 $f$가 어떻게 변하는지 이해하는 것부터 시작할 것입니다. 우리는 처음으로 기울기로 작업했을 때와 같이 추론할 것입니다. $a$를 취하고 여기에 작은 양 $\epsilon$을 더해 봅시다.

$$ 
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$ 

첫 번째 줄은 편미분의 정의로부터 따르고, 두 번째 줄은 기울기의 정의로부터 따릅니다. 식 $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$에서처럼 우리가 모든 도함수를 어디서 평가하는지 정확하게 추적하는 것은 표기법상 부담스러우므로, 우리는 종종 이를 훨씬 더 기억하기 쉬운 다음과 같은 형태로 줄여 씁니다.

$$ 
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}. $$ 

과정의 의미에 대해 생각해보는 것이 유용합니다. 우리는 $a$의 변화에 따라 $f(u(a, b), v(a, b))$ 형태의 함수가 그 값을 어떻게 바꾸는지 이해하려고 노력하고 있습니다. 이것이 일어날 수 있는 두 가지 경로가 있습니다: $a \rightarrow u \rightarrow f$인 경로와 $a \rightarrow v \rightarrow f$인 경로입니다. 우리는 연쇄 법칙을 통해 이러한 두 기여를 각각 $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$와 $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$로 계산하고 합산할 수 있습니다.

그림 :numref:`fig_chain-2`에 표시된 것처럼 오른쪽의 함수가 왼쪽에서 연결된 함수들에 의존하는 다른 함수 네트워크가 있다고 상상해 보십시오.

![연쇄 법칙의 또 다른 더 미묘한 예.](../img/chain-net2.svg)
:label:`fig_chain-2`

$rac{\partial f}{\partial y}$와 같은 것을 계산하려면, $y$에서 $f$로 가는 모든(이 경우 3개) 경로에 대해 합산해야 하며 다음을 얻습니다.

$$ 
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}. $$ 

이런 방식으로 연쇄 법칙을 이해하면 기울기가 네트워크를 통해 어떻게 흐르는지, 그리고 LSTM(:numref:`sec_lstm`)이나 잔차 레이어(:numref:`sec_resnet`)의 여러 아키텍처 선택이 기울기 흐름을 제어함으로써 학습 과정을 형성하는 데 어떻게 도움이 되는지 이해할 때 큰 보상을 얻을 수 있을 것입니다.

## 역전파 알고리즘 (The Backpropagation Algorithm)

이전 섹션의 :eqref:`eq_multi_func_def` 예제로 돌아가 봅시다.

$$ 
\begin{aligned}
f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$ 

가령 $\frac{\partial f}{\partial w}$를 계산하고 싶다면 다변수 연쇄 법칙을 적용하여 다음을 볼 수 있습니다.

$$ 
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$ 

이 분해를 사용하여 $\frac{\partial f}{\partial w}$를 계산해 봅시다. 여기서 우리에게 필요한 것은 다양한 단일 단계 부분들뿐입니다.

$$ 
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$ 

이것을 코드로 작성하면 상당히 관리하기 쉬운 식이 됩니다.

```{.python .input}
#@tab all
# 입력에서 출력으로 함수 값 계산
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    {w}, {x}, {y}, {z} 에서 f는 {f} 입니다')

# 단일 단계 부분 계산
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# 입력에서 출력으로 최종 결과 계산
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'    {w}, {x}, {y}, {z} 에서 df/dw는 {df_dw} 입니다')
```

하지만 이 방식이 여전히 $\frac{\partial f}{\partial x}$와 같은 것을 계산하기 쉽게 만들지는 않는다는 점에 유의하십시오. 그 이유는 우리가 연쇄 법칙을 적용하기로 선택한 *방식* 때문입니다. 위에서 우리가 한 일을 보면, 우리는 가능한 한 분모에 $\partial w$를 유지했습니다. 이런 식으로 우리는 $w$가 다른 모든 변수를 어떻게 바꾸는지 보면서 연쇄 법칙을 적용하기로 선택했습니다. 만약 그것이 우리가 원한 것이라면 이것은 좋은 아이디어였을 것입니다. 하지만 딥러닝에서의 동기를 다시 생각해 보십시오: 우리는 모든 파라미터가 *손실*을 어떻게 바꾸는지 보고 싶어 합니다. 본질적으로 우리는 가능할 때마다 분자에 $\partial f$를 유지하면서 연쇄 법칙을 적용하고 싶습니다!

더 명시적으로 말하자면, 우리는 다음과 같이 쓸 수 있음에 유의하십시오.

$$ 
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$ 

연쇄 법칙의 이 적용은 우리가 $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \textrm{및} \; \frac{\partial f}{\partial w}$를 명시적으로 계산하게 합니다. 다음과 같은 방정식을 포함하는 것을 막는 것은 아무것도 없습니다.

$$ 
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$ 

그리고 나서 전체 네트워크의 *임의의* 노드를 바꿀 때 $f$가 어떻게 변하는지 추적합니다. 이를 구현해 봅시다.

```{.python .input}
#@tab all
# 입력에서 출력으로 함수 값 계산
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'{w}, {x}, {y}, {z} 에서 f는 {f} 입니다')

# 위의 분해를 사용하여 도함수 계산
# 먼저 단일 단계 부분 계산
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# 이제 출력에서 입력으로 임의의 값을 바꿀 때 f가 어떻게 변하는지 계산
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'{w}, {x}, {y}, {z} 에서 df/dw는 {df_dw} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dx는 {df_dx} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dy는 {df_dy} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dz는 {df_dz} 입니다')
```

우리가 입력에서 출력으로 정방향으로 도함수를 계산하는 대신(위의 첫 번째 코드 스니펫에서 했던 것처럼), $f$에서 입력 방향으로 역방향으로 도함수를 계산한다는 사실이 이 알고리즘에 그 이름을 부여합니다: *역전파(backpropagation)*. 여기에는 두 단계가 있음에 유의하십시오.
1. 함수의 값과 단일 단계 부분들을 앞에서 뒤로 계산합니다. 위에서 하지는 않았지만, 이는 단일 *정방향 패스(forward pass)*로 결합될 수 있습니다.
2. 뒤에서 앞으로 $f$의 기울기를 계산합니다. 우리는 이를 *역방향 패스(backwards pass)*라고 부릅니다.

이것이 바로 모든 딥러닝 알고리즘이 한 번의 패스로 네트워크의 모든 가중치에 대한 손실의 기울기 계산을 가능하게 하기 위해 구현하는 것입니다. 우리가 그러한 분해를 가졌다는 것은 놀라운 사실입니다.

이것이 어떻게 캡슐화되는지 보기 위해, 이 예제를 빠르게 살펴봅시다.

```{.python .input}
#@tab mxnet
# ndarray로 초기화한 다음 기울기 첨부
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# 기울기를 추적하며 평소와 같이 계산 수행
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# 역방향 패스 실행
f.backward()

print(f'{w}, {x}, {y}, {z} 에서 df/dw는 {w.grad} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dx는 {x.grad} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dy는 {y.grad} 입니다')
print(f'{w}, {x}, {y}, {z} 에서 df/dz는 {z.grad} 입니다')
```

```{.python .input}
#@tab pytorch
# ndarray로 초기화한 다음 기울기 첨부
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# 기울기를 추적하며 평소와 같이 계산 수행
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# 역방향 패스 실행
f.backward()

print(f'{w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} 에서 df/dw는 {w.grad.data.item()} 입니다')
print(f'{w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} 에서 df/dx는 {x.grad.data.item()} 입니다')
print(f'{w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} 에서 df/dy는 {y.grad.data.item()} 입니다')
print(f'{w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} 에서 df/dz는 {z.grad.data.item()} 입니다')
```

```{.python .input}
#@tab tensorflow
# ndarray로 초기화한 다음 기울기 첨부
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# 기울기를 추적하며 평소와 같이 계산 수행
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# 역방향 패스 실행
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'{w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} 에서 df/dw는 {w_grad} 입니다')
print(f'{w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} 에서 df/dx는 {x_grad} 입니다')
print(f'{w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} 에서 df/dy는 {y_grad} 입니다')
print(f'{w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} 에서 df/dz는 {z_grad} 입니다')
```

우리가 위에서 한 모든 것은 `f.backwards()`를 호출함으로써 자동으로 수행될 수 있습니다.


## 헤시안 (Hessians)
단일 변수 미적분학에서와 마찬가지로, 함수에 대한 더 나은 근사를 얻는 방법을 파악하기 위해 고계 도함수를 고려하는 것이 유용합니다. 이는 기울기만 사용하는 것보다 더 나은 근사를 제공합니다.

여러 변수의 함수의 고계 도함수로 작업할 때 즉시 마주치는 한 가지 문제는 그 수가 매우 많다는 것입니다. $n$개 변수의 함수 $f(x_1, \ldots, x_n)$가 있다면, 우리는 $n^{2}$개의 많은 2계 도함수를 취할 수 있습니다. 즉, $i$와 $j$의 임의의 선택에 대해 다음과 같습니다.

$$ 
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$ 

이들은 전통적으로 *헤시안(Hessian)*이라고 불리는 행렬로 조립됩니다.

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$ 
:eqlabel:`eq_hess_def`

이 행렬의 모든 항목이 독립적인 것은 아닙니다. 실제로 양쪽의 *혼합 부분 도함수(mixed partials)*가 존재하고 연속적인 한, 임의의 $i$와 $j$에 대해 다음과 같이 말할 수 있습니다.

$$ 
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$ 

이는 먼저 $x_i$ 방향으로 함수를 섭동시킨 다음 $x_j$로 섭동시키는 것과, 먼저 $x_j$로 섭동시킨 다음 $x_i$로 섭동시키는 것 두 순서 모두 $f$ 출력의 동일한 최종 변화로 이어진다는 지식을 가지고 비교함으로써 따릅니다.

단일 변수에서와 마찬가지로, 우리는 이러한 도함수를 사용하여 함수가 점 근처에서 어떻게 행동하는지에 대해 훨씬 더 나은 아이디어를 얻을 수 있습니다. 특히, 단일 변수에서 보았던 것처럼 점 $\mathbf{x}_0$ 근처에서 가장 잘 맞는 이차식을 찾는 데 이를 사용할 수 있습니다.

예제를 살펴봅시다. $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$라고 가정합시다. 이것은 두 변수의 이차식에 대한 일반적인 형태입니다. 제로(zero) 지점에서의 함수의 값, 기울기, 그리고 헤시안 :eqref:`eq_hess_def`을 본다면 다음과 같습니다.

$$ 
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$ 

우리는 다음과 같이 말함으로써 원래의 다항식을 다시 얻을 수 있습니다.

$$ 
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}. $$ 

일반적으로 임의의 점 $\mathbf{x}_0$에서 이 전개를 계산한다면 다음을 알 수 있습니다.

$$ 
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$ 

이는 임의의 차원 입력에 대해 작동하며, 임의의 함수에 대해 특정 점에서의 최상의 근사 이차식을 제공합니다. 예를 들어 다음 함수를 플롯해 봅시다.

$$ 
f(x, y) = xe^{-x^2-y^2}.
$$ 

기울기와 헤시안이 다음과 같음을 계산할 수 있습니다.
$$ 
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \textrm{및} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$ 

따라서 약간의 대수를 통해 $[-1,0]^\top$에서의 근사 이차식이 다음과 같음을 알 수 있습니다.

$$ 
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$ 

```{.python .input}
#@tab mxnet
# 그리드 구성 및 함수 계산
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# (1, 0)에서 기울기와 헤시안을 사용하여 근사 이차식 계산
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# 함수 플롯
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# 그리드 구성 및 함수 계산
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# (1, 0)에서 기울기와 헤시안을 사용하여 근사 이차식 계산
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# 함수 플롯
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# 그리드 구성 및 함수 계산
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# (1, 0)에서 기울기와 헤시안을 사용하여 근사 이차식 계산
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# 함수 플롯
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

이것이 :numref:`sec_gd`에서 논의된 뉴턴 알고리즘(Newton's Algorithm)의 기초를 형성합니다. 여기서는 최상의 근사 이차식을 반복적으로 찾고 그 이차식을 정확하게 최소화함으로써 수치적 최적화를 수행합니다.

## 행렬 미적분학 맛보기 (A Little Matrix Calculus)
행렬을 포함하는 함수의 도함수는 특히 깔끔하게 나옵니다. 이 섹션은 표기법상 무거울 수 있으므로 처음 읽을 때는 건너뛸 수 있지만, 특히 딥러닝 응용 분야에서 행렬 연산이 얼마나 중심적인지를 고려할 때 일반적인 행렬 연산을 포함하는 함수의 도함수가 예상보다 훨씬 깔끔한 경우가 많다는 것을 아는 것은 유용합니다.

예제로 시작해 봅시다. 어떤 고정된 열 벡터 $\boldsymbol{\beta}$가 있다고 가정하고, 곱 함수 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$를 취하여 $\mathbf{x}$를 바꿀 때 내적이 어떻게 변하는지 이해하고 싶다고 합시다.

머신러닝에서 행렬 도함수를 다룰 때 유용하게 쓰일 표기법 중 하나는 *분모 레이아웃 행렬 도함수(denominator layout matrix derivative)*라고 불리는데, 여기서는 미분 분모에 있는 벡터, 행렬 또는 텐서의 모양으로 부분 도함수를 조립합니다. 이 경우 다음과 같이 쓸 것입니다.

$$ 
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix}, 
$$ 

열 벡터 $\mathbf{x}$의 모양과 일치시켰습니다.

함수를 구성 요소로 써보면 다음과 같습니다.

$$ 
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n. $$ 

이제 가령 $\beta_1$에 대해 부분 도함수를 취하면, 첫 번째 항을 제외하고는 모든 것이 0이며 이는 단지 $\beta_1$에 $x_1$을 곱한 것이므로 다음을 얻습니다.

$$ 
\frac{df}{dx_1} = \beta_1, $$ 

또는 더 일반적으로 다음과 같습니다.

$$ 
\frac{df}{dx_i} = \beta_i.
$$ 

이제 이를 행렬로 다시 조립하여 다음을 볼 수 있습니다.

$$ 
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$ 

이는 이 섹션 전체에서 종종 마주하게 될 행렬 미적분학에 대한 몇 가지 요소를 설명합니다.

* 첫째, 계산은 상당히 복잡해질 것입니다.
* 둘째, 최종 결과는 중간 과정보다 훨씬 깔끔하며 항상 단일 변수의 경우와 유사하게 보일 것입니다. 이 경우 $\frac{d}{dx}(bx) = b$와 $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$가 모두 유사함에 유의하십시오.
* 셋째, 전치(transposes)가 뜬금없이 나타날 수 있습니다. 이에 대한 핵심 이유는 우리가 분모의 모양과 일치시킨다는 관례 때문이며, 따라서 행렬을 곱할 때 원래 항의 모양과 다시 일치시키기 위해 전치를 취해야 할 것입니다.

직관을 계속 구축하기 위해 조금 더 어려운 계산을 시도해 봅시다. 열 벡터 $\mathbf{x}$와 정사각 행렬 $A$가 있다고 가정하고 다음을 계산하고 싶다고 합시다.

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$ 
:eqlabel:`eq_mat_goal_1`

조작하기 더 쉬운 표기법을 위해 아인슈타인 표기법을 사용하여 이 문제를 고려해 봅시다. 이 경우 함수를 다음과 같이 쓸 수 있습니다.

$$ 
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j. $$ 

도함수를 계산하려면 모든 $k$에 대해 다음의 값이 무엇인지 이해해야 합니다.

$$ 
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$ 

곱의 미분법에 의해 다음을 얻습니다.

$$ 
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$ 

$rac{dx_i}{dx_k}$와 같은 항의 경우, $i=k$일 때 1이고 그렇지 않으면 0임을 알 수 있습니다. 이는 $i$와 $k$가 다른 모든 항이 이 합계에서 사라짐을 의미하므로, 첫 번째 합계에서 남는 유일한 항은 $i=k$인 항들뿐입니다. $j=k$가 필요한 두 번째 항에 대해서도 동일한 추론이 성립합니다. 이는 다음을 제공합니다.

$$ 
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$ 

이제 아인슈타인 표기법에서 인덱스의 이름은 임의적입니다 - 이 시점에서 이 계산에 있어 $i$와 $j$가 다르다는 사실은 중요하지 않으므로, 둘 다 $i$를 사용하도록 인덱스를 다시 지정하여 다음을 볼 수 있습니다.

$$ 
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$ 

이제 여기서부터 더 나아가기 위해 약간의 연습이 필요합니다. 행렬 연산의 관점에서 이 결과를 식별해 봅시다. $a_{ki} + a_{ik}$는 $\mathbf{A} + \mathbf{A}^\top$의 $k, i$번째 성분입니다. 이는 다음을 제공합니다.

$$ 
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$ 

마찬가지로, 이 항은 이제 행렬 $\mathbf{A} + \mathbf{A}^\top$와 벡터 $\mathbf{x}$의 곱이므로 다음과 같음을 알 수 있습니다.

$$ 
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$ 

따라서 우리는 :eqref:`eq_mat_goal_1`로부터 원하는 도함수의 $k$번째 항목이 우변에 있는 벡터의 $k$번째 항목임을 알 수 있으며, 따라서 두 가지는 동일합니다. 따라서 다음을 산출합니다.

$$ 
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$ 

이것은 지난번보다 훨씬 더 많은 작업을 필요로 했지만 최종 결과는 작습니다. 그뿐만 아니라 전통적인 단일 변수 도함수에 대한 다음 계산을 고려해 보십시오.

$$ 
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$ 

동등하게 $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$입니다. 다시 말하지만, 우리는 단일 변수 결과와 다소 비슷해 보이지만 전치가 섞여 들어간 결과를 얻습니다.

이 시점에서 패턴이 다소 의심스러워 보일 것이므로 이유를 알아내려고 노력해 봅시다. 이와 같이 행렬 도함수를 취할 때, 우리가 얻는 식이 행렬과 그들의 전치의 곱과 합의 관점에서 쓸 수 있는 또 다른 행렬 식이라고 먼저 가정해 봅시다. 그러한 식이 존재한다면 모든 행렬에 대해 참이어야 할 것입니다. 특히, 행렬 곱이 단지 숫자의 곱이고, 행렬 합이 단지 합이며, 전치가 아무것도 하지 않는 $1 \times 1$ 행렬에 대해 참이어야 할 것입니다! 즉, 우리가 얻는 식은 무엇이든 단일 변수 식과 *일치해야* 합니다. 이는 약간의 연습을 통해 연관된 단일 변수 식이 어떻게 생겼어야 하는지 아는 것만으로 행렬 도함수를 종종 추측할 수 있음을 의미합니다!

이를 시도해 봅시다. $\mathbf{X}$가 $n \times m$ 행렬이고, $\mathbf{U}$가 $n \times r$이며 $\mathbf{V}$가 $r \times m$이라고 가정합시다. 다음을 계산해 봅시다.

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$ 
:eqlabel:`eq_mat_goal_2`

이 계산은 행렬 분해(matrix factorization)라는 분야에서 중요합니다. 하지만 우리에게는 그저 계산해야 할 도함수일 뿐입니다. 이것이 $1\times1$ 행렬에 대해 어떨지 상상해 봅시다. 그 경우 다음과 같은 식을 얻습니다.

$$ 
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u, $$ 

여기서 도함수는 꽤 표준적입니다. 이를 다시 행렬 식으로 변환하려고 시도하면 다음을 얻습니다.

$$ 
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}. $$ 

하지만 이를 보면 제대로 작동하지 않습니다. $\mathbf{X}$는 $n \times m$이고 $\mathbf{U}\mathbf{V}$도 마찬가지이므로, 행렬 $2(\mathbf{X} - \mathbf{U}\mathbf{V})$는 $n \times m$입니다. 반면 $\mathbf{U}$는 $n \times r$이며, 차원이 일치하지 않으므로 $n \times m$과 $n \times r$ 행렬을 곱할 수 없습니다!

우리는 $\mathbf{V}$와 모양이 같은 $r \times m$인 $\frac{d}{d\mathbf{V}}$를 얻고 싶습니다. 그래서 어떻게든 우리는 $n \times m$ 행렬과 $n \times r$ 행렬을 가져와서(아마도 전치를 사용하여) 곱해서 $r \times m$을 얻어야 합니다. 우리는 $U^\top$에 $(\mathbf{X} - \mathbf{U}\mathbf{V})$를 곱함으로써 이를 할 수 있습니다. 따라서 :eqref:`eq_mat_goal_2`에 대한 해가 다음과 같을 것이라고 추측할 수 있습니다.

$$ 
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}). $$ 

이것이 작동함을 보여주기 위해 상세한 계산을 제공하지 않는다면 태만한 일일 것입니다. 이 경험 법칙이 작동한다고 이미 믿으신다면 이 유도를 건너뛰셔도 좋습니다. 다음을 계산하려면

$$ 
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$ 

모든 $a$와 $b$에 대해 다음을 찾아야 합니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$ 

$rac{d}{dv_{ab}}$에 관한 한 $\mathbf{X}$와 $\mathbf{U}$의 모든 항목이 상수임을 상기하면, 도함수를 합계 안으로 밀어 넣고 제곱에 연쇄 법칙을 적용하여 다음을 얻을 수 있습니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$ 

이전 유도에서와 같이, $\frac{dv_{kj}}{dv_{ab}}$는 $k=a$ 및 $j=b$일 때만 0이 아님에 주목할 수 있습니다. 이 조건 중 어느 하나라도 충족되지 않으면 합계의 항은 0이 되며 우리는 이를 자유롭게 버릴 수 있습니다. 다음을 알 수 있습니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$ 

여기서 중요한 미묘함은 $k=a$라는 요구 사항이 안쪽 합계 내에서 발생하지 않는다는 것인데, 그 $k$는 안쪽 항 내에서 우리가 합산하고 있는 더미 변수이기 때문입니다. 표기법상 더 깔끔한 예로 다음 이유를 고려해 보십시오.

$$ 
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$ 

이 지점으로부터 우리는 합계의 구성 요소들을 식별하기 시작할 수 있습니다. 먼저 다음입니다.

$$ 
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$ 

따라서 합계 내부의 전체 식은 다음과 같습니다.

$$ 
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$ 

이는 우리가 이제 우리의 도함수를 다음과 같이 쓸 수 있음을 의미합니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$ 

우리는 이것이 행렬의 $a, b$ 원소처럼 보이기를 원하므로 이전 예제와 같은 기술을 사용하여 행렬 식에 도달할 수 있으며, 이는 $u_{ia}$의 인덱스 순서를 바꿔야 함을 의미합니다. $u_{ia} = [\mathbf{U}^\top]_{ai}$임을 주목한다면 다음과 같이 쓸 수 있습니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$ 

이것은 행렬 곱이며, 따라서 다음과 같이 결론지을 수 있습니다.

$$ 
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$ 

따라서 우리는 :eqref:`eq_mat_goal_2`에 대한 해를 다음과 같이 쓸 수 있습니다.

$$ 
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$ 

이것은 우리가 위에서 추측한 해와 일치합니다!

이 시점에서 "왜 내가 배운 모든 미적분학 규칙의 행렬 버전을 그냥 쓸 수 없나요? 이것이 여전히 기계적이라는 것은 분명합니다. 그냥 끝내 버립시다!"라고 묻는 것이 합리적입니다. 실제로 그러한 규칙들이 있으며 :cite:`Petersen.Pedersen.ea.2008`은 훌륭한 요약을 제공합니다. 그러나 단일 값에 비해 행렬 연산이 결합될 수 있는 수많은 방식 때문에, 단일 변수보다 훨씬 더 많은 행렬 도함수 규칙이 있습니다. 인덱스로 작업하거나 적절한 경우 자동 미분에 맡기는 것이 최선인 경우가 많습니다.

## 요약 (Summary)

* 고차원에서 우리는 1차원에서의 도함수와 동일한 목적을 수행하는 기울기를 정의할 수 있습니다. 이를 통해 입력에 임의의 작은 변화를 주었을 때 다변수 함수가 어떻게 변하는지 알 수 있습니다.
* 역전파 알고리즘은 많은 부분 도함수의 효율적인 계산을 가능하게 하기 위해 다변수 연쇄 법칙을 조직하는 방법으로 볼 수 있습니다.
* 행렬 미적분학은 행렬 식의 도함수를 간결한 방식으로 쓸 수 있게 해 줍니다.

## 연습 문제 (Exercises)
1. 열 벡터 $\boldsymbol{\beta}$가 주어졌을 때, $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$와 $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$의 도함수를 모두 계산하십시오. 왜 같은 답을 얻습니까?
2. $\mathbf{v}$를 $n$차원 벡터라고 합시다. $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$는 무엇입니까?
3. $L(x, y) = \log(e^x + e^y)$라고 합시다. 기울기를 계산하십시오. 기울기 구성 요소들의 합은 얼마입니까?
4. $f(x, y) = x^2y + xy^2$라고 합시다. 유일한 임계점이 $(0,0)$임을 보이십시오. $f(x, x)$를 고려하여 $(0,0)$이 최대값, 최소값, 아니면 둘 다 아닌지 결정하십시오.
5. 함수 $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$를 최소화한다고 가정합시다. $\nabla f = 0$ 조건을 $g$와 $h$의 관점에서 어떻게 기하학적으로 해석할 수 있습니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab: