```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 미적분 (Calculus)
:label:`sec_calculus`

오랫동안 원의 넓이를 계산하는 방법은 미스터리로 남아 있었습니다.
그러다 고대 그리스에서 수학자 아르키메데스가
원 내부에 꼭짓점 수가 증가하는 일련의 다각형을 새기는
영리한 아이디어를 생각해 냈습니다
(:numref:`fig_circle_area`).
$n$개의 꼭짓점이 있는 다각형의 경우,
우리는 $n$개의 삼각형을 얻습니다.
원을 더 미세하게 분할할수록 각 삼각형의 높이는 반지름 $r$에 가까워집니다.
동시에, 많은 수의 꼭짓점에 대해 호와 할선 사이의 비율이 1에 가까워지므로,
밑변은 $2 \pi r/n$에 가까워집니다.
따라서 다각형의 넓이는
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$에 가까워집니다.

![극한 절차로서 원의 넓이 찾기.](../img/polygon-circle.svg)
:label:`fig_circle_area`

이 극한 절차는 *미분 미적분*과 *적분 미적분*의 뿌리에 있습니다.
전자는 인수를 조작하여 함수의 값을 증가시키거나 감소시키는 방법을 알려줄 수 있습니다.
이것은 우리가 딥러닝에서 직면하는 *최적화 문제*에 유용합니다.
여기서 우리는 손실 함수를 줄이기 위해 파라미터를 반복적으로 업데이트합니다.
최적화는 모델을 훈련 데이터에 맞추는 방법을 다루며,
미적분은 그 핵심 전제 조건입니다.
그러나 우리의 궁극적인 목표는 *이전에 본 적 없는* 데이터에서 잘 수행하는 것임을 잊지 마십시오.
그 문제를 *일반화*라고 하며 다른 장의 핵심 초점이 될 것입니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## 도함수와 미분 (Derivatives and Differentiation)

간단히 말해서, *도함수(derivative)*는 인수의 변화에 대한 함수의 변화율입니다.
도함수는 각 파라미터를 무한히 작은 양만큼 *증가*시키거나 *감소*시킬 경우
손실 함수가 얼마나 빠르게 증가하거나 감소하는지 알려줄 수 있습니다.
공식적으로, 스칼라에서 스칼라로 매핑하는 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$의 경우,
[**점 $x$에서 $f$의 *도함수*는 다음과 같이 정의됩니다**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

오른쪽에 있는 이 항을 *극한(limit)*이라고 하며,
지정된 변수가 특정 값에 접근할 때
표현식의 값에 어떤 일이 일어나는지 알려줍니다.
이 극한은 섭동(perturbation) $h$와
함수 값의 변화 $f(x + h) - f(x)$ 사이의 비율이
크기를 0으로 줄일 때 무엇으로 수렴하는지 알려줍니다.

$f'(x)$가 존재할 때, $f$는 $x$에서 *미분 가능(differentiable)*하다고 합니다;
그리고 $f'(x)$가 집합(예: 구간 $[a,b]$)의 모든 $x$에 대해 존재할 때,
우리는 $f$가 이 집합에서 미분 가능하다고 말합니다. 
정확도나 수신자 조작 특성 곡선 아래 면적(AUC)과 같이 우리가 최적화하고자 하는 많은 함수를 포함하여,
모든 함수가 미분 가능한 것은 아닙니다.
그러나 손실의 도함수를 계산하는 것은 심층 신경망을 훈련하기 위한
거의 모든 알고리즘에서 중요한 단계이므로,
우리는 종종 대신 미분 가능한 *대리(surrogate)*를 최적화합니다.


우리는 도함수 $f'(x)$를
$x$에 대한 $f(x)$의 *순간* 변화율로 해석할 수 있습니다.
예제를 통해 직관을 키워봅시다.
(**$u = f(x) = 3x^2-4x$를 정의합니다.**)

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$로 설정하면, $\frac{f(x+h) - f(x)}{h}$가**] (**$h$가 $0$에 접근함에 따라 $2$에 접근하는 것을 볼 수 있습니다.**)
이 실험은 수학적 증명의 엄격함이 부족하지만,
실제로 $f'(1) = 2$임을 빠르게 확인할 수 있습니다.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

도함수에 대한 몇 가지 동등한 표기법 관례가 있습니다.
$y = f(x)$가 주어지면 다음 표현식은 동등합니다:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$ 

여기서 기호 $\frac{d}{dx}$와 $D$는 *미분 연산자*입니다. 
아래에 몇 가지 일반적인 함수의 도함수를 제시합니다:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{어떤 상수 $C$에 대해} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{단, } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$ 

미분 가능한 함수들로 구성된 함수는 종종 그 자체로 미분 가능합니다.
다음 규칙들은 미분 가능한 함수 $f$와 $g$, 그리고 상수 $C$의
합성을 다룰 때 유용합니다.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{상수 배수 규칙} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{합의 규칙} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{곱의 규칙} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{몫의 규칙} \end{aligned}$$ 

이것을 사용하여, 우리는 다음을 통해 $3 x^2 - 4x$의 도함수를 찾기 위해 규칙을 적용할 수 있습니다.

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$ 

$x = 1$을 대입하면 실제로 이 위치에서 도함수가 $2$와 같음을 보여줍니다. 
도함수는 특정 위치에서 함수의 *기울기(slope)*를 알려줍니다.

## 시각화 유틸리티 (Visualization Utilities)

[**`matplotlib` 라이브러리를 사용하여 함수의 기울기를 시각화할 수 있습니다**].
몇 가지 함수를 정의해야 합니다.
이름에서 알 수 있듯이 `use_svg_display`는
더 선명한 이미지를 위해 `matplotlib`에 SVG 형식으로 그래픽을 출력하도록 지시합니다.
주석 `#@save`는 함수, 클래스 또는 기타 코드 블록을 `d2l` 패키지에 저장하여
나중에 코드를 반복하지 않고 호출할 수 있게 해주는 특수 수정자입니다.
예: `d2l.use_svg_display()`.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Jupyter에서 플롯을 표시하기 위해 svg 형식을 사용합니다."""
    backend_inline.set_matplotlib_formats('svg')
```

편리하게도 `set_figsize`로 그림 크기를 설정할 수 있습니다.
import 문 `from matplotlib import pyplot as plt`가
`d2l` 패키지에서 `#@save`를 통해 표시되었으므로 `d2l.plt`를 호출할 수 있습니다.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """matplotlib의 그림 크기를 설정합니다."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` 함수는 축을 레이블, 범위, 스케일을 포함한 속성과 연결할 수 있습니다.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """matplotlib의 축을 설정합니다."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

이 세 가지 함수를 사용하여 여러 곡선을 겹쳐 그리는 `plot` 함수를 정의할 수 있습니다.
여기에 있는 대부분의 코드는 입력의 크기와 모양이 일치하는지 확인하는 것입니다.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """데이터 포인트를 플롯합니다."""

    def has_one_axis(X):  # X(텐서 또는 리스트)가 1개의 축을 가지면 True
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

이제 [**함수 $u = f(x)$와 $x=1$에서의 접선 $y = 2x - 3$을 플롯할 수 있습니다**],
여기서 계수 $2$는 접선의 기울기입니다.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 편도함수와 기울기 (Partial Derivatives and Gradients)
:label:`subsec_calculus-grad`

지금까지 우리는 변수가 하나뿐인 함수를 미분해 왔습니다.
딥러닝에서는 *많은* 변수를 가진 함수도 다뤄야 합니다.
이러한 *다변수* 함수에 적용되는 도함수의 개념을 간략하게 소개합니다.


$y = f(x_1, x_2, \ldots, x_n)$을 $n$개의 변수를 가진 함수라고 합시다. 
$y$의 $i$번째 파라미터 $x_i$에 대한 *편도함수(partial derivative)*는 다음과 같습니다.

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$ 


$rac{\partial y}{\partial x_i}$를 계산하기 위해, 
우리는 $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$을 상수로 취급하고
$x_i$에 대한 $y$의 도함수를 계산할 수 있습니다.
편도함수에 대한 다음 표기법 관례는 모두 일반적이며 모두 같은 의미입니다:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$ 

우리는 다변수 함수의 모든 변수에 대한 편도함수를 연결하여
함수의 *기울기(gradient)*라고 불리는 벡터를 얻을 수 있습니다.
함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 입력이
$n$차원 벡터 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$이고
출력이 스칼라라고 가정해 봅시다. 
$\\mathbf{x}$에 대한 함수 $f$의 기울기는
$n$개의 편도함수로 구성된 벡터입니다:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 


모호함이 없을 때,
$\\nabla_{\mathbf{x}} f(\mathbf{x})$는 일반적으로
$\\nabla f(\mathbf{x})$로 대체됩니다.
다음 규칙은 다변수 함수를 미분할 때 유용합니다:

* 모든 $\\mathbf{A} \in \mathbb{R}^{m \times n}$에 대해 $\\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$이고 $\\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$입니다.
* 정사각 행렬 $\\mathbf{A} \in \mathbb{R}^{n \times n}$에 대해 $\\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\\mathbf{x}$이고 특히
$\\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\\mathbf{x}$입니다.

마찬가지로, 어떤 행렬 $\\mathbf{X}$에 대해서도, 
우리는 $\\nabla_{\\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\\mathbf{X}$를 갖습니다. 




## 연쇄 법칙 (Chain Rule)

딥러닝에서 관심 있는 기울기는 종종 계산하기 어렵습니다.
우리가 깊게 중첩된 함수(함수의 (함수의...))를 다루고 있기 때문입니다.
다행히도 *연쇄 법칙(chain rule)*이 이것을 처리합니다.
단일 변수 함수로 돌아가서, $y = f(g(x))$이고
기본 함수 $y=f(u)$와 $u=g(x)$가 모두 미분 가능하다고 가정해 봅시다.
연쇄 법칙은 다음을 명시합니다.


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$ 



다변수 함수로 돌아가서,
$y = f(\\mathbf{u})$가 변수 $u_1, u_2, \ldots, u_m$을 가지고 있고,
각 $u_i = g_i(\\mathbf{x})$가 변수 $x_1, x_2, \ldots, x_n$을 가지고 있다고 가정해 봅시다.
즉, $\\mathbf{u} = g(\\mathbf{x})$입니다.
그러면 연쇄 법칙은 다음을 명시합니다.

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ 따라서 } \ \nabla_{\\mathbf{x}} y =  \\mathbf{A} \nabla_{\\mathbf{u}} y,$$ 

여기서 $\\mathbf{A} \in \mathbb{R}^{n \times m}$은
벡터 $\\mathbf{x}$에 대한 벡터 $\\mathbf{u}$의 도함수를 포함하는 *행렬*입니다.
따라서 기울기를 평가하려면 벡터-행렬 곱을 계산해야 합니다.
이것이 선형 대수가 딥러닝 시스템을 구축하는 데 있어
그토록 필수적인 구성 요소인 주요 이유 중 하나입니다. 




## 토론

우리는 깊은 주제의 겉핥기만 했지만,
이미 많은 개념이 초점에 들어왔습니다: 
첫째, 미분을 위한 합성 규칙을 일상적으로 적용할 수 있어,
기울기를 *자동으로* 계산할 수 있습니다.
이 작업은 창의성이 필요하지 않으므로 우리는 인지 능력을 다른 곳에 집중할 수 있습니다. 
둘째, 벡터 값 함수의 도함수를 계산하려면 출력에서 입력으로
변수의 종속성 그래프를 추적하면서 행렬을 곱해야 합니다. 
특히, 이 그래프는 함수를 평가할 때 *순방향(forward)*으로 순회하고
기울기를 계산할 때 *역방향(backwards)*으로 순회합니다. 
나중 챕터에서는 연쇄 법칙을 적용하기 위한 계산 절차인 역전파를 공식적으로 소개할 것입니다.

최적화의 관점에서, 기울기는 손실을 낮추기 위해
모델의 파라미터를 어떻게 이동해야 하는지 결정할 수 있게 해주며,
이 책 전체에서 사용되는 최적화 알고리즘의 각 단계는 기울기 계산을 필요로 합니다.

## 연습 문제

1. 지금까지 우리는 도함수 규칙을 당연하게 여겼습니다. 
   정의와 극한을 사용하여 (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$, (iv) $f(x) = \log x$에 대한 속성을 증명하십시오.
2. 같은 맥락에서, 첫 번째 원칙에서 곱, 합, 몫의 규칙을 증명하십시오. 
3. 상수 배수 규칙이 곱의 규칙의 특수한 경우로 뒤따른다는 것을 증명하십시오. 
4. $f(x) = x^x$의 도함수를 계산하십시오. 
5. 어떤 $x$에 대해 $f'(x) = 0$이라는 것은 무엇을 의미합니까? 
   이것이 성립할 수 있는 함수 $f$와 위치 $x$의 예를 드십시오. 
6. 함수 $y = f(x) = x^3 - \frac{1}{x}$의 그래프와 $x = 1$에서의 접선을 플롯하십시오. 
7. 함수 $f(\\mathbf{x}) = 3x_1^2 + 5e^{x_2}$의 기울기를 구하십시오. 
8. 함수 $f(\\mathbf{x}) = \|\\mathbf{x}\|_2$의 기울기는 무엇입니까? $\\mathbf{x} = \\mathbf{0}$일 때 어떻게 됩니까? 
9. $u = f(x, y, z)$이고 $x = x(a, b)$, $y = y(a, b)$, $z = z(a, b)$인 경우에 대해 연쇄 법칙을 작성할 수 있습니까? 
10. 역함수가 존재하는 함수 $f(x)$가 주어졌을 때, 역함수 $f^{-1}(x)$의 도함수를 계산하십시오. 
    여기서 우리는 $f^{-1}(f(x)) = x$이고 반대로 $f(f^{-1}(y)) = y$를 갖습니다. 
    힌트: 유도 과정에서 이 속성을 사용하십시오. 

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/197)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17969)
:end_tab:

```