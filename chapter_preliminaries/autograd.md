```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 자동 미분 (Automatic Differentiation)
:label:`sec_autograd`

:numref:`sec_calculus`에서
도함수 계산이 심층 신경망을 훈련하는 데 사용할
모든 최적화 알고리즘의 중요한 단계라는 것을 상기해 보십시오.
계산은 간단하지만, 손으로 계산하는 것은 지루하고 오류가 발생하기 쉬우며,
모델이 더 복잡해짐에 따라 이러한 문제는 커질 뿐입니다.

다행히도 모든 최신 딥러닝 프레임워크는
*자동 미분(automatic differentiation)* (종종 *autograd*로 줄임)을 제공하여
이 작업을 대신해 줍니다.
각 연속적인 함수를 통해 데이터를 전달할 때,
프레임워크는 각 값이 다른 값에 어떻게 의존하는지 추적하는
*계산 그래프(computational graph)*를 구축합니다.
도함수를 계산하기 위해, 자동 미분은
연쇄 법칙을 적용하여 이 그래프를 역방향으로 진행합니다.
이런 방식으로 연쇄 법칙을 적용하는 계산 알고리즘을 *역전파(backpropagation)*라고 합니다.

autograd 라이브러리는 지난 10년 동안 뜨거운 관심사가 되었지만,
오랜 역사를 가지고 있습니다.
사실 autograd에 대한 최초의 언급은 반세기 전으로 거슬러 올라갑니다 :cite:`Wengert.1964`.
현대 역전파의 핵심 아이디어는 1980년 박사 학위 논문 :cite:`Speelpenning.1980`으로 거슬러 올라가며
1980년대 후반에 더욱 발전되었습니다 :cite:`Griewank.1989`.
역전파는 기울기를 계산하는 기본 방법이 되었지만 유일한 옵션은 아닙니다.
예를 들어 Julia 프로그래밍 언어는 순방향 전파를 사용합니다 :cite:`Revels.Lubin.Papamarkou.2016`.
방법을 탐구하기 전에, 먼저 autograd 패키지를 마스터해 봅시다.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## 간단한 함수 (A Simple Function)

우리가 열 벡터 $\mathbf{x}$에 대해
(**함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$를 미분하는 데**) 관심이 있다고 가정해 봅시다.
시작하기 위해 `x`에 초기 값을 할당합니다.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**$\\mathbf{x}$에 대한 $y$의 기울기를 계산하기 전에,
그것을 저장할 장소가 필요합니다.**] 일반적으로 우리는 도함수를 취할 때마다 새 메모리를 할당하는 것을 피합니다.
왜냐하면 딥러닝은 동일한 파라미터에 대해
도함수를 계속해서 매우 여러 번 계산해야 하기 때문이며,
메모리가 부족할 위험이 있습니다. 벡터 $\\mathbf{x}$에 대한 스칼라 값 함수의 기울기는
$\\mathbf{x}$와 동일한 모양을 가진 벡터 값입니다.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# `attach_grad`를 호출하여 텐서의 기울기를 위한 메모리를 할당합니다
x.attach_grad()
# `x`에 대해 취한 기울기를 계산한 후, 0으로 초기화된 값을 가진
# `grad` 속성을 통해 액세스할 수 있습니다
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# x = torch.arange(4.0, requires_grad=True)를 생성할 수도 있습니다
x.requires_grad_(True)
x.grad  # 기울기는 기본적으로 None입니다
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**이제 `x`의 함수를 계산하고 결과를 `y`에 할당합니다.**)

```{.python .input  n=10}
%%tab mxnet
# 코드는 계산 그래프를 구축하기 위해 `autograd.record` 범위 안에 있습니다
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# 모든 계산을 테이프에 기록합니다
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
`backward` 메서드를 호출하여
[**이제 `x`에 대한 `y`의 기울기를 취할 수 있습니다**].
다음으로, `x`의 `grad` 속성을 통해 기울기에 액세스할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
`backward` 메서드를 호출하여
[**이제 `x`에 대한 `y`의 기울기를 취할 수 있습니다**].
다음으로, `x`의 `grad` 속성을 통해 기울기에 액세스할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
`gradient` 메서드를 호출하여
[**이제 `x`에 대한 `y`의 기울기를 계산할 수 있습니다**].
:end_tab:

:begin_tab:`jax`
`grad` 변환을 통과시켜
[**이제 `x`에 대한 `y`의 기울기를 취할 수 있습니다**].
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# `grad` 변환은 원래 함수의 기울기를 계산하는 Python 함수를 반환합니다
x_grad = grad(y)(x)
x_grad
```

(**우리는 이미 $\\mathbf{x}$에 대한 함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$의 기울기가
$4\mathbf{x}$여야 한다는 것을 알고 있습니다.**) 이제 자동 기울기 계산과 예상 결과가 동일한지 확인할 수 있습니다.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**이제 `x`의 다른 함수를 계산하고
기울기를 취해 봅시다.**] MXNet은 우리가 새 기울기를 기록할 때마다
기울기 버퍼를 재설정한다는 점에 유의하십시오.
:end_tab:

:begin_tab:`pytorch`
[**이제 `x`의 다른 함수를 계산하고
기울기를 취해 봅시다.**] PyTorch는 우리가 새 기울기를 기록할 때
자동으로 기울기 버퍼를 재설정하지 않는다는 점에 유의하십시오.
대신, 새 기울기가 이미 저장된 기울기에 추가됩니다.
이 동작은 여러 목적 함수의 합을 최적화하고 싶을 때 유용합니다.
기울기 버퍼를 재설정하려면 다음과 같이 `x.grad.zero_()`를 호출할 수 있습니다:
:end_tab:

:begin_tab:`tensorflow`
[**이제 `x`의 다른 함수를 계산하고
기울기를 취해 봅시다.**] TensorFlow는 우리가 새 기울기를 기록할 때마다
기울기 버퍼를 재설정한다는 점에 유의하십시오.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # 새로 계산된 기울기로 덮어쓰여짐
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # 기울기 재설정
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # 새로 계산된 기울기로 덮어쓰여짐
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## 비스칼라 변수의 역전파 (Backward for Non-Scalar Variables)

`y`가 벡터일 때,
벡터 `x`에 대한 `y`의 도함수의 가장 자연스러운 표현은
`y`의 각 성분의 `x`의 각 성분에 대한 편도함수를 포함하는
*자코비안(Jacobian)*이라는 행렬입니다.
마찬가지로 고차 `y`와 `x`의 경우 미분 결과는 더 높은 차수의 텐서가 될 수 있습니다.

자코비안은 일부 고급 머신러닝 기술에 등장하지만,
더 일반적으로 우리는 전체 벡터 `x`에 대한
`y`의 각 성분의 기울기를 합산하여
`x`와 동일한 모양의 벡터를 생성하기를 원합니다.
예를 들어, 우리는 종종 훈련 예제의 *배치* 중 각 예제에 대해
개별적으로 계산된 손실 함수의 값을 나타내는 벡터를 가지고 있습니다.
여기서 우리는 단지 (**각 예제에 대해 개별적으로 계산된 기울기를 합산**)하기를 원합니다.

:begin_tab:`mxnet`
MXNet은 기울기를 계산하기 전에 합계를 통해 모든 텐서를 스칼라로 축소하여 이 문제를 처리합니다.
즉, 자코비안 $\\partial_{\\mathbf{x}} \\mathbf{y}$를 반환하는 대신,
합의 기울기 $\\partial_{\\mathbf{x}} \\sum_i y_i$를 반환합니다.
:end_tab:

:begin_tab:`pytorch`
딥러닝 프레임워크마다 비스칼라 텐서의 기울기를 해석하는 방식이 다르기 때문에,
PyTorch는 혼란을 피하기 위해 몇 가지 조치를 취합니다.
비스칼라에서 `backward`를 호출하면 PyTorch에 객체를 스칼라로 축소하는 방법을 알려주지 않는 한 오류가 발생합니다.
더 공식적으로, 우리는 `backward`가 $\\partial_{\\mathbf{x}} \\mathbf{y}$ 대신
$\\mathbf{v}^\\top \\partial_{\\mathbf{x}} \\mathbf{y}$를 계산하도록 하는 어떤 벡터 $\\mathbf{v}$를 제공해야 합니다.
이 다음 부분은 혼란스러울 수 있지만, 나중에 명확해질 이유들로 인해
이 인수($\\mathbf{v}$를 나타냄)의 이름은 `gradient`입니다. 자세한 설명은 Yang Zhang의 [Medium 게시물](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29)을 참조하십시오.
:end_tab:

:begin_tab:`tensorflow`
기본적으로 TensorFlow는 합의 기울기를 반환합니다.
즉, 자코비안 $\\partial_{\\mathbf{x}} \\mathbf{y}$를 반환하는 대신,
합의 기울기 $\\partial_{\\mathbf{x}} \\sum_i y_i$를 반환합니다.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # y = sum(x * x)의 기울기와 동일
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # 더 빠름: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # y = tf.reduce_sum(x * x)와 동일
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad는 스칼라 출력 함수에 대해서만 정의됨
grad(lambda x: y(x).sum())(x)
```

## 계산 분리 (Detaching Computation)

때때로 우리는 [**일부 계산을 기록된 계산 그래프 외부로 이동**]하고 싶을 때가 있습니다.
예를 들어, 입력을 사용하여 기울기를 계산하고 싶지 않은
일부 보조 중간 항을 만든다고 가정해 봅시다.
이 경우, 최종 결과에서 해당 계산 그래프를 *분리(detach)*해야 합니다.
다음의 장난감 예제는 이것을 더 명확하게 만듭니다:
`z = x * y`이고 `y = x * x`이지만
`y`를 통해 전달되는 영향보다는 `z`에 대한 `x`의 *직접적인* 영향에 초점을 맞추고 싶다고 가정해 봅시다.
이 경우, `y`와 동일한 값을 갖지만
*출처(provenance)* (어떻게 생성되었는지)가 지워진
새 변수 `u`를 만들 수 있습니다.
따라서 `u`는 그래프에 조상이 없으며 기울기는 `u`를 통해 `x`로 흐르지 않습니다.
예를 들어 `z = x * u`의 기울기를 취하면 결과 `u`를 산출합니다
(`z = x * x * x`이기 때문에 예상할 수 있는 `3 * x * x`가 아님).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# persistent=True로 설정하여 계산 그래프를 보존합니다.
# 이를 통해 t.gradient를 두 번 이상 실행할 수 있습니다.
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax 기본 요소는 XLA 연산에 대한 Python 래퍼입니다
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

이 절차가 `z`로 이어지는 그래프에서
`y`의 조상을 분리하지만, `y`로 이어지는 계산 그래프는 유지되므로
`x`에 대한 `y`의 기울기를 계산할 수 있다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## 기울기와 Python 제어 흐름 (Gradients and Python Control Flow)

지금까지 우리는 `z = x * x * x`와 같은 함수를 통해
입력에서 출력까지의 경로가 잘 정의된 사례를 검토했습니다.
프로그래밍은 결과를 계산하는 방법에 있어 훨씬 더 많은 자유를 제공합니다.
예를 들어, 보조 변수에 의존하게 하거나 중간 결과에 대한 선택을 조건부로 만들 수 있습니다.
자동 미분을 사용하는 이점 중 하나는
(**함수의 계산 그래프를 구축하는 데 Python 제어 흐름의 미로를 통과해야 하더라도**)
(예: 조건문, 루프, 임의 함수 호출), [**결과 변수의 기울기를 여전히 계산할 수 있다는 것입니다.**]
이를 설명하기 위해 `while` 루프의 반복 횟수와 `if` 문의 평가가
모두 입력 `a`의 값에 의존하는 다음 코드 스니펫을 고려해 보십시오.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

아래에서 우리는 무작위 값을 입력으로 전달하여 이 함수를 호출합니다.
입력이 확률 변수이므로, 우리는 계산 그래프가 어떤 형태를 취할지 모릅니다.
그러나 특정 입력에 대해 `f(a)`를 실행할 때마다
특정 계산 그래프를 실현하고 이후에 `backward`를 실행할 수 있습니다.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ()) 
d = f(a)
d_grad = grad(f)(a)
```

비록 우리 함수 `f`가 데모 목적으로 약간 작위적이지만,
입력에 대한 의존성은 꽤 간단합니다: 그것은 부분적으로 정의된 스케일을 가진 `a`의 *선형* 함수입니다.
따라서 `f(a) / a`는 상수 항목의 벡터이며, 게다가 `f(a) / a`는 `a`에 대한 `f(a)`의 기울기와 일치해야 합니다.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

동적 제어 흐름은 딥러닝에서 매우 일반적입니다.
예를 들어 텍스트를 처리할 때 계산 그래프는 입력 길이에 따라 달라집니다.
이러한 경우, 기울기를 *선험적으로(a priori)* 계산하는 것이 불가능하기 때문에
자동 미분은 통계적 모델링에 필수적이 됩니다.

## 토론

여러분은 이제 자동 미분의 힘을 맛보았습니다.
도함수를 자동으로 그리고 효율적으로 계산하는 라이브러리의 개발은
딥러닝 실무자들에게 엄청난 생산성 향상 요인이 되어, 그들이 덜 하찮은 일에 집중할 수 있게 해주었습니다.
게다가 autograd를 사용하면 펜과 종이로 기울기를 계산하는 것이
엄두도 못 낼 정도로 시간이 많이 걸리는 거대한 모델을 설계할 수 있습니다.
흥미롭게도, 우리는 모델을 (통계적 의미에서) *최적화*하기 위해 autograd를 사용하지만,
autograd 라이브러리 자체의 (계산적 의미에서의) *최적화*는
프레임워크 설계자들에게 매우 중요한 관심사인 풍부한 주제입니다.
여기서 가장 신속하고 메모리 효율적인 방식으로 결과를 계산하기 위해
컴파일러와 그래프 조작 도구가 활용됩니다.

지금은 다음 기본 사항을 기억해 두십시오: (i) 도함수를 원하는 변수에 기울기를 연결합니다; (ii) 목표 값의 계산을 기록합니다; (iii) 역전파 함수를 실행합니다; (iv) 결과 기울기에 액세스합니다.


## 연습 문제

1. 2계 도함수가 1계 도함수보다 계산 비용이 훨씬 더 많이 드는 이유는 무엇입니까?
2. 역전파 함수를 실행한 후 즉시 다시 실행하면 어떻게 됩니까? 조사해 보십시오.
3. `a`에 대한 `d`의 도함수를 계산하는 제어 흐름 예제에서 변수 `a`를 무작위 벡터나 행렬로 변경하면 어떻게 됩니까? 이 시점에서 계산 `f(a)`의 결과는 더 이상 스칼라가 아닙니다. 결과에 어떤 일이 발생합니까? 이것을 어떻게 분석합니까?
4. $f(x) = \sin(x)$라고 합시다. $f$와 도함수 $f'$의 그래프를 플롯하십시오. $f'(x) = \cos(x)$라는 사실을 이용하지 말고 자동 미분을 사용하여 결과를 얻으십시오.
5. $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$이라고 합시다. $x$에서 $f(x)$까지 결과를 추적하는 종속성 그래프를 작성하십시오.
6. 연쇄 법칙을 사용하여 앞서 언급한 함수의 도함수 $\\frac{df}{dx}$를 계산하고, 이전에 구성한 종속성 그래프에 각 항을 배치하십시오.
7. 그래프와 중간 도함수 결과가 주어졌을 때, 기울기를 계산할 때 여러 가지 옵션이 있습니다. $x$에서 $f$로 시작하여 한 번, $f$에서 $x$로 추적하여 한 번 결과를 평가하십시오. $x$에서 $f$로의 경로는 일반적으로 *순방향 미분(forward differentiation)*으로 알려져 있으며, $f$에서 $x$로의 경로는 *역방향 미분(backward differentiation)*으로 알려져 있습니다.
8. 언제 순방향 미분을 사용하고 언제 역방향 미분을 사용하고 싶으십니까? 힌트: 필요한 중간 데이터의 양, 단계를 병렬화하는 능력, 관련된 행렬 및 벡터의 크기를 고려하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17970)
:end_tab:
