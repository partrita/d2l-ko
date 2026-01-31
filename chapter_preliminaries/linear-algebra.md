```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 대수 (Linear Algebra)
:label:`sec_linear-algebra`

지금까지 우리는 데이터셋을 텐서로 로드하고
기본적인 수학 연산으로 텐서를 조작할 수 있었습니다.
정교한 모델을 구축하기 시작하려면,
선형 대수의 몇 가지 도구도 필요합니다.
이 섹션은 스칼라 산술에서 시작하여 행렬 곱셈까지
가장 필수적인 개념에 대한 부드러운 소개를 제공합니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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

## 스칼라 (Scalars)


대부분의 일상적인 수학은
숫자를 한 번에 하나씩 조작하는 것으로 구성됩니다.
공식적으로 우리는 이러한 값을 *스칼라(scalars)*라고 부릅니다.
예를 들어 팰로앨토의 온도는
화씨 $72$도의 온화한 날씨입니다.
온도를 섭씨로 변환하려면
$f$를 $72$로 설정하여 표현식 $c = \frac{5}{9}(f - 32)$를 평가합니다.
이 방정식에서 값 $5$, $9$, $32$는 상수 스칼라입니다.
변수 $c$와 $f$는 일반적으로 알려지지 않은 스칼라를 나타냅니다.

우리는 일반적인 소문자(예: $x$, $y$, $z$)로 스칼라를 나타내고,
모든 (연속적인) *실수 값* 스칼라의 공간을 $\mathbb{R}$로 나타냅니다.
편의를 위해 *공간(spaces)*에 대한 엄격한 정의는 건너뛰겠습니다:
표현식 $x \in \mathbb{R}$은 $x$가 실수 값 스칼라라는 것을 말하는
공식적인 방법이라는 것만 기억하십시오.
기호 $\in$ ("in"으로 발음)은 집합의 멤버십을 나타냅니다.
예를 들어 $x, y \in \{0, 1\}$은
$x$와 $y$가 $0$ 또는 $1$ 값만 취할 수 있는 변수임을 나타냅니다.

(**스칼라는 하나의 요소만 포함하는 텐서로 구현됩니다.**) 아래에서는 두 개의 스칼라를 할당하고
친숙한 덧셈, 곱셈, 나눗셈, 거듭제곱 연산을 수행합니다.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## 벡터 (Vectors)

현재 목적을 위해, [**벡터를 고정 길이의 스칼라 배열로 생각할 수 있습니다.**] 
코드 대응물과 마찬가지로,
우리는 이러한 스칼라를 벡터의 *요소(elements)*라고 부릅니다
(*항목(entries)* 및 *성분(components)*과 동의어). 벡터가 실제 데이터셋의 예제를 나타낼 때,
그 값은 실제적인 중요성을 갖습니다.
예를 들어, 대출 채무 불이행 위험을 예측하는 모델을 훈련하는 경우,
각 신청자를 소득, 고용 기간, 이전 채무 불이행 횟수와 같은
수량에 해당하는 성분을 가진 벡터와 연관시킬 수 있습니다.
심장마비 위험을 연구하는 경우,
각 벡터는 환자를 나타낼 수 있으며
그 성분은 가장 최근의 활력 징후, 콜레스테롤 수치,
하루 운동 시간 등에 해당할 수 있습니다. 우리는 굵은 소문자(예: $\mathbf{x}$, $\mathbf{y}$, $\mathbf{z}$)로 벡터를 나타냅니다.

벡터는 1차 텐서로 구현됩니다. 일반적으로 이러한 텐서는 메모리 제한에 따라 임의의 길이를 가질 수 있습니다. 주의: 대부분의 프로그래밍 언어와 마찬가지로 Python에서 벡터 인덱스는 $0$에서 시작하며(*0 기반 인덱싱*이라고도 함), 선형 대수에서는 아래첨자가 $1$에서 시작합니다(*1 기반 인덱싱*).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

아래첨자를 사용하여 벡터의 요소를 참조할 수 있습니다.
예를 들어 $x_2$는 $\mathbf{x}$의 두 번째 요소를 나타냅니다.
$x_2$는 스칼라이므로 굵게 표시하지 않습니다.
기본적으로 우리는 요소를 수직으로 쌓아 벡터를 시각화합니다:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \ \vdots  \\x_{n}\end{bmatrix}.$$ 
:eqlabel:`eq_vec_def`

여기서 $x_1, \ldots, x_n$은 벡터의 요소입니다. 나중에 우리는 이러한 *열 벡터(column vectors)*와
요소가 수평으로 쌓인 *행 벡터(row vectors)*를 구별할 것입니다. [**인덱싱을 통해 텐서의 요소에 액세스한다**]는 것을 상기하십시오.

```{.python .input}
%%tab all
x[2]
```

벡터에 $n$개의 요소가 포함되어 있음을 나타내기 위해
$\mathbf{x} \in \mathbb{R}^n$이라고 씁니다. 공식적으로 우리는 $n$을 벡터의 *차원(dimensionality)*이라고 부릅니다.
[**코드에서 이는 텐서의 길이에 해당하며**], Python의 내장 `len` 함수를 통해 액세스할 수 있습니다.

```{.python .input}
%%tab all
len(x)
```

`shape` 속성을 통해서도 길이에 액세스할 수 있습니다.
모양은 각 축을 따른 텐서의 길이를 나타내는 튜플입니다.
(**축이 하나만 있는 텐서는 하나의 요소만 있는 모양을 갖습니다.**)

```{.python .input}
%%tab all
x.shape
```

종종 "차원(dimension)"이라는 단어는
축의 수와 특정 축을 따른 길이를 모두 의미하도록 과부하됩니다.
이러한 혼란을 피하기 위해,
우리는 축의 수를 나타낼 때는 *차수(order)*를 사용하고,
구성 요소의 수를 나타낼 때는 *차원(dimensionality)*을 독점적으로 사용합니다.


## 행렬 (Matrices)

스칼라가 0차 텐서이고
벡터가 1차 텐서인 것처럼,
행렬은 2차 텐서입니다. 우리는 굵은 대문자(예: $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{Z}$)로 행렬을 나타내고,
코드에서는 두 개의 축을 가진 텐서로 나타냅니다. 표현식 $\mathbf{A} \in \mathbb{R}^{m \times n}$은
행렬 $\mathbf{A}$가 $m$개의 행과 $n$개의 열로 배열된
$m \times n$개의 실수 값 스칼라를 포함함을 나타냅니다. $m = n$일 때, 우리는 행렬이 *정사각(square)*이라고 말합니다. 시각적으로 우리는 모든 행렬을 표로 설명할 수 있습니다. 개별 요소를 참조하려면 행과 열 인덱스를 모두 아래첨자로 사용합니다. 예:
$a_{ij}$는 $\mathbf{A}$의 $i$번째 행과 $j$번째 열에 속하는 값입니다:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \ a_{21} & a_{22} & \cdots & a_{2n} \ \vdots & \vdots & \ddots & \vdots \ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}.$$ 
:eqlabel:`eq_matrix_def`


코드에서 우리는 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$을
모양 ($m$, $n$)을 가진 2차 텐서로 나타냅니다. 원하는 모양을 `reshape`에 전달하여
[**적절한 크기의 $m \times n$ 텐서를 $m \times n$ 행렬로 변환할 수 있습니다**]:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

때때로 우리는 축을 뒤집고 싶을 때가 있습니다. 행렬의 행과 열을 교환하면,
그 결과를 *전치(transpose)*라고 합니다. 공식적으로 행렬 $\mathbf{A}$의 전치를 $\mathbf{A}^\top$로 표시하고,
$\\mathbf{B} = \mathbf{A}^\top$이면 모든 $i$와 $j$에 대해 $b_{ij} = a_{ji}$입니다. 따라서 $m \times n$ 행렬의 전치는 $n \times m$ 행렬입니다:

$$ \mathbf{A}^\top = \begin{bmatrix} a_{11} & a_{21} & \dots  & a_{m1} \ a_{12} & a_{22} & \dots  & a_{m2} \ \vdots & \vdots & \ddots  & \vdots \ a_{1n} & a_{2n} & \dots  & a_{mn} \end{bmatrix}. $$ 

코드에서는 다음과 같이 모든 (**행렬의 전치**)에 액세스할 수 있습니다:

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**대칭 행렬(Symmetric matrices)은 자신의 전치와 동일한 정사각 행렬의 하위 집합입니다:
$\\mathbf{A} = \mathbf{A}^\top$.**] 다음 행렬은 대칭입니다:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

행렬은 데이터셋을 나타내는 데 유용합니다. 일반적으로 행은 개별 레코드에 해당하고
열은 별개의 속성에 해당합니다.



## 텐서 (Tensors)

스칼라, 벡터, 행렬만으로도 머신러닝 여정을 멀리 갈 수 있지만,
결국에는 고차 [**텐서**]를 다뤄야 할 수도 있습니다. 텐서는 (**$n$차 배열로의 확장을 설명하는 일반적인 방법을 제공합니다.**) 우리는 텐서 클래스의 소프트웨어 객체도 임의의 수의 축을 가질 수 있기 때문에
정확히 "텐서"라고 부릅니다. 수학적 객체와 코드에서의 구현 모두에 *텐서*라는 단어를 사용하는 것이 혼란스러울 수 있지만,
우리의 의미는 일반적으로 문맥상 명확해야 합니다. 우리는 일반적인 텐서를 특수 글꼴(예: $\mathsf{X}$, $\mathsf{Y}$, $\mathsf{Z}$)을 사용한 대문자로 나타내며,
인덱싱 메커니즘(예: $x_{ijk}$ 및 $[\\mathsf{X}]_{1, 2i-1, 3}$)은
행렬의 메커니즘을 자연스럽게 따릅니다.

이미지로 작업하기 시작하면 텐서가 더 중요해질 것입니다. 각 이미지는 높이, 너비, *채널*에 해당하는 축을 가진 3차 텐서로 도착합니다. 각 공간 위치에서 각 색상(빨강, 초록, 파랑)의 강도는 채널을 따라 쌓입니다. 또한 이미지 모음은 코드에서 4차 텐서로 표현되며,
여기서 별개의 이미지는 첫 번째 축을 따라 인덱싱됩니다. 고차 텐서는 벡터 및 행렬과 마찬가지로
모양 구성 요소의 수를 늘려 구성됩니다.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## 텐서 산술의 기본 속성

스칼라, 벡터, 행렬, 그리고 고차 텐서는
모두 몇 가지 편리한 속성을 가지고 있습니다. 예를 들어, 요소별 연산은
피연산자와 동일한 모양을 가진 출력을 생성합니다.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # 새 메모리를 할당하여 A의 사본을 B에 할당
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # 새 메모리를 할당하여 A의 사본을 B에 할당
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # 새 메모리를 할당하여 A를 B에 복제하지 않음
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

[**두 행렬의 요소별 곱을 *하다마드 곱(Hadamard product)*이라고 합니다**] ($\\odot$로 표시). 두 행렬 $\\mathbf{A}, \\mathbf{B} \in \mathbb{R}^{m \times n}$의 하다마드 곱의 항목은 다음과 같습니다:



$$ \mathbf{A} \odot \mathbf{B} = \begin{bmatrix} a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \ a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \ \vdots & \vdots & \ddots & \vdots \ a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn} \end{bmatrix}. $$ 

```{.python .input}
%%tab all
A * B
```

[**스칼라와 텐서를 더하거나 곱하면**] 원래 텐서와 동일한 모양을 가진 결과가 생성됩니다. 여기서 텐서의 각 요소는 스칼라에 더해지거나 곱해집니다.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## 축소 (Reduction)
:label:`subsec_lin-alg-reduction`

종종 우리는 [**텐서 요소의 합을 계산하고 싶어 합니다.**] 길이가 $n$인 벡터 $\\mathbf{x}$의 요소 합을 표현하기 위해
$\\sum_{i=1}^n x_i$라고 씁니다. 이를 위한 간단한 함수가 있습니다:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

[**임의의 모양을 가진 텐서 요소의 합**]을 표현하기 위해,
우리는 단순히 모든 축에 대해 합계를 구합니다. 예를 들어, $m \times n$ 행렬 $\\mathbf{A}$의 요소 합은
$\\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$로 쓸 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

기본적으로 합계 함수를 호출하면
텐서를 모든 축을 따라 *축소(reduce)*하여
결국 스칼라를 생성합니다. 우리의 라이브러리는 또한 [**텐서가 축소되어야 할 축을 지정**]할 수 있게 해줍니다. 행(축 0)을 따라 모든 요소를 합산하려면,
`sum`에서 `axis=0`을 지정합니다. 입력 행렬이 축 0을 따라 축소되어 출력 벡터를 생성하므로,
이 축은 출력 모양에서 사라집니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1`을 지정하면 모든 열의 요소를 합산하여 열 차원(축 1)을 축소합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

합계를 통해 행과 열 모두를 따라 행렬을 축소하는 것은
행렬의 모든 요소를 합산하는 것과 같습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # A.sum()과 동일
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # tf.reduce_sum(A)와 동일
```

[**관련된 양은 *평균(mean)*이며 *평균(average)*이라고도 합니다.**] 우리는 합계를 총 요소 수로 나누어 평균을 계산합니다. 평균을 계산하는 것은 매우 일반적이기 때문에,
`sum`과 유사하게 작동하는 전용 라이브러리 함수가 있습니다.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

마찬가지로 평균을 계산하는 함수도
특정 축을 따라 텐서를 축소할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## 비축소 합계 (Non-Reduction Sum)
:label:`subsec_lin-alg-non-reduction`

합계나 평균을 계산하는 함수를 호출할 때
[**축의 수를 변경하지 않고 유지**]하는 것이 유용할 때가 있습니다. 이것은 브로드캐스트 메커니즘을 사용하고 싶을 때 중요합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

예를 들어, `sum_A`는 각 행을 합산한 후에도 두 축을 유지하므로,
우리는 (**브로드캐스팅을 사용하여 `A`를 `sum_A`로 나누어**) 각 행의 합이 $1$이 되는 행렬을 만들 수 있습니다.

```{.python .input}
%%tab all
A / sum_A
```

[**어떤 축을 따라 `A` 요소의 누적 합을 계산하고 싶다면**],
예를 들어 `axis=0` (행별로), `cumsum` 함수를 호출할 수 있습니다. 설계상, 이 함수는 어떤 축을 따라 입력 텐서를 축소하지 않습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## 내적 (Dot Products)

지금까지 우리는 요소별 연산, 합계, 평균만 수행했습니다. 이것이 우리가 할 수 있는 전부라면 선형 대수는 별도의 섹션을 가질 자격이 없을 것입니다. 다행히도 여기서부터 상황이 더 흥미로워집니다. 가장 기본적인 연산 중 하나는 내적입니다. 두 벡터 $\\mathbf{x}, \\mathbf{y} \in \mathbb{R}^d$가 주어졌을 때, 그들의 *내적(dot product)* $\\mathbf{x}^\top \\mathbf{y}$ (*내적(inner product)* $\\langle \\mathbf{x}, \\mathbf{y}  \\rangle$라고도 함)는 동일한 위치에 있는 요소들의 곱에 대한 합입니다:
$\\mathbf{x}^\top \\mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~두 벡터의 *내적*은 같은 위치에 있는 요소들의 곱에 대한 합입니다~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

동등하게, (**요소별 곱셈을 수행한 다음 합계를 구하여 두 벡터의 내적을 계산할 수 있습니다:**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

내적은 광범위한 맥락에서 유용합니다. 예를 들어, 벡터 $\\mathbf{x}  \\in \\mathbb{R}^n$으로 표시되는 일련의 값과
$\\mathbf{w} \in \\mathbb{R}^n$으로 표시되는 일련의 가중치가 주어졌을 때, 가중치 $\\mathbf{w}$에 따른 $\\mathbf{x}$ 값의 가중 합은
내적 $\\mathbf{x}^\top \\mathbf{w}$로 표현될 수 있습니다. 가중치가 음수가 아니고
합이 $1$일 때, 즉 $\\left(\\sum_{i=1}^{n} {w_i} = 1\\right)$,
내적은 *가중 평균(weighted average)*을 나타냅니다. 단위 길이를 갖도록 두 벡터를 정규화한 후,
내적은 두 벡터 사이 각도의 코사인을 나타냅니다. 이 섹션의 뒷부분에서 이 *길이* 개념을 공식적으로 소개할 것입니다.


## 행렬-벡터 곱 (Matrix--Vector Products)

이제 내적을 계산하는 방법을 알았으므로,
$m \times n$ 행렬 $\\mathbf{A}$와 $n$차원 벡터 $\\mathbf{x}$ 사이의 *곱*을 이해할 수 있습니다. 시작하기 위해, 우리는 행렬을 행 벡터로 시각화합니다

$$\\mathbf{A}= \begin{bmatrix} \\mathbf{a}^\top_{1} \\ \\mathbf{a}^\top_{2} \\ \\vdots \\ \\mathbf{a}^\top_m \\ \end{bmatrix},$$ 

여기서 각 $\\mathbf{a}^\top_{i} \in \\mathbb{R}^n$은 행렬 $\\mathbf{A}$의 $i$번째 행을 나타내는 행 벡터입니다.

[**행렬-벡터 곱 $\\mathbf{A}\\mathbf{x}$는 단순히 길이가 $m$인 열 벡터이며, 그 $i$번째 요소는 내적 $\\mathbf{a}^\top_i \\mathbf{x}$입니다:**]

$$ \mathbf{A}\\mathbf{x} = \begin{bmatrix} \\mathbf{a}^\top_{1} \\ \\mathbf{a}^\top_{2} \\ \\vdots \\ \\mathbf{a}^\top_m \\ \end{bmatrix}\\mathbf{x} = \begin{bmatrix}  \\mathbf{a}^\top_{1} \\mathbf{x}  \\ \\mathbf{a}^\top_{2} \\mathbf{x} \\ \\vdots\\ \\mathbf{a}^\top_{m} \\mathbf{x}\\end{bmatrix}. $$ 

우리는 행렬 $\\mathbf{A}\\in \\mathbb{R}^{m \times n}$과의 곱셈을 벡터를 $\\mathbb{R}^{n}$에서 $\\mathbb{R}^{m}$으로 투영하는 변환으로 생각할 수 있습니다. 이러한 변환은 놀라울 정도로 유용합니다. 예를 들어, 우리는 회전을 특정 정사각 행렬에 의한 곱셈으로 나타낼 수 있습니다. 행렬-벡터 곱은 또한 이전 레이어의 출력이 주어졌을 때 신경망의 각 레이어의 출력을 계산하는 데 관련된 핵심 계산을 설명합니다.

:begin_tab:`mxnet`
코드에서 행렬-벡터 곱을 표현하기 위해,
우리는 동일한 `dot` 함수를 사용합니다. 연산은 인수의 유형에 따라 추론됩니다. `A`의 열 차원(축 1을 따른 길이)은
`x`의 차원(길이)과 같아야 합니다.
:end_tab:

:begin_tab:`pytorch`
코드에서 행렬-벡터 곱을 표현하기 위해,
우리는 `mv` 함수를 사용합니다. `A`의 열 차원(축 1을 따른 길이)은
`x`의 차원(길이)과 같아야 합니다. Python에는 행렬-벡터 및 행렬-행렬 곱을 모두 실행할 수 있는
편의 연산자 `@`가 있습니다(인수에 따라 다름). 따라서 `A@x`라고 쓸 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
코드에서 행렬-벡터 곱을 표현하기 위해,
우리는 `matvec` 함수를 사용합니다. `A`의 열 차원(축 1을 따른 길이)은
`x`의 차원(길이)과 같아야 합니다.
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## 행렬-행렬 곱셈 (Matrix--Matrix Multiplication)

내적과 행렬-벡터 곱에 익숙해졌다면,
*행렬-행렬 곱셈*은 간단할 것입니다.

두 개의 행렬 $\\mathbf{A} \in \\mathbb{R}^{n \times k}$와
$\\mathbf{B} \in \\mathbb{R}^{k \times m}$이 있다고 가정해 봅시다:

$$\\mathbf{A}=\\begin{bmatrix}
 a_{11} & a_{12} & \\cdots & a_{1k} \\
 a_{21} & a_{22} & \\cdots & a_{2k} \\
\\vdots & \\vdots & \\ddots & \\vdots \\
 a_{n1} & a_{n2} & \\cdots & a_{nk} \\
\\end{bmatrix},
\quad
\\mathbf{B}=\\begin{bmatrix}
 b_{11} & b_{12} & \\cdots & b_{1m} \\
 b_{21} & b_{22} & \\cdots & b_{2m} \\
\\vdots & \\vdots & \\ddots & \\vdots \\
 b_{k1} & b_{k2} & \\cdots & b_{km} \\
\\end{bmatrix}.$$ 


$\\mathbf{a}^\top_{i} \in \\mathbb{R}^k$는
행렬 $\\mathbf{A}$의 $i$번째 행을 나타내는 행 벡터를 나타내고,
$\\mathbf{b}_{j} \in \\mathbb{R}^k$는
행렬 $\\mathbf{B}$의 $j$번째 열에서 나온 열 벡터를 나타냅니다:

$$\\mathbf{A}= \begin{bmatrix}
\\mathbf{a}^\top_{1} \\
\\mathbf{a}^\top_{2} \\
\\vdots \\
\\mathbf{a}^\top_n \\
\end{bmatrix}, 
\quad 
\\mathbf{B}=\\begin{bmatrix}
 \\mathbf{b}_{1} & \\mathbf{b}_{2} & \\cdots & \\mathbf{b}_{m} \\
\end{bmatrix}. $$ 


행렬 곱 $\\mathbf{C} \in \\mathbb{R}^{n \times m}$을 형성하기 위해,
우리는 각 요소 $c_{ij}$를
$\\mathbf{A}$의 $i$번째 행과
$\\mathbf{B}$의 $j$번째 열 사이의 내적,
즉 $\\mathbf{a}^\top_i \\mathbf{b}_j$로 간단히 계산합니다:

$$\\mathbf{C} = \\mathbf{AB} = \begin{bmatrix}
\\mathbf{a}^\top_{1} \\
\\mathbf{a}^\top_{2} \\
\\vdots \\
\\mathbf{a}^\top_n \\
\end{bmatrix}
\\begin{bmatrix}
 \\mathbf{b}_{1} & \\mathbf{b}_{2} & \\cdots & \\mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\\mathbf{a}^\top_{1} \\mathbf{b}_1 & \\mathbf{a}^\top_{1}\\mathbf{b}_2& \\cdots & \\mathbf{a}^\top_{1} \\mathbf{b}_m \\
 \\mathbf{a}^\top_{2}\\mathbf{b}_1 & \\mathbf{a}^\top_{2} \\mathbf{b}_2 & \\cdots & \\mathbf{a}^\top_{2} \\mathbf{b}_m \\
 \\vdots & \\vdots & \\ddots &\\vdots\\
\\mathbf{a}^\top_{n} \\mathbf{b}_1 & \\mathbf{a}^\top_{n}\\mathbf{b}_2& \\cdots& \\mathbf{a}^\top_{n} \\mathbf{b}_m
\\end{bmatrix}. $$ 

[**행렬-행렬 곱셈 $\\mathbf{AB}$를
$m$개의 행렬-벡터 곱을 수행하거나
$m \times n$개의 내적을 수행하고
결과를 연결하여 $n \times m$ 행렬을 형성하는 것으로
생각할 수 있습니다.**] 다음 스니펫에서,
우리는 `A`와 `B`에 대해 행렬 곱셈을 수행합니다. 여기서 `A`는 2개의 행과 3개의 열을 가진 행렬이고,
`B`는 3개의 행과 4개의 열을 가진 행렬입니다. 곱셈 후, 우리는 2개의 행과 4개의 열을 가진 행렬을 얻습니다.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

*행렬-행렬 곱셈*이라는 용어는 종종
*행렬 곱셈*으로 단순화되며, 하다마드 곱과 혼동해서는 안 됩니다.


## 노름 (Norms)
:label:`subsec_lin-algebra-norms`

선형 대수에서 가장 유용한 연산자 중 일부는 *노름(norms)*입니다. 비공식적으로, 벡터의 노름은 벡터가 얼마나 *큰지* 알려줍니다. 예를 들어, $\\ell_2$ 노름은 벡터의 (유클리드) 길이를 측정합니다. 여기서 우리는 벡터의 차원이 아니라 벡터 성분의 크기와 관련된 *크기* 개념을 사용하고 있습니다.

노름은 벡터를 스칼라로 매핑하고 다음 세 가지 속성을 만족하는 함수 $\\ \| \\cdot \\ \|$입니다:

1. 어떤 벡터 $\\mathbf{x}$가 주어졌을 때, 벡터의 (모든 요소)를 스칼라 $\\alpha \\in \\mathbb{R}$로 스케일링하면, 노름도 그에 따라 스케일링됩니다:
   $\\|\\alpha \\mathbf{x}\\| = |\\alpha| \\|\\mathbf{x}\\|$.
2. 어떤 벡터 $\\mathbf{x}$와 $\\mathbf{y}$에 대해서도:
   노름은 삼각 부등식을 만족합니다:
   $\\|\\mathbf{x} + \\mathbf{y}\\| \\leq \\|\\mathbf{x}\\| + \\|\\mathbf{y}\\$.
3. 벡터의 노름은 음이 아니며 벡터가 0일 때만 사라집니다:
   모든 $\\mathbf{x} \\neq 0$에 대해 $\\|\\mathbf{x}\\| > 0$.

많은 함수가 유효한 노름이며 서로 다른 노름은 서로 다른 크기 개념을 인코딩합니다. 초등학교 기하학에서 직각 삼각형의 빗변을 계산할 때 배웠던 유클리드 노름은 벡터 요소의 제곱 합의 제곱근입니다. 공식적으로 이것은 [**$\\ell_2$ *노름***]이라고 하며 다음과 같이 표현됩니다.

(**$$\\|\\mathbf{x}\\|_2 = \\sqrt{\\sum_{i=1}^n x_i^2}.$$**)

`norm` 메서드는 $\\ell_2$ 노름을 계산합니다.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**$\\ell_1$ 노름**]도 일반적이며 관련된 측정값을 맨해튼 거리라고 합니다. 정의에 따라 $\\ell_1$ 노름은 벡터 요소의 절댓값을 합산합니다:

(**$$\\|\\mathbf{x}\\|_1 = \\sum_{i=1}^n \\left|x_i \\right|.$$**)

$\\ell_2$ 노름에 비해 이상값에 덜 민감합니다. $\\ell_1$ 노름을 계산하기 위해,
우리는 절댓값 연산과 합계 연산을 구성합니다.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # jnp.abs(u).sum()과 동일
```

$\\ell_2$ 및 $\\ell_1$ 노름은 모두 더 일반적인 $\\ell_p$ *노름*의 특수한 경우입니다:

$$\\|\\mathbf{x}\\|_p = \\left(\\sum_{i=1}^n \\left|x_i \\right|^p \\right)^{1/p}.$$ 

행렬의 경우 문제는 더 복잡합니다. 결국 행렬은 개별 항목의 모음이자
벡터에 작용하여 다른 벡터로 변환하는 객체로 볼 수 있습니다. 예를 들어, 우리는 행렬-벡터 곱 $\\mathbf{X} \\mathbf{v}$가
$\\mathbf{v}$에 비해 얼마나 더 길어질 수 있는지 물을 수 있습니다. 이러한 생각은 *스펙트럼(spectral)* 노름이라고 불리는 것으로 이어집니다. 지금은 [**계산하기 훨씬 쉬운 *프로베니우스(Frobenius) 노름***]을 소개합니다. 이것은 행렬 요소의 제곱 합의 제곱근으로 정의됩니다:

[**$$\\|\\mathbf{X}\\|_\textrm{F} = \\sqrt{\\sum_{i=1}^m \\sum_{j=1}^n x_{ij}^2}.$$**]

프로베니우스 노름은 마치 행렬 모양 벡터의 $\\ell_2$ 노름인 것처럼 동작합니다. 다음 함수를 호출하면 행렬의 프로베니우스 노름이 계산됩니다.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

너무 앞서 나가고 싶지는 않지만, 우리는 이미 이러한 개념이 유용한 이유에 대한 직관을 심을 수 있습니다. 딥러닝에서 우리는 종종 최적화 문제를 해결하려고 합니다:
관찰된 데이터에 할당된 확률을 *최대화*합니다; 추천 모델과 관련된 수익을 *최대화*합니다; 예측과 실제 관찰 사이의 거리를 *최소화*합니다; 동일한 사람의 사진 표현 간의 거리를 *최소화*하는 동시에
다른 사람의 사진 표현 간의 거리를 *최대화*합니다. 딥러닝 알고리즘의 목표를 구성하는 이러한 거리는
종종 노름으로 표현됩니다.


## 토론

이 섹션에서 우리는 현대 딥러닝의 상당 부분을 이해하는 데 필요한
모든 선형 대수를 검토했습니다. 하지만 선형 대수에는 훨씬 더 많은 내용이 있으며,
그 중 많은 부분이 머신러닝에 유용합니다. 예를 들어 행렬을 인수로 분해할 수 있으며,
이러한 분해는 실제 데이터셋의 저차원 구조를 드러낼 수 있습니다. 데이터셋의 구조를 발견하고 예측 문제를 해결하기 위해
행렬 분해와 고차 텐서로의 일반화를 사용하는 데 초점을 맞춘
머신러닝의 전체 하위 분야가 있습니다. 하지만 이 책은 딥러닝에 초점을 맞춥니다. 그리고 우리는 여러분이 실제 데이터셋에 머신러닝을 적용하며
손을 더럽히고 나면 더 많은 수학을 배우고 싶어질 것이라고 믿습니다. 따라서 나중에 더 많은 수학을 소개할 권리는 보유하지만,
이 섹션은 여기서 마무리합니다.

선형 대수를 더 배우고 싶다면,
훌륭한 책과 온라인 리소스가 많이 있습니다. 더 고급 집중 코스를 원한다면
:citet:`Strang.1993`, :citet:`Kolter.2008`, 및 :citet:`Petersen.Pedersen.ea.2008`을 확인해 보십시오.

요약하자면:

* 스칼라, 벡터, 행렬, 텐서는
  선형 대수에서 사용되는 기본 수학적 객체이며
  각각 0개, 1개, 2개, 그리고 임의의 수의 축을 가지고 있습니다.
* 텐서는 인덱싱을 통해 특정 축을 따라 슬라이스하거나,
  `sum` 및 `mean`과 같은 연산을 통해 축소할 수 있습니다.
* 요소별 곱을 하다마드 곱이라고 합니다.
  반면 내적, 행렬-벡터 곱, 행렬-행렬 곱은
  요소별 연산이 아니며 일반적으로 피연산자와 다른 모양을 가진 객체를 반환합니다.
* 하다마드 곱에 비해 행렬-행렬 곱은
  계산하는 데 상당히 더 오래 걸립니다(2차 시간이 아닌 3차 시간).
* 노름은 벡터(또는 행렬)의 크기에 대한 다양한 개념을 포착하며,
  일반적으로 두 벡터 사이의 거리를 측정하기 위해 두 벡터의 차이에 적용됩니다.
* 일반적인 벡터 노름에는 $\\ell_1$ 및 $\\ell_2$ 노름이 포함되며,
   일반적인 행렬 노름에는 *스펙트럼* 및 *프로베니우스* 노름이 포함됩니다.


## 연습 문제

1. 행렬 전치의 전치는 행렬 자체임을 증명하십시오: $(\\mathbf{A}^\top)^\top = \\mathbf{A}$.
2. 두 행렬 $\\mathbf{A}$와 $\\mathbf{B}$가 주어졌을 때, 합과 전치가 교환 가능함을 보이십시오: $\\mathbf{A}^\top + \\mathbf{B}^\top = (\\mathbf{A} + \\mathbf{B})^\top$.
3. 임의의 정사각 행렬 $\\mathbf{A}$에 대해, $\\mathbf{A} + \\mathbf{A}^\top$는 항상 대칭입니까? 이전 두 연습 문제의 결과만 사용하여 결과를 증명할 수 있습니까?
4. 우리는 이 섹션에서 모양 (2, 3, 4)의 텐서 `X`를 정의했습니다. `len(X)`의 출력은 무엇입니까? 코드를 구현하지 말고 답을 쓴 다음 코드를 사용하여 답을 확인하십시오.
5. 임의의 모양을 가진 텐서 `X`의 경우, `len(X)`는 항상 `X`의 특정 축의 길이에 해당합니까? 그 축은 무엇입니까?
6. `A / A.sum(axis=1)`을 실행하고 무슨 일이 일어나는지 확인하십시오. 결과를 분석할 수 있습니까?
7. 맨해튼 시내의 두 지점 사이를 이동할 때, 좌표 측면에서, 즉 거리와 거리(avenues and streets) 측면에서 커버해야 하는 거리는 얼마입니까? 대각선으로 이동할 수 있습니까?
8. 모양 (2, 3, 4)의 텐서를 고려하십시오. 축 0, 1, 2를 따른 합계 출력의 모양은 무엇입니까?
9. 3개 이상의 축을 가진 텐서를 `linalg.norm` 함수에 입력하고 출력을 관찰하십시오. 이 함수는 임의의 모양을 가진 텐서에 대해 무엇을 계산합니까?
10. 가우스 무작위 변수로 초기화된 $\\mathbf{A} \\in \\mathbb{R}^{2^{10} \\times 2^{16}}$, $\\mathbf{B} \\in \\mathbb{R}^{2^{16} \\times 2^{5}}$, $\\mathbf{C} \\in \\mathbb{R}^{2^{5} \\times 2^{14}}$의 세 개의 큰 행렬을 고려하십시오. 곱 $\\mathbf{A} \\mathbf{B} \\mathbf{C}$를 계산하려고 합니다. $(\\mathbf{A} \\mathbf{B}) \\mathbf{C}$를 계산하는지 아니면 $\\mathbf{A} (\\mathbf{B} \\mathbf{C})$를 계산하는지에 따라 메모리 사용량과 속도에 차이가 있습니까? 왜 그렇습니까?
11. 세 개의 큰 행렬 $\\mathbf{A} \\in \\mathbb{R}^{2^{10} \\times 2^{16}}$, $\\mathbf{B} \\in \\mathbb{R}^{2^{16} \\times 2^{5}}$, $\\mathbf{C} \\in \\mathbb{R}^{2^{5} \\times 2^{16}}$을 고려하십시오. $\\mathbf{A} \\mathbf{B}$를 계산하는지 아니면 $\\mathbf{A} \\mathbf{C}^\top$를 계산하는지에 따라 속도에 차이가 있습니까? 왜 그렇습니까? 메모리를 복제하지 않고 $\\mathbf{C} = \\mathbf{B}^\top$를 초기화하면 어떻게 변합니까? 왜 그렇습니까?
12. 세 행렬 $\\mathbf{A}, \\mathbf{B}, \\mathbf{C} \\in \\mathbb{R}^{100 \\times 200}$을 고려하십시오. $[\\mathbf{A}, \\mathbf{B}, \\mathbf{C}]$를 쌓아서 3개의 축을 가진 텐서를 만듭니다. 차원(dimensionality)은 무엇입니까? 세 번째 축의 두 번째 좌표를 슬라이스하여 $\\mathbf{B}$를 복구하십시오. 답이 맞는지 확인하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17968)
:end_tab:

```