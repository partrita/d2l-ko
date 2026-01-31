```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 데이터 조작 (Data Manipulation)
:label:`sec_ndarray`

어떤 일을 하려면,
데이터를 저장하고 조작할 방법이 필요합니다.
일반적으로 데이터로 해야 할 두 가지 중요한 일이 있습니다:
(i) 데이터를 획득하는 것;
그리고 (ii) 컴퓨터 내부에 들어오면 처리하는 것.
데이터를 저장할 방법 없이 데이터를 획득하는 것은 의미가 없으므로,
시작하기 위해 *텐서(tensor)*라고도 부르는 $n$-차원 배열로
손을 더럽혀 봅시다.
이미 NumPy 과학 컴퓨팅 패키지를 알고 있다면,
이것은 식은 죽 먹기일 것입니다.
모든 최신 딥러닝 프레임워크의 *텐서 클래스*
(MXNet의 `ndarray`, PyTorch와 TensorFlow의 `Tensor`)는
몇 가지 킬러 기능이 추가된 NumPy의 `ndarray`와 유사합니다.
첫째, 텐서 클래스는 자동 미분을 지원합니다.
둘째, NumPy는 CPU에서만 실행되는 반면,
텐서 클래스는 GPU를 활용하여 수치 계산을 가속화합니다.
이러한 속성 덕분에 신경망은 코딩하기 쉽고 실행 속도도 빠릅니다.



## 시작하기 (Getting Started)

:begin_tab:`mxnet`
시작하기 위해, MXNet에서 `np` (`numpy`)와 `npx` (`numpy_extension`) 모듈을 가져옵니다.
여기서 `np` 모듈은 NumPy가 지원하는 함수를 포함하고,
`npx` 모듈은 NumPy와 유사한 환경 내에서 딥러닝을 지원하기 위해 개발된 확장 세트를 포함합니다.
텐서를 사용할 때, 우리는 거의 항상 `set_np` 함수를 호출합니다:
이것은 MXNet의 다른 구성 요소에 의한 텐서 처리의 호환성을 위한 것입니다.
:end_tab:

:begin_tab:`pytorch`
(**시작하기 위해, PyTorch 라이브러리를 가져옵니다.
패키지 이름은 `torch`입니다.**)
:end_tab:

:begin_tab:`tensorflow`
시작하기 위해, `tensorflow`를 가져옵니다.
간결함을 위해 실무자들은 종종 별칭 `tf`를 할당합니다.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**텐서는 수치 값의 (아마도 다차원) 배열을 나타냅니다.**]
1차원인 경우, 즉 데이터에 하나의 축만 필요한 경우,
텐서를 *벡터(vector)*라고 합니다.
두 개의 축을 가진 텐서를 *행렬(matrix)*이라고 합니다.
축이 $k > 2$인 경우, 우리는 전문적인 이름을 버리고
그냥 $k$차 *텐서($k^	extrm{th}$-order tensor)*라고 부릅니다.

:begin_tab:`mxnet`
MXNet은 값으로 미리 채워진 새 텐서를 생성하기 위한
다양한 함수를 제공합니다.
예를 들어 `arange(n)`을 호출하여,
0(포함)에서 시작하여 `n`(포함되지 않음)으로 끝나는
균등하게 간격을 둔 값의 벡터를 만들 수 있습니다.
기본적으로 간격 크기는 $1$입니다.
달리 명시되지 않는 한, 새 텐서는 메인 메모리에 저장되고
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch는 값으로 미리 채워진 새 텐서를 생성하기 위한
다양한 함수를 제공합니다.
예를 들어 `arange(n)`을 호출하여,
0(포함)에서 시작하여 `n`(포함되지 않음)으로 끝나는
균등하게 간격을 둔 값의 벡터를 만들 수 있습니다.
기본적으로 간격 크기는 $1$입니다.
달리 명시되지 않는 한, 새 텐서는 메인 메모리에 저장되고
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow는 값으로 미리 채워진 새 텐서를 생성하기 위한
다양한 함수를 제공합니다.
예를 들어 `range(n)`을 호출하여,
0(포함)에서 시작하여 `n`(포함되지 않음)으로 끝나는
균등하게 간격을 둔 값의 벡터를 만들 수 있습니다.
기본적으로 간격 크기는 $1$입니다.
달리 명시되지 않는 한, 새 텐서는 메인 메모리에 저장되고
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
이러한 각 값을 텐서의 *요소(element)*라고 합니다.
텐서 `x`에는 12개의 요소가 포함되어 있습니다.
`size` 속성을 통해 텐서의 총 요소 수를 검사할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이러한 각 값을 텐서의 *요소(element)*라고 합니다.
텐서 `x`에는 12개의 요소가 포함되어 있습니다.
`numel` 메서드를 통해 텐서의 총 요소 수를 검사할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
이러한 각 값을 텐서의 *요소(element)*라고 합니다.
텐서 `x`에는 12개의 요소가 포함되어 있습니다.
`size` 함수를 통해 텐서의 총 요소 수를 검사할 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**텐서의 *모양(shape)*에 액세스할 수 있습니다.**)
(각 축을 따른 길이)
`shape` 속성을 검사하여 수행합니다.
여기서는 벡터를 다루고 있으므로,
`shape`에는 단일 요소만 포함되며 크기와 동일합니다.

```{.python .input}
%%tab all
x.shape
```

우리는 `reshape`를 호출하여
[**크기나 값을 변경하지 않고 텐서의 모양을 변경할 수 있습니다.**]
예를 들어, 모양이 (12,)인 벡터 `x`를
모양이 (3, 4)인 행렬 `X`로 변환할 수 있습니다.
이 새 텐서는 모든 요소를 유지하지만 행렬로 재구성합니다.
벡터의 요소는 한 번에 한 행씩 배치되므로
`x[3] == X[0, 3]`입니다.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

`reshape`에 모든 모양 구성 요소를 지정하는 것은 중복입니다.
이미 텐서의 크기를 알고 있으므로, 나머지가 주어지면 모양의 한 구성 요소를 알아낼 수 있습니다.
예를 들어, 크기가 $n$인 텐서와 목표 모양 ($h$, $w$)가 주어지면,
$w = n/h$임을 알 수 있습니다.
모양의 한 구성 요소를 자동으로 추론하려면,
자동으로 추론해야 하는 모양 구성 요소에 `-1`을 넣을 수 있습니다.
우리의 경우 `x.reshape(3, 4)`를 호출하는 대신,
동등하게 `x.reshape(-1, 4)` 또는 `x.reshape(3, -1)`을 호출할 수 있었습니다.

실무자들은 종종 모두 0 또는 1을 포함하도록 초기화된 텐서로 작업해야 합니다.
`zeros` 함수를 통해 [**모든 요소가 0으로 설정된 텐서를 구성할 수 있습니다.**]
그리고 모양은 (2, 3, 4)입니다.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

마찬가지로 `ones`를 호출하여
모두 1인 텐서를 생성할 수 있습니다.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

우리는 종종 주어진 확률 분포에서
[**각 요소를 무작위로 (그리고 독립적으로) 샘플링**]하고 싶어 합니다.
예를 들어, 신경망의 파라미터는 종종 무작위로 초기화됩니다.
다음 스니펫은 평균이 0이고 표준 편차가 1인
표준 가우스(정규) 분포에서 추출한 요소로 텐서를 생성합니다.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# JAX에서 무작위 함수를 호출하려면 키를 지정해야 합니다.
# 무작위 함수에 동일한 키를 제공하면 항상 동일한 샘플이 생성됩니다.
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

마지막으로, 수치 리터럴을 포함하는 (아마도 중첩된) Python 리스트를 제공하여
[**각 요소에 대한 정확한 값을 제공**]함으로써 텐서를 구성할 수 있습니다.
여기서 우리는 리스트의 리스트로 행렬을 구성하는데,
가장 바깥쪽 리스트는 축 0에 해당하고 안쪽 리스트는 축 1에 해당합니다.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 인덱싱 및 슬라이싱 (Indexing and Slicing)

Python 리스트와 마찬가지로,
인덱싱(0부터 시작)을 통해 텐서 요소에 액세스할 수 있습니다.
리스트 끝을 기준으로 한 위치를 기준으로 요소에 액세스하려면
음수 인덱싱을 사용할 수 있습니다.
마지막으로, 슬라이싱(예: `X[start:stop]`)을 통해 전체 인덱스 범위에 액세스할 수 있으며,
반환된 값에는 첫 번째 인덱스(`start`)는 포함되지만 *마지막 인덱스*(`stop`)는 포함되지 않습니다.
마지막으로, $k$차 텐서에 대해 하나의 인덱스(또는 슬라이스)만 지정되면,
축 0을 따라 적용됩니다.
따라서 다음 코드에서
[**`[-1]`은 마지막 행을 선택하고 `[1:3]`은 두 번째와 세 번째 행을 선택합니다.**]

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
읽는 것 외에도, (**인덱스를 지정하여 행렬의 요소를 *쓸(write)* 수도 있습니다.**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow의 `Tensor`는 불변(immutable)이므로 할당할 수 없습니다.
TensorFlow의 `Variable`은 할당을 지원하는 변경 가능한(mutable) 상태 컨테이너입니다.
TensorFlow의 기울기는 `Variable` 할당을 통해 역방향으로 흐르지 않는다는 점을 명심하십시오.

전체 `Variable`에 값을 할당하는 것 외에도 인덱스를 지정하여 `Variable`의 요소를 쓸 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX 배열은 불변입니다. jax.numpy.ndarray.at 인덱스 업데이트 연산자는
# 해당 수정 사항이 적용된 새 배열을 만듭니다.
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

[**여러 요소에 동일한 값을 할당하려면,
할당 연산의 왼쪽에 인덱싱을 적용합니다.**]
예를 들어 `[:2, :]`는 첫 번째와 두 번째 행에 액세스하며,
여기서 `:`는 축 1(열)을 따라 모든 요소를 가져옵니다.
행렬에 대한 인덱싱을 논의했지만,
이는 벡터와 2차원 이상의 텐서에서도 작동합니다.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## 연산 (Operations)

이제 텐서를 구성하는 방법과
요소를 읽고 쓰는 방법을 알았으므로,
다양한 수학적 연산으로 텐서를 조작할 수 있습니다.
이 중 가장 유용한 것은 *요소별(elementwise)* 연산입니다.
이것은 텐서의 각 요소에 표준 스칼라 연산을 적용합니다.
두 개의 텐서를 입력으로 받는 함수의 경우,
요소별 연산은 해당 요소의 각 쌍에 표준 이진 연산자를 적용합니다.
스칼라에서 스칼라로 매핑하는 모든 함수에서
요소별 함수를 만들 수 있습니다.

수학적 표기법에서, 우리는 이러한
*단항(unary)* 스칼라 연산자(하나의 입력을 받음)를
서명 $f: \mathbb{R} \rightarrow \mathbb{R}$로 나타냅니다.
이것은 함수가 임의의 실수를 다른 실수로 매핑한다는 것을 의미합니다.
$e^x$와 같은 단항 연산자를 포함한 대부분의 표준 연산자는 요소별로 적용될 수 있습니다.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

마찬가지로, 우리는 실수의 쌍을 (단일) 실수로 매핑하는
*이진(binary)* 스칼라 연산자를
서명 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$로 나타냅니다.
*모양이 같은* 임의의 두 벡터 $\mathbf{u}$와 $\mathbf{v}$와
이진 연산자 $f$가 주어지면, 모든 $i$에 대해 $c_i \gets f(u_i, v_i)$를 설정하여
벡터 $\mathbf{c} = F(\mathbf{u},\mathbf{v})$를 생성할 수 있습니다.
여기서 $c_i, u_i, v_i$는 벡터 $\mathbf{c}, \mathbf{u}, \mathbf{v}$의 $i^\textrm{th}$ 요소입니다.
여기서 우리는 스칼라 함수를 요소별 벡터 연산으로 *리프팅(lifting)*하여
벡터 값 함수 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$를 생성했습니다.
덧셈(`+`), 뺄셈(`-`), 곱셈(`*`), 나눗셈(`/`), 거듭제곱(`**`)에 대한
일반적인 표준 산술 연산자는
임의의 모양의 동일한 모양을 가진 텐서에 대해 모두 요소별 연산으로 *리프팅*되었습니다.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

요소별 계산 외에도,
내적 및 행렬 곱셈과 같은 선형 대수 연산을 수행할 수도 있습니다.
이에 대해서는 :numref:`sec_linear-algebra`에서 자세히 설명하겠습니다.

우리는 또한 [***여러 텐서를 연결(concatenate)*하여**]
더 큰 텐서를 형성하기 위해 끝과 끝을 쌓을 수 있습니다.
텐서 리스트를 제공하고 시스템에 어떤 축을 따라 연결할지 알려주기만 하면 됩니다.
아래 예제는 열(축 1) 대신 행(축 0)을 따라 두 행렬을 연결할 때 어떤 일이 발생하는지 보여줍니다.
첫 번째 출력의 축 0 길이($6$)는 두 입력 텐서의 축 0 길이의 합($3 + 3$)이고,
두 번째 출력의 축 1 길이($8$)는 두 입력 텐서의 축 1 길이의 합($4 + 4$)임을 알 수 있습니다.

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

때로는 [*논리문*을 통해 이진 텐서를 구성]하고 싶을 때가 있습니다.
`X == Y`를 예로 들어보겠습니다.
각 위치 `i, j`에 대해 `X[i, j]`와 `Y[i, j]`가 같으면, 
결과의 해당 항목은 값 `1`을 취하고,
그렇지 않으면 값 `0`을 취합니다.

```{.python .input}
%%tab all
X == Y
```

[**텐서의 모든 요소를 합하면**] 요소가 하나만 있는 텐서가 생성됩니다.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## 브로드캐스팅 (Broadcasting)
:label:`subsec_broadcasting`

지금까지 여러분은 동일한 모양의 두 텐서에 대해
요소별 이진 연산을 수행하는 방법을 알고 있습니다.
특정 조건 하에서,
모양이 다르더라도
우리는 여전히 [***브로드캐스팅 메커니즘(broadcasting mechanism)*을 호출하여
요소별 이진 연산을 수행할 수 있습니다.**]
브로드캐스팅은 다음 두 단계 절차에 따라 작동합니다:
(i) 길이가 1인 축을 따라 요소를 복사하여 하나 또는 두 배열을 모두 확장하여
이 변환 후 두 텐서가 동일한 모양을 갖도록 합니다;
(ii) 결과 배열에 대해 요소별 연산을 수행합니다.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

`a`와 `b`는 각각 $3\times1$ 및 $1\times2$ 행렬이므로
모양이 일치하지 않습니다.
브로드캐스팅은 요소별로 더하기 전에
행렬 `a`를 열을 따라 복제하고
행렬 `b`를 행을 따라 복제하여
더 큰 $3\times2$ 행렬을 생성합니다.

```{.python .input}
%%tab all
a + b
```

## 메모리 절약 (Saving Memory)

[**연산을 실행하면 결과를 호스팅하기 위해 새 메모리가 할당될 수 있습니다.**]
예를 들어 `Y = X + Y`라고 쓰면,
`Y`가 가리키던 텐서의 참조를 해제하고
대신 `Y`가 새로 할당된 메모리를 가리키게 합니다.
우리는 참조된 객체의 정확한 메모리 주소를 제공하는
Python의 `id()` 함수로 이 문제를 시연할 수 있습니다.
`Y = Y + X`를 실행한 후,
`id(Y)`가 다른 위치를 가리킨다는 점에 유의하십시오.
이는 Python이 먼저 `Y + X`를 평가하여
결과를 위한 새 메모리를 할당한 다음
`Y`가 이 새 메모리 위치를 가리키게 하기 때문입니다.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

이것은 두 가지 이유로 바람직하지 않을 수 있습니다.
첫째, 우리는 항상 불필요하게 메모리를 할당하며 돌아다니고 싶지 않습니다.
머신러닝에서는 종종 수백 메가바이트의 파라미터를 가지고 있으며
초당 여러 번 업데이트합니다.
가능할 때마다 이러한 업데이트를 *제자리에서(in place)* 수행하기를 원합니다.
둘째, 여러 변수에서 동일한 파라미터를 가리킬 수 있습니다.
제자리에서 업데이트하지 않으면,
메모리 누수가 발생하거나 실수로 오래된 파라미터를 참조하지 않도록
이러한 모든 참조를 신중하게 업데이트해야 합니다.

:begin_tab:`mxnet, pytorch`
다행히도 (**제자리 연산을 수행하는 것**)은 쉽습니다.
슬라이스 표기법 `Y[:] = <expression>`을 사용하여
이전에 할당된 배열 `Y`에 연산 결과를 할당할 수 있습니다.
이 개념을 설명하기 위해,
`zeros_like`를 사용하여 `Y`와 같은 모양을 갖도록 초기화한 후
텐서 `Z`의 값을 덮어씁니다.
:end_tab:

:begin_tab:`tensorflow`
`Variables`는 TensorFlow에서 변경 가능한 상태 컨테이너입니다.
모델 파라미터를 저장하는 방법을 제공합니다.
`assign`을 사용하여 연산 결과를 `Variable`에 할당할 수 있습니다.
이 개념을 설명하기 위해,
`zeros_like`를 사용하여 `Y`와 같은 모양을 갖도록 초기화한 후
`Variable` `Z`의 값을 덮어씁니다.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX 배열은 제자리 연산을 허용하지 않습니다
```

:begin_tab:`mxnet, pytorch`
[**후속 계산에서 `X`의 값이 재사용되지 않는 경우,
`X[:] = X + Y` 또는 `X += Y`를 사용하여
연산의 메모리 오버헤드를 줄일 수도 있습니다.**]
:end_tab:

:begin_tab:`tensorflow`
`Variable`에 상태를 영구적으로 저장하더라도, 
모델 파라미터가 아닌 텐서에 대한 초과 할당을 피하여
메모리 사용량을 더 줄이고 싶을 수 있습니다.
TensorFlow `Tensors`는 불변이고
기울기가 `Variable` 할당을 통해 흐르지 않기 때문에, 
TensorFlow는 개별 연산을 제자리에서 실행하는 명시적인 방법을 제공하지 않습니다.

그러나 TensorFlow는 `tf.function` 데코레이터를 제공하여
실행하기 전에 컴파일되고 최적화되는 TensorFlow 그래프 내부의 계산을 래핑합니다.
이를 통해 TensorFlow는 사용되지 않는 값을 정리하고,
더 이상 필요하지 않은 이전 할당을 재사용할 수 있습니다.
이는 TensorFlow 계산의 메모리 오버헤드를 최소화합니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # 이 사용되지 않는 값은 제거됩니다
    A = X + Y  # 더 이상 필요하지 않을 때 할당이 재사용됩니다
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 다른 Python 객체로의 변환

:begin_tab:`mxnet, tensorflow`
[**NumPy 텐서(`ndarray`)로 변환하기**] 또는 그 반대로 변환하는 것은 쉽습니다.
변환된 결과는 메모리를 공유하지 않습니다.
이 사소한 불편함은 실제로 꽤 중요합니다: 
CPU나 GPU에서 연산을 수행할 때, 
Python의 NumPy 패키지가 동일한 메모리 청크로
다른 작업을 하고 싶어 할지 확인하기 위해
계산을 중단하고 기다리고 싶지 않기 때문입니다.
:end_tab:

:begin_tab:`pytorch`
[**NumPy 텐서(`ndarray`)로 변환하기**] 또는 그 반대로 변환하는 것은 쉽습니다.
토치 텐서와 NumPy 배열은 기본 메모리를 공유하며,
제자리 연산을 통해 하나를 변경하면 다른 하나도 변경됩니다.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

(**크기가 1인 텐서를 Python 스칼라로 변환**)하려면,
`item` 함수나 Python의 내장 함수를 호출할 수 있습니다.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## 요약

텐서 클래스는 딥러닝 라이브러리에서 데이터를 저장하고 조작하기 위한 주요 인터페이스입니다.
텐서는 생성 루틴, 인덱싱 및 슬라이싱, 기본 수학 연산, 브로드캐스팅, 메모리 효율적인 할당, 다른 Python 객체와의 상호 변환을 포함한 다양한 기능을 제공합니다.


## 연습 문제

1. 이 섹션의 코드를 실행하십시오. 조건문 `X == Y`를 `X < Y` 또는 `X > Y`로 변경하고 어떤 종류의 텐서를 얻을 수 있는지 확인하십시오.
2. 브로드캐스팅 메커니즘에서 요소별로 작동하는 두 텐서를 다른 모양(예: 3차원 텐서)으로 대체하십시오. 결과가 예상과 같습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17966)
:end_tab:

```