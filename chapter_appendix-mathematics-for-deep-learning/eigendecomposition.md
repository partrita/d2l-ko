# 고유 분해 (Eigendecompositions)
:label:`sec_eigendecompositions`

고유값(eigenvalues)은 선형 대수를 공부할 때 접하게 되는 가장 유용한 개념 중 하나인 경우가 많지만, 초보자로서 그 중요성을 간과하기 쉽습니다. 
아래에서는 고유 분해를 소개하고 그것이 왜 그렇게 중요한지에 대한 느낌을 전달하려고 노력합니다.

다음과 같은 항목을 가진 행렬 $A$가 있다고 가정해 봅시다.

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\ 0 & -1
\end{bmatrix}.
$$

임의의 벡터 $\mathbf{v} = [x, y]^\top$에 $A$를 적용하면, 
벡터 $\mathbf{A}\mathbf{v} = [2x, -y]^\top$를 얻습니다. 
이것은 직관적인 해석을 갖습니다: 
벡터를 $x$ 방향으로 두 배 넓게 늘리고, $y$ 방향으로 뒤집는 것입니다.

그러나 무언가가 변경되지 않고 유지되는 *일부* 벡터가 있습니다. 
즉, $[1, 0]^\top$은 $[2, 0]^\top$으로 보내지고, 
$[0, 1]^\top$은 $[0, -1]^\top$으로 보내집니다. 
이러한 벡터는 여전히 동일한 직선상에 있으며, 유일한 수정 사항은 행렬이 각각 $2$와 $-1$의 인수로 늘린다는 것입니다. 
우리는 그러한 벡터를 *고유 벡터(eigenvectors)*라고 부르고, 그것들이 늘어나는 인수를 *고유값(eigenvalues)*이라고 부릅니다.

일반적으로 다음과 같은 숫자 $\lambda$와 벡터 $\mathbf{v}$를 찾을 수 있다면

$$ 
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$ 

우리는 $\mathbf{v}$를 $A$에 대한 고유 벡터라고 하고 $\lambda$를 고유값이라고 합니다.

## 고유값 찾기 (Finding Eigenvalues)
그것들을 찾는 방법을 알아봅시다. 양변에서 $\lambda \mathbf{v}$를 빼고 벡터를 인수분해하면 위의 식이 다음과 동등함을 알 수 있습니다.

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der`가 발생하려면 $(\mathbf{A} - \lambda \mathbf{I})$가 어떤 방향을 0으로 압축해야 하므로 가역적이지 않으며, 따라서 행렬식이 0입니다. 
따라서 우리는 $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$이 되는 $\lambda$가 무엇인지 찾음으로써 *고유값*을 찾을 수 있습니다. 
고유값을 찾으면 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$를 풀어 연관된 *고유 벡터*를 찾을 수 있습니다.

### 예제 (An Example)
더 어려운 행렬로 이것을 살펴봅시다.

$$ 
\mathbf{A} = \begin{bmatrix}
2 & 1\\ 2 & 3
\end{bmatrix}.
$$ 

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$을 고려하면, 
이것이 다항식 방정식 $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$와 동등함을 알 수 있습니다. 
따라서 두 개의 고유값은 $4$와 $1$입니다. 
연관된 벡터를 찾으려면 다음을 풀어야 합니다.

$$ 
\begin{bmatrix}
2 & 1\\ 2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{및} \; 
\begin{bmatrix}
2 & 1\\ 2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$ 

우리는 이를 각각 벡터 $[1, -1]^\top$ 및 $[1, 2]^\top$로 풀 수 있습니다.

내장된 `numpy.linalg.eig` 루틴을 사용하여 코드에서 이를 확인할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy`는 고유 벡터의 길이를 1로 정규화하는 반면, 우리는 임의의 길이를 취했다는 점에 유의하십시오. 
또한 부호의 선택은 임의적입니다. 
그러나 계산된 벡터는 동일한 고유값을 가진 우리가 손으로 찾은 벡터와 평행합니다.

## 행렬 분해 (Decomposing Matrices)
이전 예제를 한 단계 더 진행해 봅시다.

$$ 
\mathbf{W} = \begin{bmatrix}
1 & 1 \\ -1 & 2
\end{bmatrix},
$$ 

를 행렬 $\mathbf{A}$의 고유 벡터들이 열인 행렬이라고 합시다. 

$$ 
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\ 0 & 4
\end{bmatrix},
$$ 

를 대각선에 연관된 고유값이 있는 행렬이라고 합시다. 
그러면 고유값과 고유 벡터의 정의는 우리에게 다음을 알려줍니다.

$$ 
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$ 

행렬 $W$는 가역적이므로 양변의 오른쪽에 $W^{-1}$을 곱하면 다음과 같이 쓸 수 있음을 알 수 있습니다.

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$ 
:eqlabel:`eq_eig_decomp`

다음 섹션에서 이것의 몇 가지 좋은 결과를 보겠지만, 지금은 선형 독립인 고유 벡터의 전체 컬렉션을 찾을 수 있는 한(그래서 $W$가 가역적이 되도록) 그러한 분해가 존재한다는 것만 알면 됩니다.

## 고유 분해에 대한 연산 (Operations on Eigendecompositions)
고유 분해 :eqref:`eq_eig_decomp`의 한 가지 좋은 점은 우리가 보통 접하는 많은 연산을 고유 분해의 관점에서 깔끔하게 쓸 수 있다는 것입니다. 첫 번째 예로 다음을 고려하십시오.

$$ 
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ 번}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ 번}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ 번}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$ 

이것은 행렬의 임의의 양수 거듭제곱에 대해, 고유값을 동일한 거듭제곱으로 올림으로써 고유 분해를 얻을 수 있음을 알려줍니다. 
음수 거듭제곱에 대해서도 동일하게 나타낼 수 있으므로, 행렬을 반전시키고 싶다면 다음만 고려하면 됩니다.

$$ 
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$ 

즉, 각 고유값만 반전시키면 됩니다. 
이것은 각 고유값이 0이 아닌 한 작동하므로, 가역적이라는 것이 0인 고유값이 없는 것과 같음을 알 수 있습니다.

실제로 $\lambda_1, \ldots, \lambda_n$이 행렬의 고유값이라면 그 행렬의 행렬식은

$$ 
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$ 

또는 모든 고유값의 곱임을 추가 작업으로 보일 수 있습니다. 
이는 $\mathbf{W}$가 어떤 늘림을 하든 $W^{-1}$이 이를 되돌리므로 결국 발생하는 유일한 늘림은 대각 원소들의 곱으로 부피를 늘리는 대각 행렬 $\boldsymbol{\Sigma}$에 의한 곱셈이기 때문에 직관적으로 말이 됩니다.

마지막으로, 랭크(rank)가 행렬의 선형 독립인 열의 최대 개수였음을 상기하십시오. 
고유 분해를 자세히 검토함으로써, 랭크가 $\mathbf{A}$의 0이 아닌 고유값의 개수와 같음을 알 수 있습니다.

예제는 계속될 수 있지만 요점은 명확합니다: 
고유 분해는 많은 선형 대수 계산을 단순화할 수 있으며 많은 수치 알고리즘과 우리가 선형 대수에서 수행하는 많은 분석의 기초가 되는 근본적인 연산입니다.

## 대칭 행렬의 고유 분해 (Eigendecompositions of Symmetric Matrices)
위의 과정이 작동할 만큼 충분한 선형 독립 고유 벡터를 항상 찾을 수 있는 것은 아닙니다. 예를 들어 행렬

$$ 
\mathbf{A} = \begin{bmatrix}
1 & 1 \\ 0 & 1
\end{bmatrix},
$$ 

은 단 하나의 고유 벡터, 즉 $(1, 0)^\top$만 갖습니다. 
그러한 행렬을 다루기 위해서는 우리가 다룰 수 있는 것보다 더 발전된 기술(예: 조르단 표준형 또는 특이값 분해)이 필요합니다. 
우리는 종종 고유 벡터의 전체 집합의 존재를 보장할 수 있는 행렬들에 주의를 집중해야 할 것입니다.

가장 흔히 마주치는 가족은 *대칭 행렬*입니다. 이는 $\mathbf{A} = \mathbf{A}^\top$인 행렬들입니다. 
이 경우 우리는 $W$를 *직교 행렬(orthogonal matrix)*—모든 열이 서로 직각인 길이 1의 벡터인 행렬, 즉 $\mathbf{W}^\top = \mathbf{W}^{-1}$—로 취할 수 있으며 모든 고유값은 실수가 됩니다. 
따라서 이 특별한 경우에 우리는 :eqref:`eq_eig_decomp`를 다음과 같이 쓸 수 있습니다.

$$ 
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$ 

## 거시고린 원판 정리 (Gershgorin Circle Theorem)
고유값은 종종 직관적으로 추론하기 어렵습니다. 
임의의 행렬이 제시되면 그것을 계산하지 않고는 고유값이 무엇인지에 대해 말할 수 있는 것이 거의 없습니다. 
하지만 가장 큰 값들이 대각선에 있다면 잘 근사하기 쉽게 만들 수 있는 정리가 하나 있습니다.

$\mathbf{A} = (a_{ij})$를 임의의 정사각 행렬($n\times n$)이라고 합시다. 
우리는 $r_i = \sum_{j \neq i} |a_{ij}|$로 정의할 것입니다. 
$\\mathcal{D}_i$가 복소평면에서 중심이 $a_{ii}$이고 반지름이 $r_i$인 원판을 나타낸다고 합시다. 
그러면 $\mathbf{A}$의 모든 고유값은 $\\mathcal{D}_i$ 중 하나에 포함됩니다.

이것은 이해하기에 조금 많을 수 있으므로 예제를 살펴봅시다. 
행렬을 고려해 보십시오:

$$ 
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\ 0.1 & 3.0 & 0.2 & 0.3 \\ 0.1 & 0.2 & 5.0 & 0.5 \\ 0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$ 

우리는 $r_1 = 0.3, r_2 = 0.6, r_3 = 0.8, r_4 = 0.9$를 갖습니다. 
행렬은 대칭이므로 모든 고유값은 실수입니다. 
이는 우리의 모든 고유값이 다음 범위 중 하나에 있게 됨을 의미합니다.

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


수치 계산을 수행하면 고유값이 대략 $0.99, 2.97, 4.95, 9.08$임을 알 수 있으며, 
모두 제공된 범위 내에 편안하게 들어옵니다.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

이런 식으로 고유값을 근사할 수 있으며, 
대각선이 다른 모든 원소보다 상당히 큰 경우 근사는 상당히 정확할 것입니다.

작은 일이지만, 고유 분해와 같이 복잡하고 미묘한 주제에 대해 우리가 할 수 있는 어떤 직관적인 파악이라도 얻는 것은 좋습니다.

## 유용한 응용: 반복 맵의 성장 (A Useful Application: The Growth of Iterated Maps)

이제 고유 벡터가 원칙적으로 무엇인지 이해했으므로, 
신경망 동작의 중심 문제인 적절한 가중치 초기화에 대한 깊은 이해를 제공하는 데 그것들이 어떻게 사용될 수 있는지 살펴봅시다.

### 장기 행동으로서의 고유 벡터 (Eigenvectors as Long Term Behavior)

심층 신경망 초기화에 대한 완전한 수학적 조사는 텍스트의 범위를 벗어나지만, 
고유값이 이러한 모델이 어떻게 작동하는지 이해하는 데 어떻게 도움이 되는지 여기에서 장난감 버전을 볼 수 있습니다. 
우리가 알고 있듯이, 신경망은 선형 변환 레이어와 비선형 연산을 산재시켜 작동합니다. 
여기서 단순함을 위해 비선형성이 없다고 가정하고 변환이 단일 반복 행렬 연산 $A$라고 가정합시다. 그러면 우리 모델의 출력은 다음과 같습니다.

$$ 
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$ 

이러한 모델이 초기화될 때 $A$는 가우스(Gaussian) 항목을 가진 무작위 행렬로 취해지므로, 그중 하나를 만들어 봅시다. 
구체적으로 평균 0, 분산 1인 가우스 분포 $5 \times 5$ 행렬로 시작합니다.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### 무작위 데이터에서의 행동 (Behavior on Random Data)
우리의 장난감 모델에서 단순함을 위해, 
우리가 공급하는 데이터 벡터 $\mathbf{v}_{in}$이 무작위 5차원 가우스 벡터라고 가정합시다. 
어떤 일이 일어나기를 원하는지 생각해 봅시다. 
문맥을 위해 일반적인 머신러닝 문제를 생각해 봅시다. 
우리는 이미지와 같은 입력 데이터를 이미지가 고양이 사진일 확률과 같은 예측으로 바꾸려고 노력하고 있습니다. 
만약 $\mathbf{A}$를 반복적으로 적용하여 무작위 벡터를 매우 길게 늘린다면, 
입력의 작은 변화가 출력의 큰 변화로 증폭될 것입니다 - 입력 이미지의 아주 작은 수정이 매우 다른 예측으로 이어질 것입니다. 
이것은 옳지 않은 것 같습니다!

반대로 $\mathbf{A}$가 무작위 벡터를 더 짧게 수축시킨다면, 
많은 레이어를 거친 후 벡터는 본질적으로 아무것도 아닌 것으로 수축할 것이고, 
출력은 입력에 의존하지 않을 것입니다. 이것 또한 분명히 옳지 않습니다!

우리는 출력이 입력에 따라 변하되 너무 많이 변하지 않도록 하기 위해 
성장과 쇠퇴 사이의 좁은 길을 걸어야 합니다!

무작위 입력 벡터에 대해 행렬 $\mathbf{A}$를 반복적으로 곱할 때 어떤 일이 일어나는지 살펴보고 노름(norm)을 추적해 봅시다.

```{.python .input}
#@tab mxnet
# `A`를 반복적으로 적용한 후 노름의 시퀀스 계산
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# `A`를 반복적으로 적용한 후 노름의 시퀀스 계산
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# `A`를 반복적으로 적용한 후 노름의 시퀀스 계산
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

노름이 통제할 수 없이 커지고 있습니다! 
실제로 몫의 리스트를 취하면 패턴을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
# 노름의 스케일링 인자 계산
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# 노름의 스케일링 인자 계산
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# 노름의 스케일링 인자 계산
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

위의 계산 마지막 부분을 보면, 무작위 벡터가 `1.974459321485[...]`라는 인수로 늘어나는 것을 볼 수 있습니다. 
끝부분이 약간씩 바뀌기는 하지만 늘어나는 인수는 안정적입니다.

### 고유 벡터와 다시 연결하기 (Relating Back to Eigenvectors)

우리는 고유 벡터와 고유값이 무언가가 늘어나는 양에 대응한다는 것을 보았지만, 그것은 특정 벡터와 특정 늘림에 대한 것이었습니다. 
$\mathbf{A}$에 대해 그것들이 무엇인지 살펴봅시다. 
여기서 약간의 주의사항이 있습니다: 그것들을 모두 보려면 복소수로 가야 한다는 것이 밝혀졌습니다. 
이것들을 늘림과 회전으로 생각할 수 있습니다. 
복소수의 노름(실수부와 허수부의 제곱 합의 제곱근)을 취함으로써 그 늘림 인수를 측정할 수 있습니다. 그것들을 정렬해 봅시다.

```{.python .input}
#@tab mxnet
# 고유값 계산
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# 고유값 계산
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# 고유값 계산
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### 관찰 (An Observation)

우리는 여기서 약간 예상치 못한 일이 일어나는 것을 봅니다: 
무작위 벡터에 적용된 우리 행렬 $\mathbf{A}$의 장기적인 늘림에 대해 우리가 이전에 식별한 그 숫자가 *정확히* 
(소수점 아래 13자리까지 정확하게!) 
$\mathbf{A}$의 가장 큰 고유값입니다. 
이것은 분명히 우연이 아닙니다!

하지만 이제 기하학적으로 무슨 일이 일어나고 있는지 생각해보면 이해가 되기 시작합니다. 
무작위 벡터를 고려해 보십시오. 이 무작위 벡터는 모든 방향을 조금씩 가리키고 있으므로, 
특히 $\mathbf{A}$의 가장 큰 고유값과 연관된 고유 벡터와 동일한 방향을 적어도 조금은 가리킵니다. 
이것은 너무 중요해서 *주 고유값(principal eigenvalue)* 및 *주 고유 벡터(principal eigenvector)*라고 불립니다. 
$\mathbf{A}$를 적용한 후, 우리의 무작위 벡터는 가능한 모든 고유 벡터와 연관되어 모든 가능한 방향으로 늘어나지만, 
무엇보다도 이 주 고유 벡터와 연관된 방향으로 가장 많이 늘어납니다. 
이것이 의미하는 바는 $A$를 적용한 후 우리의 무작위 벡터가 더 길어지고 주 고유 벡터와 더 일치하는 방향을 가리킨다는 것입니다. 
행렬을 여러 번 적용한 후 주 고유 벡터와의 일치는 점점 더 가까워져서, 
모든 실질적인 목적을 위해 우리의 무작위 벡터는 주 고유 벡터로 변환됩니다! 
실제로 이 알고리즘은 행렬의 가장 큰 고유값과 고유 벡터를 찾기 위한 *거듭제곱 반복(power iteration)*으로 알려진 것의 기초입니다. 자세한 내용은 예를 들어 :cite:`Golub.Van-Loan.1996`을 참조하십시오.

### 정규화 수정 (Fixing the Normalization)

이제 위의 논의로부터 우리는 무작위 벡터가 전혀 늘어나거나 줄어들지 않기를 원한다는 결론을 내렸습니다. 
우리는 전체 과정 동안 무작위 벡터가 거의 동일한 크기를 유지하기를 바랍니다. 
그렇게 하기 위해 이제 우리는 가장 큰 고유값이 대신 1이 되도록 주 고유값으로 우리 행렬의 스케일을 조정합니다. 
이 경우 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab mxnet
# 행렬 `A`의 스케일 조정
A /= norm_eigs[-1]

# 동일한 실험 다시 수행
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# 행렬 `A`의 스케일 조정
A /= norm_eigs[-1]

# 동일한 실험 다시 수행
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# 행렬 `A`의 스케일 조정
A /= norm_eigs[-1]

# 동일한 실험 다시 수행
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

이전처럼 연속된 노름 사이의 비율을 플롯할 수도 있으며 실제로 안정화되는 것을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
# 비율도 플롯
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# 비율도 플롯
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# 비율도 플롯
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 토론 (Discussion)

우리는 이제 우리가 원했던 것을 정확히 봅니다! 
행렬을 주 고유값으로 정규화한 후, 
무작위 데이터가 이전처럼 폭발하지 않고 오히려 결국 특정 값으로 평형을 이룹니다. 
이런 일들을 제1원리부터 할 수 있으면 좋겠는데, 
수학을 깊이 파고들면 독립적인 평균 0, 분산 1인 가우스 항목을 가진 큰 무작위 행렬의 가장 큰 고유값은 
*원형 법칙(circular law)*이라는 매혹적인 사실 덕분에 평균적으로 대략 $\sqrt{n}$, 
또는 우리의 경우 $\sqrt{5} \approx 2.2$가 된다는 것을 알 수 있습니다 :cite:`Ginibre.1965`. 
무작위 행렬의 고유값(및 특이값이라고 불리는 관련 객체) 사이의 관계는 :citet:`Pennington.Schoenholz.Ganguli.2017` 및 후속 작업에서 논의된 것처럼 신경망의 적절한 초기화와 깊은 관계가 있음이 밝혀졌습니다.

## 요약 (Summary)
* 고유 벡터는 방향을 바꾸지 않고 행렬에 의해 늘어나는 벡터입니다.
* 고유값은 행렬의 적용에 의해 고유 벡터가 늘어나는 양입니다.
* 행렬의 고유 분해는 많은 연산을 고유값에 대한 연산으로 축소할 수 있게 해줍니다.
* 거시고린 원판 정리는 행렬의 고유값에 대한 근사치를 제공할 수 있습니다.
* 반복되는 행렬 거듭제곱의 행동은 주로 가장 큰 고유값의 크기에 달려 있습니다. 이 이해는 신경망 초기화 이론에서 많은 응용 분야를 갖습니다.

## 연습 문제 (Exercises)
1. 다음 행렬의 고유값과 고유 벡터는 무엇입니까?
$$ 
\mathbf{A} = \begin{bmatrix}
2 & 1 \\ 1 & 2
\end{bmatrix}?
$$ 
2. 다음 행렬의 고유값과 고유 벡터는 무엇이며, 이 예제는 이전 예제와 비교하여 무엇이 이상합니까?
$$ 
\mathbf{A} = \begin{bmatrix}
2 & 1 \\ 0 & 2
\end{bmatrix}.
$$ 
3. 고유값을 계산하지 않고, 다음 행렬의 가장 작은 고유값이 $0.5$보다 작을 수 있습니까? *참고*: 이 문제는 머릿속으로 풀 수 있습니다.
$$ 
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\ 0.1 & 1.0 & 0.1 & 0.2 \\ 0.3 & 0.1 & 5.0 & 0.0 \\ 1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$ 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab: