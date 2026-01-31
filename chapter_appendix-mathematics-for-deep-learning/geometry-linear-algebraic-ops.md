# 기하 및 선형 대수 연산 (Geometry and Linear Algebraic Operations)
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-algebra`에서 우리는 선형 대수의 기초를 접했고, 데이터를 변환하는 일반적인 연산을 표현하는 데 선형 대수가 어떻게 사용될 수 있는지 보았습니다. 
선형 대수는 우리가 딥러닝과 더 넓은 머신러닝에서 수행하는 많은 작업의 핵심적인 수학적 기둥 중 하나입니다. 
:numref:`sec_linear-algebra`는 현대 딥러닝 모델의 메커니즘을 전달하기에 충분한 도구를 포함하고 있었지만, 이 주제에는 훨씬 더 많은 것이 있습니다. 
이 섹션에서는 선형 대수 연산의 몇 가지 기하학적 해석을 강조하고 고유값과 고유벡터를 포함한 몇 가지 기본 개념을 소개하면서 더 깊이 파고들 것입니다.

## 벡터의 기하학 (Geometry of Vectors)
먼저 벡터의 두 가지 공통된 기하학적 해석, 즉 공간에서의 점 또는 방향으로의 해석에 대해 논의해야 합니다. 
근본적으로 벡터는 아래의 Python 리스트와 같은 숫자들의 목록입니다.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

수학자들은 이것을 *열* 벡터 또는 *행* 벡터로 쓰는 경우가 많습니다. 즉, 다음과 같습니다.

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$ 

또는

$$ 
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$ 

데이터 예제가 열 벡터이고 가중 합을 형성하는 데 사용되는 가중치가 행 벡터인 경우처럼, 이들은 종종 다른 해석을 갖습니다. 
그러나 유연하게 대처하는 것이 유익할 수 있습니다. 
:numref:`sec_linear-algebra`에서 설명했듯이, 단일 벡터의 기본 방향은 열 벡터이지만, 테이블 형식의 데이터셋을 나타내는 행렬의 경우 행렬의 각 데이터 예제를 행 벡터로 취급하는 것이 더 일반적입니다.

벡터가 주어졌을 때 우리가 부여해야 할 첫 번째 해석은 공간에서의 점입니다. 
2차원 또는 3차원에서 우리는 벡터의 구성 요소를 사용하여 *원점*이라고 불리는 고정된 기준점과 비교하여 공간에서 점의 위치를 정의함으로써 이러한 점들을 시각화할 수 있습니다. 이는 :numref:`fig_grid`에서 볼 수 있습니다.

![벡터를 평면상의 점으로 시각화한 그림. 벡터의 첫 번째 구성 요소는 $\mathit{x}$-좌표를 제공하고, 두 번째 구성 요소는 $\mathit{y}$-좌표를 제공합니다. 고차원도 유사하지만 시각화하기는 훨씬 어렵습니다.](../img/grid-points.svg)
:label:`fig_grid`

이러한 기하학적 관점은 문제를 더 추상적인 수준에서 고려할 수 있게 해줍니다. 
사진을 고양이 또는 개로 분류하는 것과 같은 극복할 수 없어 보이는 문제에 더 이상 직면하지 않고, 작업을 공간에서의 점들의 모음으로 추상적으로 간주하기 시작하고 작업을 두 개의 뚜렷한 점 클러스터를 분리하는 방법을 발견하는 것으로 상상할 수 있습니다.

이와 병행하여 사람들이 종종 벡터에 대해 갖는 두 번째 관점이 있습니다: 공간에서의 방향입니다. 
벡터 $\mathbf{v} = [3,2]^\top$를 원점에서 오른쪽으로 3단위, 위로 2단위인 위치로 생각할 수 있을 뿐만 아니라, 오른쪽으로 3걸음, 위로 2걸음을 걷는 방향 자체로 생각할 수도 있습니다. 
이런 식으로 우리는 그림 :numref:`fig_arrow`의 모든 벡터를 동일하게 간주합니다.

![모든 벡터는 평면상의 화살표로 시각화될 수 있습니다. 이 경우 그려진 모든 벡터는 벡터 $(3,2)^\top$의 표현입니다.](../img/par-vec.svg)
:label:`fig_arrow`

이러한 전환의 이점 중 하나는 벡터 덧셈 행위에 대한 시각적 의미를 부여할 수 있다는 것입니다. 
특히, 우리는 한 벡터가 가리키는 방향을 따르고 나서 다른 벡터가 가리키는 방향을 따릅니다. 이는 :numref:`fig_add-vec`에서 볼 수 있습니다.

![한 벡터를 따르고 나서 다른 벡터를 따름으로써 벡터 덧셈을 시각화할 수 있습니다.](../img/vec-add.svg)
:label:`fig_add-vec`

벡터 뺄셈도 비슷한 해석을 갖습니다. 
$\\mathbf{u} = \\mathbf{v} + (\\mathbf{u}-\\mathbf{v})$라는 항등식을 고려함으로써, 우리는 벡터 $\\mathbf{u}-\\mathbf{v}$가 점 $\\mathbf{v}$에서 점 $\\mathbf{u}$로 우리를 데려다주는 방향임을 알 수 있습니다.


## 내적과 각도 (Dot Products and Angles)
:numref:`sec_linear-algebra`에서 보았듯이, 
두 개의 열 벡터 $\\mathbf{u}$와 $\\mathbf{v}$를 취하면 다음과 같이 계산하여 내적을 형성할 수 있습니다.

$$\\mathbf{u}^\top\\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def`는 대칭적이므로 고전적인 곱셈의 표기법을 모방하여 다음과 같이 씁니다.

$$ 
\\mathbf{u}\cdot\\mathbf{v} = \\mathbf{u}^\top\\mathbf{v} = \\mathbf{v}^\top\\mathbf{u},
$$ 

벡터의 순서를 바꿔도 동일한 답이 나온다는 사실을 강조하기 위함입니다.

내적 :eqref:`eq_dot_def`는 기하학적 해석도 허용합니다: 이는 두 벡터 사이의 각도와 밀접하게 관련되어 있습니다. :numref:`fig_angle`에 표시된 각도를 고려하십시오.

![평면상의 임의의 두 벡터 사이에는 잘 정의된 각도 $\\theta$가 있습니다. 우리는 이 각도가 내적과 밀접하게 묶여 있음을 알게 될 것입니다.](../img/vec-angle.svg)
:label:`fig_angle`

시작하기 위해 두 개의 특정 벡터를 고려해 봅시다.

$$ 
\\mathbf{v} = (r,0) \; \\textrm{및} \; \\mathbf{w} = (s\cos(\\theta), s \sin(\\theta)).
$$ 

벡터 $\\mathbf{v}$는 길이가 $r$이고 $x$축과 평행하며, 벡터 $\\mathbf{w}$는 길이가 $s$이고 $x$축과 각도 $\\theta$를 이룹니다. 
이 두 벡터의 내적을 계산하면 다음과 같음을 알 수 있습니다.

$$ 
\\mathbf{v}\cdot\\mathbf{w} = rs\cos(\\theta) = \|\\mathbf{v}\|\|\\mathbf{w}\|\cos(\\theta).
$$ 

간단한 대수적 조작을 통해 항을 재배열하면 다음을 얻을 수 있습니다.

$$ 
\\theta = \arccos\left(rac{\\mathbf{v}\cdot\\mathbf{w}}{\|\\mathbf{v}\|\|\\mathbf{w}\|}\right).
$$ 

요컨대, 이 두 특정 벡터에 대해 내적과 노름의 조합은 두 벡터 사이의 각도를 알려줍니다. 이 사실은 일반적으로도 참입니다. 여기서 식을 유도하지는 않겠지만, 
$\\|\\mathbf{v} - \\mathbf{w}\\|^2$를 두 가지 방식(하나는 내적으로, 다른 하나는 코사인 법칙을 사용한 기하학적 방식)으로 쓰는 것을 고려하면 전체 관계를 얻을 수 있습니다. 
실제로 임의의 두 벡터 $\\mathbf{v}$와 $\\mathbf{w}$에 대해 두 벡터 사이의 각도는 다음과 같습니다.

$$\\theta = \arccos\left(rac{\\mathbf{v}\cdot\\mathbf{w}}{\|\\mathbf{v}\|\|\\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

이것은 계산의 어떤 것도 2차원을 참조하지 않기 때문에 멋진 결과입니다. 
실제로 우리는 이를 3차원 또는 300만 차원에서 문제없이 사용할 수 있습니다.

간단한 예로 두 벡터 사이의 각도를 계산하는 방법을 살펴봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

지금은 사용하지 않겠지만, 각도가 $\\pi/2$ (또는 동등하게 $90^{\\circ}$)인 벡터를 *직교(orthogonal)*한다고 지칭한다는 점을 알아두면 유용합니다. 
위의 방정식을 조사하면 $\\theta = \\pi/2$일 때 이런 일이 발생하며, 이는 $\\cos(\\theta) = 0$과 같음을 알 수 있습니다. 
이것이 발생할 수 있는 유일한 방법은 내적 자체가 0인 경우이며, 두 벡터는 $\\mathbf{v}\\.\\mathbf{w} = 0$인 경우에만 직교합니다. 
이는 객체를 기하학적으로 이해할 때 유용한 공식이 될 것입니다.

각도를 계산하는 것이 왜 유용한지 묻는 것은 합리적입니다. 
답은 우리가 데이터에 기대하는 일종의 불변성(invariance)에 있습니다. 
이미지와 모든 픽셀 값은 같지만 밝기가 $10\%$인 복제 이미지를 고려해 보십시오. 
개별 픽셀의 값은 일반적으로 원래 값과 거리가 멉니다. 
따라서 원본 이미지와 어두운 이미지 사이의 거리를 계산하면 거리가 클 수 있습니다. 
그러나 대부분의 머신러닝 응용 프로그램에서 *콘텐츠*는 동일합니다. 고양이/개 분류기 입장에서는 여전히 고양이 이미지입니다. 
그러나 각도를 고려하면 임의의 벡터 $\\mathbf{v}$에 대해 $\\mathbf{v}$와 $0.1\cdot\\mathbf{v}$ 사이의 각도가 0임을 알 수 있습니다. 
이는 벡터의 스케일을 조정해도 동일한 방향을 유지하고 길이만 변경된다는 사실에 대응합니다. 각도는 어두운 이미지를 동일한 것으로 간주합니다.

이와 같은 예는 어디에나 있습니다. 
텍스트에서 우리는 똑같은 말을 하는 두 배 긴 문서를 작성하더라도 논의되는 주제가 바뀌지 않기를 원할 수 있습니다. 
어떤 인코딩(예: 어떤 어휘의 단어 발생 횟수 계산)의 경우 이는 문서를 인코딩하는 벡터를 두 배로 늘리는 것에 대응하므로 다시 각도를 사용할 수 있습니다.

### 코사인 유사도 (Cosine Similarity)
각도를 사용하여 두 벡터의 근접성을 측정하는 머신러닝 맥락에서, 실무자들은 다음과 같은 부분을 지칭하기 위해 *코사인 유사도(cosine similarity)*라는 용어를 채택합니다.
$$ 
\\cos(\\theta) = \frac{\\mathbf{v}\\.\\mathbf{w}}{\|\\mathbf{v}\|\|\\mathbf{w}\|}. 
$$ 

코사인은 두 벡터가 같은 방향을 가리킬 때 최대값 $1$, 반대 방향을 가리킬 때 최소값 $-1$, 두 벡터가 직교할 때 값 $0$을 갖습니다. 
고차원 벡터의 성분이 평균 $0$으로 무작위로 샘플링되면 코사인은 거의 항상 $0$에 가깝습니다.


## 초평면 (Hyperplanes)

벡터로 작업하는 것 외에도 선형 대수에서 멀리 나아가기 위해 이해해야 할 또 다른 핵심 객체는 *초평면(hyperplane)*입니다. 이는 직선(2차원) 또는 평면(3차원)을 고차원으로 일반화한 것입니다. 
$d$차원 벡터 공간에서 초평면은 $d-1$차원을 가지며 공간을 두 개의 반공간(half-spaces)으로 나눕니다.

예제부터 시작하겠습니다. 
열 벡터 $\\mathbf{w}=[2,1]^\top$가 있다고 가정합시다. 우리는 "$\\mathbf{w}\\.\\mathbf{v} = 1$인 점 $\\mathbf{v}$는 무엇인가?"를 알고 싶습니다. 
위의 내적과 각도 사이의 연결 :eqref:`eq_angle_forumla`을 상기하면, 이것은 다음과 동등함을 알 수 있습니다.
$$ 
\|\\mathbf{v}\|\|\\mathbf{w}\|\cos(\\theta) = 1 \; \\iff \; \|\\mathbf{v}\|\cos(\\theta) = \frac{1}{\|\\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$ 

![삼각법을 상기하면, 공식 $\\|\\mathbf{v}\|\cos(\\theta)$가 $\\mathbf{w}$ 방향으로의 벡터 $\\mathbf{v}$의 투영(projection) 길이임을 알 수 있습니다.](../img/proj-vec.svg)
:label:`fig_vector-project`

이 식의 기하학적 의미를 고려하면, 이는 $\\mathbf{w}$ 방향으로의 $\\mathbf{v}$의 투영 길이가 정확히 $1/\\|\\mathbf{w}\\|$\\$임을 알 수 있습니다. 이는 :numref:`fig_vector-project`에 나와 있습니다. 
이것이 참인 모든 점의 집합은 벡터 $\\mathbf{w}$에 직각인 직선입니다. 
원한다면 이 직선의 방정식을 찾아 $2x + y = 1$ 또는 동등하게 $y = 1 - 2x$임을 알 수 있습니다.

이제 $\\mathbf{w}\\.\\mathbf{v} > 1$ 또는 $\\mathbf{w}\\.\\mathbf{v} < 1$인 점들의 집합에 대해 물으면 어떻게 되는지 살펴봅시다. 
이들은 각각 투영이 $1/\\|\\mathbf{w}\\|$\\$보다 길거나 짧은 경우임을 알 수 있습니다. 
따라서 이 두 부등식은 직선의 양쪽을 정의합니다. 
이런 식으로 우리는 공간을 두 반쪽으로 나누는 방법을 찾았습니다. 여기서 한쪽의 모든 점은 임계값 미만의 내적을 갖고 다른 쪽은 그 이상을 갖습니다. 이는 :numref:`fig_space-division`에서 볼 수 있습니다.

![이제 식의 부등식 버전을 고려하면, 초평면(이 경우 단순한 직선)이 공간을 두 반쪽으로 분리하는 것을 알 수 있습니다.](../img/space-division.svg)
:label:`fig_space-division`

고차원에서의 이야기도 거의 같습니다. 
이제 $\\mathbf{w} = [1,2,3]^\top$를 취하고 3차원에서 $\\mathbf{w}\\.\\mathbf{v} = 1$인 점들에 대해 물으면, 주어진 벡터 $\\mathbf{w}$에 직각인 평면을 얻습니다. 
두 부등식은 다시 :numref:`fig_higher-division`에 표시된 것처럼 평면의 양쪽을 정의합니다.

![모든 차원의 초평면은 공간을 두 반쪽으로 분리합니다.](../img/space-division-3d.svg)
:label:`fig_higher-division`

이 시점에서 시각화 능력은 다하겠지만, 십 차원, 백 차원 또는 수십억 차원에서 이 작업을 수행하는 것을 막을 수 있는 것은 아무것도 없습니다. 
이는 머신러닝 모델을 생각할 때 종종 발생합니다. 
예를 들어, 우리는 :numref:`sec_softmax`의 선형 분류 모델을 서로 다른 타겟 클래스를 분리하는 초평면을 찾는 방법으로 이해할 수 있습니다. 
이러한 맥락에서 그러한 초평면은 종종 *결정 평면(decision planes)*이라고 불립니다. 
심층 학습 분류 모델의 대부분은 softmax에 공급되는 선형 레이어로 끝나므로, 심층 신경망의 역할을 타겟 클래스가 초평면에 의해 깨끗하게 분리될 수 있도록 하는 비선형 임베딩을 찾는 것으로 해석할 수 있습니다.

수작업으로 만든 예제를 제공하기 위해, Fashion-MNIST 데이터셋(:numref:`sec_fashion_mnist`에서 본)의 티셔츠와 바지의 작은 이미지를 분류하기 위해 단순히 평균들 사이의 벡터를 취하여 결정 평면을 정의하고 대략적인 임계값을 눈대중으로 잡음으로써 합리적인 모델을 생성할 수 있음에 주목하십시오. 먼저 데이터를 로드하고 평균을 계산하겠습니다.

```{.python .input}
#@tab mxnet
# 데이터셋 로드
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# 평균 계산
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# 데이터셋 로드
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# 평균 계산
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# 데이터셋 로드
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# 평균 계산
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

이러한 평균들을 자세히 살펴보는 것이 유익할 수 있으므로 어떤 모습인지 그려봅시다. 이 경우 평균이 실제로 티셔츠의 흐릿한 이미지와 닮았음을 알 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# 평균 티셔츠 플롯
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# 평균 티셔츠 플롯
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

두 번째 경우에도 평균이 바지의 흐릿한 이미지와 닮았음을 다시 알 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# 평균 바지 플롯
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# 평균 바지 플롯
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

완전한 머신러닝 솔루션에서는 데이터셋에서 임계값을 학습할 것입니다. 이 경우 저는 단순히 훈련 데이터에서 수동으로 잘 어울려 보이는 임계값을 눈대중으로 잡았습니다.

```{.python .input}
#@tab mxnet
# 눈대중 임계값을 사용한 테스트 세트 정확도 출력
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# 정확도
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# 눈대중 임계값을 사용한 테스트 세트 정확도 출력
w = (ave_1 - ave_0).T
# '@'는 pytorch의 행렬 곱셈 연산자입니다.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# 정확도
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# 눈대중 임계값을 사용한 테스트 세트 정확도 출력
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# 정확도
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## 선형 변환의 기하학 (Geometry of Linear Transformations)

:numref:`sec_linear-algebra` 및 위의 논의를 통해, 우리는 벡터, 길이 및 각도의 기하학에 대해 확고한 이해를 얻었습니다. 
그러나 우리가 논의를 생략한 중요한 객체가 하나 있는데, 그것은 행렬로 표현되는 선형 변환의 기하학적 이해입니다. 
잠재적으로 다른 두 고차원 공간 사이에서 데이터를 변환하기 위해 행렬이 무엇을 할 수 있는지 완전히 내면화하려면 상당한 연습이 필요하며, 이는 이 부록의 범위를 벗어납니다. 
그러나 우리는 2차원에서 직관을 구축하기 시작할 수 있습니다.

어떤 행렬이 있다고 가정해 봅시다.

$$ 
\\mathbf{A} = \begin{bmatrix}
 a & b \\ c & d
\end{bmatrix}.
$$ 

이것을 임의의 벡터 $\\mathbf{v} = [x, y]^\top$에 적용하고 싶다면 다음과 같이 곱합니다.

$$ 
\begin{aligned}
\\mathbf{A}\\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\\begin{bmatrix}a \\ c\end{bmatrix} + y\\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\\mathbf{A}\\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\\mathbf{A}\\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$ 

이것은 명확한 것이 다소 난해해진 이상한 계산처럼 보일 수 있습니다. 
그러나 이것은 행렬이 *임의의* 벡터를 변환하는 방식을 *두 개의 특정 벡터* $[1,0]^\top$ 및 $[0,1]^\top$를 변환하는 방식의 관점에서 쓸 수 있음을 알려줍니다. 
이는 잠시 고려해 볼 가치가 있습니다. 
우리는 본질적으로 무한한 문제(임의의 실수 쌍에 어떤 일이 일어나는지)를 유한한 문제(이 특정 벡터들에게 어떤 일이 일어나는지)로 축소했습니다. 이러한 벡터들은 *기저(basis)*의 한 예이며, 우리 공간의 모든 벡터를 이러한 *기저 벡터*들의 가중 합으로 쓸 수 있습니다.

특정 행렬을 사용할 때 어떤 일이 발생하는지 그려봅시다.

$$ 
\\mathbf{A} = \begin{bmatrix}
 1 & 2 \\ -1 & 3
\end{bmatrix}.
$$ 

특정 벡터 $\\mathbf{v} = [2, -1]^\top$를 보면, 이것은 $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$임을 알 수 있으며, 따라서 행렬 $A$가 이를 $2(\\mathbf{A}[1,0]^\top) + -1(\\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$로 보낼 것임을 알 수 있습니다. 
만약 우리가 이 로직을 신중하게 따라간다면, 가령 모든 정수 쌍 점들의 그리드를 고려함으로써, 행렬 곱셈이 그리드를 비틀고(skew), 회전하고(rotate), 스케일을 조정할 수 있음을 알 수 있습니다. 하지만 그리드 구조는 :numref:`fig_grid-transform`에서 보듯이 유지되어야 합니다.

![주어진 기저 벡터에 작용하는 행렬 $\\mathbf{A}$. 전체 그리드가 그와 함께 어떻게 이동하는지 주목하십시오.](../img/grid-transform.svg)
:label:`fig_grid-transform`

이것은 행렬로 표현되는 선형 변환에 대해 내면화해야 할 가장 중요한 직관적인 점입니다. 
행렬은 공간의 일부 부분을 다른 부분보다 다르게 왜곡할 수 없습니다. 
그들이 할 수 있는 전부는 우리 공간의 원래 좌표를 비틀고, 회전하고, 스케일을 조정하는 것뿐입니다.

일부 왜곡은 심할 수 있습니다. 예를 들어 행렬

$$ 
\\mathbf{B} = \begin{bmatrix}
 2 & -1 \\ 4 & -2
\end{bmatrix},
$$ 

은 전체 2차원 평면을 하나의 직선으로 압축합니다. 
그러한 변환을 식별하고 작업하는 것은 나중 섹션의 주제이지만, 기하학적으로 이것이 우리가 위에서 본 변환 유형과 근본적으로 다르다는 것을 알 수 있습니다. 
예를 들어, 행렬 $\\mathbf{A}$의 결과는 원래 그리드로 "다시 구부러질" 수 있습니다. 하지만 행렬 $\\mathbf{B}$의 결과는 그럴 수 없습니다. 벡터 $[1,2]^\top$가 어디서 왔는지(가령 $[1,1]^\top$에서 왔는지 아니면 $[0, -1]^\top$에서 왔는지) 결코 알 수 없기 때문입니다.

이 그림은 $2\times2$ 행렬에 대한 것이었지만, 학습한 교훈을 고차원으로 가져가는 것을 방해하는 것은 아무것도 없습니다. 
$[1,0, \ldots,0]$과 같은 유사한 기저 벡터를 취하고 우리 행렬이 그것들을 어디로 보내는지 본다면, 우리가 다루고 있는 어떤 차원 공간에서든 행렬 곱셈이 전체 공간을 어떻게 왜곡하는지 느끼기 시작할 수 있습니다.

## 선형 종속 (Linear Dependence)

행렬을 다시 고려해 보십시오.

$$ 
\\mathbf{B} = \begin{bmatrix}
 2 & -1 \\ 4 & -2
\end{bmatrix}.
$$ 

이것은 전체 평면을 단일 직선 $y = 2x$ 위에 살도록 압축합니다. 
이제 질문이 생깁니다: 행렬 자체만 보고 이를 감지할 수 있는 방법이 있을까요? 
답은 실제로 가능하다는 것입니다. $\\mathbf{b}_1 = [2,4]^\top$ 및 $\\mathbf{b}_2 = [-1, -2]^\top$를 $\\mathbf{B}$의 두 열이라고 합시다. 
행렬 $\\mathbf{B}$에 의해 변환된 모든 것은 행렬 열의 가중 합(예: $a_1\\mathbf{b}_1 + a_2\\mathbf{b}_2$)으로 쓸 수 있음을 기억하십시오. 
우리는 이를 *선형 결합(linear combination)*이라고 부릅니다. 
$\\mathbf{b}_1 = -2\cdot\\mathbf{b}_2$라는 사실은 이러한 두 열의 임의의 선형 결합을 다음과 같이 $\\mathbf{b}_2$에 대해서만 쓸 수 있음을 의미합니다.

$$ 
 a_1\\mathbf{b}_1 + a_2\\mathbf{b}_2 = -2a_1\\mathbf{b}_2 + a_2\\mathbf{b}_2 = (a_2-2a_1)\\mathbf{b}_2.
$$ 

이는 열 중 하나가 공간에서 고유한 방향을 정의하지 않기 때문에 어떤 의미에서 중복된다는 것을 의미합니다. 
이 행렬이 전체 평면을 단일 직선으로 무너뜨리는 것을 이미 보았기 때문에 이것은 우리를 너무 놀라게 해서는 안 됩니다. 
더욱이, 선형 종속 $\\mathbf{b}_1 = -2\cdot\\mathbf{b}_2$가 이를 포착함을 알 수 있습니다. 
두 벡터 사이에서 이를 더 대칭적으로 만들기 위해 다음과 같이 쓰겠습니다.

$$ 
\\mathbf{b}_1  + 2\cdot\\mathbf{b}_2 = 0.
$$ 

일반적으로 벡터 모음 $\\mathbf{v}_1, \ldots, \\mathbf{v}_k$는 *모두 0은 아닌* 계수 $a_1, \ldots, a_k$가 존재하여 다음을 만족할 때 *선형 종속(linearly dependent)*이라고 합니다.

$$ 
\sum_{i=1}^k a_i\\mathbf{v_i} = 0.
$$ 

이 경우, 우리는 벡터 중 하나를 다른 것들의 어떤 조합의 관점에서 풀 수 있으며, 효과적으로 그것을 중복으로 만들 수 있습니다. 
따라서 행렬 열의 선형 종속은 우리 행렬이 공간을 어떤 낮은 차원으로 압축하고 있다는 사실에 대한 증거입니다. 
선형 종속이 없으면 벡터들이 *선형 독립(linearly independent)*하다고 합니다. 
행렬의 열이 선형 독립이면 압축이 발생하지 않으며 작업을 되돌릴 수 있습니다.

## 랭크 (Rank)

일반적인 $n\times m$ 행렬이 있을 때, 행렬이 어떤 차원 공간으로 매핑되는지 묻는 것은 합리적입니다. 
*랭크(rank)*라고 알려진 개념이 우리의 답이 될 것입니다. 
이전 섹션에서 우리는 선형 종속이 공간이 더 낮은 차원으로 압축되는 것을 증명한다는 점을 언급했으므로, 이를 사용하여 랭크 개념을 정의할 수 있습니다. 
특히, 행렬 $\\mathbf{A}$의 랭크는 모든 열 부분 집합 중에서 선형 독립인 열의 최대 개수입니다. 예를 들어 행렬

$$ 
\\mathbf{B} = \begin{bmatrix}
 2 & 4 \\ -1 & -2
\end{bmatrix},
$$ 

은 두 열이 선형 종속이지만 어느 한 열 자체는 선형 종속이 아니므로 $\\textrm{rank}(B)=1$을 갖습니다. 
더 도전적인 예로 다음을 고려할 수 있습니다.

$$ 
\\mathbf{C} = \begin{bmatrix}
 1& 3 & 0 & -1 & 0 \\ -1 & 0 & 1 & 1 & -1 \\ 0 & 3 & 1 & 0 & -1 \\ 2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$ 

그리고 예를 들어 처음 두 열은 선형 독립이지만 세 개의 열로 구성된 네 가지 모음은 모두 종속적임을 보임으로써 $\\mathbf{C}$가 랭크 2임을 보여줄 수 있습니다.

설명된 이 절차는 매우 비효율적입니다. 
주어진 행렬의 모든 열 부분 집합을 살펴봐야 하므로 열 수에 따라 잠재적으로 기하급수적입니다. 
나중에 행렬의 랭크를 계산하는 보다 계산 효율적인 방법을 보겠지만, 지금은 개념이 잘 정의되어 있고 그 의미를 이해하는 것으로 충분합니다.

## 가역성 (Invertibility)

우리는 위에서 선형 종속 열을 가진 행렬에 의한 곱셈은 되돌릴 수 없음을 보았습니다. 즉, 항상 입력을 복구할 수 있는 역작용이 없습니다. 
그러나 풀 랭크 행렬(즉, 랭크가 $n$인 $n \times n$ 행렬 $\\mathbf{A}$)에 의한 곱셈은 항상 되돌릴 수 있어야 합니다. 행렬

$$ 
\\mathbf{I} = \begin{bmatrix}
 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1
\end{bmatrix}.
$$ 

를 고려해 보십시오. 이는 대각선을 따라 1이 있고 다른 곳은 0인 행렬입니다. 
우리는 이를 *단위(identity)* 행렬이라고 부릅니다. 
적용했을 때 우리 데이터를 변경하지 않고 그대로 두는 행렬입니다. 
우리 행렬 $\\mathbf{A}$가 한 일을 되돌리는 행렬을 찾으려면, 다음을 만족하는 행렬 $\\mathbf{A}^{-1}$를 찾고 싶습니다.

$$ 
\\mathbf{A}^{-1}\\mathbf{A} = \\mathbf{A}\\mathbf{A}^{-1} =  \\mathbf{I}.
$$ 

이것을 시스템으로 보면, $n \times n$개의 미지수($\\mathbf{A}^{-1}$의 항목들)와 $n \times n$개의 방정식(곱 $\\mathbf{A}^{-1}\\mathbf{A}$의 모든 항목과 $\\mathbf{I}$의 모든 항목 사이에 성립해야 하는 등식)이 있으므로 일반적으로 해가 존재할 것으로 기대해야 합니다. 
실제로 다음 섹션에서 *행렬식(determinant)*이라는 수량을 보게 될 텐데, 이는 행렬식이 0이 아닌 한 해를 찾을 수 있다는 속성을 가지고 있습니다. 우리는 그러한 행렬 $\\mathbf{A}^{-1}$를 *역(inverse)* 행렬이라고 부릅니다. 
예를 들어 $\\mathbf{A}$가 일반적인 $2 \times 2$ 행렬인 경우

$$ 
\\mathbf{A} = \begin{bmatrix}
 a & b \\ c & d
\end{bmatrix},
$$ 

역행렬은 다음과 같음을 알 수 있습니다.

$$ 
  \\frac{1}{ad-bc}  \\begin{bmatrix}
 d & -b \\ -c & a
\end{bmatrix}.
$$ 

위의 공식에 의해 주어진 역행렬을 곱하는 것이 실제로 작동하는지 확인함으로써 이를 테스트할 수 있습니다.

```{.python .input}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### 수치적 문제 (Numerical Issues)
행렬의 역행렬이 이론적으로는 유용하지만, 대부분의 경우 실제 문제 해결을 위해 행렬 역행렬을 *사용*하고 싶지는 않다는 점을 말해야 합니다. 
일반적으로 다음과 같은 선형 방정식을 푸는 데 훨씬 더 수치적으로 안정적인 알고리즘이 있습니다.

$$ 
\\mathbf{A}\\mathbf{x} = \\mathbf{b},
$$ 

역행렬을 계산하고 곱하여 다음을 얻는 것보다 말입니다.

$$ 
\\mathbf{x} = \\mathbf{A}^{-1}\\mathbf{b}.
$$ 

작은 수로 나누는 것이 수치적 불안정성을 초래할 수 있는 것처럼, 랭크가 낮은 것에 가까운 행렬의 반전도 마찬가지입니다.

게다가 행렬 $\\mathbf{A}$가 *희소(sparse)*한 경우가 흔하며, 이는 소량의 0이 아닌 값만 포함하고 있음을 의미합니다. 
예제를 탐구해 보면 이것이 역행렬이 희소하다는 것을 의미하지는 않는다는 것을 알게 될 것입니다. 
$\mathbf{A}$가 500만 개의 0이 아닌 항목만 있는 100만 x 100만 행렬이라 하더라도(따라서 해당 500만 개만 저장하면 됨), 역행렬은 일반적으로 거의 모든 항목이 0이 아니어서 100만^2개 항목, 즉 1조 개의 항목을 모두 저장해야 합니다!

선형 대수로 작업할 때 자주 마주치는 까다로운 수치적 문제를 끝까지 파고들 시간은 없지만, 언제 주의를 기울여야 하는지에 대한 직관을 제공하고 싶었으며, 일반적으로 실전에서 역행렬 계산을 피하는 것이 좋은 경험 법칙입니다.

## 행렬식 (Determinant)
선형 대수의 기하학적 관점은 *행렬식(determinant)*으로 알려진 근본적인 수량을 해석하는 직관적인 방법을 제공합니다. 
이전의 그리드 이미지를 고려하되 이제 강조된 영역이 있습니다 (:numref:`fig_grid-filled`).

![그리드를 다시 왜곡하는 행렬 $\\mathbf{A}$. 이번에는 강조된 사각형에 특별히 주의를 기울이고 싶습니다.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

강조된 사각형을 보십시오. 이것은 $(0, 1)$ 및 $(1, 0)$에 의해 주어진 모서리를 가진 사각형이므로 넓이가 1입니다. 
$\mathbf{A}$가 이 사각형을 변환한 후 평행사변형이 되는 것을 알 수 있습니다. 
이 평행사변형이 우리가 시작했을 때와 동일한 넓이를 가져야 할 이유는 없으며, 실제로 여기에 표시된 특정 사례인

$$ 
\\mathbf{A} = \begin{bmatrix}
 1 & 2 \\ -1 & 3
\end{bmatrix},
$$ 

에서 이 평행사변형의 넓이를 계산하고 넓이가 5임을 얻는 것은 좌표 기하학의 연습 문제입니다.

일반적으로 행렬이 다음과 같은 경우

$$ 
\\mathbf{A} = \begin{bmatrix}
 a & b \\ c & d
\end{bmatrix},
$$ 

약간의 계산을 통해 결과 평행사변형의 넓이가 $ad-bc$임을 알 수 있습니다. 이 넓이를 *행렬식*이라고 합니다.

예제 코드로 이를 빠르게 확인해 봅시다.

```{.python .input}
#@tab mxnet
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

눈치 빠른 분들은 이 식이 0이거나 심지어 음수일 수 있다는 것을 알아차릴 것입니다. 음수 항의 경우 수학에서 일반적으로 취하는 관례의 문제입니다: 
행렬이 도형을 뒤집으면 넓이가 음수가 된다고 말합니다. 
이제 행렬식이 0일 때 우리는 더 많은 것을 배우게 됨을 살펴봅시다.

다음을 고려해 보십시오.

$$ 
\\mathbf{B} = \begin{bmatrix}
 2 & 4 \\ -1 & -2
\end{bmatrix}.
$$ 

이 행렬의 행렬식을 계산하면 $2\cdot(-2 ) - 4\cdot(-1) = 0$을 얻습니다. 
우리의 이해를 고려하면 이것은 이치에 맞습니다. 
$\mathbf{B}$는 원래 이미지의 사각형을 넓이가 0인 선분으로 무너뜨립니다. 
실제로 변환 후 넓이가 0이 되는 유일한 방법은 더 낮은 차원 공간으로 압축되는 것입니다. 
따라서 우리는 다음과 같은 결과가 참임을 알 수 있습니다: 
행렬 $A$는 행렬식이 0이 아닐 때만 가역적입니다.

마지막 코멘트로, 평면에 그려진 임의의 도형이 있다고 상상해 보십시오. 
컴퓨터 과학자처럼 생각하여 그 도형을 작은 사각형들의 모음으로 분해할 수 있으며, 따라서 도형의 넓이는 본질적으로 분해된 사각형의 수입니다. 
이제 그 도형을 행렬로 변환하면 이러한 사각형 각각을 평행사변형으로 보내며, 각 평행사변형은 행렬식에 의해 주어진 넓이를 갖습니다. 
우리는 임의의 도형에 대해 행렬식이 행렬이 도형의 넓이를 조정하는 (부호가 있는) 숫자임을 알 수 있습니다.

더 큰 행렬에 대한 행렬식을 계산하는 것은 힘들 수 있지만 직관은 같습니다. 
행렬식은 $n\times n$ 행렬이 $n$차원 부피를 조정하는 인수로 남아 있습니다.

## 텐서 및 일반적인 선형 대수 연산 (Tensors and Common Linear Algebra Operations)

:numref:`sec_linear-algebra`에서 텐서의 개념이 소개되었습니다. 
이 섹션에서는 텐서 축약(tensor contraction, 행렬 곱셈의 텐서 등가물)에 대해 더 깊이 파고들어, 이것이 수많은 행렬 및 벡터 연산에 대해 어떻게 통합된 관점을 제공할 수 있는지 살펴볼 것입니다.

행렬과 벡터의 경우 데이터를 변환하기 위해 그것들을 곱하는 방법을 알았습니다. 
텐서가 우리에게 유용하려면 유사한 정의가 필요합니다. 행렬 곱셈을 생각해 보십시오.

$$ 
\\mathbf{C} = \\mathbf{A}\\mathbf{B},
$$ 

또는 동등하게

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$ 

이 패턴은 텐서에 대해 반복할 수 있는 패턴입니다. 
텐서의 경우 보편적으로 선택할 수 있는 합산 대상이 하나만 있는 것은 아니므로 합산하려는 인덱스를 정확하게 지정해야 합니다. 
예를 들어 다음을 고려할 수 있습니다.

$$ 
 y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$ 

그러한 변환을 *텐서 축약*이라고 합니다. 
이는 행렬 곱셈만으로 가능한 것보다 훨씬 더 유연한 변환 가족을 나타낼 수 있습니다.

자주 사용되는 표기법상 단순화를 위해, 합계가 식에서 두 번 이상 발생하는 정확히 해당 인덱스들에 대한 것임을 알 수 있으므로 사람들은 종종 *아인슈타인 표기법(Einstein notation)*으로 작업합니다. 여기서 합계는 모든 반복되는 인덱스에 대해 암시적으로 취해집니다. 
이는 다음과 같은 간결한 표현을 제공합니다.

$$ 
 y_{il} = x_{ijkl}a_{jk}.
$$ 

### 선형 대수의 일반적인 예 (Common Examples from Linear Algebra)

이전에 보았던 선형 대수 정의 중 얼마나 많은 것들이 이 압축된 텐서 표기법으로 표현될 수 있는지 살펴봅시다.

* $\\mathbf{v} \cdot \\mathbf{w} = \sum_i v_iw_i$
* $\\|\\mathbf{v}\\|_2^{2} = \sum_i v_iv_i$
* $(\\mathbf{A}\\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\\mathbf{A}\\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\\textrm{tr}(\\mathbf{A}) = \sum_i a_{ii}$

이런 식으로 우리는 무수한 특수 표기법을 짧은 텐서 표현으로 대체할 수 있습니다.

### 코드로 표현하기 (Expressing in Code)
텐서는 코드에서도 유연하게 조작될 수 있습니다. 
:numref:`sec_linear-algebra`에서 본 것처럼 다음과 같이 텐서를 생성할 수 있습니다.

```{.python .input}
#@tab mxnet
# 텐서 정의
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# 모양 출력
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# 텐서 정의
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# 모양 출력
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# 텐서 정의
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# 모양 출력
A.shape, B.shape, v.shape
```

아인슈타인 합계(Einstein summation)가 직접 구현되었습니다. 
아인슈타인 합계에서 발생하는 인덱스는 문자열로 전달될 수 있으며, 그 뒤에 작용할 텐서가 옵니다. 
예를 들어 행렬 곱셈을 구현하려면 위에서 본 아인슈타인 합계($\\mathbf{A}\\mathbf{v} = a_{ij}v_j$)를 고려하고 인덱스 자체를 스트립하여 구현을 얻을 수 있습니다.

```{.python .input}
#@tab mxnet
# 행렬 곱셈 재구현
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# 행렬 곱셈 재구현
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# 행렬 곱셈 재구현
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

이것은 매우 유연한 표기법입니다. 
예를 들어 전통적으로 다음과 같이 쓰일 계산을 하려면

$$ 
 c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\\mathbf{a}_{il}v_j.
$$ 

아인슈타인 합계를 통해 다음과 같이 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

이 표기법은 인간이 읽기 쉽고 효율적이지만, 어떤 이유로든 프로그래밍 방식으로 텐서 축약을 생성해야 하는 경우에는 번거롭습니다. 
이러한 이유로 `einsum`은 각 텐서에 대해 정수 인덱스를 제공하는 대안 표기법을 제공합니다. 
예를 들어 동일한 텐서 축약은 다음과 같이 쓸 수도 있습니다.

```{.python .input}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch는 이 유형의 표기법을 지원하지 않습니다.
```

```{.python .input}
#@tab tensorflow
# TensorFlow는 이 유형의 표기법을 지원하지 않습니다.
```

두 표기법 중 어느 것이든 코드에서 텐서 축약을 간결하고 효율적으로 표현할 수 있게 해줍니다.

## 요약 (Summary)
* 벡터는 기하학적으로 공간에서의 점 또는 방향으로 해석될 수 있습니다.
* 내적은 임의의 고차원 공간에 대한 각도 개념을 정의합니다.
* 초평면은 직선과 평면의 고차원 일반화입니다. 분류 작업의 마지막 단계로 자주 사용되는 결정 평면을 정의하는 데 사용될 수 있습니다.
* 행렬 곱셈은 기본 좌표의 균일한 왜곡으로 기하학적으로 해석될 수 있습니다. 그들은 벡터를 변환하는 매우 제한적이지만 수학적으로 깨끗한 방법을 나타냅니다.
* 선형 종속은 벡터 모음이 우리가 기대하는 것보다 낮은 차원 공간에 있을 때(가령 2차원 공간에 3개의 벡터가 있을 때)를 알려주는 방법입니다. 행렬의 랭크는 선형 독립인 열의 최대 부분 집합의 크기입니다.
* 행렬의 역행렬이 정의되면 행렬 반전을 통해 첫 번째 행렬의 작용을 되돌리는 다른 행렬을 찾을 수 있습니다. 행렬 반전은 이론적으로 유용하지만 수치적 불안정성 때문에 실전에서는 주의가 필요합니다.
* 행렬식은 행렬이 공간을 얼마나 확장하거나 축소하는지 측정할 수 있게 해줍니다. 0이 아닌 행렬식은 가역(비특이) 행렬을 의미하고 0 값 행렬식은 행렬이 비가역(특이)임을 의미합니다.
* 텐서 축약 및 아인슈타인 합계는 머신러닝에서 볼 수 있는 많은 계산을 표현하기 위한 깔끔하고 명확한 표기법을 제공합니다.

## 연습 문제 (Exercises)
1. 다음 사이의 각도는 얼마입니까?
$$ 
\\vec v_1 = \begin{bmatrix}
 1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
 3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$ 
2. 맞음 또는 틀림: $\\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ 및 $\\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$은 서로의 역행렬입니까?
3. 평면에 넓이가 $100\textrm{m}^2$인 도형을 그린다고 가정해 봅시다. 행렬

$$ 
\\begin{bmatrix}
 2 & 3\\ 1 & 2
\end{bmatrix}.
$$ 
로 도형을 변환한 후의 넓이는 얼마입니까?
4. 다음 벡터 집합 중 선형 독립인 것은 무엇입니까?
 * $\\left\{\\begin{pmatrix}1\\0\\-1\end{pmatrix}, \\begin{pmatrix}2\\1\\-1\end{pmatrix}, \\begin{pmatrix}3\\1\\1\end{pmatrix}\\right\}$
 * $\\left\{\\begin{pmatrix}3\\1\\1\end{pmatrix}, \\begin{pmatrix}1\\1\\1\end{pmatrix}, \\begin{pmatrix}0\\0\\0\end{pmatrix}\\right\}$
 * $\\left\{\\begin{pmatrix}1\\1\\0\end{pmatrix}, \\begin{pmatrix}0\\1\\-1\end{pmatrix}, \\begin{pmatrix}1\\0\\1\end{pmatrix}\\right\}$
5. 어떤 값 $a, b, c, d$의 선택에 대해 행렬이 $A = \\begin{bmatrix}c\\d\end{bmatrix}\\.\\begin{bmatrix}a & b\end{bmatrix}$로 쓰여졌다고 가정합니다. 맞음 또는 틀림: 그러한 행렬의 행렬식은 항상 0입니까?
6. 벡터 $e_1 = \\begin{bmatrix}1\\0\end{bmatrix}$ 및 $e_2 = \\begin{bmatrix}0\\1\end{bmatrix}$은 직교합니다. $Ae_1$과 $Ae_2$가 직교하도록 하는 행렬 $A$에 대한 조건은 무엇입니까?
7. 임의의 행렬 $A$에 대해 $\\textrm{tr}(\\mathbf{A}^4)$를 아인슈타인 표기법으로 어떻게 쓸 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1085)
:end_tab: