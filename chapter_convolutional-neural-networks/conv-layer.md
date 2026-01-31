```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 이미지를 위한 합성곱 (Convolutions for Images)
:label:`sec_conv_layer`

이제 이론적으로 합성곱 레이어가 어떻게 작동하는지 이해했으므로, 실제로 어떻게 작동하는지 볼 준비가 되었습니다. 
이미지 데이터에서 구조를 탐색하기 위한 효율적인 아키텍처로서의 합성곱 신경망에 대한 동기를 바탕으로, 우리는 실행 예제로 이미지를 고수합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 상호 상관 연산 (The Cross-Correlation Operation)

엄밀히 말하면 합성곱 레이어라는 이름은 잘못된 것입니다. 그들이 표현하는 연산은 상호 상관으로 더 정확하게 설명되기 때문입니다. 
:numref:`sec_why-conv`의 합성곱 레이어 설명에 따르면, 그러한 레이어에서 입력 텐서와 커널 텐서는 (**상호 상관 연산**)을 통해 결합되어 출력 텐서를 생성합니다.

지금은 채널을 무시하고 2차원 데이터와 은닉 표현으로 이것이 어떻게 작동하는지 봅시다. 
:numref:`fig_correlation`에서 입력은 높이 3, 너비 3인 2차원 텐서입니다. 
우리는 텐서의 모양을 $3 \times 3$ 또는 ($3$, $3$)으로 표시합니다. 
커널의 높이와 너비는 모두 2입니다. 
*커널 윈도우* (또는 *합성곱 윈도우*)의 모양은 커널의 높이와 너비로 주어집니다(여기서는 $2 \times 2$).

![2차원 상호 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

2차원 상호 상관 연산에서는 입력 텐서의 왼쪽 상단 모서리에 위치한 합성곱 윈도우로 시작하여 왼쪽에서 오른쪽으로, 위에서 아래로 입력 텐서를 가로질러 밉니다. 
합성곱 윈도우가 특정 위치로 미끄러질 때, 해당 윈도우에 포함된 입력 하위 텐서와 커널 텐서가 요소별로 곱해지고 결과 텐서가 합산되어 단일 스칼라 값을 산출합니다. 
이 결과는 해당 위치에서 출력 텐서의 값을 제공합니다. 
여기서 출력 텐서는 높이 2, 너비 2를 가지며 4개의 요소는 2차원 상호 상관 연산에서 파생됩니다:

$$ 
0\times0+1\times1+3\times2+4\times3=19,\
1\times0+2\times1+4\times2+5\times3=25,\
3\times0+4\times1+6\times2+7\times3=37,\
4\times0+5\times1+7\times2+8\times3=43. 
$$

각 축을 따라 출력 크기는 입력 크기보다 약간 작습니다. 
커널의 너비와 높이가 $1$보다 크기 때문에, 커널이 이미지 내에 완전히 들어맞는 위치에 대해서만 상호 상관을 적절하게 계산할 수 있습니다. 
출력 크기는 입력 크기 $n_\textrm{h} \times n_\textrm{w}$에서 합성곱 커널 크기 $k_\textrm{h} \times k_\textrm{w}$를 뺀 값으로 다음과 같이 주어집니다.

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$ 

이는 이미지를 가로질러 합성곱 커널을 "이동"할 충분한 공간이 필요하기 때문입니다. 
나중에 커널을 이동할 충분한 공간이 있도록 경계 주위에 0으로 이미지를 패딩하여 크기를 변경하지 않고 유지하는 방법을 볼 것입니다. 
다음으로 입력 텐서 `X`와 커널 텐서 `K`를 받아 출력 텐서 `Y`를 반환하는 `corr2d` 함수로 이 과정을 구현합니다.

```{.python .input}
%%tab mxnet
def corr2d(X, K):  #@save
    """2D 상호 상관을 계산합니다."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab pytorch
def corr2d(X, K):  #@save
    """2D 상호 상관을 계산합니다."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab jax
def corr2d(X, K):  #@save
    """2D 상호 상관을 계산합니다."""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y
```

```{.python .input}
%%tab tensorflow
def corr2d(X, K):  #@save
    """2D 상호 상관을 계산합니다."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

:numref:`fig_correlation`의 입력 텐서 `X`와 커널 텐서 `K`를 구성하여 2차원 상호 상관 연산의 [**위 구현 출력을 검증**]할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 합성곱 레이어 (Convolutional Layers)

합성곱 레이어는 입력과 커널을 상호 상관시키고 스칼라 편향을 더하여 출력을 생성합니다. 
합성곱 레이어의 두 파라미터는 커널과 스칼라 편향입니다. 
합성곱 레이어를 기반으로 모델을 훈련할 때, 일반적으로 완전 연결 레이어와 마찬가지로 커널을 무작위로 초기화합니다.

이제 위에서 정의한 `corr2d` 함수를 기반으로 [**2차원 합성곱 레이어를 구현**]할 준비가 되었습니다. 
`__init__` 생성자 메서드에서 `weight`와 `bias`를 두 모델 파라미터로 선언합니다. 
순전파 메서드는 `corr2d` 함수를 호출하고 편향을 더합니다.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
%%tab jax
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

$h \times w$ 합성곱 또는 $h \times w$ 합성곱 커널에서, 합성곱 커널의 높이와 너비는 각각 $h$와 $w$입니다. 
우리는 또한 $h \times w$ 합성곱 커널을 가진 합성곱 레이어를 단순히 $h \times w$ 합성곱 레이어라고 부릅니다.


## 이미지의 객체 가장자리 감지 (Object Edge Detection in Images)

픽셀 변화의 위치를 찾아 [**합성곱 레이어의 간단한 응용: 이미지의 객체 가장자리 감지**]를 분석해 봅시다. 
먼저 $6\times 8$ 픽셀의 "이미지"를 구성합니다. 
가운데 네 열은 검은색($0$)이고 나머지는 흰색($1$)입니다.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

```{.python .input}
%%tab jax
X = jnp.ones((6, 8))
X = X.at[:, 2:6].set(0)
X
```

다음으로 높이가 1이고 너비가 2인 커널 `K`를 구성합니다. 
입력과 상호 상관 연산을 수행할 때, 수평으로 인접한 요소가 같으면 출력은 0입니다. 그렇지 않으면 출력은 0이 아닙니다. 
이 커널은 유한 차분 연산자의 특수한 경우라는 점에 유의하십시오. 위치 $(i,j)$에서 $x_{i,j} - x_{(i+1),j}$를 계산합니다. 즉, 수평으로 인접한 픽셀 값의 차이를 계산합니다. 이것은 수평 방향의 1계 도함수의 이산 근사입니다. 결국 함수 $f(i,j)$에 대해 그 도함수는 $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$입니다. 이것이 실제로 어떻게 작동하는지 봅시다.

```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

인수 `X`(입력)와 `K`(커널)로 상호 상관 연산을 수행할 준비가 되었습니다. 
보시다시피, [**흰색에서 검은색으로 변하는 가장자리는 $1$로, 검은색에서 흰색으로 변하는 가장자리는 $-1$로 감지합니다.**] 
다른 모든 출력은 값 $0$을 취합니다.

```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

이제 전치된 이미지에 커널을 적용할 수 있습니다. 
예상대로 사라집니다. [**커널 `K`는 수직 가장자리만 감지합니다.**]

```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## 커널 학습하기 (Learning a Kernel)

이것이 우리가 찾고 있는 것이 정확히 무엇인지 안다면 유한 차분 `[1, -1]`로 가장자리 감지기를 설계하는 것은 깔끔합니다. 
하지만 더 큰 커널을 보고 연속적인 합성곱 레이어를 고려할 때, 각 필터가 무엇을 해야 하는지 수동으로 정확하게 지정하는 것은 불가능할 수 있습니다.

이제 입력-출력 쌍만 보고 [**`X`에서 `Y`를 생성한 커널을 학습할 수 있는지**] 봅시다. 
먼저 합성곱 레이어를 구성하고 커널을 무작위 텐서로 초기화합니다. 
다음으로 각 반복에서 제곱 오차를 사용하여 `Y`를 합성곱 레이어의 출력과 비교합니다. 
그런 다음 기울기를 계산하여 커널을 업데이트할 수 있습니다. 
단순함을 위해 다음에서는 2차원 합성곱 레이어에 대한 내장 클래스를 사용하고 편향을 무시합니다.

```{.python .input}
%%tab mxnet
# 1개의 출력 채널과 모양 (1, 2)의 커널을 가진 2차원 합성곱 레이어를 구성합니다.
# 단순함을 위해 여기서는 편향을 무시합니다
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# 2차원 합성곱 레이어는 (예제, 채널, 높이, 너비) 형식의 4차원 입력 및 출력을 사용합니다.
# 여기서 배치 크기(배치의 예제 수)와 채널 수는 모두 1입니다
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # 학습률

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # 커널 업데이트
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# 1개의 출력 채널과 모양 (1, 2)의 커널을 가진 2차원 합성곱 레이어를 구성합니다.
# 단순함을 위해 여기서는 편향을 무시합니다
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# 2차원 합성곱 레이어는 (예제, 채널, 높이, 너비) 형식의 4차원 입력 및 출력을 사용합니다.
# 여기서 배치 크기(배치의 예제 수)와 채널 수는 모두 1입니다
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 학습률

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 커널 업데이트
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# 1개의 출력 채널과 모양 (1, 2)의 커널을 가진 2차원 합성곱 레이어를 구성합니다.
# 단순함을 위해 여기서는 편향을 무시합니다
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# 2차원 합성곱 레이어는 (예제, 높이, 너비, 채널) 형식의 4차원 입력 및 출력을 사용합니다.
# 여기서 배치 크기(배치의 예제 수)와 채널 수는 모두 1입니다
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # 학습률

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # 커널 업데이트
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
%%tab jax
# 1개의 출력 채널과 모양 (1, 2)의 커널을 가진 2차원 합성곱 레이어를 구성합니다.
# 단순함을 위해 여기서는 편향을 무시합니다
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')

# 2차원 합성곱 레이어는 (예제, 높이, 너비, 채널) 형식의 4차원 입력 및 출력을 사용합니다.
# 여기서 배치 크기(배치의 예제 수)와 채널 수는 모두 1입니다
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2  # 학습률

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    # 커널 업데이트
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

10번 반복 후 오차가 작은 값으로 떨어졌음에 유의하십시오. 이제 [**우리가 학습한 커널 텐서를 살펴봅시다.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
%%tab jax
params['params']['kernel'].reshape((1, 2))
```

실제로 학습된 커널 텐서는 우리가 앞서 정의한 커널 텐서 `K`와 놀랍도록 가깝습니다.

## 상호 상관과 합성곱 (Cross-Correlation and Convolution)

:numref:`sec_why-conv`에서 상호 상관 연산과 합성곱 연산 사이의 대응 관계에 대한 관찰을 상기해 보십시오. 
여기서 2차원 합성곱 레이어를 계속 고려해 봅시다. 
만약 그러한 레이어가 상호 상관 대신 :eqref:`eq_2d-conv-discrete`에 정의된 엄격한 합성곱 연산을 수행한다면 어떨까요? 
엄격한 *합성곱* 연산의 출력을 얻으려면, 2차원 커널 텐서를 수평 및 수직으로 모두 뒤집은 다음 입력 텐서와 *상호 상관* 연산을 수행하기만 하면 됩니다.

딥러닝에서는 데이터로부터 커널을 학습하기 때문에, 그러한 레이어가 엄격한 합성곱 연산을 수행하든 상호 상관 연산을 수행하든 합성곱 레이어의 출력은 영향을 받지 않는다는 점에 주목할 가치가 있습니다.

이를 설명하기 위해, 합성곱 레이어가 *상호 상관*을 수행하고 :numref:`fig_correlation`의 커널을 학습한다고 가정합시다. 여기서는 행렬 $\mathbf{K}$로 표시됩니다. 
다른 조건이 변하지 않는다고 가정할 때, 이 레이어가 대신 엄격한 *합성곱*을 수행한다면, 학습된 커널 $\mathbf{K}'$는 $\mathbf{K}'$가 수평 및 수직으로 모두 뒤집힌 후 $\mathbf{K}$와 같아질 것입니다. 
즉, 합성곱 레이어가 :numref:`fig_correlation`의 입력과 $\mathbf{K}'$에 대해 엄격한 *합성곱*을 수행할 때, :numref:`fig_correlation`의 동일한 출력(입력과 $\mathbf{K}$의 상호 상관)을 얻게 됩니다.

딥러닝 문헌의 표준 용어를 따르기 위해, 엄밀히 말하면 약간 다르지만 상호 상관 연산을 계속해서 합성곱이라고 부를 것입니다. 
또한 레이어 표현이나 합성곱 커널을 나타내는 텐서의 항목(또는 구성 요소)을 지칭하기 위해 *요소*라는 용어를 사용합니다.


## 특성 맵과 수용 영역 (Feature Map and Receptive Field)

:numref:`subsec_why-conv-channels`에서 설명한 바와 같이, :numref:`fig_correlation`의 합성곱 레이어 출력은 때때로 *특성 맵(feature map)*이라고 불립니다. 후속 레이어에 대한 공간 차원(예: 너비 및 높이)의 학습된 표현(특성)으로 간주될 수 있기 때문입니다. 
CNN에서 어떤 레이어의 요소 $x$에 대해, 그 *수용 영역(receptive field)*은 순전파 동안 $x$의 계산에 영향을 줄 수 있는 (모든 이전 레이어의) 모든 요소를 말합니다. 
수용 영역은 입력의 실제 크기보다 클 수 있다는 점에 유의하십시오.

:numref:`fig_correlation`을 계속 사용하여 수용 영역을 설명해 봅시다. 
$2 \times 2$ 합성곱 커널이 주어졌을 때, 음영 처리된 출력 요소(값 $19$)의 수용 영역은 입력의 음영 처리된 부분에 있는 4개의 요소입니다. 
이제 $2 \times 2$ 출력을 $\mathbf{Y}$로 표시하고, $\mathbf{Y}$를 입력으로 받아 단일 요소 $z$를 출력하는 추가적인 $2 \times 2$ 합성곱 레이어가 있는 더 깊은 CNN을 고려해 봅시다. 
이 경우, $\mathbf{Y}$에 대한 $z$의 수용 영역은 $\mathbf{Y}$의 4개 요소를 모두 포함하는 반면, 입력에 대한 수용 영역은 9개의 입력 요소를 모두 포함합니다. 
따라서 특성 맵의 어떤 요소가 더 넓은 영역의 입력 특성을 감지하기 위해 더 큰 수용 영역이 필요한 경우, 우리는 더 깊은 네트워크를 구축할 수 있습니다.


수용 영역이라는 이름은 신경생리학에서 유래했습니다. 
다양한 자극을 사용하여 다양한 동물에 대해 수행된 일련의 실험들 :cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968`은 소위 시각 피질이 해당 자극에 반응하는 것을 탐구했습니다. 
대체로 그들은 낮은 수준이 가장자리 및 관련 모양에 반응한다는 것을 발견했습니다. 
나중에 :citet:`Field.1987`은 합성곱 커널이라고밖에 부를 수 없는 것으로 자연 이미지에 대한 이 효과를 설명했습니다. 
우리는 놀라운 유사성을 설명하기 위해 :numref:`field_visual`에 핵심 그림을 다시 인쇄합니다.

![ :citet:`Field.1987`에서 가져온 그림 및 캡션: 6개의 다른 채널로 코딩하는 예. (왼쪽) 각 채널과 관련된 6가지 유형의 센서 예. (오른쪽) (가운데) 이미지를 (왼쪽)에 표시된 6개 센서로 합성곱. 개별 센서의 응답은 센서 크기에 비례하는 거리(점으로 표시됨)에서 이러한 필터링된 이미지를 샘플링하여 결정됩니다. 이 다이어그램은 짝수 대칭 센서의 응답만 보여줍니다.](../img/field-visual.png)
:label:`field_visual`

밝혀진 바로는, 이 관계는 예를 들어 :citet:`Kuzovkin.Vicente.Petton.ea.2018`에서 입증된 바와 같이 이미지 분류 작업에 대해 훈련된 네트워크의 더 깊은 레이어에 의해 계산된 특성에도 적용됩니다. 
합성곱은 생물학과 코드 모두에서 컴퓨터 비전을 위한 믿을 수 없을 정도로 강력한 도구임이 입증되었다고 말하는 것으로 충분합니다. 
따라서 (지나고 나서 보면) 그것들이 딥러닝의 최근 성공을 예고했다는 것은 놀라운 일이 아닙니다.

## 요약 (Summary)

합성곱 레이어에 필요한 핵심 계산은 상호 상관 연산입니다. 우리는 간단한 중첩 for-루프만으로 그 값을 계산할 수 있음을 보았습니다. 다중 입력 및 다중 출력 채널이 있는 경우, 우리는 채널 간에 행렬-행렬 연산을 수행합니다. 보시다시피 계산은 간단하며, 가장 중요한 것은 고도로 *국소적*이라는 점입니다. 이는 상당한 하드웨어 최적화를 제공하며 컴퓨터 비전의 많은 최근 결과는 그 덕분에 가능했습니다. 결국 칩 설계자가 합성곱 최적화와 관련하여 메모리보다는 빠른 계산에 투자할 수 있다는 것을 의미합니다. 이것이 다른 응용 분야에 대한 최적의 설계로 이어지지는 않을지라도, 유비쿼터스하고 저렴한 컴퓨터 비전의 문을 엽니다.

합성곱 자체의 측면에서 보면, 가장자리 및 선 감지, 이미지 흐리기 또는 선명하게 하기 등 다양한 목적으로 사용될 수 있습니다. 
가장 중요한 것은 통계학자(또는 엔지니어)가 적절한 필터를 발명할 필요가 없다는 것입니다. 
대신 데이터로부터 간단히 *학습*할 수 있습니다. 
이것은 특성 엔지니어링 휴리스틱을 증거 기반 통계로 대체합니다. 
마지막으로, 아주 기쁘게도 이러한 필터는 심층 네트워크를 구축하는 데 유리할 뿐만 아니라 뇌의 수용 영역 및 특성 맵과도 일치합니다. 
이것은 우리가 올바른 길을 가고 있다는 확신을 줍니다.

## 연습 문제 (Exercises)

1. 대각선 가장자리가 있는 이미지 `X`를 구성하십시오.
    1. 이 섹션의 커널 `K`를 적용하면 어떻게 됩니까?
    2. `X`를 전치하면 어떻게 됩니까?
    3. `K`를 전치하면 어떻게 됩니까?
2. 커널을 수동으로 설계해 보십시오.
    1. 방향 벡터 $\mathbf{v} = (v_1, v_2)$가 주어졌을 때, $\mathbf{v}$에 직교하는 가장자리, 즉 $(v_2, -v_1)$ 방향의 가장자리를 감지하는 가장자리 감지 커널을 유도하십시오.
    2. 2계 도함수에 대한 유한 차분 연산자를 유도하십시오. 이와 관련된 합성곱 커널의 최소 크기는 얼마입니까? 이미지의 어떤 구조가 가장 강하게 반응합니까?
    3. 흐림(blur) 커널을 어떻게 설계하시겠습니까? 왜 그런 커널을 사용하고 싶을까요?
    4. 차수 $d$의 도함수를 얻기 위한 커널의 최소 크기는 얼마입니까?
3. 우리가 만든 `Conv2D` 클래스에 대한 기울기를 자동으로 찾으려고 할 때 어떤 종류의 오류 메시지가 표시됩니까?
4. 입력 및 커널 텐서를 변경하여 상호 상관 연산을 행렬 곱셈으로 어떻게 표현합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/271)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17996)
:end_tab: