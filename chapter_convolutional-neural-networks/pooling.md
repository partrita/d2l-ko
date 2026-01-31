```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 풀링 (Pooling)
:label:`sec_pooling`

많은 경우 우리의 궁극적인 작업은 이미지에 대한 전역적인 질문을 던집니다. 
예를 들어 *고양이가 포함되어 있습니까?* 결과적으로 우리 최종 레이어의 유닛은 전체 입력에 민감해야 합니다. 
점차적으로 정보를 집계하여 점점 더 거친 맵을 생성함으로써, 
우리는 처리의 중간 레이어에서 합성곱 레이어의 모든 이점을 유지하면서 궁극적으로 전역 표현을 학습한다는 목표를 달성합니다. 
네트워크에서 더 깊이 들어갈수록 각 은닉 노드가 민감한 수용 영역(입력에 비해)이 커집니다. 공간 해상도를 줄이면 합성곱 커널이 더 큰 유효 영역을 커버하기 때문에 이 프로세스가 가속화됩니다.

더욱이 가장자리와 같은 하위 수준 특성을 감지할 때(:numref:`sec_conv_layer`에서 논의됨), 
우리는 종종 표현이 평행 이동에 어느 정도 불변하기를 원합니다. 
예를 들어 흑백의 뚜렷한 경계가 있는 이미지 `X`를 가져와 전체 이미지를 오른쪽으로 한 픽셀 이동하면, 즉 `Z[i, j] = X[i, j + 1]`이면, 
새 이미지 `Z`에 대한 출력은 상당히 다를 수 있습니다. 
가장자리가 한 픽셀 이동했을 것입니다. 
실제로 객체는 정확히 같은 위치에 거의 나타나지 않습니다. 
사실 삼각대와 정지된 물체가 있어도 셔터의 움직임으로 인한 카메라의 진동으로 모든 것이 픽셀 정도 이동할 수 있습니다
(고급 카메라는 이 문제를 해결하기 위한 특수 기능을 갖추고 있습니다).

이 섹션에서는 합성곱 레이어의 위치에 대한 민감도를 완화하고 공간적으로 표현을 다운샘플링하는 두 가지 목적을 수행하는 *풀링 레이어(pooling layers)*를 소개합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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

## 최대 풀링과 평균 풀링 (Maximum Pooling and Average Pooling)

합성곱 레이어와 마찬가지로 *풀링* 연산자는 고정된 모양의 윈도우로 구성되며, 스트라이드에 따라 입력의 모든 영역 위로 미끄러지며 고정 모양 윈도우(때로는 *풀링 윈도우*라고도 함)가 지나가는 각 위치에 대해 단일 출력을 계산합니다. 
그러나 합성곱 레이어의 입력 및 커널의 상호 상관 계산과 달리, 풀링 레이어에는 파라미터가 없습니다(*커널*이 없음). 
대신 풀링 연산자는 결정론적이며, 일반적으로 풀링 윈도우에 있는 요소의 최댓값 또는 평균값을 계산합니다. 
이러한 연산을 각각 *최대 풀링(maximum pooling)* (줄여서 *max-pooling*) 및 *평균 풀링(average pooling)*이라고 합니다.

*평균 풀링*은 본질적으로 CNN만큼이나 오래되었습니다. 아이디어는 이미지를 다운샘플링하는 것과 유사합니다. 저해상도 이미지를 위해 두 번째(또는 세 번째) 픽셀마다 값을 취하는 대신, 인접한 픽셀에 대해 평균을 내어 더 나은 신호 대 잡음비를 가진 이미지를 얻을 수 있습니다. 우리는 여러 인접 픽셀의 정보를 결합하고 있기 때문입니다. *최대 풀링*은 :citet:`Riesenhuber.Poggio.1999`에서 인지 신경과학의 맥락에서 객체 인식을 위해 정보 집계가 계층적으로 집계될 수 있는 방법을 설명하기 위해 도입되었습니다. 음성 인식에는 이미 이전 버전이 있었습니다 :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. 거의 모든 경우에 최대 풀링이 평균 풀링보다 선호됩니다.

두 경우 모두 상호 상관 연산자와 마찬가지로 풀링 윈도우가 입력 텐서의 왼쪽 상단에서 시작하여 왼쪽에서 오른쪽으로, 위에서 아래로 미끄러지는 것으로 생각할 수 있습니다. 
풀링 윈도우가 닿는 각 위치에서 최대 풀링이 사용되는지 평균 풀링이 사용되는지에 따라 윈도우에 있는 입력 하위 텐서의 최댓값 또는 평균값을 계산합니다.


![풀링 윈도우 모양이 $2\times 2$인 최대 풀링. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 텐서 요소입니다: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`의 출력 텐서는 높이 2, 너비 2를 갖습니다. 
4개의 요소는 각 풀링 윈도우의 최댓값에서 파생됩니다:

$$
\max(0, 1, 3, 4)=4,\
\max(1, 2, 4, 5)=5,\
\max(3, 4, 6, 7)=7,\
\max(4, 5, 7, 8)=8.\
$$

더 일반적으로 해당 크기의 영역에 대해 집계하여 $p \times q$ 풀링 레이어를 정의할 수 있습니다. 가장자리 감지 문제로 돌아가서, 합성곱 레이어의 출력을 $2\times 2$ 최대 풀링의 입력으로 사용합니다. 
`X`를 합성곱 레이어 입력의 입력으로, `Y`를 풀링 레이어 출력으로 표시합니다. 
`X[i, j]`, `X[i, j + 1]`, `X[i+1, j]`, `X[i+1, j + 1]`의 값이 다른지 여부에 관계없이 풀링 레이어는 항상 `Y[i, j] = 1`을 출력합니다. 
즉, $2\times 2$ 최대 풀링 레이어를 사용하면 합성곱 레이어가 인식한 패턴이 높이 또는 너비에서 한 요소 이상 움직이지 않는 경우 여전히 감지할 수 있습니다.

아래 코드에서는 `pool2d` 함수에서 (**풀링 레이어의 순전파를 구현**)합니다. 
이 함수는 :numref:`sec_conv_layer`의 `corr2d` 함수와 유사합니다. 
그러나 커널이 필요하지 않으며 입력의 각 영역에 대한 최댓값 또는 평균으로 출력을 계산합니다.

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

:numref:`fig_pooling`의 입력 텐서 `X`를 구성하여 [**2차원 최대 풀링 레이어의 출력을 검증**]할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

또한 (**평균 풀링 레이어**)를 실험해 볼 수도 있습니다.

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**패딩과 스트라이드 (Padding and Stride)**]

합성곱 레이어와 마찬가지로 풀링 레이어는 출력 모양을 변경합니다. 
그리고 이전과 마찬가지로 입력을 패딩하고 스트라이드를 조정하여 원하는 출력 모양을 얻도록 작업을 조정할 수 있습니다. 
딥러닝 프레임워크의 내장 2차원 최대 풀링 레이어를 통해 풀링 레이어에서 패딩과 스트라이드의 사용을 시연할 수 있습니다. 
먼저 모양이 4차원인 입력 텐서 `X`를 구성합니다. 여기서 예제 수(배치 크기)와 채널 수는 모두 1입니다.

:begin_tab:`tensorflow`
다른 프레임워크와 달리 TensorFlow는 *채널-마지막(channels-last)* 입력을 선호하고 이에 최적화되어 있음에 유의하십시오.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

풀링은 영역의 정보를 집계하므로 (**딥러닝 프레임워크는 기본적으로 풀링 윈도우 크기와 스트라이드를 일치시킵니다.**) 예를 들어 `(3, 3)` 모양의 풀링 윈도우를 사용하면 기본적으로 `(3, 3)`의 스트라이드 모양을 얻습니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# 풀링에는 모델 파라미터가 없으므로 초기화가 필요하지 않습니다
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# 풀링에는 모델 파라미터가 없으므로 초기화가 필요하지 않습니다
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# 풀링에는 모델 파라미터가 없으므로 초기화가 필요하지 않습니다
pool2d(X)
```

```{.python .input}
%%tab jax
# 풀링에는 모델 파라미터가 없으므로 초기화가 필요하지 않습니다
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

말할 필요도 없이, 필요한 경우 프레임워크 기본값을 재정의하기 위해 [**스트라이드와 패딩을 수동으로 지정할 수 있습니다**].

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

물론 아래 예제와 같이 임의의 높이와 너비를 가진 임의의 직사각형 풀링 윈도우를 지정할 수 있습니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## 다중 채널 (Multiple Channels)

다중 채널 입력 데이터를 처리할 때, 
[**풀링 레이어는**] 합성곱 레이어처럼 채널에 대해 입력을 합산하는 대신 [**각 입력 채널을 개별적으로 풀링합니다**]. 
이는 풀링 레이어의 출력 채널 수가 입력 채널 수와 같다는 것을 의미합니다. 
아래에서 채널 차원에서 텐서 `X`와 `X + 1`을 연결하여 두 개의 채널이 있는 입력을 구성합니다.

:begin_tab:`tensorflow`
TensorFlow는 채널-마지막 구문으로 인해 마지막 차원을 따라 연결해야 한다는 점에 유의하십시오.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# 채널-마지막 구문으로 인해 `dim=3`을 따라 연결
X = d2l.concat([X, X + 1], 3)
X
```

보시다시피 풀링 후에도 출력 채널 수는 여전히 2개입니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
TensorFlow 풀링의 출력이 언뜻 보기에 다르게 보일 수 있지만, 수치적으로는 MXNet 및 PyTorch와 동일한 결과가 제시된다는 점에 유의하십시오. 
차이점은 차원성에 있으며, 출력을 수직으로 읽으면 다른 구현과 동일한 출력을 산출합니다.
:end_tab:

## 요약 (Summary)

풀링은 매우 간단한 연산입니다. 이름이 나타내는 대로 정확히 값의 윈도우에 대한 결과를 집계합니다. 스트라이드 및 패딩과 같은 모든 합성곱 의미론은 이전과 동일한 방식으로 적용됩니다. 풀링은 채널에 무관심합니다. 즉, 채널 수를 변경하지 않고 각 채널에 개별적으로 적용됩니다. 마지막으로, 두 가지 인기 있는 풀링 선택 중 최대 풀링이 평균 풀링보다 선호되는데, 이는 출력에 어느 정도의 불변성을 부여하기 때문입니다. 인기 있는 선택은 공간 해상도를 1/4로 줄이기 위해 $2 \times 2$의 풀링 윈도우 크기를 선택하는 것입니다. 

풀링 외에도 해상도를 줄이는 방법은 더 많이 있습니다. 예를 들어 확률적 풀링 :cite:`Zeiler.Fergus.2013` 및 부분 최대 풀링 :cite:`Graham.2014`에서 집계는 무작위화와 결합됩니다. 이는 일부 경우에 정확도를 약간 향상시킬 수 있습니다. 마지막으로, 나중에 주의 메커니즘에서 볼 수 있듯이, 쿼리와 표현 벡터 간의 정렬을 사용하는 등 출력에 대해 집계하는 더 세련된 방법이 있습니다. 


## 연습 문제 (Exercises)

1. 합성곱을 통해 평균 풀링을 구현하십시오. 
2. 최대 풀링은 합성곱만으로 구현될 수 없음을 증명하십시오. 
3. 최대 풀링은 ReLU 연산, 즉 $\textrm{ReLU}(x) = \max(0, x)$을 사용하여 달성할 수 있습니다.
    1. ReLU 연산만 사용하여 $\max (a, b)$를 표현하십시오.
    1. 이를 사용하여 합성곱 및 ReLU 레이어를 통해 최대 풀링을 구현하십시오. 
    1. $2 \times 2$ 합성곱에는 몇 개의 채널과 레이어가 필요합니까? $3 \times 3$ 합성곱에는 몇 개가 필요합니까?
4. 풀링 레이어의 계산 비용은 얼마입니까? 풀링 레이어에 대한 입력 크기가 $c\times h\times w$이고, 풀링 윈도우 모양이 $p_\textrm{h}\times p_\textrm{w}$이며 패딩 $(p_\textrm{h}, p_\textrm{w})$와 스트라이드 $(s_\textrm{h}, s_\textrm{w})$를 갖는다고 가정합니다. 
5. 최대 풀링과 평균 풀링이 다르게 작동할 것으로 예상하는 이유는 무엇입니까? 
6. 별도의 최소 풀링 레이어가 필요합니까? 다른 연산으로 대체할 수 있습니까? 
7. 풀링에 소프트맥스 연산을 사용할 수 있습니다. 왜 인기가 없을까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/274)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17999)
:end_tab: