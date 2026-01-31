```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 패딩과 스트라이드 (Padding and Stride)
:label:`sec_padding`

:numref:`fig_correlation`의 합성곱 예제를 상기해 보십시오. 
입력은 높이와 너비가 모두 3이었고 합성곱 커널은 높이와 너비가 모두 2였으며, $2\times2$ 차원의 출력 표현을 산출했습니다.
입력 모양이 $n_\textrm{h}\times n_\textrm{w}$이고 합성곱 커널 모양이 $k_\textrm{h}\times k_\textrm{w}$라고 가정하면, 
출력 모양은 $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$이 됩니다: 
합성곱을 적용할 픽셀이 떨어질 때까지만 합성곱 커널을 이동할 수 있습니다.

다음에서는 출력 크기에 대한 더 많은 제어권을 제공하는 패딩 및 스트라이드 합성곱을 포함한 여러 기술을 살펴볼 것입니다. 
동기를 부여하자면, 커널은 일반적으로 $1$보다 큰 너비와 높이를 가지므로, 많은 연속적인 합성곱을 적용한 후에는 입력보다 상당히 작은 출력으로 끝나는 경향이 있습니다. 
$240 \times 240$ 픽셀 이미지로 시작하는 경우, 10개의 $5 \times 5$ 합성곱 레이어는 이미지를 $200 \times 200$ 픽셀로 줄여 이미지의 $30 \%$를 잘라내고 원본 이미지 경계에 있는 흥미로운 정보를 없애버립니다. 
*패딩(padding)*은 이 문제를 처리하는 가장 인기 있는 도구입니다. 
다른 경우에는 차원을 대폭 줄이고 싶을 수도 있습니다. 예를 들어 원래 입력 해상도가 다루기 힘들다고 생각되는 경우입니다. 
*스트라이드 합성곱(strided convolutions)*은 이러한 경우에 도움이 될 수 있는 인기 있는 기술입니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 패딩 (Padding)

위에서 설명한 바와 같이, 합성곱 레이어를 적용할 때 한 가지 까다로운 문제는 이미지 주변의 픽셀을 잃어버리는 경향이 있다는 것입니다. 합성곱 커널 크기와 이미지 내 위치의 함수로서 픽셀 활용도를 나타내는 :numref:`img_conv_reuse`를 고려해 보십시오. 모서리의 픽셀은 거의 사용되지 않습니다. 

![각각 $1 \times 1$, $2 \times 2$, $3 \times 3$ 크기의 합성곱에 대한 픽셀 활용도.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

우리는 일반적으로 작은 커널을 사용하므로 주어진 합성곱에 대해 몇 개의 픽셀만 잃을 수 있지만, 많은 연속적인 합성곱 레이어를 적용함에 따라 이것이 누적될 수 있습니다. 
이 문제에 대한 한 가지 간단한 해결책은 입력 이미지의 경계 주위에 채우기 픽셀을 추가하여 이미지의 유효 크기를 늘리는 것입니다. 
일반적으로 추가 픽셀의 값을 0으로 설정합니다. 
:numref:`img_conv_pad`에서는 $3 \times 3$ 입력을 패딩하여 크기를 $5 \times 5$로 늘립니다. 
해당 출력은 $4 \times 4$ 행렬로 증가합니다. 
음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+0\times1+0\times2+0\times3=0$.

![패딩이 있는 2차원 상호 상관.](../img/conv-pad.svg)
:label:`img_conv_pad`

일반적으로 총 $p_\textrm{h}$ 행의 패딩(대략 위쪽에 절반, 아래쪽에 절반)과 총 $p_\textrm{w}$ 열의 패딩(대략 왼쪽에 절반, 오른쪽에 절반)을 추가하면 출력 모양은 다음과 같습니다.

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$ 

즉, 출력의 높이와 너비가 각각 $p_\textrm{h}$와 $p_\textrm{w}$만큼 증가합니다.

많은 경우, 입력과 출력의 높이와 너비를 같게 만들기 위해 $p_\textrm{h}=k_\textrm{h}-1$ 및 $p_\textrm{w}=k_\textrm{w}-1$로 설정하고 싶을 것입니다. 
이렇게 하면 네트워크를 구성할 때 각 레이어의 출력 모양을 예측하기가 더 쉬워집니다. 
여기서 $k_\textrm{h}$가 홀수라고 가정하면, 높이의 양쪽에 $p_\textrm{h}/2$ 행을 패딩합니다. 
$k_\textrm{h}$가 짝수인 경우, 한 가지 가능성은 입력의 위쪽에 $\lceil p_\textrm{h}/2\rceil$ 행을 패딩하고 아래쪽에 $\lfloor p_\textrm{h}/2\rfloor$ 행을 패딩하는 것입니다. 
너비의 양쪽도 같은 방식으로 패딩합니다.

CNN은 일반적으로 1, 3, 5 또는 7과 같이 홀수 높이 및 너비 값을 가진 합성곱 커널을 사용합니다. 
홀수 커널 크기를 선택하면 위쪽과 아래쪽에 같은 수의 행을, 왼쪽과 오른쪽에 같은 수의 열을 패딩하면서 차원을 보존할 수 있다는 이점이 있습니다.

더욱이 홀수 커널을 사용하고 차원을 정확하게 보존하기 위해 패딩하는 이 관행은 사무적인 이점을 제공합니다. 
모든 2차원 텐서 `X`에 대해, 커널 크기가 홀수이고 모든 측면의 패딩 행과 열 수가 동일하여 입력과 동일한 높이와 너비를 가진 출력을 생성할 때, 
우리는 출력 `Y[i, j]`가 `X[i, j]`를 중심으로 하는 윈도우와 입력 및 합성곱 커널의 상호 상관에 의해 계산된다는 것을 알고 있습니다.

다음 예제에서는 높이와 너비가 3인 2차원 합성곱 레이어를 생성하고 (**모든 면에 1픽셀의 패딩을 적용합니다.**)
높이와 너비가 8인 입력이 주어지면 출력의 높이와 너비도 8임을 알 수 있습니다.

```{.python .input}
%%tab mxnet
# 합성곱을 계산하기 위한 도우미 함수를 정의합니다.
# 합성곱 레이어 가중치를 초기화하고 입력 및 출력에 대해 해당하는 차원 상승 및 축소를 수행합니다.
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)은 배치 크기와 채널 수가 모두 1임을 나타냅니다
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 처음 두 차원을 제거합니다: 예제 및 채널
    return Y.reshape(Y.shape[2:])

# 양쪽에 1행과 1열이 패딩되므로 총 2개의 행 또는 열이 추가됩니다
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# 합성곱을 계산하기 위한 도우미 함수를 정의합니다.
# 합성곱 레이어 가중치를 초기화하고 입력 및 출력에 대해 해당하는 차원 상승 및 축소를 수행합니다.
def comp_conv2d(conv2d, X):
    # (1, 1)은 배치 크기와 채널 수가 모두 1임을 나타냅니다
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 처음 두 차원을 제거합니다: 예제 및 채널
    return Y.reshape(Y.shape[2:])

# 양쪽에 1행과 1열이 패딩되므로 총 2개의 행 또는 열이 추가됩니다
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# 합성곱을 계산하기 위한 도우미 함수를 정의합니다.
# 합성곱 레이어 가중치를 초기화하고 입력 및 출력에 대해 해당하는 차원 상승 및 축소를 수행합니다.
def comp_conv2d(conv2d, X):
    # (1, 1)은 배치 크기와 채널 수가 모두 1임을 나타냅니다
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # 처음 두 차원을 제거합니다: 예제 및 채널
    return tf.reshape(Y, Y.shape[1:3])
# 양쪽에 1행과 1열이 패딩되므로 총 2개의 행 또는 열이 추가됩니다
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# 합성곱을 계산하기 위한 도우미 함수를 정의합니다.
# 합성곱 레이어 가중치를 초기화하고 입력 및 출력에 대해 해당하는 차원 상승 및 축소를 수행합니다.
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1)은 배치 크기와 채널 수가 모두 1임을 나타냅니다
    key = jax.random.PRNGKey(d2l.get_seed())
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # 차원을 제거합니다: 예제 및 채널
    return Y.reshape(Y.shape[1:3])
# 양쪽에 1행과 1열이 패딩되므로 총 2개의 행 또는 열이 추가됩니다
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

합성곱 커널의 높이와 너비가 다른 경우, [**높이와 너비에 대해 다른 패딩 숫자를 설정**]하여 출력과 입력이 동일한 높이와 너비를 갖도록 만들 수 있습니다.

```{.python .input}
%%tab mxnet
# 높이 5, 너비 3인 합성곱 커널을 사용합니다.
# 높이와 너비의 양쪽 패딩은 각각 2와 1입니다
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# 높이 5, 너비 3인 합성곱 커널을 사용합니다.
# 높이와 너비의 양쪽 패딩은 각각 2와 1입니다
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# 높이 5, 너비 3인 합성곱 커널을 사용합니다.
# 높이와 너비의 양쪽 패딩은 각각 2와 1입니다
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# 높이 5, 너비 3인 합성곱 커널을 사용합니다.
# 높이와 너비의 양쪽 패딩은 각각 2와 1입니다
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## 스트라이드 (Stride)

상호 상관을 계산할 때, 입력 텐서의 왼쪽 상단 모서리에서 합성곱 윈도우를 시작한 다음, 아래쪽과 오른쪽의 모든 위치로 밉니다. 
이전 예제에서는 기본적으로 한 번에 한 요소씩 이동했습니다. 
그러나 때로는 계산 효율성을 위해 또는 다운샘플링을 원하기 때문에 중간 위치를 건너뛰고 윈도우를 한 번에 두 개 이상의 요소로 이동합니다. 이는 합성곱 커널이 클 경우 기본 이미지의 넓은 영역을 캡처하므로 특히 유용합니다.

우리는 슬라이드당 가로지르는 행과 열의 수를 *스트라이드(stride)*라고 합니다. 
지금까지는 높이와 너비 모두에 대해 1의 스트라이드를 사용했습니다. 
때로는 더 큰 스트라이드를 사용하고 싶을 수 있습니다. 
:numref:`img_conv_stride`는 세로로 3, 가로로 2의 스트라이드를 갖는 2차원 상호 상관 연산을 보여줍니다. 
음영 처리된 부분은 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$. 
첫 번째 열의 두 번째 요소가 생성될 때 합성곱 윈도우가 아래로 세 행 이동함을 알 수 있습니다. 
첫 번째 행의 두 번째 요소가 생성될 때 합성곱 윈도우가 오른쪽으로 두 열 이동합니다. 
합성곱 윈도우가 입력에서 오른쪽으로 두 열 더 이동하면, 입력 요소가 윈도우를 채울 수 없으므로 출력이 없습니다(다른 패딩 열을 추가하지 않는 한).

![높이와 너비에 대해 각각 3과 2의 스트라이드를 갖는 상호 상관.](../img/conv-stride.svg)
:label:`img_conv_stride`

일반적으로 높이에 대한 스트라이드가 $s_\textrm{h}$이고 너비에 대한 스트라이드가 $s_\textrm{w}$일 때 출력 모양은 다음과 같습니다.

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$ 

$p_\textrm{h}=k_\textrm{h}-1$ 및 $p_\textrm{w}=k_\textrm{w}-1$로 설정하면 출력 모양은 $\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$로 단순화될 수 있습니다. 
한 단계 더 나아가 입력 높이와 너비가 높이와 너비의 스트라이드로 나누어떨어지면 출력 모양은 $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$가 됩니다.

아래에서는 [**높이와 너비 모두의 스트라이드를 2로 설정**]하여 입력 높이와 너비를 반으로 줄입니다.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

(**약간 더 복잡한 예제**)를 살펴봅시다.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## 요약 및 토론 (Summary and Discussion)

패딩은 출력의 높이와 너비를 늘릴 수 있습니다. 이는 종종 출력의 바람직하지 않은 축소를 피하기 위해 출력에 입력과 동일한 높이와 너비를 제공하는 데 사용됩니다. 또한 모든 픽셀이 똑같이 자주 사용되도록 합니다. 일반적으로 입력 높이와 너비의 양쪽에 대칭 패딩을 선택합니다. 이 경우 $(p_\textrm{h}, p_\textrm{w})$ 패딩이라고 합니다. 가장 일반적으로 $p_\textrm{h} = p_\textrm{w}$로 설정하며, 이 경우 단순히 패딩 $p$를 선택한다고 말합니다. 

비슷한 관례가 스트라이드에도 적용됩니다. 수평 스트라이드 $s_\textrm{h}$와 수직 스트라이드 $s_\textrm{w}$가 일치하면 단순히 스트라이드 $s$라고 합니다. 스트라이드는 출력의 해상도를 줄일 수 있습니다. 예를 들어 $n > 1$인 경우 출력의 높이와 너비를 입력 높이와 너비의 $1/n$로 줄입니다. 기본적으로 패딩은 0이고 스트라이드는 1입니다. 

지금까지 논의한 모든 패딩은 단순히 이미지를 0으로 확장했습니다. 이것은 달성하기 쉽기 때문에 상당한 계산상의 이점이 있습니다. 더욱이 연산자는 추가 메모리를 할당할 필요 없이 이 패딩을 암시적으로 활용하도록 엔지니어링될 수 있습니다. 동시에, 단순히 "공백"이 어디에 있는지 학습함으로써 CNN이 이미지 내의 암시적 위치 정보를 인코딩할 수 있게 합니다. 제로 패딩 외에도 많은 대안이 있습니다. :citet:`Alsallakh.Kokhlikyan.Miglani.ea.2020`는 이에 대한 광범위한 개요를 제공했습니다(아티팩트가 발생하지 않는 한 0이 아닌 패딩을 사용해야 하는 명확한 경우는 없지만). 


## 연습 문제 (Exercises)

1. 커널 크기 $(3, 5)$, 패딩 $(0, 1)$, 스트라이드 $(3, 4)$인 이 섹션의 마지막 코드 예제를 감안할 때, 
   출력 모양을 계산하여 실험 결과와 일치하는지 확인하십시오.
2. 오디오 신호의 경우 스트라이드 2는 무엇에 해당합니까?
3. 미러 패딩, 즉 경계 값을 단순히 미러링하여 텐서를 확장하는 패딩을 구현하십시오.
4. 1보다 큰 스트라이드의 계산상의 이점은 무엇입니까?
5. 1보다 큰 스트라이드의 통계적 이점은 무엇일까요?
6. $\frac{1}{2}$의 스트라이드를 어떻게 구현하시겠습니까? 이것은 무엇에 해당합니까? 언제 유용할까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/272)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17997)
:end_tab:
