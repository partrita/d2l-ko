```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다중 입력 및 다중 출력 채널 (Multiple Input and Multiple Output Channels)
:label:`sec_channels`

우리는 각 이미지를 구성하는 다중 채널(예: 컬러 이미지는 빨강, 초록, 파랑의 양을 나타내는 표준 RGB 채널을 가짐)과 :numref:`subsec_why-conv-channels`에서 다중 채널을 위한 합성곱 레이어를 설명했지만, 
지금까지 모든 수치 예제를 단일 입력 및 단일 출력 채널로 작업하여 단순화했습니다. 
이를 통해 입력, 합성곱 커널, 출력을 각각 2차원 텐서로 생각할 수 있었습니다.

채널을 혼합에 추가하면 입력과 은닉 표현 모두 3차원 텐서가 됩니다. 
예를 들어 각 RGB 입력 이미지는 $3\times h\times w$ 모양을 갖습니다. 
크기가 3인 이 축을 *채널(channel)* 차원이라고 합니다. 채널의 개념은 CNN 자체만큼이나 오래되었습니다. 예를 들어 LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`는 채널을 사용합니다. 
이 섹션에서는 다중 입력 및 다중 출력 채널이 있는 합성곱 커널을 더 깊이 살펴봅니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 다중 입력 채널 (Multiple Input Channels)

입력 데이터에 다중 채널이 포함된 경우, 입력 데이터와 상호 상관을 수행할 수 있도록 입력 데이터와 동일한 수의 입력 채널을 가진 합성곱 커널을 구성해야 합니다. 
입력 데이터의 채널 수가 $c_\textrm{i}$라고 가정하면, 합성곱 커널의 입력 채널 수도 $c_\textrm{i}$여야 합니다. 합성곱 커널의 윈도우 모양이 $k_\textrm{h}\times k_\textrm{w}$인 경우, $c_\textrm{i}=1$일 때 합성곱 커널을 $k_\textrm{h}\times k_\textrm{w}$ 모양의 2차원 텐서로 생각할 수 있습니다.

그러나 $c_\textrm{i}>1$인 경우, *모든* 입력 채널에 대해 $k_\textrm{h}\times k_\textrm{w}$ 모양의 텐서를 포함하는 커널이 필요합니다. 이 $c_\textrm{i}$ 텐서들을 함께 연결하면 $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ 모양의 합성곱 커널이 생성됩니다. 
입력과 합성곱 커널 각각에 $c_\textrm{i}$ 채널이 있으므로, 각 채널에 대해 입력의 2차원 텐서와 합성곱 커널의 2차원 텐서에 대해 상호 상관 연산을 수행하고, $c_\textrm{i}$ 결과를 함께 더하여(채널에 대해 합산) 2차원 텐서를 산출할 수 있습니다. 
이것이 다중 채널 입력과 다중 입력 채널 합성곱 커널 간의 2차원 상호 상관 결과입니다.

:numref:`fig_conv_multi_in`은 두 개의 입력 채널이 있는 2차원 상호 상관의 예를 제공합니다. 
음영 처리된 부분은 첫 번째 출력 요소뿐만 아니라 출력 계산에 사용된 입력 및 커널 텐서 요소입니다:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![두 개의 입력 채널을 사용한 상호 상관 계산.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


여기서 무슨 일이 일어나고 있는지 정말로 이해했는지 확인하기 위해, 우리는 (**다중 입력 채널을 사용한 상호 상관 연산을 직접 구현**)할 수 있습니다. 
우리가 하는 일은 채널당 상호 상관 연산을 수행한 다음 결과를 더하는 것뿐이라는 점에 유의하십시오.

```{.python .input}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # K의 0번째 차원(채널)을 먼저 반복한 다음 더합니다
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # K의 0번째 차원(채널)을 먼저 반복한 다음 더합니다
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

:numref:`fig_conv_multi_in`의 값에 해당하는 입력 텐서 `X`와 커널 텐서 `K`를 구성하여 상호 상관 연산의 (**출력을 검증**)할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 다중 출력 채널 (Multiple Output Channels)
:label:`subsec_multi-output-channels`

입력 채널 수에 관계없이 지금까지 우리는 항상 하나의 출력 채널로 끝났습니다. 
그러나 :numref:`subsec_why-conv-channels`에서 논의했듯이 각 레이어에 다중 채널을 갖는 것이 필수적임이 밝혀졌습니다. 
가장 인기 있는 신경망 아키텍처에서는 실제로 신경망 깊이 들어갈수록 채널 차원을 늘리는데, 일반적으로 공간 해상도를 더 큰 *채널 깊이*와 교환하기 위해 다운샘플링합니다. 
직관적으로 각 채널이 다른 특성 세트에 반응한다고 생각할 수 있습니다. 
현실은 이것보다 조금 더 복잡합니다. 순진한 해석은 표현이 픽셀당 또는 채널당 독립적으로 학습됨을 시사할 것입니다. 
대신 채널은 공동으로 유용하도록 최적화됩니다. 
이는 단일 채널을 가장자리 감지기에 매핑하는 대신 채널 공간의 어떤 방향이 가장자리 감지에 해당할 수 있음을 의미합니다.

입력 및 출력 채널 수를 각각 $c_\textrm{i}$ 및 $c_\textrm{o}$로, 커널의 높이와 너비를 $k_\textrm{h}$ 및 $k_\textrm{w}$로 표시합니다. 
다중 채널 출력을 얻기 위해, *모든* 출력 채널에 대해 $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ 모양의 커널 텐서를 생성할 수 있습니다. 
출력 채널 차원에서 이들을 연결하여 합성곱 커널의 모양이 $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$가 되도록 합니다. 
상호 상관 연산에서 각 출력 채널의 결과는 해당 출력 채널에 해당하는 합성곱 커널에서 계산되며 입력 텐서의 모든 채널에서 입력을 받습니다.

아래와 같이 [**다중 채널의 출력을 계산**]하는 상호 상관 함수를 구현합니다.

```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # K의 0번째 차원을 반복하고 매번 입력 X와 상호 상관 연산을 수행합니다.
    # 모든 결과가 함께 쌓입니다
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

`K`에 대한 커널 텐서를 `K+1` 및 `K+2`와 연결하여 3개의 출력 채널이 있는 간단한 합성곱 커널을 구성합니다.

```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

아래에서는 입력 텐서 `X`와 커널 텐서 `K`에 대해 상호 상관 연산을 수행합니다. 
이제 출력에는 3개의 채널이 포함됩니다. 
첫 번째 채널의 결과는 이전 입력 텐서 `X`와 다중 입력 채널, 단일 출력 채널 커널의 결과와 일치합니다.

```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ 합성곱 레이어 ($1\times 1$ Convolutional Layer)
:label:`subsec_1x1`

처음에는 [**$1 \times 1$ 합성곱**], 즉 $k_\textrm{h} = k_\textrm{w} = 1$이 별로 의미가 없어 보일 수 있습니다. 
결국 합성곱은 인접한 픽셀을 상관시킵니다. 
$1 \times 1$ 합성곱은 분명히 그렇지 않습니다. 
그럼에도 불구하고 이들은 복잡한 심층 네트워크 설계에 때때로 포함되는 인기 있는 연산입니다 :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`. 
이것이 실제로 무엇을 하는지 자세히 살펴봅시다.

최소 윈도우가 사용되기 때문에 $1\times 1$ 합성곱은 높이 및 너비 차원에서 인접 요소 간의 상호 작용으로 구성된 패턴을 인식하는 더 큰 합성곱 레이어의 능력을 잃습니다. 
$1\times 1$ 합성곱의 유일한 계산은 채널 차원에서 발생합니다.

:numref:`fig_conv_1x1`은 3개의 입력 채널과 2개의 출력 채널을 가진 $1\times 1$ 합성곱 커널을 사용한 상호 상관 계산을 보여줍니다. 
입력과 출력의 높이와 너비가 동일하다는 점에 유의하십시오. 
출력의 각 요소는 입력 이미지의 *동일한 위치*에 있는 요소들의 선형 결합에서 파생됩니다. 
$1\times 1$ 합성곱 레이어를 모든 단일 픽셀 위치에 적용되는 완전 연결 레이어로 생각하여 $c_\textrm{i}$개의 해당 입력 값을 $c_\textrm{o}$개의 출력 값으로 변환하는 것으로 생각할 수 있습니다. 
이것은 여전히 합성곱 레이어이므로 가중치는 픽셀 위치 전체에 묶여 있습니다. 
따라서 $1\times 1$ 합성곱 레이어에는 $c_\textrm{o}\times c_\textrm{i}$ 가중치(더하기 편향)가 필요합니다. 또한 합성곱 레이어 뒤에는 일반적으로 비선형성이 뒤따릅니다. 이렇게 하면 $1 \times 1$ 합성곱이 단순히 다른 합성곱으로 접혀 들어가는 것을 방지할 수 있습니다. 

![3개의 입력 채널과 2개의 출력 채널이 있는 $1\times 1$ 합성곱 커널을 사용하는 상호 상관 계산. 입력과 출력은 높이와 너비가 같습니다.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

이것이 실제로 작동하는지 확인해 봅시다: 완전 연결 레이어를 사용하여 $1 \times 1$ 합성곱을 구현합니다. 
유일한 것은 행렬 곱셈 전후에 데이터 모양을 약간 조정해야 한다는 것입니다.

```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # 완전 연결 레이어에서의 행렬 곱셈
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

$1\times 1$ 합성곱을 수행할 때, 위의 함수는 이전에 구현된 상호 상관 함수 `corr2d_multi_in_out`과 동일합니다. 
샘플 데이터로 이것을 확인해 봅시다.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (3, 3, 3)) + 0 * 1
K = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 3, 1, 1)) + 0 * 1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## 토론 (Discussion)

채널을 사용하면 두 가지 장점을 결합할 수 있습니다: 상당한 비선형성을 허용하는 MLP와 특성의 *국소적* 분석을 허용하는 합성곱입니다. 특히 채널을 통해 CNN은 가장자리 및 모양 감지기와 같은 여러 특성을 동시에 추론할 수 있습니다. 또한 평행 이동 불변성 및 지역성에서 발생하는 급격한 파라미터 감소와 컴퓨터 비전에서 표현력 있고 다양한 모델의 필요성 사이의 실용적인 절충안을 제공합니다. 

하지만 이 유연성에는 대가가 따릅니다. 크기 $(h \times w)$의 이미지가 주어졌을 때, $k \times k$ 합성곱을 계산하는 비용은 $\mathcal{O}(h \cdot w \cdot k^2)$입니다. $c_\textrm{i}$ 및 $c_\textrm{o}$ 입력 및 출력 채널의 경우 이는 각각 $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$로 증가합니다. $5 \times 5$ 커널과 $128$개의 입력 및 출력 채널이 있는 $256 \times 256$ 픽셀 이미지의 경우 이는 530억 개 이상의 연산에 해당합니다(곱셈과 덧셈을 별도로 계산). 나중에 우리는 채널별 연산이 블록 대각선이어야 한다는 요구 사항을 통해 비용을 줄이는 효과적인 전략을 만날 것입니다. 이는 ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`와 같은 아키텍처로 이어집니다. 

## 연습 문제 (Exercises)

1. 각각 크기 $k_1$과 $k_2$인 두 개의 합성곱 커널이 있다고 가정합니다(중간에 비선형성 없음). 
    1. 연산 결과가 단일 합성곱으로 표현될 수 있음을 증명하십시오. 
    1. 동등한 단일 합성곱의 차원은 무엇입니까? 
    1. 역이 성립합니까? 즉, 항상 합성곱을 두 개의 더 작은 합성곱으로 분해할 수 있습니까? 
2. 모양 $c_\textrm{i}\times h\times w$의 입력과 모양 $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$의 합성곱 커널, 패딩 $(p_\textrm{h}, p_\textrm{w})$, 스트라이드 $(s_\textrm{h}, s_\textrm{w})$를 가정합니다. 
    1. 순전파에 대한 계산 비용(곱셈 및 덧셈)은 얼마입니까? 
    2. 메모리 사용량은 얼마입니까? 
    3. 역방향 계산에 대한 메모리 사용량은 얼마입니까? 
    4. 역전파에 대한 계산 비용은 얼마입니까? 
3. 입력 채널 수 $c_\textrm{i}$와 출력 채널 수 $c_\textrm{o}$를 모두 두 배로 늘리면 계산 횟수는 몇 배로 증가합니까? 패딩을 두 배로 늘리면 어떻게 됩니까? 
4. 이 섹션의 마지막 예제에 있는 변수 `Y1`과 `Y2`는 정확히 같습니까? 그 이유는 무엇입니까? 
5. 합성곱 윈도우가 $1 \times 1$이 아니더라도 합성곱을 행렬 곱셈으로 표현하십시오. 
6. 여러분의 임무는 $k \times k$ 커널로 빠른 합성곱을 구현하는 것입니다. 알고리즘 후보 중 하나는 소스를 수평으로 스캔하여 $k$ 너비 스트립을 읽고 한 번에 하나의 값을 $1$ 너비 출력 스트립으로 계산하는 것입니다. 대안은 $k + \Delta$ 너비 스트립을 읽고 $\Delta$ 너비 출력 스트립을 계산하는 것입니다. 후자가 더 바람직한 이유는 무엇입니까? $\Delta$를 얼마나 크게 선택해야 하는지에 대한 제한이 있습니까? 
7. $c \times c$ 행렬이 있다고 가정합니다. 
    1. 행렬이 $b$개의 블록으로 나뉘어 있는 경우 블록 대각 행렬과 곱하는 것이 얼마나 더 빠릅니까? 
    2. $b$개의 블록을 갖는 것의 단점은 무엇입니까? 어떻게 고칠 수 있습니까(적어도 부분적으로)? 

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/273)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17998)
:end_tab: