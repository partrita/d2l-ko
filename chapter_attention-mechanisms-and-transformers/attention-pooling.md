# 유사성에 의한 어텐션 풀링 (Attention Pooling by Similarity)

:label:`sec_attention-pooling`

이제 주의 메커니즘의 주요 구성 요소를 소개했으므로, 이를 다소 고전적인 설정, 즉 커널 밀도 추정을 통한 회귀 및 분류에서 사용해 봅시다 :cite:`Nadaraya.1964,Watson.1964`. 이 우회로는 단순히 추가 배경을 제공합니다: 전적으로 선택 사항이며 필요한 경우 건너뛸 수 있습니다. 
핵심적으로 나다라야-왓슨(Nadaraya--Watson) 추정기는 쿼리 $\mathbf{q}$와 키 $\mathbf{k}$를 연관시키는 유사성 커널 $\alpha(\mathbf{q}, \mathbf{k})$에 의존합니다. 일반적인 커널들은 다음과 같습니다.

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{가우시안;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{박스카(Boxcar);} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{에파네치니코프(Epanechikov).}`
$$`

우리가 선택할 수 있는 더 많은 선택지가 있습니다. 더 광범위한 검토와 커널 선택이 때때로 *파젠 윈도우(Parzen Windows)*라고도 불리는 커널 밀도 추정 :cite:`parzen1957consistent`과 어떻게 관련되는지는 [위키피디아 문서](https://en.wikipedia.org/wiki/Kernel_(statistics))를 참조하십시오. 모든 커널은 휴리스틱하며 튜닝될 수 있습니다. 예를 들어 글로벌 기준뿐만 아니라 좌표별 기준으로 너비를 조정할 수 있습니다. 어쨌든 이들 모두는 회귀와 분류 모두에 대해 다음과 같은 방정식으로 이어집니다:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$`

특성과 레이블에 대한 관찰 $(\mathbf{x}_i, y_i)$이 있는 (스칼라) 회귀의 경우, $\mathbf{v}_i = y_i$는 스칼라, $\mathbf{k}_i = \mathbf{x}_i$는 벡터이며, 쿼리 $\mathbf{q}$는 $f$가 평가되어야 할 새로운 위치를 나타냅니다. (다중 클래스) 분류의 경우 $y_i$의 원-핫 인코딩을 사용하여 $\mathbf{v}_i$를 얻습니다. 이 추정기의 편리한 속성 중 하나는 훈련이 필요 없다는 것입니다. 더욱이 데이터 양이 증가함에 따라 커널을 적절히 좁히면 이 접근 방식은 일관성이 있습니다 :cite:`mack1982weak`. 즉, 통계적으로 최적인 어떤 솔루션으로 수렴할 것입니다. 몇 가지 커널을 검사하는 것으로 시작해 봅시다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**커널과 데이터 (Kernels and Data)**]

이 섹션에서 정의된 모든 커널 $\alpha(\mathbf{k}, \mathbf{q})$는 *평행 이동 및 회전 불변*입니다. 즉, $\mathbf{k}$와 $\mathbf{q}$를 같은 방식으로 이동하고 회전해도 $\alpha$의 값은 변하지 않습니다. 단순함을 위해 우리는 스칼라 인수 $k, q \in \mathbb{R}$을 선택하고 키 $k = 0$을 원점으로 잡습니다. 이는 다음을 산출합니다:

```{.python .input}
%%tab all
# 몇 가지 커널 정의
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

서로 다른 커널은 범위와 매끄러움에 대한 서로 다른 개념에 해당합니다. 예를 들어 박스카 커널은 거리 1(또는 달리 정의된 하이퍼파라미터) 이내의 관찰에만 무차별적으로 주의를 기울입니다. 

나다라야-왓슨 추정이 작동하는 것을 보기 위해 몇 가지 훈련 데이터를 정의해 봅시다. 다음에서는 다음과 같은 종속성을 사용합니다.

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$`

여기서 $\epsilon$은 평균 0과 단위 분산을 갖는 정규 분포에서 추출됩니다. 40개의 훈련 예제를 추출합니다.

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**나다라야-왓슨 회귀를 통한 어텐션 풀링 (Attention Pooling via Nadaraya--Watson Regression)**]

데이터와 커널이 준비되었으므로 필요한 것은 커널 회귀 추정치를 계산하는 함수뿐입니다. 우리는 또한 약간의 진단을 수행하기 위해 상대적인 커널 가중치를 얻고 싶습니다. 따라서 먼저 모든 훈련 특성(공변량) `x_train`과 모든 검증 특성 `x_val` 사이의 커널을 계산합니다. 이는 행렬을 산출하며, 이를 나중에 정규화합니다. 훈련 레이블 `y_train`과 곱하면 추정치를 얻습니다.

:eqref:`eq_attention_pooling`의 어텐션 풀링을 상기해 보십시오. 각 검증 특성을 쿼리로 하고, 각 훈련 특성-레이블 쌍을 키-값 쌍으로 합시다. 결과적으로 정규화된 상대 커널 가중치(아래의 `attention_w`)가 *주의 가중치(attention weights)*가 됩니다.

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # 각 열/행은 각 쿼리/키에 해당합니다
    k = d2l.astype(kernel(dists), d2l.float32)
    # 각 쿼리에 대한 키들에 대한 정규화
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

서로 다른 커널이 생성하는 추정치의 종류를 살펴봅시다.

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

가장 눈에 띄는 것은 세 가지 사소하지 않은 커널(Gaussian, Boxcar, Epanechikov) 모두가 실제 함수에서 그리 멀지 않은 상당히 실행 가능한 추정치를 생성한다는 것입니다. 사소한 추정치 $f(x) = \frac{1}{n} \sum_i y_i$로 이어지는 상수(constant) 커널만이 다소 비현실적인 결과를 생성합니다. 주의 가중치를 좀 더 자세히 살펴봅시다:

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

시각화는 Gaussian, Boxcar, Epanechikov에 대한 추정치가 왜 매우 유사한지 명확하게 보여줍니다: 커널의 함수 형태가 다름에도 불구하고 매우 유사한 주의 가중치로부터 도출되기 때문입니다. 이것이 항상 그런 것인지에 대한 의문이 생깁니다.

## [**어텐션 풀링 조정하기 (Adapting Attention Pooling)**]

가우시안 커널을 다른 너비의 커널로 교체할 수 있습니다. 즉, $\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$를 사용할 수 있습니다. 여기서 $\sigma^2$은 커널의 너비를 결정합니다. 이것이 결과에 영향을 미치는지 봅시다.

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

명백히 커널이 좁을수록 추정치가 덜 매끄럽습니다. 동시에 국소적인 변화에 더 잘 적응합니다. 해당 주의 가중치를 살펴봅시다.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

예상대로 커널이 좁을수록 큰 주의 가중치의 범위가 좁아집니다. 또한 동일한 너비를 선택하는 것이 이상적이지 않을 수 있음이 분명합니다. 실제로 :citet:`Silverman86`은 국소 밀도에 의존하는 휴리스틱을 제안했습니다. 더 많은 그러한 "트릭"들이 제안되었습니다. 예를 들어 :citet:`norelli2022asif`는 교차 모달 이미지 및 텍스트 표현을 설계하기 위해 유사한 최근접 이웃 보간 기술을 사용했습니다.

예리한 독자는 왜 우리가 반세기 이상 된 방법에 대해 이렇게 깊이 있게 파고드는지 의아해할 수 있습니다. 첫째, 그것은 현대 주의 메커니즘의 가장 초기 전구체 중 하나입니다. 둘째, 시각화에 좋습니다. 셋째, 그리고 마찬가지로 중요한 것은 수작업으로 만든 주의 메커니즘의 한계를 보여줍니다. 훨씬 더 나은 전략은 쿼리와 키에 대한 표현을 학습함으로써 메커니즘을 *학습*하는 것입니다. 이것이 우리가 다음 섹션에서 착수할 작업입니다.


## 요약 (Summary)

나다라야-왓슨 커널 회귀는 현재 주의 메커니즘의 초기 전구체입니다. 
분류 또는 회귀를 위해 훈련이나 튜닝 없이 직접 사용할 수 있습니다. 
주의 가중치는 쿼리와 키 사이의 유사성(또는 거리)에 따라, 그리고 얼마나 많은 유사한 관찰이 가능한지에 따라 할당됩니다.

## 연습 문제 (Exercises)

1. 파젠 윈도우(Parzen windows) 밀도 추정치는 $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$로 주어집니다. 이진 분류의 경우 파젠 윈도우에 의해 얻은 함수 $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$이 나다라야-왓슨 분류와 동일함을 증명하십시오. 
2. 나다라야-왓슨 회귀에서 커널 너비에 대한 좋은 값을 학습하기 위해 확률적 경사 하강법을 구현하십시오. 
    1. $(f(\mathbf{x_i}) - y_i)^2$를 직접 최소화하기 위해 위의 추정치를 사용하면 어떤 일이 발생합니까? 힌트: $y_i$는 $f$를 계산하는 데 사용되는 항의 일부입니다.
    2. $f(\mathbf{x}_i)$ 추정치에서 $(\mathbf{x}_i, y_i)$를 제거하고 커널 너비에 대해 최적화하십시오. 여전히 과대적합이 관찰됩니까?
3. 모든 $\mathbf{x}$가 단위 구 위에 있다고 가정합시다. 즉, 모두 $\|\mathbf{x}\| = 1$을 만족합니다. 지수 내의 $\|\mathbf{x} - \mathbf{x}_i\|^2$ 항을 단순화할 수 있습니까? 힌트: 나중에 이것이 내적 주의(dot product attention)와 매우 밀접한 관련이 있음을 보게 될 것입니다. 
4. :citet:`mack1982weak`가 나다라야-왓슨 추정이 일관됨을 증명했음을 상기하십시오. 데이터를 더 많이 얻을수록 주의 메커니즘의 스케일을 얼마나 빨리 줄여야 합니까? 답변에 대한 직관을 제공하십시오. 데이터의 차원성에 의존합니까? 어떻게 그렇습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18026)
:end_tab:
