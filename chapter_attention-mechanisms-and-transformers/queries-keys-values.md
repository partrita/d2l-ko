```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# 쿼리, 키, 값 (Queries, Keys, and Values)
:label:`sec_queries-keys-values`

지금까지 우리가 검토한 모든 네트워크는 입력이 잘 정의된 크기인 것에 결정적으로 의존했습니다. 
예를 들어 ImageNet의 이미지는 $224 \times 224$ 픽셀 크기이며 CNN은 특히 이 크기에 맞춰져 있습니다. 
자연어 처리에서도 RNN의 입력 크기는 잘 정의되어 있고 고정되어 있습니다. 가변 크기는 한 번에 하나의 토큰을 순차적으로 처리하거나 특별히 설계된 합성곱 커널을 통해 처리됩니다 :cite:`Kalchbrenner.Grefenstette.Blunsom.2014`. 
이러한 접근 방식은 :numref:`sec_seq2seq`의 텍스트 변환과 같이 입력이 정보 내용이 변하는 진정한 가변 크기일 때 심각한 문제로 이어질 수 있습니다 :cite:`Sutskever.Vinyals.Le.2014`. 
특히 긴 시퀀스의 경우 이미 생성되었거나 네트워크에서 확인된 모든 것을 추적하기가 상당히 어려워집니다. :citet:`yang2016neural`이 제안한 것과 같은 명시적인 추적 휴리스틱조차도 제한된 이점만 제공합니다.

이를 데이터베이스와 비교해 보십시오. 가장 단순한 형태의 데이터베이스는 키($k$)와 값($v$)의 모음입니다. 
예를 들어, 우리의 데이터베이스 $\mathcal{D}$는 성이 키이고 이름이 값인 \{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")} 튜플로 구성될 수 있습니다. 
우리는 $\mathcal{D}$에 대해 연산을 수행할 수 있습니다. 예를 들어 "Li"에 대한 정확한 쿼리($q$)는 값 "Mu"를 반환할 것입니다. 
("Li", "Mu")가 $\mathcal{D}$의 레코드가 아니라면 유효한 답변이 없을 것입니다. 근사 매칭도 허용한다면 대신 ("Lipton", "Zachary")를 검색할 것입니다. 이 아주 간단하고 사소한 예제는 그럼에도 불구하고 우리에게 몇 가지 유용한 것들을 가르쳐 줍니다:

* 우리는 데이터베이스 크기에 관계없이 유효하도록 ($k$,$v$) 쌍에 대해 작동하는 쿼리 $q$를 설계할 수 있습니다. 
* 동일한 쿼리가 데이터베이스의 내용에 따라 다른 답변을 받을 수 있습니다. 
* 큰 상태 공간(데이터베이스)에서 작동하기 위해 실행되는 "코드"는 상당히 간단할 수 있습니다(예: 정확한 매칭, 근사 매칭, 상위 $k$개).
* 작업을 효과적으로 만들기 위해 데이터베이스를 압축하거나 단순화할 필요가 없습니다.

딥러닝을 설명하기 위한 목적이 아니었다면 분명히 여기서 간단한 데이터베이스를 소개하지 않았을 것입니다. 
실제로 이것은 지난 10년 동안 딥러닝에 도입된 가장 흥미로운 개념 중 하나인 *주의(attention) 메커니즘*으로 이어집니다 :cite:`Bahdanau.Cho.Bengio.2014`. 
기계 번역에 대한 구체적인 응용은 나중에 다룰 것입니다. 지금은 다음을 고려하십시오: $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$를 $m$개의 *키*와 *값* 튜플로 구성된 데이터베이스라고 합시다. 또한 $\mathbf{q}$를 *쿼리*라고 합시다. 그러면 $\mathcal{D}$에 대한 *주의(attention)*를 다음과 같이 정의할 수 있습니다.

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$ 
:eqlabel:`eq_attention_pooling`

여기서 $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ ($i = 1, \ldots, m$)는 스칼라 주의 가중치입니다. 이 연산 자체는 일반적으로 *어텐션 풀링(attention pooling)*이라고 합니다. *주의(attention)*라는 이름은 이 연산이 가중치 $\alpha$가 유의미한(즉, 큰) 항들에 특별한 주의를 기울인다는 사실에서 유래했습니다. 이와 같이 $\mathcal{D}$에 대한 주의는 데이터베이스에 포함된 값들의 선형 결합을 생성합니다. 사실 이것은 단 하나의 가중치만 제외하고 모두 0인 특수한 경우로 위의 예제를 포함합니다. 우리는 몇 가지 특수한 경우를 가집니다:

* 가중치 $\alpha(\mathbf{q}, \mathbf{k}_i)$가 0 이상입니다. 이 경우 주의 메커니즘의 출력은 값 $\mathbf{v}_i$들에 의해 생성된 볼록 원뿔(convex cone)에 포함됩니다. 
* 가중치 $\alpha(\mathbf{q}, \mathbf{k}_i)$가 볼록 조합(convex combination)을 이룹니다. 즉, $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$이고 모든 $i$에 대해 $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$입니다. 이것이 딥러닝에서 가장 일반적인 설정입니다.
* 가중치 중 정확히 하나만 1이고 나머지는 모두 0입니다. 이것은 전통적인 데이터베이스 쿼리와 유사합니다.
* 모든 가중치가 동일합니다. 즉, 모든 $i$에 대해 $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$입니다. 이는 전체 데이터베이스에 걸쳐 평균을 내는 것과 같으며, 딥러닝에서는 평균 풀링(average pooling)이라고도 합니다. 

가중치의 합이 1이 되도록 보장하는 일반적인 전략은 다음을 통해 정규화하는 것입니다.

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{{\sum_j} \alpha(\mathbf{q}, \mathbf{k}_j)}.$$ 

특히 가중치가 0 이상이 되도록 보장하기 위해 지수화(exponentiation)를 사용할 수 있습니다. 즉, 이제 *임의의* 함수 $a(\mathbf{q}, \mathbf{k})$를 선택한 다음 다항 모델에 사용되는 소프트맥스 연산을 다음과 같이 적용할 수 있습니다.

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(a(\mathbf{q}, \mathbf{k}_j))}. $$
:eqlabel:`eq_softmax_attention`

이 연산은 모든 딥러닝 프레임워크에서 즉시 사용할 수 있습니다. 미분 가능하고 기울기가 결코 사라지지 않는데, 이는 모델에서 바람직한 속성입니다. 하지만 위에서 소개한 주의 메커니즘이 유일한 옵션은 아닙니다. 예를 들어 강화 학습 방법을 사용하여 훈련할 수 있는 미분 불가능한 주의 모델을 설계할 수 있습니다 :cite:`Mnih.Heess.Graves.ea.2014`. 예상할 수 있듯이 그러한 모델을 훈련하는 것은 매우 복잡합니다. 결과적으로 현대 주의 연구의 대부분은 :numref:`fig_qkv`에 설명된 프레임워크를 따릅니다. 따라서 우리는 이러한 미분 가능한 메커니즘 패밀리에 설명을 집중합니다. 

![주의 메커니즘은 쿼리 $\mathbf{q}$와 키 $\mathbf{k}_\mathit{i}$ 사이의 호환성에 따라 가중치가 파생되는 어텐션 풀링을 통해 값 $\mathbf{v}_\mathit{i}$에 대한 선형 결합을 계산합니다.](../img/qkv.svg)
:label:`fig_qkv`

상당히 놀라운 점은 키와 값의 집합에 대해 실행되는 실제 "코드", 즉 쿼리가 작동할 공간이 상당히 큼에도 불구하고 매우 간결할 수 있다는 것입니다. 이것은 학습해야 할 파라미터가 너무 많이 필요하지 않기 때문에 네트워크 레이어에 바람직한 속성입니다. 마찬가지로 편리한 점은 어텐션 풀링 연산이 수행되는 방식을 변경할 필요 없이 주의 메커니즘이 임의로 큰 데이터베이스에서 작동할 수 있다는 사실입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=2}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## 시각화 (Visualization)

주의 메커니즘의 장점 중 하나는 특히 가중치가 0 이상이고 합이 1일 때 상당히 직관적일 수 있다는 것입니다. 이 경우 우리는 큰 가중치를 모델이 관련성 있는 구성 요소를 선택하는 방법으로 *해석*할 수 있습니다. 이것은 좋은 직관이지만, 그것이 단지 *직관*일 뿐이라는 것을 기억하는 것이 중요합니다. 그럼에도 불구하고 다양한 쿼리를 적용할 때 주어진 키 세트에 미치는 영향을 시각화하고 싶을 수 있습니다. 이 함수는 나중에 유용하게 사용될 것입니다.

따라서 `show_heatmaps` 함수를 정의합니다. 이 함수는 입력으로 (주의 가중치의) 행렬을 받는 것이 아니라 4개의 축을 가진 텐서를 받아 다양한 쿼리와 가중치 배열을 허용합니다. 결과적으로 입력 `matrices`는 (표시할 행 수, 표시할 열 수, 쿼리 수, 키 수) 모양을 갖습니다. 이는 나중에 Transformer를 설계하는 작동 방식을 시각화하고 싶을 때 유용할 것입니다.

```{.python .input  n=17}
%%tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """행렬의 히트맵을 보여줍니다."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if tab.selected('jax'):
                pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

간단한 정상성 확인으로, 쿼리와 키가 동일할 때만 주의 가중치가 1인 경우를 나타내는 단위 행렬을 시각화해 보겠습니다.

```{.python .input  n=20}
%%tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

## 요약 (Summary)

주의 메커니즘을 사용하면 많은 (키, 값) 쌍에서 데이터를 집계할 수 있습니다. 지금까지 우리의 논의는 데이터를 풀링하는 방법을 설명하는 꽤 추상적인 것이었습니다. 우리는 아직 그 신비로운 쿼리, 키, 값이 어디에서 발생할 수 있는지 설명하지 않았습니다. 여기서 약간의 직관이 도움이 될 수 있습니다: 예를 들어, 회귀 설정에서 쿼리는 회귀가 수행되어야 할 위치에 해당할 수 있습니다. 키는 과거 데이터가 관찰된 위치이고 값은 (회귀) 값 자체입니다. 이것은 우리가 다음 섹션에서 공부할 소위 나다라야-왓슨(Nadaraya--Watson) 추정기 :cite:`Nadaraya.1964,Watson.1964`입니다. 

설계상 주의 메커니즘은 신경망이 집합에서 요소를 선택하고 표현들에 대해 관련 가중 합을 구성할 수 있게 하는 *미분 가능한* 제어 수단을 제공합니다.

## 연습 문제 (Exercises)

1. 클래식 데이터베이스에서 사용되는 근사 (키, 쿼리) 매칭을 다시 구현하고 싶다면 어떤 주의 함수를 선택하시겠습니까? 
2. 주의 함수가 $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$로 주어지고 $i = 1, \ldots, m$에 대해 $\mathbf{k}_i = \mathbf{v}_i$라고 가정합시다. :eqref:`eq_softmax_attention`의 소프트맥스 정규화를 사용할 때 키들에 대한 확률 분포를 $p(\mathbf{k}_i; \mathbf{q})$로 표시합시다. $\nabla_{\mathbf{q}} \mathop{\textrm{Attention}}(\mathbf{q}, \mathcal{D}) = \textrm{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$임을 증명하십시오.
3. 주의 메커니즘을 사용하여 미분 가능한 검색 엔진을 설계하십시오. 
4. Squeeze and Excitation Networks :cite:`Hu.Shen.Sun.2018`의 설계를 검토하고 주의 메커니즘의 렌즈를 통해 해석해 보십시오. 

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1710)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18024)
:end_tab:

```