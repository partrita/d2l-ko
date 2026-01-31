```{.python .input}
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 어텐션 스코어 함수 (Attention Scoring Functions)
:label:`sec_attention-scoring-functions`


:numref:`sec_attention-pooling`에서 우리는 쿼리와 키 사이의 상호 작용을 모델링하기 위해 가우시안 커널을 포함한 여러 가지 거리 기반 커널을 사용했습니다. 
알고 보니 거리 함수는 내적(dot product)보다 계산 비용이 약간 더 많이 듭니다. 
따라서 주의 가중치가 음수가 되지 않도록 보장하는 소프트맥스 연산과 함께, 
계산이 더 간단한 :eqref:`eq_softmax_attention` 및 :numref:`fig_attention_output`의 *어텐션 스코어 함수(attention scoring functions)* $a$에 많은 노력이 기울여졌습니다. 

![주의 가중 평균으로 어텐션 풀링의 출력을 계산합니다. 여기서 가중치는 어텐션 스코어 함수 $\mathit{a}$와 소프트맥스 연산으로 계산됩니다.](../img/attention-output.svg)
:label:`fig_attention_output`

```{.python .input}
%%tab mxnet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
```

## [**내적 주의 (Dot Product Attention)**] 


가우시안 커널의 주의 함수(지수화 제외)를 잠시 검토해 봅시다:

$$ 
 a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2. 
$$ 

먼저, 마지막 항은 $\mathbf{q}$에만 의존한다는 점에 유의하십시오. 따라서 모든 $(\mathbf{q}, \mathbf{k}_i)$ 쌍에 대해 동일합니다. 
:eqref:`eq_softmax_attention`에서와 같이 주의 가중치를 1로 정규화하면 이 항은 완전히 사라집니다. 
둘째, 배치 및 레이어 정규화(나중에 논의됨)는 모두 잘 제한된, 종종 일정한 노름 $\|\mathbf{k}_i\|$을 갖는 활성화로 이어집니다. 
예를 들어 키 $\mathbf{k}_i$가 레이어 정규화에 의해 생성된 경우입니다. 
따라서 결과에 큰 변화 없이 $a$의 정의에서 이를 생략할 수 있습니다.

마지막으로, 지수 함수의 인수의 크기 정도를 제어해야 합니다. 
쿼리 $\mathbf{q} \in \mathbb{R}^d$와 키 $\mathbf{k}_i \in \mathbb{R}^d$의 모든 요소가 평균이 0이고 분산이 1인 독립적이고 동일하게 추출된 확률 변수라고 가정합니다. 
두 벡터 사이의 내적은 평균이 0이고 분산이 $d$입니다. 
벡터 길이에 관계없이 내적의 분산이 1로 유지되도록 하기 위해, *스케일드 내적 주의(scaled dot product attention)* 스코어 함수를 사용합니다. 
즉, 내적을 $1/\sqrt{d}$로 재조정합니다. 
따라서 우리는 예를 들어 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`에서 사용되는 첫 번째로 흔히 사용되는 주의 함수에 도달합니다:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$ 
:eqlabel:`eq_dot_product_attention`

주의 가중치 $\alpha$는 여전히 정규화가 필요하다는 점에 유의하십시오. 
우리는 소프트맥스 연산을 사용하여 :eqref:`eq_softmax_attention`을 통해 이를 더욱 단순화할 수 있습니다:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$ 
:eqlabel:`eq_attn-scoring-alpha`

알고 보니 모든 인기 있는 주의 메커니즘은 소프트맥스를 사용하므로, 이 장의 나머지 부분에서는 그것으로 제한하겠습니다.

## 편의 함수 (Convenience Functions) 

주의 메커니즘을 효율적으로 배포하기 위해 몇 가지 함수가 필요합니다. 
여기에는 가변 길이 문자열(자연어 처리에 흔함)을 처리하기 위한 도구와 미니배치에 대한 효율적인 평가를 위한 도구(배치 행렬 곱셈)가 포함됩니다.


### [**마스킹된 소프트맥스 연산 (Masked Softmax Operation)**] 

주의 메커니즘의 가장 인기 있는 응용 중 하나는 시퀀스 모델입니다. 따라서 우리는 서로 다른 길이의 시퀀스를 처리할 수 있어야 합니다. 
어떤 경우에는 그러한 시퀀스들이 동일한 미니배치에 포함될 수 있으며, 더 짧은 시퀀스에 대해 더미 토큰으로 패딩해야 합니다(:numref:`sec_machine_translation`의 예 참조). 
이러한 특수 토큰은 의미를 전달하지 않습니다. 예를 들어 다음과 같은 세 문장이 있다고 가정해 봅시다:

```
Dive  into  Deep    Learning 
Learn to    code    <blank>
Hello world <blank> <blank>
```


우리 주의 모델에서 공백을 원하지 않으므로, 단순히 $\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$를 실제 문장 길이인 $l \leq n$까지인 $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$로 제한하면 됩니다. 
이것은 매우 흔한 문제이므로 *마스킹된 소프트맥스 연산(masked softmax operation)*이라는 이름이 있습니다.

구현해 봅시다. 실제로 구현에서는 $i > l$에 대해 $\mathbf{v}_i$의 값을 0으로 설정하여 아주 살짝 속임수를 씁니다. 
게다가 실제로 기울기와 값에 대한 기여를 사라지게 하기 위해 주의 가중치를 $-10^{6}$과 같은 큰 음수로 설정합니다. 
이는 선형 대수 커널과 연산자가 GPU에 고도로 최적화되어 있으며, 조건부(if then else) 문이 있는 코드보다는 계산에서 약간 낭비하는 것이 더 빠르기 때문입니다.

```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """마지막 축의 요소를 마스킹하여 소프트맥스 연산을 수행합니다."""
    # X: 3D 텐서, valid_lens: 1D 또는 2D 텐서
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 마지막 축에서 마스킹된 요소를 매우 큰 음수 값으로 교체합니다.
        # 이 값의 지수화 출력은 0이 됩니다.
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """마지막 축의 요소를 마스킹하여 소프트맥스 연산을 수행합니다."""
    # X: 3D 텐서, valid_lens: 1D 또는 2D 텐서 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 마지막 축에서 마스킹된 요소를 매우 큰 음수 값으로 교체합니다.
        # 이 값의 지수화 출력은 0이 됩니다.
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """마지막 축의 요소를 마스킹하여 소프트맥스 연산을 수행합니다."""
    # X: 3D 텐서, valid_lens: 1D 또는 2D 텐서
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # 마지막 축에서 마스킹된 요소를 매우 큰 음수 값으로 교체합니다.
        # 이 값의 지수화 출력은 0이 됩니다.
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """마지막 축의 요소를 마스킹하여 소프트맥스 연산을 수행합니다."""
    # X: 3D 텐서, valid_lens: 1D 또는 2D 텐서
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 마지막 축에서 마스킹된 요소를 매우 큰 음수 값으로 교체합니다.
        # 이 값의 지수화 출력은 0이 됩니다.
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

[**이 함수가 어떻게 작동하는지 설명**]하기 위해, 
유효 길이가 각각 2와 3인 크기 $2 \times 4$의 두 예제 미니배치를 고려해 보십시오. 
마스킹된 소프트맥스 연산의 결과로, 각 벡터 쌍에 대해 유효 길이를 벗어난 값은 모두 0으로 마스킹됩니다.

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)), jnp.array([2, 3]))
```

모든 예제의 두 벡터 각각에 대해 유효 길이를 지정하기 위해 더 세밀한 제어가 필요한 경우, 유효 길이의 2차원 텐서를 사용하면 됩니다. 이는 다음을 산출합니다:

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### 배치 행렬 곱셈 (Batch Matrix Multiplication)
:label:`subsec_batch_dot`

또 다른 흔히 사용되는 연산은 행렬 배치를 서로 곱하는 것입니다. 이는 쿼리, 키, 값의 미니배치가 있을 때 유용합니다. 더 구체적으로 다음과 같다고 가정합니다.

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}. 
$$

그러면 배치 행렬 곱셈(BMM)은 다음과 같이 요소별 곱을 계산합니다.

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$ 
:eqlabel:`eq_batch-matrix-mul`

딥러닝 프레임워크에서 작동하는 것을 봅시다.

```{.python .input}
%%tab mxnet
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(npx.batch_dot(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab pytorch
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab tensorflow
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(tf.matmul(Q, K).numpy(), (2, 3, 6))
```

```{.python .input}
%%tab jax
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

## [**스케일드 내적 주의 (Scaled Dot Product Attention)**] 

:eqref:`eq_dot_product_attention`에서 소개된 내적 주의로 돌아가 봅시다. 
일반적으로 쿼리와 키가 모두 동일한 벡터 길이 $d$를 가질 것을 요구하지만, 
$\mathbf{q}^\top \mathbf{k}$를 두 공간 사이의 변환을 위해 적절히 선택된 행렬 $\mathbf{M}$에 대해 $\mathbf{q}^\top \mathbf{M} \mathbf{k}$로 대체함으로써 이를 쉽게 해결할 수 있습니다. 지금은 차원이 일치한다고 가정합니다.

실제로 우리는 효율성을 위해 $n$개의 쿼리와 $m$개의 키-값 쌍에 대한 주의를 계산하는 것과 같이 미니배치를 종종 생각합니다. 여기서 쿼리와 키의 길이는 $d$이고 값의 길이는 $v$입니다. 
쿼리 $\mathbf Q\in\mathbb R^{n\times d}$, 키 $\mathbf K\in\mathbb R^{m\times d}$, 값 $\mathbf V\in\mathbb R^{m\times v}$의 스케일드 내적 주의는 다음과 같이 쓸 수 있습니다.

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$ 
:eqlabel:`eq_softmax_QK_V`

이를 미니배치에 적용할 때 :eqref:`eq_batch-matrix-mul`에서 소개한 배치 행렬 곱셈이 필요하다는 점에 유의하십시오. 다음 스케일드 내적 주의 구현에서는 모델 정규화를 위해 드롭아웃을 사용합니다.

```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """스케일드 내적 주의."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # queries의 모양: (batch_size, 쿼리 수, d)
    # keys의 모양: (batch_size, 키-값 쌍의 수, d)
    # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
    # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # keys의 마지막 두 차원을 바꾸기 위해 transpose_b=True로 설정합니다
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """스케일드 내적 주의."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # queries의 모양: (batch_size, 쿼리 수, d)
    # keys의 모양: (batch_size, 키-값 쌍의 수, d)
    # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
    # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # keys.transpose(1, 2)를 사용하여 keys의 마지막 두 차원을 바꿉니다
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """스케일드 내적 주의."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # queries의 모양: (batch_size, 쿼리 수, d)
    # keys의 모양: (batch_size, 키-값 쌍의 수, d)
    # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
    # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class DotProductAttention(nn.Module):  #@save
    """스케일드 내적 주의."""
    dropout: float

    # queries의 모양: (batch_size, 쿼리 수, d)
    # keys의 모양: (batch_size, 키-값 쌍의 수, d)
    # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
    # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # keys.swapaxes(1, 2)를 사용하여 keys의 마지막 두 차원을 바꿉니다
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

[**`DotProductAttention` 클래스가 어떻게 작동하는지 설명**]하기 위해, 
가산 주의(additive attention)에 대한 이전 장난감 예제의 것과 동일한 키, 값, 유효 길이를 사용합니다. 
우리 예제의 목적을 위해 미니배치 크기가 2, 총 10개의 키와 값, 그리고 값의 차원이 4라고 가정합니다. 
마지막으로 관찰당 유효 길이가 각각 2와 6이라고 가정합니다. 
이를 감안할 때 출력이 $2 \times 1 \times 4$ 텐서, 즉 미니배치의 예제당 한 행이 될 것으로 기대합니다.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
keys = tf.random.normal(shape=(2, 10, 2))
values = tf.random.normal(shape=(2, 10, 4))
valid_lens = tf.constant([2, 6])

attention = DotProductAttention(dropout=0.5)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 2))
keys = jax.random.normal(d2l.get_key(), (2, 10, 2))
values = jax.random.normal(d2l.get_key(), (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

주의 가중치가 각각 두 번째와 여섯 번째 열 너머에서 실제로 사라지는지 확인해 봅시다(유효 길이를 2와 6으로 설정했기 때문입니다).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**가산 주의 (Additive Attention)**]
:label:`subsec_additive-attention`

쿼리 $\mathbf{q}$와 키 $\mathbf{k}$가 서로 다른 차원의 벡터인 경우, $\mathbf{q}^\top \mathbf{M} \mathbf{k}$를 통해 불일치를 해결하기 위해 행렬을 사용하거나, 스코어 함수로 가산 주의를 사용할 수 있습니다. 
또 다른 이점은 이름에서 알 수 있듯이 주의가 가산적이라는 것입니다. 이는 약간의 계산 절약으로 이어질 수 있습니다. 
쿼리 $\mathbf{q} \in \mathbb{R}^q$와 키 $\mathbf{k} \in \mathbb{R}^k$가 주어졌을 때, *가산 주의(additive attention)* 스코어 함수 :cite:`Bahdanau.Cho.Bengio.2014`는 다음과 같이 주어집니다.

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$ 
:eqlabel:`eq_additive-attn`

여기서 $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$, $\mathbf w_v\in\mathbb R^{h}$는 학습 가능한 파라미터입니다. 
이 항은 비음수성과 정규화를 모두 보장하기 위해 소프트맥스에 공급됩니다. 
:eqref:`eq_additive-attn`에 대한 동등한 해석은 쿼리와 키가 연결되어 단일 은닉층이 있는 MLP에 공급된다는 것입니다. 
$\tanh$를 활성화 함수로 사용하고 편향 항을 비활성화하여 가산 주의를 다음과 같이 구현합니다:

```{.python .input}
%%tab mxnet
class AdditiveAttention(nn.Block):  #@save
    """가산 주의."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # flatten=False를 사용하여 마지막 축만 변환하여 다른 축의 모양이 동일하게 유지되도록 합니다
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 차원 확장 후, queries의 모양: (batch_size, 쿼리 수, 1, num_hiddens)
        # keys의 모양: (batch_size, 1, 키-값 쌍의 수, num_hiddens)
        # 브로드캐스팅을 사용하여 이들을 더합니다
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # self.w_v의 출력은 하나뿐이므로 모양에서 마지막 1차원 항목을 제거합니다.
        # scores의 모양: (batch_size, 쿼리 수, 키-값 쌍의 수)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class AdditiveAttention(nn.Module):  #@save
    """가산 주의."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 차원 확장 후, queries의 모양: (batch_size, 쿼리 수, 1, num_hiddens)
        # keys의 모양: (batch_size, 1, 키-값 쌍의 수, num_hiddens)
        # 브로드캐스팅을 사용하여 이들을 더합니다
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v의 출력은 하나뿐이므로 모양에서 마지막 1차원 항목을 제거합니다.
        # scores의 모양: (batch_size, 쿼리 수, 키-값 쌍의 수)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """가산 주의."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 차원 확장 후, queries의 모양: (batch_size, 쿼리 수, 1, num_hiddens)
        # keys의 모양: (batch_size, 1, 키-값 쌍의 수, num_hiddens)
        # 브로드캐스팅을 사용하여 이들을 더합니다
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # self.w_v의 출력은 하나뿐이므로 모양에서 마지막 1차원 항목을 제거합니다.
        # scores의 모양: (batch_size, 쿼리 수, 키-값 쌍의 수)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class AdditiveAttention(nn.Module):  #@save
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 차원 확장 후, queries의 모양: (batch_size, 쿼리 수, 1, num_hiddens)
        # keys의 모양: (batch_size, 1, 키-값 쌍의 수, num_hiddens)
        # 브로드캐스팅을 사용하여 이들을 더합니다
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # self.w_v의 출력은 하나뿐이므로 모양에서 마지막 1차원 항목을 제거합니다.
        # scores의 모양: (batch_size, 쿼리 수, 키-값 쌍의 수)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # values의 모양: (batch_size, 키-값 쌍의 수, 값 차원)
        return dropout_layer(attention_weights)@values, attention_weights
```

[**`AdditiveAttention`이 어떻게 작동하는지 살펴봅시다.**] 
장난감 예제에서 쿼리, 키, 값의 크기를 각각 $(2, 1, 20)$, $(2, 10, 2)$, $(2, 10, 4)$로 선택합니다. 
이는 이제 쿼리가 20차원이라는 점을 제외하고는 `DotProductAttention`에 대한 우리의 선택과 동일합니다. 
마찬가지로 미니배치의 시퀀스에 대해 $(2, 6)$을 유효 길이로 선택합니다.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 20))

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

주의 함수를 검토할 때 `DotProductAttention`의 결과와 질적으로 매우 유사한 동작을 보입니다. 
즉, 선택된 유효 길이 $(2, 6)$ 내의 항만 0이 아닙니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 요약 (Summary)

이 섹션에서는 두 가지 핵심 어텐션 스코어 함수인 내적 주의와 가산 주의를 소개했습니다. 
그들은 가변 길이의 시퀀스를 가로질러 집계하기 위한 효과적인 도구입니다. 
특히 내적 주의는 현대 Transformer 아키텍처의 주류입니다. 
쿼리와 키가 서로 다른 길이의 벡터인 경우 대신 가산 주의 스코어 함수를 사용할 수 있습니다. 
이러한 레이어를 최적화하는 것은 최근 몇 년 동안의 핵심 발전 분야 중 하나입니다. 예를 들어 [NVIDIA의 Transformer 라이브러리](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)와 Megatron :cite:`shoeybi2019megatron`은 주의 메커니즘의 효율적인 변형에 결정적으로 의존합니다. 나중에 섹션에서 Transformer를 검토하면서 이에 대해 좀 더 자세히 알아볼 것입니다.

## 연습 문제 (Exercises)

1. `DotProductAttention` 코드를 수정하여 거리 기반 주의(distance-based attention)를 구현하십시오. 효율적인 구현을 위해 키의 제곱 노름 $\|\mathbf{k}_i\|^2$만 필요하다는 점에 유의하십시오. 
2. 차원을 조정하기 위해 행렬을 채택하여 서로 다른 차원의 쿼리와 키를 허용하도록 내적 주의를 수정하십시오. 
3. 계산 비용이 키, 쿼리, 값의 차원 및 수에 따라 어떻게 확장됩니까? 메모리 대역폭 요구 사항은 어떻습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1064)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3867)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18027)
:end_tab: