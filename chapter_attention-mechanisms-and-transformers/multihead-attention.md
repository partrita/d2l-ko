```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 멀티 헤드 어텐션 (Multi-Head Attention)
:label:`sec_multihead-attention`


실제로 동일한 쿼리, 키, 값 세트가 주어졌을 때, 
우리는 모델이 시퀀스 내의 다양한 범위(예: 짧은 범위 대 긴 범위)의 의존성을 캡처하는 것과 같이 동일한 주의 메커니즘의 서로 다른 행동들로부터 얻은 지식을 결합하기를 원할 수 있습니다. 
따라서 우리의 주의 메커니즘이 쿼리, 키, 값의 서로 다른 표현 하위 공간을 공동으로 사용하도록 허용하는 것이 유익할 수 있습니다. 


이를 위해 단일 어텐션 풀링을 수행하는 대신, 
쿼리, 키, 값을 $h$개의 독립적으로 학습된 선형 투영(linear projection)으로 변환할 수 있습니다. 
그런 다음 이 $h$개의 투영된 쿼리, 키, 값이 병렬로 어텐션 풀링에 공급됩니다. 
마지막으로 $h$개의 어텐션 풀링 출력이 연결(concatenate)되고 
다른 학습된 선형 투영으로 변환되어 최종 출력을 생성합니다. 
이 설계를 *멀티 헤드 어텐션(multi-head attention)*이라고 하며, $h$개의 어텐션 풀링 출력 각각을 *헤드(head)*라고 합니다 :cite:`Vaswani.Shazeer.Parmar.ea.2017`. 
학습 가능한 선형 변환을 수행하기 위해 완전 연결 레이어를 사용하는 멀티 헤드 어텐션을 :numref:`fig_multi-head-attention`에서 설명합니다.

![여러 헤드가 연결된 다음 선형 변환되는 멀티 헤드 어텐션.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
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
```

## 모델

멀티 헤드 어텐션의 구현을 제공하기 전에 이 모델을 수학적으로 공식화해 봅시다. 
쿼리 $\mathbf q \in \mathbb R^{d_q}$, 키 $\mathbf k \in \mathbb R^{d_k}$, 값 $\mathbf v \in \mathbb R^{d_v}$가 주어지면, 
각 어텐션 헤드 $\mathbf h_i$ ($i = 1, \ldots, h$)는 다음과 같이 계산됩니다.

$$\mathbf h_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$ 

여기서 
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$,
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$,
$\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$
는 학습 가능한 파라미터이고 
$f$는 :numref:`sec_attention-scoring-functions`의 가산 주의 및 스케일드 내적 주의와 같은 어텐션 풀링입니다. 
멀티 헤드 어텐션 출력은 $h$개 헤드의 연결에 대한 학습 가능한 파라미터 $\mathbf W_o\in\mathbb R^{p_o\times h p_v}$를 통한 또 다른 선형 변환입니다:

$$\mathbf W_o \begin{bmatrix}\mathbf h_1
\vdots
\\\mathbf h_h
\end{bmatrix} \in \mathbb{R}^{p_o}.$$ 

이 설계를 바탕으로 각 헤드는 입력의 서로 다른 부분에 주의를 기울일 수 있습니다. 
단순한 가중 평균보다 더 정교한 함수를 표현할 수 있습니다.

## 구현 (Implementation)

우리 구현에서는 멀티 헤드 어텐션의 [**각 헤드에 대해 스케일드 내적 주의를 선택**]합니다. 
계산 비용과 파라미터화 비용의 급격한 증가를 피하기 위해 $p_q = p_k = p_v = p_o / h$로 설정합니다. 
쿼리, 키, 값에 대한 선형 변환의 출력 수를 $p_q h = p_k h = p_v h = p_o$로 설정하면 $h$개의 헤드를 병렬로 계산할 수 있음에 유의하십시오. 
다음 구현에서 $p_o$는 `num_hiddens` 인수를 통해 지정됩니다.

```{.python .input}
%%tab mxnet
class MultiHeadAttention(d2l.Module):  #@save
    """멀티 헤드 어텐션."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values의 모양: 
        # (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens)
        # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
        # 전치 후 출력 queries, keys, values의 모양: 
        # (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, 
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # axis 0에서 첫 번째 항목(스칼라 또는 벡터)을 num_heads번 복사하고, 
            # 그다음 항목을 복사하는 식으로 진행합니다
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # 출력 모양: (batch_size * num_heads, 쿼리 수, 
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # output_concat의 모양: (batch_size, 쿼리 수, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab pytorch
class MultiHeadAttention(d2l.Module):  #@save
    """멀티 헤드 어텐션."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values의 모양: 
        # (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens)
        # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
        # 전치 후 출력 queries, keys, values의 모양: 
        # (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, 
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # axis 0에서 첫 번째 항목(스칼라 또는 벡터)을 num_heads번 복사하고, 
            # 그다음 항목을 복사하는 식으로 진행합니다
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 출력 모양: (batch_size * num_heads, 쿼리 수, 
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat의 모양: (batch_size, 쿼리 수, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab tensorflow
class MultiHeadAttention(d2l.Module):  #@save
    """멀티 헤드 어텐션."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # queries, keys, values의 모양: 
        # (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens)
        # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
        # 전치 후 출력 queries, keys, values의 모양: 
        # (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, 
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            # axis 0에서 첫 번째 항목(스칼라 또는 벡터)을 num_heads번 복사하고, 
            # 그다음 항목을 복사하는 식으로 진행합니다
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # 출력 모양: (batch_size * num_heads, 쿼리 수, 
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        
        # output_concat의 모양: (batch_size, 쿼리 수, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab jax
class MultiHeadAttention(nn.Module):  #@save
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool = False

    def setup(self):
        self.attention = d2l.DotProductAttention(self.dropout)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        # queries, keys, values의 모양: 
        # (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens)
        # valid_lens의 모양: (batch_size,) 또는 (batch_size, 쿼리 수)
        # 전치 후 출력 queries, keys, values의 모양: 
        # (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, 
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # axis 0에서 첫 번째 항목(스칼라 또는 벡터)을 num_heads번 복사하고, 
            # 그다음 항목을 복사하는 식으로 진행합니다
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        # 출력 모양: (batch_size * num_heads, 쿼리 수, 
        # num_hiddens / num_heads)
        output, attention_weights = self.attention(
            queries, keys, values, valid_lens, training=training)
        # output_concat의 모양: (batch_size, 쿼리 수, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights
```

[**여러 헤드의 병렬 계산**]을 허용하기 위해 위의 `MultiHeadAttention` 클래스는 아래에 정의된 두 가지 전치(transposition) 메서드를 사용합니다. 
구체적으로 `transpose_output` 메서드는 `transpose_qkv` 메서드의 연산을 반전시킵니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """여러 어텐션 헤드의 병렬 계산을 위한 전치."""
    # 입력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens).
    # 출력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # 출력 X의 모양: (batch_size, num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)
    # 출력 모양: (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """transpose_qkv 연산을 반전시킵니다."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """여러 어텐션 헤드의 병렬 계산을 위한 전치."""
    # 입력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens).
    # 출력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # 출력 X의 모양: (batch_size, num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # 출력 모양: (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """transpose_qkv 연산을 반전시킵니다."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """여러 어텐션 헤드의 병렬 계산을 위한 전치."""
    # 입력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens).
    # 출력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_heads, num_hiddens / num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
    # 출력 X의 모양: (batch_size, num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    # 출력 모양: (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """transpose_qkv 연산을 반전시킵니다."""
    X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """여러 어텐션 헤드의 병렬 계산을 위한 전치."""
    # 입력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_hiddens).
    # 출력 X의 모양: (batch_size, 쿼리 또는 키-값 쌍의 수, num_heads, num_hiddens / num_heads)
    X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
    # 출력 X의 모양: (batch_size, num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    X = jnp.transpose(X, (0, 2, 1, 3))
    # 출력 모양: (batch_size * num_heads, 쿼리 또는 키-값 쌍의 수, num_hiddens / num_heads)
    return X.reshape((-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """transpose_qkv 연산을 반전시킵니다."""
    X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
    X = jnp.transpose(X, (0, 2, 1, 3))
    return X.reshape((X.shape[0], X.shape[1], -1))
```

키와 값이 동일한 장난감 예제를 사용하여 [**구현된**] `MultiHeadAttention` 클래스를 [**테스트해 봅시다.**] 
결과적으로 멀티 헤드 어텐션 출력의 모양은 (`batch_size`, `num_queries`, `num_hiddens`)입니다.

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, Y, Y, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## 요약 (Summary)

멀티 헤드 어텐션은 쿼리, 키, 값의 서로 다른 표현 하위 공간을 통해 동일한 어텐션 풀링의 지식을 결합합니다. 
멀티 헤드 어텐션의 여러 헤드를 병렬로 계산하려면 적절한 텐서 조작이 필요합니다. 


## 연습 문제 (Exercises)

1. 이 실험에서 여러 헤드의 주의 가중치를 시각화하십시오.
2. 멀티 헤드 어텐션 기반의 훈련된 모델이 있고, 예측 속도를 높이기 위해 덜 중요한 어텐션 헤드를 프루닝(pruning)하고 싶다고 가정해 봅시다. 어텐션 헤드의 중요도를 측정하기 위해 실험을 어떻게 설계할 수 있을까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1635)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3869)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18029)
:end_tab:

```