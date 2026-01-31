```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 트랜스포머 아키텍처 (The Transformer Architecture)
:label:`sec_transformer`

우리는 :numref:`subsec_cnn-rnn-self-attention`에서 CNN, RNN, 셀프 어텐션을 비교했습니다. 특히 셀프 어텐션은 병렬 계산과 가장 짧은 최대 경로 길이(maximum path length)라는 장점을 모두 가지고 있습니다. 따라서 셀프 어텐션을 사용하여 깊은 아키텍처를 설계하는 것은 매우 매력적입니다.

입력 표현을 위해 여전히 RNN에 의존했던 초기 셀프 어텐션 모델(:cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017`)과 달리, 트랜스포머(Transformer) 모델은 합성곱이나 순환 레이어 없이 오로지 어텐션 메커니즘에만 기반합니다(:cite:`Vaswani.Shazeer.Parmar.ea.2017`). 원래는 텍스트 데이터의 시퀀스 투 시퀀스(sequence-to-sequence) 학습을 위해 제안되었지만, 트랜스포머는 언어, 비전, 음성, 강화 학습 등 현대 딥러닝 응용 분야 전반에 걸쳐 널리 사용되고 있습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
import pandas as pd
```

## 모델 (Model)

인코더-디코더 아키텍처의 한 사례로서, 트랜스포머의 전체 아키텍처는 :numref:`fig_transformer`에 나와 있습니다. 보시는 바와 같이 트랜스포머는 인코더와 디코더로 구성됩니다. :numref:`fig_s2s_attention_details`의 시퀀스 투 시퀀스 학습을 위한 바다나우(Bahdanau) 어텐션과 대조적으로, 입력(소스) 및 출력(타겟) 시퀀스 임베딩은 셀프 어텐션 기반의 모듈을 쌓아 올린 인코더와 디코더에 공급되기 전에 포지셔널 인코딩(positional encoding)과 더해집니다.

![트랜스포머 아키텍처.](../img/transformer.svg)
:width:`320px`
:label:`fig_transformer` 

이제 :numref:`fig_transformer`에 있는 트랜스포머 아키텍처의 개요를 제공합니다. 높은 수준에서 볼 때, 트랜스포머 인코더는 여러 개의 동일한 레이어가 쌓인 형태이며, 각 레이어에는 두 개의 서브레이어($\textrm{sublayer}$라고 함)가 있습니다. 첫 번째는 멀티 헤드 셀프 어텐션 풀링이고, 두 번째는 포지션 와이즈(positionwise) 피드포워드 네트워크입니다. 구체적으로 인코더 셀프 어텐션에서 쿼리, 키, 값은 모두 이전 인코더 레이어의 출력에서 옵니다. :numref:`sec_resnet`의 ResNet 설계에서 영감을 받아, 두 서브레이어 주위에는 잔차 연결(residual connection)이 사용됩니다. 트랜스포머에서는 시퀀스의 모든 위치에 있는 임의의 입력 $\mathbf{x} \in \mathbb{R}^d$에 대해 $\textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$여야 잔차 연결 $\mathbf{x} + \textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$가 가능합니다. 잔차 연결을 통한 이 덧셈 직후에는 레이어 정규화(layer normalization)가 수행됩니다(:cite:`Ba.Kiros.Hinton.2016`). 결과적으로 트랜스포머 인코더는 입력 시퀀스의 각 위치에 대해 $d$차원 벡터 표현을 출력합니다.

트랜스포머 디코더 또한 잔차 연결과 레이어 정규화가 포함된 여러 개의 동일한 레이어 스택입니다. 인코더에서 설명한 두 서브레이어 외에도, 디코더는 이 두 레이어 사이에 인코더-디코더 어텐션이라고 알려진 세 번째 서브레이어를 삽입합니다. 인코더-디코더 어텐션에서 쿼리는 디코더의 셀프 어텐션 서브레이어 출력에서 오고, 키와 값은 트랜스포머 인코더의 출력에서 옵니다. 디코더 셀프 어텐션에서 쿼리, 키, 값은 모두 이전 디코더 레이어의 출력에서 옵니다. 그러나 디코더의 각 위치는 해당 위치까지의 디코더 내 모든 위치에만 어텐션을 수행할 수 있습니다. 이 *마스크된(masked)* 어텐션은 자기 회귀(autoregressive) 속성을 유지하여, 예측이 이미 생성된 출력 토큰에만 의존하도록 보장합니다.

우리는 이미 :numref:`sec_multihead-attention`에서 스케일드 닷 프로덕트(scaled dot product) 기반의 멀티 헤드 어텐션을, :numref:`subsec_positional-encoding`에서 포지셔널 인코딩을 설명하고 구현했습니다. 다음에서는 트랜스포머 모델의 나머지 부분을 구현할 것입니다.


## 포지션 와이즈 피드포워드 네트워크 (Positionwise Feed-Forward Networks)
:label:`subsec_positionwise-ffn`

포지션 와이즈 피드포워드 네트워크는 동일한 MLP를 사용하여 모든 시퀀스 위치에서의 표현을 변환합니다. 이것이 우리가 이를 *포지션 와이즈*라고 부르는 이유입니다. 아래 구현에서 (배치 크기, 타임스텝 수 또는 토큰 단위 시퀀스 길이, 은닉 유닛 수 또는 특성 차원) 형태의 입력 `X`는 2개 레이어의 MLP에 의해 (배치 크리, 타임스텝 수, `ffn_num_outputs`) 형태의 출력 텐서로 변환됩니다.

```{.python .input}
%%tab mxnet
class PositionWiseFFN(nn.Block):  #@save
    """포지션 와이즈 피드포워드 네트워크."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
%%tab pytorch
class PositionWiseFFN(nn.Module):  #@save
    """포지션 와이즈 피드포워드 네트워크."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab tensorflow
class PositionWiseFFN(tf.keras.layers.Layer):  #@save
    """포지션 와이즈 피드포워드 네트워크."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab jax
class PositionWiseFFN(nn.Module):  #@save
    """포지션 와이즈 피드포워드 네트워크."""
    ffn_num_hiddens: int
    ffn_num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.ffn_num_hiddens)
        self.dense2 = nn.Dense(self.ffn_num_outputs)

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))
```

다음 예제는 [**텐서의 가장 안쪽 차원이**] 포지션 와이즈 피드포워드 네트워크의 출력 수로 변경됨을 보여줍니다. 동일한 MLP가 모든 위치에서 변환을 수행하므로, 이러한 모든 위치에서의 입력이 같으면 출력 또한 동일합니다.

```{.python .input}
%%tab mxnet
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab pytorch
ffn = PositionWiseFFN(4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab jax
ffn = PositionWiseFFN(4, 8)
ffn.init_with_output(d2l.get_key(), jnp.ones((2, 3, 4)))[0][0]
```

## 잔차 연결 및 레이어 정규화 (Residual Connection and Layer Normalization)

이제 :numref:`fig_transformer`의 "add & norm" 구성 요소에 집중해 봅시다. 이 섹션의 시작 부분에서 설명했듯이, 이는 잔차 연결 직후에 레이어 정규화가 뒤따르는 구조입니다. 두 가지 모두 효과적인 깊은 아키텍처의 핵심입니다.

:numref:`sec_batch_norm`에서 우리는 배치 정규화(batch normalization)가 미니배치 내의 예제들에 걸쳐 어떻게 중심을 재조정하고 스케일을 조정하는지 설명했습니다. :numref:`subsec_layer-normalization-in-bn`에서 논의했듯이, 레이어 정규화는 특성 차원에 걸쳐 정규화한다는 점을 제외하면 배치 정규화와 동일하며, 따라서 스케일 독립성과 배치 크기 독립성의 이점을 누릴 수 있습니다. 컴퓨터 비전에서의 널리 퍼진 응용에도 불구하고, 배치 정규화는 입력이 종종 가변 길이 시퀀스인 자연어 처리 작업에서 레이어 정규화보다 실무적으로 덜 효과적인 경우가 많습니다.

다음 코드 스니펫은 [**레이어 정규화와 배치 정규화에 의한 서로 다른 차원에 걸친 정규화를 비교합니다.**]

```{.python .input}
%%tab mxnet
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# 훈련 모드에서 X로부터 평균과 분산을 계산합니다.
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab pytorch
ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d()
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# 훈련 모드에서 X로부터 평균과 분산을 계산합니다.
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X, training=True))
```

```{.python .input}
%%tab jax
ln = nn.LayerNorm()
bn = nn.BatchNorm()
X = d2l.tensor([[1, 2], [2, 3]], dtype=d2l.float32)
# 훈련 모드에서 X로부터 평균과 분산을 계산합니다.
print('layer norm:', ln.init_with_output(d2l.get_key(), X)[0],
      '\nbatch norm:', bn.init_with_output(d2l.get_key(), X,
                                           use_running_average=False)[0])
```

이제 [**잔차 연결과 그 뒤를 잇는 레이어 정규화를 사용하여**] `AddNorm` 클래스를 구현할 수 있습니다. 정규화를 위해 드롭아웃(dropout)도 적용됩니다.

```{.python .input}
%%tab mxnet
class AddNorm(nn.Block):  #@save
    """잔차 연결과 그 뒤를 잇는 레이어 정규화."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab pytorch
class AddNorm(nn.Module):  #@save
    """잔차 연결과 그 뒤를 잇는 레이어 정규화."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab tensorflow
class AddNorm(tf.keras.layers.Layer):  #@save
    """잔차 연결과 그 뒤를 잇는 레이어 정규화."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

```{.python .input}
%%tab jax
class AddNorm(nn.Module):  #@save
    """잔차 연결과 그 뒤를 잇는 레이어 정규화."""
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
```

잔차 연결은 두 입력의 모양이 같아야 하므로, [**덧셈 연산 후의 출력 텐서도 같은 모양을 가집니다.**]

```{.python .input}
%%tab mxnet
add_norm = AddNorm(0.5)
add_norm.initialize()
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab pytorch
add_norm = AddNorm(4, 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab tensorflow
# Normalized_shape은: [i for i in range(len(input.shape))][1:] 입니다.
add_norm = AddNorm([1, 2], 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(tf.ones(shape), tf.ones(shape), training=False),
                shape)
```

```{.python .input}
%%tab jax
add_norm = AddNorm(0.5)
shape = (2, 3, 4)
output, _ = add_norm.init_with_output(d2l.get_key(), d2l.ones(shape),
                                      d2l.ones(shape))
d2l.check_shape(output, shape)
```

## 인코더 (Encoder)
:label:`subsec_transformer-encoder`

트랜스포머 인코더를 조립하기 위한 모든 필수 구성 요소가 준비되었으므로, [**인코더 내의 단일 레이어**]를 구현하는 것부터 시작하겠습니다. 다음 `TransformerEncoderBlock` 클래스는 두 개의 서브레이어, 즉 멀티 헤드 셀프 어텐션과 포지션 와이즈 피드포워드 네트워크를 포함하며, 두 서브레이어 주위에는 잔차 연결과 그 뒤를 잇는 레이어 정규화가 사용됩니다.

```{.python .input}
%%tab mxnet
class TransformerEncoderBlock(nn.Block):  #@save
    """트랜스포머 인코더 블록."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab pytorch
class TransformerEncoderBlock(nn.Module):  #@save
    """트랜스포머 인코더 블록."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab tensorflow
class TransformerEncoderBlock(tf.keras.layers.Layer):  #@save
    """트랜스포머 인코더 블록."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

```{.python .input}
%%tab jax
class TransformerEncoderBlock(nn.Module):  #@save
    """트랜스포머 인코더 블록."""
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.addnorm1 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm2 = AddNorm(self.dropout)

    def __call__(self, X, valid_lens, training=False):
        output, attention_weights = self.attention(X, X, X, valid_lens,
                                                   training=training)
        Y = self.addnorm1(X, output, training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training), attention_weights
```

보시는 바와 같이, [**트랜스포머 인코더의 어떤 레이어도 입력의 모양을 변경하지 않습니다.**]

```{.python .input}
%%tab mxnet
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
d2l.check_shape(encoder_blk(X, valid_lens, training=False), X.shape)
```

```{.python .input}
%%tab jax
X = jnp.ones((2, 100, 24))
valid_lens = jnp.array([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
(output, _), _ = encoder_blk.init_with_output(d2l.get_key(), X, valid_lens,
                                              training=False)
d2l.check_shape(output, X.shape)
```

다음 [**트랜스포머 인코더**] 구현에서, 우리는 위에서 만든 `TransformerEncoderBlock` 클래스의 인스턴스를 `num_blks`개만큼 쌓습니다. 값이 항상 -1과 1 사이인 고정된 포지셔널 인코딩을 사용하므로, 입력 임베딩과 포지셔널 인코딩을 합치기 전에 스케일을 맞추기 위해 학습 가능한 입력 임베딩 값에 임베딩 차원의 제곱근을 곱합니다.

```{.python .input}
%%tab mxnet
class TransformerEncoder(d2l.Encoder):  #@save
    """트랜스포머 인코더."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
        self.initialize()

    def forward(self, X, valid_lens):
        # 포지셔널 인코딩 값이 -1과 1 사이이므로, 임베딩 값들을 합치기 전에
        # 스케일을 맞추기 위해 임베딩 차원의 제곱근을 곱합니다.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab pytorch
class TransformerEncoder(d2l.Encoder):  #@save
    """트랜스포머 인코더."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # 포지셔널 인코딩 값이 -1과 1 사이이므로, 임베딩 값들을 합치기 전에
        # 스케일을 맞추기 위해 임베딩 차원의 제곱근을 곱합니다.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab tensorflow
class TransformerEncoder(d2l.Encoder):  #@save
    """트랜스포머 인코더."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerEncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_blks)]

    def call(self, X, valid_lens, **kwargs):
        # 포지셔널 인코딩 값이 -1과 1 사이이므로, 임베딩 값들을 합치기 전에
        # 스케일을 맞추기 위해 임베딩 차원의 제곱근을 곱합니다.
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab jax
class TransformerEncoder(d2l.Encoder):  #@save
    """트랜스포머 인코더."""
    vocab_size: int
    num_hiddens:int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens,
                                                   self.dropout)
        self.blks = [TransformerEncoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads,
                                             self.dropout, self.use_bias)
                     for _ in range(self.num_blks)]

    def __call__(self, X, valid_lens, training=False):
        # 포지셔널 인코딩 값이 -1과 1 사이이므로, 임베딩 값들을 합치기 전에
        # 스케일을 맞추기 위해 임베딩 차원의 제곱근을 곱합니다.
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X, training=training)
        attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, attention_w = blk(X, valid_lens, training=training)
            attention_weights[i] = attention_w
        # 중간 변수를 캡처하기 위해 Flax sow API가 사용됩니다.
        self.sow('intermediates', 'enc_attention_weights', attention_weights)
        return X
```

아래에서 [**2개 레이어의 트랜스포머 인코더를 생성하기 위해**] 하이퍼파라미터를 지정합니다. 트랜스포머 인코더 출력의 모양은 (배치 크기, 타임스텝 수, `num_hiddens`)입니다.

```{.python .input}
%%tab mxnet
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(np.ones((2, 100)), valid_lens), (2, 100, 24))
```

```{.python .input}
%%tab pytorch
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))
```

```{.python .input}
%%tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
d2l.check_shape(encoder(tf.ones((2, 100)), valid_lens, training=False),
                (2, 100, 24))
```

```{.python .input}
%%tab jax
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder.init_with_output(d2l.get_key(),
                                         jnp.ones((2, 100), dtype=jnp.int32),
                                         valid_lens)[0],
                (2, 100, 24))
```

## 디코더 (Decoder)

:numref:`fig_transformer`에 표시된 것처럼, [**트랜스포머 디코더는 여러 개의 동일한 레이어로 구성됩니다.**] 각 레이어는 아래의 `TransformerDecoderBlock` 클래스에서 구현되며, 세 개의 서브레이어(디코더 셀프 어텐션, 인코더-디코더 어텐션, 포지션 와이즈 피드포워드 네트워크)를 포함합니다. 이러한 서브레이어들은 주위에 잔차 연결과 그 뒤를 잇는 레이어 정규화를 사용합니다.

이 섹션의 앞부분에서 설명했듯이, 마스크된 멀티 헤드 디코더 셀프 어텐션(첫 번째 서브레이어)에서 쿼리, 키, 값은 모두 이전 디코더 레이어의 출력에서 옵니다. 시퀀스 투 시퀀스 모델을 훈련할 때 출력 시퀀스의 모든 위치(타임스텝)에 있는 토큰은 알려져 있습니다. 그러나 예측 중에는 출력 시퀀스가 토큰 단위로 생성되므로, 임의의 디코더 타임스텝에서 생성된 토큰들만 디코더 셀프 어텐션에서 사용될 수 있습니다. 디코더에서 자기 회귀를 유지하기 위해, 마스크된 셀프 어텐션은 임의의 쿼리가 해당 쿼리 위치까지의 디코더 내 위치들에만 어텐션을 수행하도록 `dec_valid_lens`를 지정합니다.

```{.python .input}
%%tab mxnet
class TransformerDecoderBlock(nn.Block):
    # 트랜스포머 디코더의 i번째 블록
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 훈련 중에는 임의의 출력 시퀀스의 모든 토큰이 동시에 처리되므로,
        # 초기화된 대로 state[2][self.i]는 None입니다. 예측 중에 토큰 단위로
        # 출력 시퀀스를 디코딩할 때, state[2][self.i]는 현재 타임스텝까지
        # i번째 블록에서 디코딩된 출력의 표현을 포함합니다.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens의 모양: (batch_size, num_steps), 여기서 모든
            # 행은 [1, 2, ..., num_steps] 입니다.
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None
        # 셀프 어텐션
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 인코더-디코더 어텐션. enc_outputs의 모양:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab pytorch
class TransformerDecoderBlock(nn.Module):
    # 트랜스포머 디코더의 i번째 블록
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 훈련 중에는 임의의 출력 시퀀스의 모든 토큰이 동시에 처리되므로,
        # 초기화된 대로 state[2][self.i]는 None입니다. 예측 중에 토큰 단위로
        # 출력 시퀀스를 디코딩할 때, state[2][self.i]는 현재 타임스텝까지
        # i번째 블록에서 디코딩된 출력의 표현을 포함합니다.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens의 모양: (batch_size, num_steps), 여기서 모든
            # 행은 [1, 2, ..., num_steps] 입니다.
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 셀프 어텐션
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 인코더-디코더 어텐션. enc_outputs의 모양:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab tensorflow
class TransformerDecoderBlock(tf.keras.layers.Layer):
    # 트랜스포머 디코더의 i번째 블록
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 훈련 중에는 임의의 출력 시퀀스의 모든 토큰이 동시에 처리되므로,
        # 초기화된 대로 state[2][self.i]는 None입니다. 예측 중에 토큰 단위로
        # 출력 시퀀스를 디코딩할 때, state[2][self.i]는 현재 타임스텝까지
        # i번째 블록에서 디코딩된 출력의 표현을 포함합니다.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens의 모양: (batch_size, num_steps), 여기서 모든
            # 행은 [1, 2, ..., num_steps] 입니다.
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), 
                           shape=(-1, num_steps)), repeats=batch_size, axis=0)
        else:
            dec_valid_lens = None
        # 셀프 어텐션
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, 
                             **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # 인코더-디코더 어텐션. enc_outputs의 모양:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,
                             **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

```{.python .input}
%%tab jax
class TransformerDecoderBlock(nn.Module):
    # 트랜스포머 디코더의 i번째 블록
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    i: int

    def setup(self):
        self.attention1 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm1 = AddNorm(self.dropout)
        self.attention2 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm2 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm3 = AddNorm(self.dropout)

    def __call__(self, X, state, training=False):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 훈련 중에는 임의의 출력 시퀀스의 모든 토큰이 동시에 처리되므로,
        # 초기화된 대로 state[2][self.i]는 None입니다. 예측 중에 토큰 단위로
        # 출력 시퀀스를 디코딩할 때, state[2][self.i]는 현재 타임스텝까지
        # i번째 블록에서 디코딩된 출력의 표현을 포함합니다.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = jnp.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens의 모양: (batch_size, num_steps), 여기서 모든
            # 행은 [1, 2, ..., num_steps] 입니다.
            dec_valid_lens = jnp.tile(jnp.arange(1, num_steps + 1),
                                      (batch_size, 1))
        else:
            dec_valid_lens = None
        # 셀프 어텐션
        X2, attention_w1 = self.attention1(X, key_values, key_values,
                                           dec_valid_lens, training=training)
        Y = self.addnorm1(X, X2, training=training)
        # 인코더-디코더 어텐션. enc_outputs의 모양:
        # (batch_size, num_steps, num_hiddens)
        Y2, attention_w2 = self.attention2(Y, enc_outputs, enc_outputs,
                                           enc_valid_lens, training=training)
        Z = self.addnorm2(Y, Y2, training=training)
        return self.addnorm3(Z, self.ffn(Z), training=training), state, attention_w1, attention_w2
```

인코더-디코더 어텐션에서의 스케일드 닷 프로덕트 연산과 잔차 연결에서의 덧셈 연산을 용이하게 하기 위해, [**디코더의 특성 차원(`num_hiddens`)은 인코더의 특성 차원과 동일합니다.**]

```{.python .input}
%%tab mxnet
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab pytorch
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab tensorflow
decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state, training=False)[0], X.shape)
```

```{.python .input}
%%tab jax
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk.init_with_output(d2l.get_key(), X, valid_lens)[0][0],
         valid_lens, [None]]
d2l.check_shape(decoder_blk.init_with_output(d2l.get_key(), X, state)[0][0],
                X.shape)
```

이제 `TransformerDecoderBlock`의 `num_blks`개 인스턴스로 구성된 [**전체 트랜스포머 디코더를 구축합니다.**] 마지막에 완전 연결 레이어는 `vocab_size`개의 가능한 모든 출력 토큰에 대한 예측을 계산합니다. 디코더 셀프 어텐션 가중치와 인코더-디코더 어텐션 가중치는 나중에 시각화하기 위해 저장됩니다.

```{.python .input}
%%tab mxnet
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add(TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize()

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 디코더 셀프 어텐션 가중치
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # 인코더-디코더 어텐션 가중치
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 디코더 셀프 어텐션 가중치
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # 인코더-디코더 어텐션 가중치
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerDecoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, i)
                     for i in range(num_blks)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        # 디코더에 2개의 어텐션 레이어
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # 디코더 셀프 어텐션 가중치
            self._attention_weights[0][i] = (
                blk.attention1.attention.attention_weights)
            # 인코더-디코더 어텐션 가중치
            self._attention_weights[1][i] = (
                blk.attention2.attention.attention_weights)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class TransformerDecoder(nn.Module):
    vocab_size: int
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens,
                                                   self.dropout)
        self.blks = [TransformerDecoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads, self.dropout, i)
                     for i in range(self.num_blks)]
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def __call__(self, X, state, training=False):
        X = self.embedding(X) * jnp.sqrt(jnp.float32(self.num_hiddens))
        X = self.pos_encoding(X, training=training)
        attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state, attention_w1, attention_w2 = blk(X, state,
                                                       training=training)
            # 디코더 셀프 어텐션 가중치
            attention_weights[0][i] = attention_w1
            # 인코더-디코더 어텐션 가중치
            attention_weights[1][i] = attention_w2
        # 중간 변수를 캡처하기 위해 Flax sow API가 사용됩니다.
        self.sow('intermediates', 'dec_attention_weights', attention_weights)
        return self.dense(X), state
```

## 훈련 (Training)

트랜스포머 아키텍처를 따라 인코더-디코더 모델을 인스턴스화해 봅시다. 여기서는 트랜스포머 인코더와 트랜스포머 디코더 모두 4-헤드 어텐션을 사용하는 2개 레이어를 갖도록 지정합니다. :numref:`sec_seq2seq_training`에서와 같이, 영어-프랑스어 기계 번역 데이터셋에서 시퀀스 투 시퀀스 학습을 위해 트랜스포머 모델을 훈련합니다.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
if tab.selected('tensorflow'):
    key_size, query_size, value_size = 256, 256, 256
    norm_shape = [2]
if tab.selected('pytorch', 'mxnet', 'jax'):
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001, training=True)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = TransformerEncoder(
            len(data.src_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        decoder = TransformerDecoder(
            len(data.tgt_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

훈련 후에는 트랜스포머 모델을 사용하여 [**몇 가지 영어 문장을**] 프랑스어로 번역하고 BLEU 점수를 계산합니다.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(
        trainer.state.params, data.build(engs, fras), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

마지막 영어 문장을 프랑스어로 번역할 때 [**트랜스포머 어텐션 가중치를 시각화해 봅시다.**] 인코더 셀프 어텐션 가중치의 모양은 (인코더 레이어 수, 어텐션 헤드 수, `num_steps` 또는 쿼리 수, `num_steps` 또는 키-값 쌍의 수)입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
enc_attention_weights = d2l.concat(model.encoder.attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

```{.python .input}
%%tab jax
_, (dec_attention_weights, enc_attention_weights) = model.predict_step(
    trainer.state.params, data.build([engs[-1]], [fras[-1]]),
    data.num_steps, True)
enc_attention_weights = d2l.concat(enc_attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

인코더 셀프 어텐션에서 쿼리와 키는 모두 동일한 입력 시퀀스에서 옵니다. 패딩 토큰은 의미를 갖지 않으므로, 입력 시퀀스의 유효 길이가 지정되면 어떤 쿼리도 패딩 토큰의 위치에 어텐션을 수행하지 않습니다. 다음에서는 멀티 헤드 어텐션 가중치의 두 레이어가 행별로 표시됩니다. 각 헤드는 쿼리, 키, 값의 별도 표현 서브스페이스에 기반하여 독립적으로 어텐션을 수행합니다.

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.show_heatmaps(
    enc_attention_weights, xlabel='키 위치', ylabel='쿼리 위치',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
%%tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='키 위치',
    ylabel='쿼리 위치', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**디코더 셀프 어텐션 가중치와 인코더-디코더 어텐션 가중치를 시각화하려면 더 많은 데이터 조작이 필요합니다.**] 예를 들어, 마스크된 어텐션 가중치를 0으로 채웁니다. 디코더 셀프 어텐션 가중치와 인코더-디코더 어텐션 가중치 모두 동일한 쿼리(시퀀스 시작 토큰 뒤에 출력 토큰들이 오고 필요한 경우 시퀀스 종료 토큰이 따름)를 갖는다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, (
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
shape = (-1, 2, num_blks, num_heads, data.num_steps)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, shape)
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
```

```{.python .input}
%%tab jax
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape(
    (-1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab all
d2l.check_shape(dec_self_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
d2l.check_shape(dec_inter_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

디코더 셀프 어텐션의 자기 회귀 속성 때문에, 어떤 쿼리도 쿼리 위치 이후의 키-값 쌍에 어텐션을 수행하지 않습니다.

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :],
    xlabel='키 위치', ylabel='쿼리 위치',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

인코더 셀프 어텐션의 경우와 유사하게, 입력 시퀀스의 지정된 유효 길이를 통해 [**출력 시퀀스의 어떤 쿼리도 입력 시퀀스의 패딩 토큰에 어텐션을 수행하지 않습니다.**]

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='키 위치',
    ylabel='쿼리 위치', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

트랜스포머 아키텍처는 원래 시퀀스 투 시퀀스 학습을 위해 제안되었지만, 나중에 책에서 보게 되듯이 트랜스포머 인코더나 트랜스포머 디코더는 각각 개별적으로 서로 다른 딥러닝 작업에 자주 사용됩니다.


## 요약 (Summary)

* 트랜스포머는 인코더-디코더 아키텍처의 한 사례이지만, 실제로는 인코더나 디코더 중 하나만 개별적으로 사용될 수도 있습니다.
* 트랜스포머 아키텍처에서 멀티 헤드 셀프 어텐션은 입력 시퀀스와 출력 시퀀스를 표현하는 데 사용되지만, 디코더는 마스크된 버전을 통해 자기 회귀 속성을 유지해야 합니다.
* 트랜스포머의 잔차 연결과 레이어 정규화는 매우 깊은 모델을 훈련하는 데 중요합니다.
* 트랜스포머 모델의 포지션 와이즈 피드포워드 네트워크는 동일한 MLP를 사용하여 모든 시퀀스 위치의 표현을 변환합니다.


## 연습 문제 (Exercises)

1. 실험에서 더 깊은 트랜스포머를 훈련해 보십시오. 훈련 속도와 번역 성능에 어떤 영향을 미칩니까?
2. 트랜스포머에서 스케일드 닷 프로덕트 어텐션을 가산(additive) 어텐션으로 바꾸는 것이 좋은 생각일까요? 왜 그렇습니까?
3. 언어 모델링의 경우 트랜스포머 인코더, 디코더 중 무엇을 사용해야 할까요, 아니면 둘 다 사용해야 할까요? 이 방법을 어떻게 설계하시겠습니까?
4. 입력 시퀀스가 매우 길면 트랜스포머가 어떤 어려움에 직면할 수 있습니까? 왜 그렇습니까?
5. 트랜스포머의 계산 및 메모리 효율성을 어떻게 개선하시겠습니까? 힌트: :citet:`Tay.Dehghani.Bahri.ea.2020`의 서베이 논문을 참조할 수 있습니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1066)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3871)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18031)
:end_tab:

```