```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 기계 번역을 위한 시퀀스-투-시퀀스 학습 (Sequence-to-Sequence Learning for Machine Translation)
:label:`sec_seq2seq`

입력과 출력이 각각 정렬되지 않은 가변 길이 시퀀스로 구성된 기계 번역(:numref:`sec_machine_translation`에서 논의됨)과 같은 소위 시퀀스-투-시퀀스 문제에서, 
우리는 일반적으로 인코더-디코더 아키텍처(:numref:`sec_encoder-decoder`)에 의존합니다. 
이 섹션에서는 인코더와 디코더가 모두 RNN으로 구현된 인코더-디코더 아키텍처를 기계 번역 작업에 적용하는 것을 시연합니다 :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`.

여기서 인코더 RNN은 가변 길이 시퀀스를 입력으로 받아 고정 모양의 은닉 상태로 변환합니다. 
나중에 :numref:`chap_attention-and-transformers`에서는 전체 입력을 하나의 고정 길이 표현으로 압축할 필요 없이 인코딩된 입력에 액세스할 수 있게 해 주는 주의(attention) 메커니즘을 소개할 것입니다. 

그런 다음 출력 시퀀스를 한 번에 한 토큰씩 생성하기 위해, 
별도의 RNN으로 구성된 디코더 모델은 
입력 시퀀스와 출력의 이전 토큰 모두가 주어졌을 때 각 후속 타겟 토큰을 예측합니다. 
훈련 중에 디코더는 일반적으로 공식 "ground truth" 레이블의 이전 토큰에 조건을 겁니다. 
그러나 테스트 시에는 이미 예측된 토큰에 디코더의 각 출력을 조건부로 설정하고 싶을 것입니다. 
인코더를 무시한다면 시퀀스-투-시퀀스 아키텍처의 디코더는 일반적인 언어 모델처럼 동작합니다. 
:numref:`fig_seq2seq`는 기계 번역에서 시퀀스-투-시퀀스 학습을 위해 두 개의 RNN을 사용하는 방법을 보여줍니다.


![RNN 인코더와 RNN 디코더를 사용한 시퀀스-투-시퀀스 학습.](../img/seq2seq.svg)
:label:`fig_seq2seq`

:numref:`fig_seq2seq`에서 특수 "&lt;eos&gt;" 토큰은 시퀀스의 끝을 표시합니다. 
이 토큰이 생성되면 모델은 예측을 중단할 수 있습니다. 
RNN 디코더의 초기 타임 스텝에는 알아야 할 두 가지 특별한 설계 결정이 있습니다: 
첫째, 모든 입력을 특수 문장 시작("&lt;bos&gt;") 토큰으로 시작합니다. 
둘째, 매 디코딩 타임 스텝마다 인코더의 최종 은닉 상태를 디코더에 공급할 수 있습니다 :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`. 
:citet:`Sutskever.Vinyals.Le.2014`와 같은 일부 다른 설계에서는 RNN 인코더의 최종 은닉 상태를 첫 번째 디코딩 단계에서만 디코더의 은닉 상태를 시작하는 데 사용합니다.

```{.python .input}
%%tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import math
import optax
```

## 교사 강요 (Teacher Forcing)

입력 시퀀스에서 인코더를 실행하는 것은 비교적 간단하지만, 
디코더의 입력과 출력을 처리하는 데는 더 많은 주의가 필요합니다. 
가장 일반적인 접근 방식은 때때로 *교사 강요(teacher forcing)*라고 불립니다. 
여기서 원래 타겟 시퀀스(토큰 레이블)가 디코더에 입력으로 제공됩니다. 
더 구체적으로, 
특수 문장 시작 토큰과 마지막 토큰을 제외한 원래 타겟 시퀀스가 디코더의 입력으로 연결되며, 
디코더 출력(훈련용 레이블)은 한 토큰만큼 이동된 원래 타겟 시퀀스입니다: 
"&lt;bos&gt;", "Ils", "regardent", "." $\rightarrow$
"Ils", "regardent", ".", "&lt;eos&gt;" (:numref:`fig_seq2seq`).

:numref:`subsec_loading-seq-fixed-len`의 구현은 교사 강요를 위한 훈련 데이터를 준비했으며, 
여기서 자기 지도 학습을 위해 토큰을 이동시키는 것은 :numref:`sec_language-model`의 언어 모델 훈련과 유사합니다. 
대안적인 접근 방식은 이전 타임 스텝에서 *예측된* 토큰을 디코더의 현재 입력으로 공급하는 것입니다. 


다음에서는 :numref:`fig_seq2seq`에 묘사된 설계를 더 자세히 설명합니다. 
우리는 :numref:`sec_machine_translation`에서 소개된 영어-프랑스어 데이터셋에서 기계 번역을 위해 이 모델을 훈련할 것입니다.

## 인코더 (Encoder)

인코더는 가변 길이의 입력 시퀀스를 고정 모양의 *문맥 변수(context variable)* $\mathbf{c}$로 변환함을 상기하십시오 (:numref:`fig_seq2seq` 참조).


단일 시퀀스 예제(배치 크기 1)를 고려하십시오. 
입력 시퀀스가 $x_1, \ldots, x_T$이고 $x_t$가 $t^{\textrm{th}}$번째 토큰이라고 가정합니다. 
타임 스텝 $t$에서 RNN은 $x_t$에 대한 입력 특성 벡터 $\mathbf{x}_t$와 이전 타임 스텝의 은닉 상태 $\mathbf{h} _{t-1}$을 현재 은닉 상태 $\mathbf{h}_t$로 변환합니다. 
우리는 함수 $f$를 사용하여 RNN의 순환 레이어의 변환을 표현할 수 있습니다:

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

일반적으로 인코더는 사용자 정의 함수 $q$를 통해 모든 타임 스텝의 은닉 상태를 문맥 변수로 변환합니다:

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$ 

예를 들어 :numref:`fig_seq2seq`에서 문맥 변수는 입력 시퀀스의 최종 토큰을 처리한 후의 인코더 RNN의 표현에 해당하는 은닉 상태 $\mathbf{h}_T$일 뿐입니다.

이 예제에서 우리는 단방향 RNN을 사용하여 인코더를 설계했습니다. 여기서 은닉 상태는 해당 타임 스텝 및 그 이전의 입력 하위 시퀀스에만 의존합니다. 
양방향 RNN을 사용하여 인코더를 구성할 수도 있습니다. 
이 경우 은닉 상태는 타임 스텝 이전과 이후의 하위 시퀀스(현재 타임 스텝의 입력 포함)에 의존하며, 이는 전체 시퀀스의 정보를 인코딩합니다.


이제 [**RNN 인코더를 구현**]해 봅시다. 
입력 시퀀스의 각 토큰에 대한 특성 벡터를 얻기 위해 *임베딩 레이어(embedding layer)*를 사용한다는 점에 유의하십시오. 
임베딩 레이어의 가중치는 행렬이며, 행의 수는 입력 어휘 크기(`vocab_size`)에 해당하고 열의 수는 특성 벡터의 차원(`embed_size`)에 해당합니다. 
임의의 입력 토큰 인덱스 $i$에 대해, 임베딩 레이어는 가중치 행렬의 $i^{\textrm{th}}$번째 행(0부터 시작)을 가져와 특성 벡터를 반환합니다. 
여기서는 다층 GRU로 인코더를 구현합니다.

```{.python .input}
%%tab mxnet
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.initialize(init.Xavier())
            
    def forward(self, X, *args):
        # X 모양: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs 모양: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs 모양: (num_steps, batch_size, num_hiddens)
        # state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab pytorch
def init_seq2seq(module):  #@save
    """시퀀스-투-시퀀스 학습을 위한 가중치 초기화."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
```

```{.python .input}
%%tab pytorch
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)
            
    def forward(self, X, *args):
        # X 모양: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs 모양: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs 모양: (num_steps, batch_size, num_hiddens)
        # state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab tensorflow
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
            
    def call(self, X, *args):
        # X 모양: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs 모양: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs 모양: (num_steps, batch_size, num_hiddens)
        # state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab jax
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)

    def __call__(self, X, *args, training=False):
        # X 모양: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        # embs 모양: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs, training=training)
        # outputs 모양: (num_steps, batch_size, num_hiddens)
        # state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

구체적인 예를 들어 [**위의 인코더 구현을 설명해 봅시다.**] 
아래에서는 은닉 유닛 수가 16인 2층 GRU 인코더를 인스턴스화합니다. 
시퀀스 입력 미니배치 `X`(배치 크기 $=4$; 타임 스텝 수 $=9$)가 주어지면, 모든 타임 스텝에서 최종 레이어의 은닉 상태(인코더의 순환 레이어에 의해 반환된 `enc_outputs`)는 (타임 스텝 수, 배치 크기, 은닉 유닛 수) 모양의 텐서입니다.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = d2l.zeros((batch_size, num_steps))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    enc_outputs, enc_state = encoder(X)
if tab.selected('jax'):
    (enc_outputs, enc_state), _ = encoder.init_with_output(d2l.get_key(), X)

d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
```

여기서는 GRU를 사용하고 있으므로, 최종 타임 스텝에서 다층 은닉 상태의 모양은 (은닉층 수, 배치 크기, 은닉 유닛 수)입니다.

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(enc_state, num_layers)
    d2l.check_shape(enc_state[0], (batch_size, num_hiddens))
```

## [**디코더 (Decoder)**]
:label:`sec_seq2seq_decoder`

타겟 출력 시퀀스 $y_1, y_2, \ldots, y_{T'}$가 주어졌을 때, 각 타임 스텝 $t'$에 대해(입력 시퀀스 타임 스텝과 구별하기 위해 $t^\prime$를 사용함), 
디코더는 타겟의 이전 토큰들 $y_1, \ldots, y_{t'}$와 문맥 변수 $\mathbf{c}$에 조건부로 설정된, $y_{t'+1}$ 단계에서 발생 가능한 각 토큰에 예측 확률을 할당합니다. 즉, $P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$.

타겟 시퀀스의 후속 토큰 $t'+1$을 예측하기 위해, RNN 디코더는 이전 단계의 타겟 토큰 $y_{t^\prime}$, 이전 타임 스텝의 은닉 RNN 상태 $\mathbf{s}_{t^\prime-1}$, 그리고 문맥 변수 $\mathbf{c}$를 입력으로 취하여 현재 타임 스텝의 은닉 상태 $\mathbf{s}_{t^\prime}$로 변환합니다. 
우리는 함수 $g$를 사용하여 디코더 은닉층의 변환을 표현할 수 있습니다:

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$ 
:eqlabel:`eq_seq2seq_s_t`

디코더의 은닉 상태를 얻은 후, 출력 레이어와 소프트맥스 연산을 사용하여 후속 출력 토큰 ${t'+1}$에 대한 예측 분포 $p(y_{t^{\prime}+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$를 계산할 수 있습니다.

:numref:`fig_seq2seq`를 따라 아래와 같이 디코더를 구현할 때, 인코더의 최종 타임 스텝 은닉 상태를 직접 사용하여 디코더의 은닉 상태를 초기화합니다. 
이는 RNN 인코더와 RNN 디코더가 동일한 수의 레이어와 은닉 유닛을 가질 것을 요구합니다. 
인코딩된 입력 시퀀스 정보를 추가로 통합하기 위해, 문맥 변수는 모든 타임 스텝에서 디코더 입력과 연결됩니다. 
출력 토큰의 확률 분포를 예측하기 위해, RNN 디코더의 최종 레이어 은닉 상태를 변환하는 완전 연결 레이어를 사용합니다.

```{.python .input}
%%tab mxnet
class Seq2SeqDecoder(d2l.Decoder):
    """시퀀스-투-시퀀스 학습을 위한 RNN 디코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs 

    def forward(self, X, state):
        # X 모양: (batch_size, num_steps)
        # embs 모양: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context 모양: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context를 (num_steps, batch_size, num_hiddens)로 브로드캐스트
        context = np.tile(context, (embs.shape[0], 1, 1))
        # 특성 차원에서 연결
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs 모양: (batch_size, num_steps, vocab_size)
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab pytorch
class Seq2SeqDecoder(d2l.Decoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 디코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
                           num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X 모양: (batch_size, num_steps)
        # embs 모양: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context 모양: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context를 (num_steps, batch_size, num_hiddens)로 브로드캐스트
        context = context.repeat(embs.shape[0], 1, 1)
        # 특성 차원에서 연결
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs 모양: (batch_size, num_steps, vocab_size)
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 디코더."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = tf.keras.layers.Dense(vocab_size)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def call(self, X, state):
        # X 모양: (batch_size, num_steps)
        # embs 모양: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context 모양: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context를 (num_steps, batch_size, num_hiddens)로 브로드캐스트
        context = tf.tile(tf.expand_dims(context, 0), (embs.shape[0], 1, 1))
        # 특성 차원에서 연결
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.transpose(self.dense(outputs), (1, 0, 2))
        # outputs 모양: (batch_size, num_steps, vocab_size)
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab jax
class Seq2SeqDecoder(d2l.Decoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 디코더."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def __call__(self, X, state, training=False):
        # X 모양: (batch_size, num_steps)
        # embs 모양: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context 모양: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context를 (num_steps, batch_size, num_hiddens)로 브로드캐스트
        context = jnp.tile(context, (embs.shape[0], 1, 1))
        # 특성 차원에서 연결
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state,
                                         training=training)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs 모양: (batch_size, num_steps, vocab_size)
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

구현된 디코더를 설명하기 위해, 
아래에서는 앞서 언급한 인코더와 동일한 하이퍼파라미터로 이를 인스턴스화합니다. 
보시다시피 디코더의 출력 모양은 (배치 크기, 타임 스텝 수, 어휘 크기)가 되며, 여기서 텐서의 최종 차원은 예측된 토큰 분포를 저장합니다.

```{.python .input}
%%tab all
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
if tab.selected('jax'):
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(), X)[0])
    (dec_outputs, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                       state)


d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(state[1], num_layers)
    d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

위의 RNN 인코더-디코더 모델의 레이어들은 :numref:`fig_seq2seq_details`에 요약되어 있습니다.

![RNN 인코더-디코더 모델의 레이어들.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`



## 시퀀스-투-시퀀스 학습을 위한 인코더-디코더 (Encoder--Decoder for Sequence-to-Sequence Learning)



코드에서 모든 것을 합치면 다음과 같습니다:

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더-디코더."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        # Adam 최적화기가 여기서 사용됩니다
        if tab.selected('mxnet'):
            return gluon.Trainer(self.parameters(), 'adam',
                                 {'learning_rate': self.lr})
        if tab.selected('pytorch'):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        if tab.selected('tensorflow'):
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
```

```{.python .input}
%%tab jax
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """시퀀스-투-시퀀스 학습을 위한 RNN 인코더-디코더."""
    encoder: nn.Module
    decoder: nn.Module
    tgt_pad: int
    lr: float

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        # Adam 최적화기가 여기서 사용됩니다
        return optax.adam(learning_rate=self.lr)
```

## 마스킹이 있는 손실 함수 (Loss Function with Masking)

각 타임 스텝에서 디코더는 출력 토큰에 대한 확률 분포를 예측합니다. 
언어 모델링과 마찬가지로, 분포를 얻기 위해 소프트맥스를 적용하고 최적화를 위해 교차 엔트로피 손실을 계산할 수 있습니다. 
:numref:`sec_machine_translation`에서 특수 패딩 토큰이 시퀀스 끝에 추가되어 다양한 길이의 시퀀스가 동일한 모양의 미니배치로 효율적으로 로드될 수 있음을 상기하십시오. 
그러나 패딩 토큰의 예측은 손실 계산에서 제외되어야 합니다. 
이를 위해, 관련 없는 예측과 0의 곱셈이 0이 되도록 [**관련 없는 항목을 0 값으로 마스킹**]할 수 있습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(Seq2Seq)
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=False):
    Y_hat = state.apply_fn({'params': params}, *X,
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    l = fn(Y_hat, Y)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask), {}
```

## [**훈련 (Training)**]
:label:`sec_seq2seq_training`

이제 기계 번역 데이터셋에서 시퀀스-투-시퀀스 학습을 위해 [**RNN 인코더-디코더 모델을 생성하고 훈련**]할 수 있습니다.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128) 
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
if tab.selected('jax'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**예측 (Prediction)**]

각 단계에서 출력 시퀀스를 예측하기 위해, 
이전 타임 스텝에서 예측된 토큰이 입력으로 디코더에 공급됩니다. 
한 가지 간단한 전략은 각 단계에서 예측할 때 디코더에 의해 가장 높은 확률이 할당된 토큰을 샘플링하는 것입니다. 
훈련에서와 마찬가지로 초기 타임 스텝에서는 문장 시작("&lt;bos&gt;") 토큰이 디코더에 공급됩니다. 
이 예측 과정은 :numref:`fig_seq2seq_predict`에 설명되어 있습니다. 
문장 끝("&lt;eos&gt;") 토큰이 예측되면 출력 시퀀스의 예측이 완료됩니다.


![RNN 인코더-디코더를 사용하여 토큰별로 출력 시퀀스 예측하기.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

다음 섹션에서는 빔 검색(:numref:`sec_beam-search`)에 기반한 더 정교한 전략을 소개할 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    if tab.selected('mxnet', 'pytorch'):
        batch = [d2l.to(a, device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    if tab.selected('mxnet', 'pytorch'):
        enc_all_outputs = self.encoder(src, src_valid_len)
    if tab.selected('tensorflow'):
        enc_all_outputs = self.encoder(src, src_valid_len, training=False)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
    for _ in range(num_steps):
        if tab.selected('mxnet', 'pytorch'):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
        if tab.selected('tensorflow'):
            Y, dec_state = self.decoder(outputs[-1], dec_state, training=False)
        outputs.append(d2l.argmax(Y, 2))
        # 주의 가중치 저장 (나중에 다룰 예정)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return d2l.concat(outputs[1:], 1), attention_weights
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, params, batch, num_steps,
                 save_attention_weights=False):
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs, inter_enc_vars = self.encoder.apply(
        {'params': params['encoder']}, src, src_valid_len, training=False,
        mutable='intermediates')
    # 인코더 주의 가중치를 포함하는 inter_enc_vars가 비어 있지 않으면 저장합니다. (나중에 다룰 예정)
    enc_attention_weights = []
    if bool(inter_enc_vars) and save_attention_weights:
        # intermediates 컬렉션에 저장된 인코더 주의 가중치
        enc_attention_weights = inter_enc_vars[
            'intermediates']['enc_attention_weights'][0]

    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:,0], 1), ], []
    for _ in range(num_steps):
        (Y, dec_state), inter_dec_vars = self.decoder.apply(
            {'params': params['decoder']}, outputs[-1], dec_state,
            training=False, mutable='intermediates')
        outputs.append(d2l.argmax(Y, 2))
        # 주의 가중치 저장 (나중에 다룰 예정)
        if save_attention_weights:
            # intermediates 컬렉션에 저장된 디코더 주의 가중치
            dec_attention_weights = inter_dec_vars[
                'intermediates']['dec_attention_weights'][0]
            attention_weights.append(dec_attention_weights)
    return d2l.concat(outputs[1:], 1), (attention_weights,
                                        enc_attention_weights)
```

## 예측된 시퀀스의 평가 (Evaluation of Predicted Sequences)

예측된 시퀀스를 타겟 시퀀스(ground truth)와 비교하여 평가할 수 있습니다. 
하지만 두 시퀀스 간의 유사성을 비교하기 위한 적절한 척도는 정확히 무엇일까요?


BLEU(Bilingual Evaluation Understudy)는 원래 기계 번역 결과를 평가하기 위해 제안되었지만 :cite:`Papineni.Roukos.Ward.ea.2002`, 
다양한 응용 프로그램에서 출력 시퀀스의 품질을 측정하는 데 널리 사용되어 왔습니다. 
원칙적으로 예측된 시퀀스의 임의의 $n$-gram(:numref:`subsec_markov-models-and-n-grams`)에 대해, BLEU는 이 $n$-gram이 타겟 시퀀스에 나타나는지 여부를 평가합니다.

$p_n$을 $n$-gram의 정밀도(precision)라고 하며, 
예측된 시퀀스와 타겟 시퀀스에서 일치하는 $n$-gram의 수를 예측된 시퀀스의 $n$-gram 수로 나눈 비율로 정의됩니다. 
설명하자면, 타겟 시퀀스 $A, B, C, D, E, F$와 예측된 시퀀스 $A, B, B, C, D$가 주어지면, 
우리는 $p_1 = 4/5, p_2 = 3/4, p_3 = 1/3, p_4 = 0$을 갖습니다. 
이제 $\textrm{len}_\textrm{label}$과 $\textrm{len}_\textrm{pred}$를 
각각 타겟 시퀀스와 예측된 시퀀스의 토큰 수라고 합시다. 
그러면 BLEU는 다음과 같이 정의됩니다.

$$ \exp\left(\min\left(0, 1 - \frac{\textrm{len}_\textrm{label}}{\textrm{len}_\textrm{pred}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$ 
:eqlabel:`eq_bleu`

여기서 $k$는 매칭을 위한 가장 긴 $n$-gram입니다.

:eqref:`eq_bleu`의 BLEU 정의에 따르면, 예측된 시퀀스가 타겟 시퀀스와 같을 때마다 BLEU는 1입니다. 
더욱이, 
더 긴 $n$-gram을 일치시키는 것이 더 어렵기 때문에, 
BLEU는 더 긴 $n$-gram이 높은 정밀도를 가질 때 더 큰 가중치를 할당합니다. 
구체적으로 $p_n$이 고정되어 있을 때, $p_n^{1/2^n}$은 $n$이 커짐에 따라 증가합니다(원본 논문은 $p_n^{1/n}$을 사용합니다). 
또한, 
더 짧은 시퀀스를 예측하는 것이 더 높은 $p_n$ 값을 산출하는 경향이 있으므로, 
:eqref:`eq_bleu`에서 곱셈 항 앞의 계수는 더 짧은 예측된 시퀀스에 페널티를 줍니다. 
예를 들어 $k=2$일 때 타겟 시퀀스 $A, B, C, D, E, F$와 예측된 시퀀스 $A, B$가 주어지면, 
$p_1 = p_2 = 1$임에도 불구하고 페널티 계수 $\exp(1-6/2) \approx 0.14$가 BLEU를 낮춥니다.

우리는 [**BLEU 척도를 구현**]합니다.

```{.python .input}
%%tab all
def bleu(pred_seq, label_seq, k):  #@save
    """BLEU를 계산합니다."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

마지막으로, 
훈련된 RNN 인코더-디코더를 사용하여 [**몇 가지 영어 문장을 프랑스어로 번역**]하고 결과의 BLEU를 계산합니다.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(trainer.state.params, data.build(engs, fras),
                                  data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)        
    print(f'{en} => {translation}, bleu,'
          f'{bleu(" ".join(translation), fr, k=2):.3f}')
```

## 요약 (Summary)

인코더-디코더 아키텍처의 설계에 따라 두 개의 RNN을 사용하여 시퀀스-투-시퀀스 학습을 위한 모델을 설계할 수 있습니다. 
인코더-디코더 훈련에서 교사 강요 접근 방식은 (예측과 대조적으로) 원래 출력 시퀀스를 디코더에 공급합니다. 
인코더와 디코더를 구현할 때 다층 RNN을 사용할 수 있습니다. 
마스크를 사용하여 손실을 계산할 때와 같이 관련 없는 계산을 걸러낼 수 있습니다. 
출력 시퀀스를 평가하기 위해 BLEU는 예측된 시퀀스와 타겟 시퀀스 사이의 $n$-gram을 일치시키는 인기 있는 척도입니다.


## 연습 문제 (Exercises)

1. 하이퍼파라미터를 조정하여 번역 결과를 개선할 수 있습니까?
2. 손실 계산에서 마스크를 사용하지 않고 실험을 다시 실행해 보십시오. 어떤 결과가 관찰됩니까? 왜 그런가요?
3. 인코더와 디코더의 레이어 수나 은닉 유닛 수가 다른 경우, 디코더의 은닉 상태를 어떻게 초기화할 수 있을까요?
4. 훈련에서 교사 강요를 디코더에 이전 타임 스텝의 예측을 공급하는 것으로 대체하십시오. 이것이 성능에 어떤 영향을 미칩니까?
5. GRU를 LSTM으로 교체하여 실험을 다시 실행하십시오.
6. 디코더의 출력 레이어를 설계하는 다른 방법이 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1062)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3865)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18022)
:end_tab:


```