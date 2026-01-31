```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 바다나우 주의 메커니즘 (The Bahdanau Attention Mechanism)
:label:`sec_seq2seq_attention`

:numref:`sec_seq2seq`에서 기계 번역을 접했을 때, 우리는 두 개의 RNN을 기반으로 시퀀스-투-시퀀스 학습을 위한 인코더-디코더 아키텍처를 설계했습니다 :cite:`Sutskever.Vinyals.Le.2014`. 
구체적으로 RNN 인코더는 가변 길이 시퀀스를 *고정 모양*의 문맥 변수로 변환합니다. 
그런 다음 RNN 디코더는 생성된 토큰과 문맥 변수를 기반으로 출력(타겟) 시퀀스를 토큰별로 생성합니다.

약간의 세부 사항을 추가하여 :numref:`fig_seq2seq_details`를 반복한 :numref:`fig_s2s_attention_state`를 상기해 보십시오. 
관습적으로 RNN에서 소스 시퀀스에 대한 모든 관련 정보는 인코더에 의해 어떤 내부의 *고정 차원* 상태 표현으로 번역됩니다. 
번역된 시퀀스를 생성하기 위해 디코더가 완전하고 독점적인 정보 소스로 사용하는 것이 바로 이 상태입니다. 
다시 말해, 시퀀스-투-시퀀스 메커니즘은 중간 상태를 입력으로 사용된 문자열의 충분 통계량(sufficient statistic)으로 취급합니다.

![시퀀스-투-시퀀스 모델. 인코더에 의해 생성된 상태는 인코더와 디코더 간에 공유되는 유일한 정보 조각입니다.](../img/seq2seq-state.svg)
:label:`fig_s2s_attention_state`

이것이 짧은 시퀀스에는 꽤 합리적이지만, 책의 장이나 아주 긴 문장과 같이 긴 시퀀스에는 불가능하다는 것이 분명합니다. 
결국 머지않아 소스 시퀀스에서 중요한 모든 것을 저장하기에는 중간 표현에 단순히 충분한 "공간"이 없게 될 것입니다. 
결과적으로 디코더는 길고 복잡한 문장을 번역하는 데 실패할 것입니다. 
이를 처음 접한 사람 중 한 명은 손글씨 텍스트를 생성하기 위해 RNN을 설계하려고 시도한 :citet:`Graves.2013`였습니다. 
소스 텍스트의 길이는 임의적이므로, 그들은 정렬이 한 방향으로만 이동하는 훨씬 더 긴 펜 자국과 텍스트 문자를 정렬하기 위해 미분 가능한 주의 모델을 설계했습니다. 
이는 차례로 음성 인식의 디코딩 알고리즘(예: 은닉 마르코프 모델 :cite:`rabiner1993fundamentals`)을 활용합니다.

정렬하는 법을 배우는 아이디어에서 영감을 받아, :citet:`Bahdanau.Cho.Bengio.2014`는 단방향 정렬 제한이 *없는* 미분 가능한 주의 모델을 제안했습니다. 
토큰을 예측할 때 모든 입력 토큰이 관련이 있는 것은 아니라면, 모델은 현재 예측과 관련이 있다고 간주되는 입력 시퀀스의 일부에만 정렬(또는 주의를 기울임)합니다. 
이것은 다음 토큰을 생성하기 전에 현재 상태를 업데이트하는 데 사용됩니다. 
설명은 꽤 평범해 보이지만, 이 *바다나우 주의 메커니즘(Bahdanau attention mechanism)*은 틀림없이 지난 10년 동안 딥러닝에서 가장 영향력 있는 아이디어 중 하나로 변모하여 Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`와 많은 관련 새로운 아키텍처를 탄생시켰습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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

우리는 :numref:`sec_seq2seq`의 시퀀스-투-시퀀스 아키텍처, 특히 :eqref:`eq_seq2seq_s_t`에 의해 도입된 표기법을 따릅니다. 
핵심 아이디어는 소스 문장을 요약하는 문맥 변수 $\mathbf{c}$인 상태를 고정된 상태로 유지하는 대신, 원래 텍스트(인코더 은닉 상태 $\mathbf{h}_{t}$)와 이미 생성된 텍스트(디코더 은닉 상태 $\mathbf{s}_{t'-1}$) 모두의 함수로서 동적으로 업데이트하는 것입니다. 
이는 임의의 디코딩 타임 스텝 $t'$ 이후에 업데이트되는 $\mathbf{c}_{t'}$를 산출합니다. 
입력 시퀀스의 길이가 $T$라고 가정합시다. 이 경우 문맥 변수는 어텐션 풀링의 출력입니다.

$$\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}.$$ 

우리는 $\mathbf{s}_{t' - 1}$을 쿼리로 사용하고, $\mathbf{h}_{t}$를 키와 값 모두로 사용했습니다. 
$\mathbf{c}_{t'}$는 상태 $\mathbf{s}_{t'}$를 생성하고 새 토큰을 생성하는 데 사용됩니다: :eqref:`eq_seq2seq_s_t`를 참조하십시오. 
특히 주의 가중치 $\alpha$는 :eqref:`eq_additive-attn`에 의해 정의된 가산 주의 스코어 함수를 사용하여 :eqref:`eq_attn-scoring-alpha`에서와 같이 계산됩니다. 
주의를 사용하는 이 RNN 인코더-디코더 아키텍처는 :numref:`fig_s2s_attention_details`에 묘사되어 있습니다. 
나중에 이 모델은 디코더에서 이미 생성된 토큰을 추가 문맥으로 포함하도록 수정되었습니다(즉, 주의 합계가 $T$에서 멈추지 않고 $t'-1$까지 진행됨). 예를 들어 음성 인식에 적용된 이 전략에 대한 설명은 :citet:`chan2015listen`을 참조하십시오.

![바다나우 주의 메커니즘을 사용한 RNN 인코더-디코더 모델의 레이어들.](../img/seq2seq-details-attention.svg)
:label:`fig_s2s_attention_details`

## 주의가 있는 디코더 정의하기 (Defining the Decoder with Attention)

주의가 있는 RNN 인코더-디코더를 구현하기 위해 디코더만 재정의하면 됩니다(주의 함수에서 생성된 기호를 생략하면 설계가 단순해집니다). 
상당히 당연한 이름인 `AttentionDecoder` 클래스를 정의함으로써 [**주의가 있는 디코더를 위한 기본 인터페이스**]부터 시작하겠습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class AttentionDecoder(d2l.Decoder):  #@save
    """주의 기반 디코더를 위한 기본 인터페이스."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

우리는 `Seq2SeqAttentionDecoder` 클래스에서 [**RNN 디코더를 구현**]해야 합니다. 
디코더의 상태는 다음과 같이 초기화됩니다: 
(i) 모든 타임 스텝에서 인코더의 마지막 레이어의 은닉 상태(주의를 위한 키와 값으로 사용됨); 
(ii) 최종 타임 스텝에서 모든 레이어에서의 인코더의 은닉 상태(디코더의 은닉 상태를 초기화하는 역할); 
(iii) 어텐션 풀링에서 패딩 토큰을 제외하기 위한 인코더의 유효 길이. 
각 디코딩 타임 스텝에서 이전 타임 스텝에서 얻은 디코더의 최종 레이어 은닉 상태가 주의 메커니즘의 쿼리로 사용됩니다. 
주의 메커니즘의 출력과 입력 임베딩은 모두 연결되어 RNN 디코더의 입력으로 사용됩니다.

```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs 모양: (num_steps, batch_size, num_hiddens).
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs 모양: (batch_size, num_steps, num_hiddens).
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 출력 X 모양: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # query 모양: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # context 모양: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 특성 차원에서 연결
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # x를 (1, batch_size, embed_size + num_hiddens)로 재구성
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 완전 연결 레이어 변환 후 outputs 모양:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs 모양: (num_steps, batch_size, num_hiddens).
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs 모양: (batch_size, num_steps, num_hiddens).
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 출력 X 모양: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query 모양: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context 모양: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 특성 차원에서 연결
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # x를 (1, batch_size, embed_size + num_hiddens)로 재구성
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 완전 연결 레이어 변환 후 outputs 모양:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs 모양: (batch_size, num_steps, num_hiddens). 
        # 리스트 hidden_state의 길이는 num_layers이며, 그 요소의 모양은 (batch_size, num_hiddens)입니다
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # 출력 enc_outputs 모양: # (batch_size, num_steps, num_hiddens)
        # 리스트 hidden_state의 길이는 num_layers이며, 그 요소의 모양은 (batch_size, num_hiddens)입니다
        enc_outputs, hidden_state, enc_valid_lens = state
        # 출력 X 모양: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # 입력 X 모양: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # query 모양: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # context 모양: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # 특성 차원에서 연결
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # 완전 연결 레이어 변환 후 outputs 모양:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class Seq2SeqAttentionDecoder(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.attention = d2l.AdditiveAttention(self.num_hiddens, self.dropout)
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.dense = nn.Dense(self.vocab_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout=self.dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs 모양: (num_steps, batch_size, num_hiddens). 
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # 주의 가중치는 상태의 일부로 반환됩니다. None으로 초기화합니다
        return (outputs.transpose(1, 0, 2), hidden_state, enc_valid_lens)

    @nn.compact
    def __call__(self, X, state, training=False):
        # enc_outputs 모양: (batch_size, num_steps, num_hiddens). 
        # hidden_state 모양: (num_layers, batch_size, num_hiddens)
        # 상태의 Attention 값 무시
        enc_outputs, hidden_state, enc_valid_lens = state
        # 출력 X 모양: (num_steps, batch_size, embed_size)
        X = self.embedding(X).transpose(1, 0, 2)
        outputs, attention_weights = [], []
        for x in X:
            # query 모양: (batch_size, 1, num_hiddens)
            query = jnp.expand_dims(hidden_state[-1], axis=1)
            # context 모양: (batch_size, 1, num_hiddens)
            context, attention_w = self.attention(query, enc_outputs,
                                                  enc_outputs, enc_valid_lens,
                                                  training=training)
            # 특성 차원에서 연결
            x = jnp.concatenate((context, jnp.expand_dims(x, axis=1)), axis=-1)
            # x를 (1, batch_size, embed_size + num_hiddens)로 재구성
            out, hidden_state = self.rnn(x.transpose(1, 0, 2), hidden_state,
                                         training=training)
            outputs.append(out)
            attention_weights.append(attention_w)

        # Flax sow API는 중간 변수를 캡처하는 데 사용됩니다
        self.sow('intermediates', 'dec_attention_weights', attention_weights)

        # 완전 연결 레이어 변환 후 outputs 모양:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(jnp.concatenate(outputs, axis=0))
        return outputs.transpose(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
```

다음에서는 각각 7개 타임 스텝 길이의 4개 시퀀스 미니배치를 사용하여 주의가 있는 [**구현된 디코더를 테스트**]합니다.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
if tab.selected('jax'):
    X = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(),
                                                        X, training=False)[0],
                               None)
    (output, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                  state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**훈련 (Training)**]

이제 새 디코더를 지정했으므로 :numref:`sec_seq2seq_training`과 유사하게 진행할 수 있습니다: 
하이퍼파라미터를 지정하고, 일반 인코더와 주의가 있는 디코더를 인스턴스화하고, 기계 번역을 위해 이 모델을 훈련합니다.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

모델이 훈련된 후, 
[**몇 가지 영어 문장을 프랑스어로 번역**]하고 BLEU 점수를 계산하는 데 사용합니다.

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

마지막 영어 문장을 번역할 때 [**주의 가중치를 시각화**]해 봅시다. 
각 쿼리가 키-값 쌍에 대해 균일하지 않은 가중치를 할당하는 것을 볼 수 있습니다. 
이는 각 디코딩 단계에서 입력 시퀀스의 서로 다른 부분이 어텐션 풀링에서 선택적으로 집계됨을 보여줍니다.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
if tab.selected('jax'):
    _, (dec_attention_weights, _) = model.predict_step(
        trainer.state.params, data.build([engs[-1]], [fras[-1]]),
        data.num_steps, True)
attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# 문장 끝 토큰을 포함하기 위해 1을 더합니다
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='키 위치', ylabel='쿼리 위치')
```

```{.python .input}
%%tab pytorch
# 문장 끝 토큰을 포함하기 위해 1을 더합니다
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='키 위치', ylabel='쿼리 위치')
```

```{.python .input}
%%tab tensorflow
# 문장 끝 토큰을 포함하기 위해 1을 더합니다
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='키 위치', ylabel='쿼리 위치')
```

```{.python .input}
%%tab jax
# 문장 끝 토큰을 포함하기 위해 1을 더합니다
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='키 위치', ylabel='쿼리 위치')
```

## 요약 (Summary)

토큰을 예측할 때 모든 입력 토큰이 관련이 있는 것은 아니라면, 바다나우 주의 메커니즘이 있는 RNN 인코더-디코더는 입력 시퀀스의 서로 다른 부분을 선택적으로 집계합니다. 이는 상태(문맥 변수)를 가산 주의 풀링의 출력으로 취급함으로써 달성됩니다. 
RNN 인코더-디코더에서 바다나우 주의 메커니즘은 이전 타임 스텝의 디코더 은닉 상태를 쿼리로, 모든 타임 스텝의 인코더 은닉 상태를 키와 값 모두로 취급합니다.


## 연습 문제 (Exercises)

1. 실험에서 GRU를 LSTM으로 교체하십시오.
2. 가산 주의 스코어 함수를 스케일드 내적으로 교체하도록 실험을 수정하십시오. 그것이 훈련 효율성에 어떤 영향을 미칩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3868)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18028)
:end_tab:


```