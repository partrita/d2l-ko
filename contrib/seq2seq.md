# 시퀀스-투-시퀀스 (Sequence to Sequence)
:label:`sec_seq2seq`

시퀀스-투-시퀀스(seq2seq) 모델은 :numref:`fig_seq2seq`에서 시연된 것처럼, 시퀀스 입력을 받아 시퀀스 출력을 생성하기 위해 인코더-디코더 아키텍처를 기반으로 합니다. 인코더와 디코더 모두 가변 길이의 시퀀스 입력을 처리하기 위해 순환 신경망(RNN)을 사용합니다. 인코더의 은닉 상태는 인코더에서 디코더로 정보를 전달하기 위해 디코더의 은닉 상태를 초기화하는 데 직접 사용됩니다.

![시퀀스-투-시퀀스 모델 아키텍처.](../img/seq2seq.svg)
:label:`fig_seq2seq`

인코더와 디코더의 레이어들은 :numref:`fig_seq2seq_details`에 설명되어 있습니다.

![인코더와 디코더의 레이어들.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

이 섹션에서는 기계 번역 데이터셋에서 훈련하기 위한 seq2seq 모델을 설명하고 구현할 것입니다.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
from queue import PriorityQueue

npx.set_np()
```

## 인코더 (Encoder)

seq2seq의 인코더는 시퀀스 정보를 $\mathbf{c}$에 인코딩하여 가변 길이의 입력을 고정 길이의 문맥 벡터(context vector) $\mathbf{c}$로 변환할 수 있음을 상기하십시오. 우리는 대개 인코더 내에 RNN 레이어를 사용합니다.
입력 시퀀스 $x_1, \ldots, x_T$가 있고, 여기서 $x_t$가 $t^\mathrm{th}$번째 단어라고 가정합시다. 타임스텝 $t$에서 RNN은 두 개의 벡터를 입력으로 받습니다: $x_t$의 특성 벡터 $\mathbf{x}_t$와 이전 타임스텝의 은닉 상태 $\mathbf{h}_{t-1}$입니다. RNN의 은닉 상태 변환을 함수 $f$로 표시해 봅시다:

$$\mathbf{h}_t = f (\mathbf{x}_t, \mathbf{h}_{t-1}).$$

다음으로, 인코더는 모든 은닉 상태의 정보를 캡처하여 함수 $q$를 통해 문맥 벡터 $\mathbf{c}$에 인코딩합니다:

$$\mathbf{c} = q (\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

예를 들어, $q$를 $q (\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$로 선택하면 문맥 벡터는 최종 은닉 상태 $\mathbf{h}_T$가 됩니다.

지금까지 위에서 설명한 것은 각 타임스텝의 은닉 상태가 이전 타임스텝들에만 의존하는 단방향 RNN입니다. 우리는 또한 순차적 입력을 인코딩하기 위해 GRU, LSTM 및 양방향 RNN과 같은 다른 형태의 RNN을 사용할 수도 있습니다.

이제 seq2seq의 인코더를 구현해 봅시다. 여기서는 입력 언어의 단어 인덱스에 따라 특성 벡터를 얻기 위해 단어 임베딩 레이어를 사용합니다. 이러한 특성 벡터들은 다층 LSTM에 공급될 것입니다. 인코더의 입력은 시퀀스들의 배치이며, 이는 (배치 크기, 시퀀스 길이) 모양의 2차원 텐서입니다. 인코더는 LSTM 출력(즉, 모든 타임스텝의 은닉 상태)뿐만 아니라 최종 타임스텝의 은닉 상태와 메모리 셀을 모두 반환합니다.

```{.python .input  n=2}
#@tab mxnet
#@save
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)  # X 모양: (batch_size, seq_len, embed_size)
        # RNN은 첫 번째 축이 타임스텝(즉, seq_len)이어야 합니다
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        out, state = self.rnn(X, state)
        # out 모양: (seq_len, batch_size, num_hiddens)
        # state 모양: (num_layers, batch_size, num_hiddens),
        # 여기서 "state"는 은닉 상태와 메모리 셀을 포함합니다
        return out, state
```

다음으로, 배치 크기가 4이고 타임스텝이 7인 미니배치 시퀀스 입력을 생성할 것입니다. LSTM 유닛의 은닉층 수는 2이고 은닉 유닛 수는 16이라고 가정합니다. 입력에 대해 순전파 계산을 수행한 후 인코더가 반환하는 출력 모양은 (타임스텝 수, 배치 크기, 은닉 유닛 수)입니다. 최종 타임스텝에서 게이트 순환 유닛의 다층 은닉 상태 모양은 (은닉층 수, 배치 크기, 은닉 유닛 수)입니다. 게이트 순환 유닛의 경우, `state` 리스트는 은닉 상태인 단 하나의 요소만 포함합니다. 장단기 메모리(LSTM)가 사용되는 경우, `state` 리스트는 메모리 셀이라는 또 다른 요소도 포함합니다.

```{.python .input  n=3}
#@tab mxnet
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = np.zeros((4, 7))
output, state = encoder(X)
output.shape
```

LSTM이 사용되므로, `state` 리스트는 동일한 모양(은닉층 수, 배치 크기, 은닉 유닛 수)을 가진 은닉 상태와 메모리 셀을 모두 포함합니다. 그러나 GRU가 사용되는 경우, `state` 리스트는 최종 타임스텝에서의 은닉 상태(모양: (은닉층 수, 배치 크기, 은닉 유닛 수))라는 단 하나의 요소만 포함하게 됩니다.

```{.python .input  n=4}
#@tab mxnet
len(state), state[0].shape, state[1].shape
```

## 디코더 (Decoder)
:label:`sec_seq2seq_decoder`

방금 소개했듯이, 문맥 벡터 $\mathbf{c}$는 전체 입력 시퀀스 $x_1, \ldots, x_T$의 정보를 인코딩합니다. 훈련 세트에 주어진 출력이 $y_1, \ldots, y_{T'}$라고 가정합시다. 각 타임스텝 $t'$에서 출력 $y_{t'}$의 조건부 확률은 이전 출력 시퀀스 $y_1, \ldots, y_{t'-1}$와 문맥 벡터 $\mathbf{c}$에 의존합니다. 즉,

$$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$

따라서 우리는 디코더로 또 다른 RNN을 사용할 수 있습니다. 타임스텝 $t'$에서 디코더는 세 개의 입력, 즉 $y_{t'-1}$의 특성 벡터 $\mathbf{y}_{t'-1}$, 문맥 벡터 $\mathbf{c}$, 그리고 이전 타임스텝의 은닉 상태 $\mathbf{s}_{t'-1}$를 사용하여 은닉 상태 $\mathbf{s}_{t'}$를 업데이트합니다. 디코더 내의 RNN의 은닉 상태 변환을 함수 $g$로 표시해 봅시다:

$$\mathbf{s}_{t'} = g(\mathbf{y}_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1}).$$


디코더를 구현할 때, 인코더의 최종 타임스텝 은닉 상태를 디코더의 초기 은닉 상태로 직접 사용합니다. 이는 인코더와 디코더 RNN이 동일한 수의 레이어와 은닉 유닛을 가질 것을 요구합니다. 디코더의 LSTM 순전파 계산은 인코더의 것과 유사합니다. 유일한 차이점은 LSTM 레이어 뒤에 은닉 크기가 어휘 크기인 밀집(dense) 레이어를 추가한다는 것입니다. 밀집 레이어는 각 단어에 대한 신뢰도 점수를 예측합니다.

```{.python .input  n=5}
#@tab mxnet
#@save
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).swapaxes(0, 1)
        out, state = self.rnn(X, state)
        # 손실 계산을 단순화하기 위해 배치를 첫 번째 차원으로 만듭니다
        out = self.dense(out).swapaxes(0, 1)
        return out, state
```

인코더와 동일한 하이퍼파라미터로 디코더를 생성합니다. 보시다시피 출력 모양이 (배치 크기, 시퀀스 길이, 어휘 크기)로 변경되었습니다.

```{.python .input  n=6}
#@tab mxnet
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8,
                         num_hiddens=16, num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
out, state = decoder(X, state)
out.shape, len(state), state[0].shape, state[1].shape
```

## 손실 함수 (The Loss Function)

각 타임스텝마다 디코더는 단어를 예측하기 위해 어휘 크기의 신뢰도 점수 벡터를 출력합니다. 언어 모델링과 유사하게, 소프트맥스를 적용하여 확률을 얻은 다음 크로스 엔트로피 손실을 사용하여 손실을 계산할 수 있습니다. 타겟 문장들이 동일한 길이를 갖도록 패딩을 추가했지만, 패딩 기호에 대해서는 손실을 계산할 필요가 없음에 유의하십시오.

일부 항목을 필터링하는 손실 함수를 구현하기 위해, `SequenceMask`라는 연산자를 사용합니다. 이는 첫 번째 차원(`axis=0`) 또는 두 번째 차원(`axis=1`)을 마스킹하도록 지정할 수 있습니다. 두 번째 차원이 선택되면, 유효 길이 벡터 `len`과 2차원 입력 `X`가 주어졌을 때 이 연산자는 모든 $i$에 대해 `X[i, len[i]:] = 0`으로 설정합니다.

```{.python .input  n=7}
#@tab mxnet
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

$n$차원 텐서 $X$에 적용하면 `X[i, len[i]:, :, ..., :] = 0`으로 설정합니다. 또한 아래와 같이 $-1$과 같은 채우기 값을 지정할 수 있습니다.

```{.python .input  n=8}
#@tab mxnet
X = np.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

이제 마스킹된 버전의 소프트맥스 크로스 엔트로피 손실을 구현할 수 있습니다. 각 Gluon 손실 함수는 예제별 가중치를 지정할 수 있으며 기본값은 1입니다. 그러면 제거하고 싶은 각 예제에 대해 단순히 가중치 0을 사용할 수 있습니다. 따라서 우리의 사용자 정의 손실 함수는 각 시퀀스에서 일부 실패한 요소를 무시하기 위해 추가적인 `valid_len` 인수를 받습니다.

```{.python .input  n=9}
#@tab mxnet
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    # pred 모양: (batch_size, seq_len, vocab_size)
    # label 모양: (batch_size, seq_len)
    # valid_len 모양: (batch_size, )
    def forward(self, pred, label, valid_len):
        # weights 모양: (batch_size, seq_len, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

정상성 확인을 위해, 동일한 세 개의 시퀀스를 생성하고 첫 번째 시퀀스는 4개 요소를 유지하고, 두 번째는 2개, 마지막은 하나도 유지하지 않습니다. 그러면 첫 번째 예제 손실은 두 번째보다 2배 커야 하고, 마지막 손실은 0이어야 합니다.

```{.python .input  n=10}
#@tab mxnet
loss = MaskedSoftmaxCELoss()
loss(np.ones((3, 4, 10)), np.ones((3, 4)), np.array([4, 2, 0]))
```

## 훈련 (Training)
:label:`sec_seq2seq_training`

훈련 중에 타겟 시퀀스의 길이가 $n$이라면, 처음 $n-1$개의 토큰을 디코더의 입력으로 공급하고, 마지막 $n-1$개의 토큰은 정답 레이블로 사용됩니다.

```{.python .input  n=11}
#@tab mxnet
#@save
def train_s2s_ch9(model, data_iter, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], ylim=[0, 0.25])
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_ctx(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen-1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_vlen.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

다음으로, 모델 인스턴스를 생성하고 하이퍼파라미터를 설정합니다. 그런 다음 모델을 훈련할 수 있습니다.

```{.python .input  n=12}
#@tab mxnet
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
batch_size, num_steps = 64, 10
lr, num_epochs, ctx = 0.005, 300, d2l.try_gpu()

src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)
train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
```

## 예측 (Predicting)

여기서는 출력 시퀀스를 생성하기 위해 가장 간단한 방법인 그리디 검색(greedy search)을 구현합니다. :numref:`fig_seq2seq_predict`에서 설명한 것처럼, 예측 중에 타임스텝 0에서 훈련 때와 동일한 "<bos>" 토큰을 디코더에 공급합니다. 그러나 이후 타임스텝의 입력 토큰은 이전 타임스텝의 예측된 토큰입니다.

![그리디 검색으로 예측하는 시퀀스-투-시퀀스 모델](../img/seq2seq_predict.svg)
:label:`fig_seq2seq_predict`

```{.python .input  n=16}
#@tab mxnet
#@save
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward
#@save
def predict_s2s_ch9_beam(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                         beam_width, ctx):
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    enc_valid_len = np.array([len(src_tokens)], ctx=ctx)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = np.array(src_tokens, ctx=ctx)
    # batch_size 차원 추가
    enc_outputs = model.encoder(np.expand_dims(enc_X, axis=0),
                                enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=ctx), axis=0)
    
    node = BeamSearchNode(dec_state, None, dec_X, 0, 1)
    nodes = PriorityQueue()
    decoded_batch = []
    nodes.put((-node.eval(), node))
    #while True:
    for _ in range(num_steps):
        # 디코딩이 너무 오래 걸리면 포기함
        score, n = nodes.get()
        dec_X = n.wordid
        dec_state = n.h
        if n.wordid.item() == tgt_vocab['<eos>'] and n.prevNode != None:
            endnodes = (score, n)
            break
        Y, dec_state = model.decoder(dec_X, dec_state)
        indexes = npx.topk(Y, k=beam_width)
        nextnodes = []
        for new_k in range(beam_width):
            decoded_t = indexes[:,:,new_k]
            log_p = Y.reshape(-1)[decoded_t].item()
            node = BeamSearchNode(dec_state, n, decoded_t, n.logp + log_p, n.length + 1)
            score = -node.eval()
            nextnodes.append((score, node))
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            
    if len(endnodes) == 0:
        endnodes = nodes.get()
    score, n = endnodes
    predict_tokens = []
    if int(n.wordid) != tgt_vocab['<eos>']:
        predict_tokens.append(int(n.wordid))
    # 역추적
    while n.prevNode != None:
        n = n.prevNode
        if int(n.wordid) != tgt_vocab['<bos>']:
            predict_tokens.append(int(n.wordid))
    predict_tokens = predict_tokens[::-1]
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))
```

몇 가지 예제를 시도해 봅니다:

```{.python .input  n=204}
#@tab mxnet
for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !']:
    print(sentence + ' => ' + predict_s2s_ch9_beam(
        model, sentence, src_vocab, tgt_vocab, num_steps, 3, ctx))
```

## 요약 (Summary)

* 시퀀스-투-시퀀스(seq2seq) 모델은 시퀀스 입력을 받아 시퀀스 출력을 생성하기 위해 인코더-디코더 아키텍처를 기반으로 합니다.
* 인코더와 디코더 모두에 대해 여러 LSTM 레이어를 사용합니다.


## 연습 문제 (Exercises)

1. 신경 기계 번역 외에 seq2seq의 다른 사용 사례를 생각할 수 있습니까?
2. 이 섹션의 예제에서 입력 시퀀스가 더 길어지면 어떻게 될까요?
3. 손실 함수에서 `SequenceMask`를 사용하지 않으면 어떤 일이 발생할 수 있습니까?


## [토론](https://discuss.mxnet.io/t/4357)

![](../img/qr_seq2seq.svg)