# 감정 분석: 순환 신경망 사용 (Sentiment Analysis: Using Recurrent Neural Networks)
:label:`sec_sentiment_rnn` 


단어 유사성 및 유추 작업과 마찬가지로,
사전 훈련된 단어 벡터를
감정 분석에도 적용할 수 있습니다.
:numref:`sec_sentiment`의 IMDb 리뷰 데이터셋은
그리 크지 않으므로,
대규모 코퍼스에서
사전 훈련된 텍스트 표현을 사용하면
모델의 과대적합을 줄일 수 있습니다.
:numref:`fig_nlp-map-sa-rnn`에 설명된
구체적인 예로서,
우리는 사전 훈련된 GloVe 모델을 사용하여 각 토큰을 표현하고,
이러한 토큰 표현을
다층 양방향 RNN에 공급하여
텍스트 시퀀스 표현을 얻습니다.
이 표현은
감정 분석 출력으로 변환됩니다 :cite:`Maas.Daly.Pham.ea.2011`.
동일한 다운스트림 응용 프로그램에 대해,
나중에 다른 아키텍처 선택을 고려할 것입니다.

![이 섹션에서는 감정 분석을 위해 사전 훈련된 GloVe를 RNN 기반 아키텍처에 공급합니다.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN으로 단일 텍스트 표현하기 (Representing Single Text with RNNs)

감정 분석과 같은 텍스트 분류 작업에서,
가변 길이 텍스트 시퀀스는
고정 길이 범주로 변환됩니다.
다음 `BiRNN` 클래스에서,
텍스트 시퀀스의 각 토큰은
임베딩 레이어(`self.embedding`)를 통해
개별적인 사전 훈련된 GloVe 표현을 얻지만,
전체 시퀀스는
양방향 RNN(`self.encoder`)에 의해 인코딩됩니다.
더 구체적으로,
초기 및 최종 타임 스텝에서의
양방향 LSTM의 (마지막 레이어의) 은닉 상태가
텍스트 시퀀스의 표현으로 연결(concatenated)됩니다.
이 단일 텍스트 표현은
두 개의 출력("긍정" 및 "부정")이 있는
완전 연결 레이어(`self.decoder`)에 의해
출력 범주로 변환됩니다.

```{.python .input}
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # `bidirectional`을 True로 설정하여 양방향 RNN을 얻습니다
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # `inputs`의 모양은 (배치 크기, 타임 스텝 수)입니다.
        # LSTM은 입력의 첫 번째 차원이 시간 차원일 것을 요구하므로,
        # 토큰 표현을 얻기 전에 입력을 전치합니다.
        # 출력 모양은 (타임 스텝 수, 배치 크기, 단어 벡터 차원)입니다
        embeddings = self.embedding(inputs.T)
        # 다른 타임 스텝에서의 마지막 은닉층의 은닉 상태를 반환합니다.
        # `outputs`의 모양은 (타임 스텝 수, 배치 크기, 2 * 은닉 유닛 수)입니다
        outputs = self.encoder(embeddings)
        # 초기 및 최종 타임 스텝의 은닉 상태를 연결하여 완전 연결 레이어의 입력으로 사용합니다.
        # 그 모양은 (배치 크기, 4 * 은닉 유닛 수)입니다
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # `bidirectional`을 True로 설정하여 양방향 RNN을 얻습니다
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # `inputs`의 모양은 (배치 크기, 타임 스텝 수)입니다.
        # LSTM은 입력의 첫 번째 차원이 시간 차원일 것을 요구하므로,
        # 토큰 표현을 얻기 전에 입력을 전치합니다.
        # 출력 모양은 (타임 스텝 수, 배치 크기, 단어 벡터 차원)입니다
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 다른 타임 스텝에서의 마지막 은닉층의 은닉 상태를 반환합니다.
        # `outputs`의 모양은 (타임 스텝 수, 배치 크기, 2 * 은닉 유닛 수)입니다
        outputs, _ = self.encoder(embeddings)
        # 초기 및 최종 타임 스텝의 은닉 상태를 연결하여 완전 연결 레이어의 입력으로 사용합니다.
        # 그 모양은 (배치 크기, 4 * 은닉 유닛 수)입니다
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

감정 분석을 위해 단일 텍스트를 표현하는 두 개의 은닉층이 있는 양방향 RNN을 구성해 봅시다.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## 사전 훈련된 단어 벡터 로드 (Loading Pretrained Word Vectors)

아래에서는 어휘의 토큰에 대해 사전 훈련된 100차원(`embed_size`와 일치해야 함) GloVe 임베딩을 로드합니다.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

어휘의 모든 토큰에 대한 벡터의 모양을 출력합니다.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

우리는 이 사전 훈련된 단어 벡터를 사용하여
리뷰의 토큰을 표현하며,
훈련 중에 이 벡터들을 업데이트하지 않을 것입니다.

```{.python .input}
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## 모델 훈련 및 평가 (Training and Evaluating the Model)

이제 감정 분석을 위해 양방향 RNN을 훈련할 수 있습니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

훈련된 모델 `net`을 사용하여 텍스트 시퀀스의 감정을 예측하는 다음 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def predict_sentiment(net, vocab, sequence):
    """텍스트 시퀀스의 감정을 예측합니다."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """텍스트 시퀀스의 감정을 예측합니다."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

마지막으로, 훈련된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측해 봅시다.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약 (Summary)

* 사전 훈련된 단어 벡터는 텍스트 시퀀스의 개별 토큰을 나타낼 수 있습니다.
* 양방향 RNN은 초기 및 최종 타임 스텝의 은닉 상태를 연결하는 것과 같이 텍스트 시퀀스를 나타낼 수 있습니다. 이 단일 텍스트 표현은 완전 연결 레이어를 사용하여 범주로 변환될 수 있습니다.



## 연습 문제 (Exercises)

1. 에포크 수를 늘리십시오. 훈련 및 테스트 정확도를 향상시킬 수 있습니까? 다른 하이퍼파라미터를 조정하는 것은 어떻습니까?
2. 300차원 GloVe 임베딩과 같은 더 큰 사전 훈련된 단어 벡터를 사용해 보십시오. 분류 정확도가 향상됩니까?
3. spaCy 토큰화를 사용하여 분류 정확도를 향상시킬 수 있습니까? spaCy를 설치(`pip install spacy`)하고 영어 패키지를 설치(`python -m spacy download en`)해야 합니다. 코드에서 먼저 spaCy를 가져옵니다(`import spacy`). 그런 다음 spaCy 영어 패키지를 로드합니다(`spacy_en = spacy.load('en')`). 마지막으로 함수 `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`를 정의하고 원래 `tokenizer` 함수를 교체합니다. GloVe와 spaCy에서 구문 토큰의 형태가 다르다는 점에 유의하십시오. 예를 들어 구문 토큰 "new york"은 GloVe에서는 "new-york" 형태를 취하고 spaCy 토큰화 후에는 "new york" 형태를 취합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab: