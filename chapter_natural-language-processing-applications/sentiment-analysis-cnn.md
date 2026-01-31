# 감정 분석: 합성곱 신경망 사용하기 (Sentiment Analysis: Using Convolutional Neural Networks)
:label:`sec_sentiment_cnn` 

:numref:`chap_cnn`에서 우리는 인접한 픽셀과 같은 국소적 특성(local features)에 적용되는 2차원 CNN을 사용하여 2차원 이미지 데이터를 처리하는 메커니즘을 조사했습니다. 원래 컴퓨터 비전을 위해 설계되었지만, CNN은 자연어 처리에도 널리 사용됩니다. 간단히 말해서, 모든 텍스트 시퀀스를 1차원 이미지로 생각하면 됩니다. 이러한 방식으로 1차원 CNN은 텍스트의 $n$-그램과 같은 국소적 특성을 처리할 수 있습니다.

이 섹션에서는 *textCNN* 모델을 사용하여 단일 텍스트를 표현하기 위한 CNN 아키텍처를 설계하는 방법을 보여줄 것입니다(:cite:`Kim.2014`). 감정 분석을 위해 GloVe 사전 훈련을 받은 RNN 아키텍처를 사용하는 :numref:`fig_nlp-map-sa-rnn`과 비교할 때, :numref:`fig_nlp-map-sa-cnn`에서의 유일한 차이점은 아키텍처 선택에 있습니다.

![이 섹션에서는 감정 분석을 위해 사전 훈련된 GloVe를 CNN 기반 아키텍처에 공급합니다.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## 1차원 합성곱 (One-Dimensional Convolutions)

모델을 소개하기 전에 1차원 합성곱이 어떻게 작동하는지 살펴봅시다. 이것은 교차 상관(cross-correlation) 연산에 기반한 2차원 합성곱의 특수한 사례일 뿐임을 명심하십시오.

![1차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

:numref:`fig_conv1d`에 표시된 것처럼, 1차원 사례에서 합성곱 창(convolution window)은 입력 텐서 전체를 왼쪽에서 오른쪽으로 슬라이딩합니다. 슬라이딩하는 동안 특정 위치의 합성곱 창에 포함된 입력 서브텐서(예: :numref:`fig_conv1d`의 0과 1)와 커널 텐서(예: :numref:`fig_conv1d`의 1과 2)가 요소별로 곱해집니다. 이러한 곱셈의 합은 출력 텐서의 대응하는 위치에서 단일 스칼라 값(예: :numref:`fig_conv1d`에서 $0\times1+1\times2=2$)을 제공합니다.

다음 `corr1d` 함수에서 1차원 교차 상관을 구현합니다. 입력 텐서 `X`와 커널 텐`K`가 주어지면 출력 텐서 `Y`를 반환합니다.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

우리는 :numref:`fig_conv1d`의 입력 텐서 `X`와 커널 텐서 `K`를 구성하여 위의 1차원 교차 상관 구현의 출력을 검증할 수 있습니다.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

채널이 여러 개인 1차원 입력의 경우, 합성곱 커널은 동일한 수의 입력 채널을 가져야 합니다. 그런 다음 각 채널에 대해 입력의 1차원 텐서와 합성곱 커널의 1차원 텐서에 대해 교차 상관 연산을 수행하고, 모든 채널에 대해 결과를 합산하여 1차원 출력 텐서를 생성합니다. :numref:`fig_conv1d_channel`은 3개의 입력 채널이 있는 1차원 교차 상관 연산을 보여줍니다.

![3개의 입력 채널이 있는 1차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

우리는 여러 입력 채널에 대한 1차원 교차 상관 연산을 구현하고 :numref:`fig_conv1d_channel`의 결과를 검증할 수 있습니다.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # 먼저 `X`와 `K`의 0번째 차원(채널 차원)을 반복합니다. 그런 다음 그것들을 함께 더합니다.
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

다중 입력 채널 1차원 교차 상관은 단일 입력 채널 2차원 교차 상관과 동일합니다. 이를 설명하기 위해, :numref:`fig_conv1d_channel`의 다중 입력 채널 1차원 교차 상관의 등가 형태는 :numref:`fig_conv1d_2d`의 단일 입력 채널 2차원 교차 상관이며, 여기서 합성곱 커널의 높이는 입력 텐서의 높이와 같아야 합니다.

![단일 입력 채널이 있는 2차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

:numref:`fig_conv1d` 및 :numref:`fig_conv1d_channel`의 출력은 모두 채널이 하나뿐입니다. :numref:`subsec_multi-output-channels`에서 설명한 다중 출력 채널이 있는 2차원 합성곱과 마찬가지로, 1차원 합성곱에 대해서도 다중 출력 채널을 지정할 수 있습니다.


## 맥스 오버 타임 풀링 (Max-Over-Time Pooling)

마찬가지로 풀링을 사용하여 시퀀스 표현에서 가장 높은 값을 타임스텝 전체에서 가장 중요한 특성으로 추출할 수 있습니다. textCNN에서 사용되는 *맥스 오버 타임 풀링(max-over-time pooling)*은 1차원 글로벌 맥스 풀링(global max-pooling)처럼 작동합니다(:cite:`Collobert.Weston.Bottou.ea.2011`). 각 채널이 서로 다른 타임스텝의 값을 저장하는 다중 채널 입력의 경우, 각 채널의 출력은 해당 채널의 최대값입니다. 맥스 오버 타임 풀링을 사용하면 채널마다 타임스텝 수가 달라도 된다는 점에 유의하십시오.


## textCNN 모델

1차원 합성곱과 맥스 오버 타임 풀링을 사용하여 textCNN 모델은 개별 사전 훈련된 토큰 표현을 입력으로 받은 다음, 다운스트림 애플리케이션을 위한 시퀀스 표현을 얻고 변환합니다.

$d$차원 벡터로 표현된 $n$개의 토큰이 있는 단일 텍스트 시퀀스의 경우, 입력 텐서의 너비, 높이 및 채널 수는 각각 $n, 1, d$입니다. textCNN 모델은 다음과 같이 입력을 출력으로 변환합니다:

1. 여러 개의 1차원 합성곱 커널을 정의하고 입력에 대해 별도로 합성곱 연산을 수행합니다. 너비가 다른 합성곱 커널은 서로 다른 수의 인접 토큰 사이의 국소적 특성을 포착할 수 있습니다.
2. 모든 출력 채널에 대해 맥스 오버 타임 풀링을 수행한 다음, 모든 스칼라 풀링 출력을 벡터로 연결합니다.
3. 완전 연결 레이어를 사용하여 연결된 벡터를 출력 카테고리로 변환합니다. 과대적합을 줄이기 위해 드롭아웃을 사용할 수 있습니다.

![textCNN의 모델 아키텍처.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn`은 구체적인 예와 함께 textCNN의 모델 아키텍처를 설명합니다. 입력은 11개의 토큰이 있는 문장이며, 각 토큰은 6차원 벡터로 표현됩니다. 따라서 너비 11의 6채널 입력이 있습니다. 너비 2와 4의 두 개의 1차원 합성곱 커널을 정의하며, 각각 4개와 5개의 출력 채널을 갖습니다. 이들은 너비가 $11-2+1=10$인 4개의 출력 채널과 너비가 $11-4+1=8$인 5개의 출력 채널을 생성합니다. 이 9개 채널의 너비가 다르더라도 맥스 오버 타임 풀링은 연결된 9차원 벡터를 제공하며, 이는 최종적으로 이진 감정 예측을 위한 2차원 출력 벡터로 변환됩니다.


### 모델 정의 (Defining the Model)

우리는 다음 클래스에서 textCNN 모델을 구현합니다. :numref:`sec_sentiment_rnn`의 양방향 RNN 모델과 비교할 때, 순환 레이어를 합성곱 레이어로 교체하는 것 외에도 두 개의 임베딩 레이어를 사용합니다. 하나는 학습 가능한 가중치가 있고 다른 하나는 고정된 가중치가 있습니다.

```{.python .input}
#@tab mxnet
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 훈련되지 않을 임베딩 레이어
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 맥스 오버 타임 풀링 레이어는 파라미터가 없으므로 이 인스턴스를 공유할 수 있습니다.
        self.pool = nn.GlobalMaxPool1D()
        # 여러 개의 1차원 합성곱 레이어 생성
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # (배치 크기, 토큰 수, 토큰 벡터 차원) 모양의 두 임베딩 레이어 출력을 벡터를 따라 연결합니다.
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # 1차원 합성곱 레이어의 입력 형식에 따라 두 번째 차원이 채널을 저장하도록 텐서를 재정렬합니다.
        embeddings = embeddings.transpose(0, 2, 1)
        # 각 1차원 합성곱 레이어에 대해, 맥스 오버 타임 풀링 후 (배치 크기, 채널 수, 1) 모양의 텐서가 얻어집니다.
        # 마지막 차원을 제거하고 채널을 따라 연결합니다.
        encoding = np.concatenate([ 
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 훈련되지 않을 임베딩 레이어
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 맥스 오버 타임 풀링 레이어는 파라미터가 없으므로 이 인스턴스를 공유할 수 있습니다.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 여러 개의 1차원 합성곱 레이어 생성
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # (배치 크기, 토큰 수, 토큰 벡터 차원) 모양의 두 임베딩 레이어 출력을 벡터를 따라 연결합니다.
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 1차원 합성곱 레이어의 입력 형식에 따라 두 번째 차원이 채널을 저장하도록 텐서를 재정렬합니다.
        embeddings = embeddings.permute(0, 2, 1)
        # 각 1차원 합성곱 레이어에 대해, 맥스 오버 타임 풀링 후 (배치 크기, 채널 수, 1) 모양의 텐서가 얻어집니다.
        # 마지막 차원을 제거하고 채널을 따라 연결합니다.
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

textCNN 인스턴스를 생성해 봅시다. 이는 커널 너비가 3, 4, 5이고 모두 100개의 출력 채널을 갖는 3개의 합성곱 레이어를 가집니다.

```{.python .input}
#@tab mxnet
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights);
```

### 사전 훈련된 단어 벡터 로드 (Loading Pretrained Word Vectors)

:numref:`sec_sentiment_rnn`과 마찬가지로 사전 훈련된 100차원 GloVe 임베딩을 초기화된 토큰 표현으로 로드합니다. 이러한 토큰 표현(임베딩 가중치)은 `embedding`에서 훈련되고 `constant_embedding`에서 고정됩니다.

```{.python .input}
#@tab mxnet
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### 모델 훈련 및 평가 (Training and Evaluating the Model)

이제 감정 분석을 위해 textCNN 모델을 훈련할 수 있습니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

아래에서는 훈련된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측합니다.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약 (Summary)

* 1차원 CNN은 텍스트의 $n$-그램과 같은 국소적 특성을 처리할 수 있습니다.
* 다중 입력 채널 1차원 교차 상관은 단일 입력 채널 2차원 교차 상관과 동일합니다.
* 맥스 오버 타임 풀링을 사용하면 채널마다 타임스텝 수가 달라도 됩니다.
* textCNN 모델은 1차원 합성곱 레이어와 맥스 오버 타임 풀링 레이어를 사용하여 개별 토큰 표현을 다운스트림 애플리케이션 출력으로 변환합니다.


## 연습 문제 (Exercises)

1. 하이퍼파라미터를 튜닝하고 분류 정확도 및 계산 효율성 측면에서 :numref:`sec_sentiment_rnn`과 이 섹션의 두 감정 분석 아키텍처를 비교해 보십시오.
2. :numref:`sec_sentiment_rnn`의 연습 문제에서 소개된 방법을 사용하여 모델의 분류 정확도를 더 향상시킬 수 있습니까?
3. 입력 표현에 포지셔널 인코딩을 추가하십시오. 분류 정확도가 향상됩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1425)
:end_tab:
