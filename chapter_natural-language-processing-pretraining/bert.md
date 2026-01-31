# 트랜스포머로부터의 양방향 인코더 표현 (BERT)
:label:`sec_bert`

우리는 자연어 이해를 위한 여러 단어 임베딩 모델을 소개했습니다.
사전 훈련 후, 출력은 각 행이 미리 정의된 어휘의 단어를 나타내는 벡터인 행렬로 생각할 수 있습니다.
사실, 이러한 단어 임베딩 모델은 모두 *문맥 독립적(context-independent)*입니다.
이 속성을 설명하는 것으로 시작하겠습니다.


## 문맥 독립적 표현에서 문맥 의존적 표현으로 (From Context-Independent to Context-Sensitive)

:numref:`sec_word2vec_pretraining`과 :numref:`sec_synonyms`의 실험을 상기해 보십시오.
예를 들어, word2vec과 GloVe는 단어의 문맥에 관계없이 동일한 단어에 동일한 사전 훈련된 벡터를 할당합니다(문맥이 있는 경우).
공식적으로, 임의의 토큰 $x$의 문맥 독립적 표현은 $x$만을 입력으로 취하는 함수 $f(x)$입니다.
자연어의 다의어와 복잡한 의미론의 풍부함을 고려할 때,
문맥 독립적 표현은 명백한 한계를 가지고 있습니다.
예를 들어, "a crane is flying"과 "a crane driver came"이라는 문맥에서 "crane"이라는 단어는 완전히 다른 의미를 갖습니다(전자는 '학', 후자는 '기중기').
따라서 동일한 단어라도 문맥에 따라 다른 표현이 할당되어야 할 수 있습니다.

이것은 단어의 표현이 문맥에 의존하는 *문맥 의존적(context-sensitive)* 단어 표현의 개발에 동기를 부여했습니다.
따라서 토큰 $x$의 문맥 의존적 표현은 $x$와 그 문맥 $c(x)$ 모두에 의존하는 함수 $f(x, c(x))$입니다.
인기 있는 문맥 의존적 표현으로는
TagLM(language-model-augmented sequence tagger) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe(Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017`,
ELMo(Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`가 있습니다.

예를 들어, 전체 시퀀스를 입력으로 취함으로써,
ELMo는 입력 시퀀스의 각 단어에 표현을 할당하는 함수입니다.
구체적으로, ELMo는 사전 훈련된 양방향 LSTM의 모든 중간 레이어 표현을 출력 표현으로 결합합니다.
그런 다음 ELMo 표현은 다운스트림 작업의 기존 지도 모델에 추가 특징으로 추가됩니다. 예를 들어 기존 모델의 토큰에 대한 원본 표현(예: GloVe)과 ELMo 표현을 연결하는 방식입니다.
한편으로, 사전 훈련된 양방향 LSTM 모델의 모든 가중치는 ELMo 표현이 추가된 후 고정됩니다.
반면, 기존 지도 모델은 주어진 작업에 맞게 특별히 맞춤화됩니다.
당시 서로 다른 작업에 대해 서로 다른 최적의 모델을 활용하고 ELMo를 추가함으로써,
감정 분석, 자연어 추론, 의미역 결정(semantic role labeling), 상호참조 해결(coreference resolution), 개체명 인식(named entity recognition), 질문 응답 등 6가지 자연어 처리 작업 전반에 걸쳐 최첨단 기술을 향상시켰습니다.


## 작업별에서 작업 불가지론적으로 (From Task-Specific to Task-Agnostic)

ELMo가 다양한 자연어 처리 작업에 대한 솔루션을 크게 개선했지만,
각 솔루션은 여전히 *작업별(task-specific)* 아키텍처에 의존합니다.
그러나 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 실제로 쉽지 않습니다.
GPT(Generative Pre-Training) 모델은 문맥 의존적 표현을 위한 일반적인 *작업 불가지론적(task-agnostic)* 모델을 설계하려는 노력을 나타냅니다 :cite:`Radford.Narasimhan.Salimans.ea.2018`.
트랜스포머 디코더를 기반으로 구축된 GPT는 텍스트 시퀀스를 표현하는 데 사용될 언어 모델을 사전 훈련합니다.
GPT를 다운스트림 작업에 적용할 때, 언어 모델의 출력은 작업의 레이블을 예측하기 위해 추가된 선형 출력 레이어에 공급됩니다.
사전 훈련된 모델의 파라미터를 고정하는 ELMo와 극명하게 대조적으로,
GPT는 다운스트림 작업의 지도 학습 중에 사전 훈련된 트랜스포머 디코더의 *모든* 파라미터를 미세 조정합니다.
GPT는 자연어 추론, 질문 응답, 문장 유사성 및 분류의 12가지 작업에서 평가되었으며,
모델 아키텍처에 최소한의 변경만으로 그중 9가지 작업에서 최첨단 기술을 개선했습니다.

그러나 언어 모델의 자기 회귀 특성으로 인해,
GPT는 앞만 봅니다(왼쪽에서 오른쪽으로).
"i went to the bank to deposit cash"와 "i went to the bank to sit down"이라는 문맥에서,
"bank"는 그 왼쪽 문맥에 민감하므로,
GPT는 "bank"에 대해 동일한 표현을 반환하지만, 실제로는 다른 의미를 갖습니다.


## BERT: 두 세계의 장점 결합 (BERT: Combining the Best of Both Worlds)

우리가 보았듯이,
ELMo는 문맥을 양방향으로 인코딩하지만 작업별 아키텍처를 사용합니다.
반면 GPT는 작업 불가지론적이지만 문맥을 왼쪽에서 오른쪽으로 인코딩합니다.
두 세계의 장점을 결합하여,
BERT(Bidirectional Encoder Representations from Transformers)는
문맥을 양방향으로 인코딩하고 광범위한 자연어 처리 작업에 대해 최소한의 아키텍처 변경만 요구합니다 :cite:`Devlin.Chang.Lee.ea.2018`.
사전 훈련된 트랜스포머 인코더를 사용하여,
BERT는 양방향 문맥을 기반으로 모든 토큰을 표현할 수 있습니다.
다운스트림 작업의 지도 학습 중에,
BERT는 두 가지 측면에서 GPT와 유사합니다.
첫째, BERT 표현은 추가된 출력 레이어에 공급되며, 모든 토큰에 대해 예측하는 것과 전체 시퀀스에 대해 예측하는 것과 같이 작업의 성격에 따라 모델 아키텍처에 최소한의 변경만 가합니다.
둘째, 사전 훈련된 트랜스포머 인코더의 모든 파라미터가 미세 조정되는 반면, 추가 출력 레이어는 처음부터 훈련됩니다.
:numref:`fig_elmo-gpt-bert`는 ELMo, GPT, BERT 간의 차이점을 묘사합니다.

![ELMo, GPT, BERT 비교.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`


BERT는 (i) 단일 텍스트 분류(예: 감정 분석), (ii) 텍스트 쌍 분류(예: 자연어 추론), (iii) 질문 응답, (iv) 텍스트 태깅(예: 개체명 인식)의 광범위한 범주 아래 11가지 자연어 처리 작업에서 최첨단 기술을 더욱 향상시켰습니다.
모두 2018년에 제안된 문맥 의존적 ELMo에서 작업 불가지론적 GPT와 BERT에 이르기까지,
개념적으로 간단하면서도 경험적으로 강력한 자연어에 대한 심층 표현의 사전 훈련은 다양한 자연어 처리 작업에 대한 솔루션에 혁명을 일으켰습니다.

이 장의 나머지 부분에서는 BERT의 사전 훈련에 대해 자세히 알아볼 것입니다.
:numref:`chap_nlp_app`에서 자연어 처리 응용 프로그램이 설명될 때,
다운스트림 응용 프로그램을 위한 BERT의 미세 조정을 설명할 것입니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**입력 표현 (Input Representation)**]
:label:`subsec_bert_input_rep`

자연어 처리에서,
어떤 작업(예: 감정 분석)은 단일 텍스트를 입력으로 취하고,
다른 작업(예: 자연어 추론)에서는 입력이 텍스트 시퀀스 쌍입니다.
BERT 입력 시퀀스는 단일 텍스트와 텍스트 쌍을 명확하게 나타냅니다.
전자의 경우,
BERT 입력 시퀀스는
특수 분류 토큰 “&lt;cls&gt;”,
텍스트 시퀀스의 토큰,
그리고 특수 구분 토큰 “&lt;sep&gt;”의 연결입니다.
후자의 경우,
BERT 입력 시퀀스는
“&lt;cls&gt;”, 첫 번째 텍스트 시퀀스의 토큰,
“&lt;sep&gt;”, 두 번째 텍스트 시퀀스의 토큰, 그리고 “&lt;sep&gt;”의 연결입니다.
우리는 "BERT 입력 시퀀스"라는 용어를 다른 유형의 "시퀀스"와 일관되게 구별할 것입니다.
예를 들어, 하나의 *BERT 입력 시퀀스*는 하나의 *텍스트 시퀀스* 또는 두 개의 *텍스트 시퀀스*를 포함할 수 있습니다.

텍스트 쌍을 구별하기 위해,
학습된 세그먼트 임베딩 $\mathbf{e}_A$와 $\mathbf{e}_B$가
각각 첫 번째 시퀀스와 두 번째 시퀀스의 토큰 임베딩에 추가됩니다.
단일 텍스트 입력의 경우 $\mathbf{e}_A$만 사용됩니다.

다음 `get_tokens_and_segments`는 한 문장 또는 두 문장을 입력으로 받아
BERT 입력 시퀀스의 토큰과 해당 세그먼트 ID를 반환합니다.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """BERT 입력 시퀀스의 토큰과 세그먼트 ID를 가져옵니다."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0과 1은 각각 세그먼트 A와 B를 표시합니다
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT는 양방향 아키텍처로 트랜스포머 인코더를 선택합니다.
트랜스포머 인코더에서 흔히 그렇듯이,
위치 임베딩이 BERT 입력 시퀀스의 모든 위치에 추가됩니다.
그러나 원래 트랜스포머 인코더와 달리,
BERT는 *학습 가능한* 위치 임베딩을 사용합니다.
요약하자면, :numref:`fig_bert-input`은
BERT 입력 시퀀스의 임베딩이
토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합임을 보여줍니다.

![BERT 입력 시퀀스의 임베딩은 토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합입니다.](../img/bert-input.svg)
:label:`fig_bert-input`

다음 [**`BERTEncoder` 클래스**]는 :numref:`sec_transformer`에서 구현된 `TransformerEncoder` 클래스와 유사합니다.
`TransformerEncoder`와 달리, `BERTEncoder`는
세그먼트 임베딩과 학습 가능한 위치 임베딩을 사용합니다.

```{.python .input}
#@tab mxnet
#@save
class BERTEncoder(nn.Block):
    """BERT 인코더."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # BERT에서 위치 임베딩은 학습 가능하므로, 충분히 긴 위치 임베딩 파라미터를 생성합니다
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # `X`의 모양은 다음 코드 스니펫에서 변경되지 않은 상태로 유지됩니다:
        # (배치 크기, 최대 시퀀스 길이, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT 인코더."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # BERT에서 위치 임베딩은 학습 가능하므로, 충분히 긴 위치 임베딩 파라미터를 생성합니다
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # `X`의 모양은 다음 코드 스니펫에서 변경되지 않은 상태로 유지됩니다:
        # (배치 크기, 최대 시퀀스 길이, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

어휘 크기가 10,000이라고 가정합니다.
[**`BERTEncoder`의 순방향 추론**]을 시연하기 위해,
인스턴스를 생성하고 파라미터를 초기화해 봅시다.

```{.python .input}
#@tab mxnet
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_blks, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
ffn_num_input, num_blks, dropout = 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
```

`tokens`를 길이 8의 BERT 입력 시퀀스 2개로 정의합니다.
여기서 각 토큰은 어휘의 인덱스입니다.
입력 `tokens`를 사용한 `BERTEncoder`의 순방향 추론은
각 토큰이 하이퍼파라미터 `num_hiddens`에 의해 미리 정의된 길이의 벡터로 표현되는 인코딩된 결과를 반환합니다.
이 하이퍼파라미터는 일반적으로 트랜스포머 인코더의 *은닉 크기(hidden size)* (은닉 유닛 수)라고 합니다.

```{.python .input}
#@tab mxnet
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 사전 훈련 작업 (Pretraining Tasks)
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder`의 순방향 추론은 입력 텍스트의 각 토큰과 삽입된 특수 토큰 “&lt;cls&gt;” 및 “&lt;seq&gt;”의 BERT 표현을 제공합니다.
다음으로, 우리는 이 표현들을 사용하여 BERT 사전 훈련을 위한 손실 함수를 계산할 것입니다.
사전 훈련은 마스킹된 언어 모델링(masked language modeling)과 다음 문장 예측(next sentence prediction)이라는 두 가지 작업으로 구성됩니다.

### [**마스킹된 언어 모델링 (Masked Language Modeling)**]
:label:`subsec_mlm`

:numref:`sec_language-model`에 설명된 대로,
언어 모델은 왼쪽의 문맥을 사용하여 토큰을 예측합니다.
각 토큰을 표현하기 위해 문맥을 양방향으로 인코딩하기 위해,
BERT는 무작위로 토큰을 마스킹하고 양방향 문맥의 토큰을 사용하여
자기 지도 방식으로 마스킹된 토큰을 예측합니다.
이 작업을 *마스킹된 언어 모델(masked language model)*이라고 합니다.

이 사전 훈련 작업에서,
토큰의 15%가 예측을 위한 마스킹된 토큰으로 무작위로 선택됩니다.
레이블을 사용하여 부정행위 없이 마스킹된 토큰을 예측하기 위해,
가장 간단한 접근 방식은 BERT 입력 시퀀스에서 항상 특수 “&lt;mask&gt;” 토큰으로 대체하는 것입니다.
그러나 인공적인 특수 토큰 “&lt;mask&gt;”는 미세 조정 단계에서는 절대 나타나지 않습니다.
사전 훈련과 미세 조정 사이의 이러한 불일치를 피하기 위해,
토큰이 예측을 위해 마스킹되는 경우(예: "this movie is great"에서 "great"이 마스킹되고 예측되도록 선택됨),
입력에서 다음과 같이 대체됩니다:

* 80%의 경우 특수 “&lt;mask&gt;” 토큰으로 대체됩니다(예: "this movie is great"가 "this movie is &lt;mask&gt;"가 됨).
* 10%의 경우 무작위 토큰으로 대체됩니다(예: "this movie is great"가 "this movie is drink"가 됨).
* 10%의 경우 변경되지 않은 원래 토큰으로 유지됩니다(예: "this movie is great"가 "this movie is great"가 됨).

15% 중 10%의 시간에는 무작위 토큰이 삽입된다는 점에 유의하십시오.
이러한 가끔 발생하는 노이즈는 BERT가 양방향 문맥 인코딩에서 마스킹된 토큰에 덜 편향되도록(특히 레이블 토큰이 변경되지 않은 상태로 유지될 때) 장려합니다.

우리는 BERT 사전 훈련의 마스킹된 언어 모델 작업에서 마스킹된 토큰을 예측하기 위해 다음 `MaskLM` 클래스를 구현합니다.
예측에는 1개의 은닉층 MLP(`self.mlp`)가 사용됩니다.
순방향 추론에서, 이는 `BERTEncoder`의 인코딩된 결과와 예측을 위한 토큰 위치라는 두 가지 입력을 받습니다.
출력은 해당 위치에서의 예측 결과입니다.

```{.python .input}
#@tab mxnet
#@save
class MaskLM(nn.Block):
    """BERT의 마스킹된 언어 모델 작업."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # `batch_size` = 2, `num_pred_positions` = 3이라고 가정하면,
        # `batch_idx`는 `np.array([0, 0, 0, 1, 1, 1])`입니다
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """BERT의 마스킹된 언어 모델 작업."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # `batch_size` = 2, `num_pred_positions` = 3이라고 가정하면,
        # `batch_idx`는 `torch.tensor([0, 0, 0, 1, 1, 1])`입니다
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

[**`MaskLM`의 순방향 추론**]을 보여주기 위해,
인스턴스 `mlm`을 생성하고 초기화합니다.
`BERTEncoder`의 순방향 추론에서 나온 `encoded_X`는 2개의 BERT 입력 시퀀스를 나타냄을 상기하십시오.
`mlm_positions`를 `encoded_X`의 BERT 입력 시퀀스 중 하나에서 예측할 3개의 인덱스로 정의합니다.
`mlm`의 순방향 추론은 `encoded_X`의 모든 마스킹된 위치 `mlm_positions`에서의 예측 결과 `mlm_Y_hat`을 반환합니다.
각 예측에 대해 결과의 크기는 어휘 크기와 같습니다.

```{.python .input}
#@tab mxnet
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

마스크 아래의 예측된 토큰 `mlm_Y_hat`의 정답 레이블 `mlm_Y`를 사용하여,
BERT 사전 훈련에서 마스킹된 언어 모델 작업의 크로스 엔트로피 손실을 계산할 수 있습니다.

```{.python .input}
#@tab mxnet
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### [**다음 문장 예측 (Next Sentence Prediction)**]
:label:`subsec_nsp`

마스킹된 언어 모델링은 단어를 표현하기 위해 양방향 문맥을 인코딩할 수 있지만,
텍스트 쌍 간의 논리적 관계를 명시적으로 모델링하지는 않습니다.
두 텍스트 시퀀스 간의 관계를 이해하는 데 도움을 주기 위해,
BERT는 사전 훈련에서 *다음 문장 예측*이라는 이진 분류 작업을 고려합니다.
사전 훈련을 위한 문장 쌍을 생성할 때,
절반은 실제로 연속된 문장이며 "True" 레이블이 붙고,
나머지 절반은 두 번째 문장이 코퍼스에서 무작위로 샘플링되며 "False" 레이블이 붙습니다.

다음 `NextSentencePred` 클래스는 1개의 은닉층 MLP를 사용하여
BERT 입력 시퀀스에서 두 번째 문장이 첫 번째 문장의 다음 문장인지 여부를 예측합니다.
트랜스포머 인코더의 셀프 어텐션으로 인해,
특수 토큰 “&lt;cls&gt;”의 BERT 표현은 입력의 두 문장을 모두 인코딩합니다.
따라서 MLP 분류기의 출력 레이어(`self.output`)는 `X`를 입력으로 받습니다.
여기서 `X`는 인코딩된 “&lt;cls&gt;” 토큰을 입력으로 하는 MLP 은닉층의 출력입니다.

```{.python .input}
#@tab mxnet
#@save
class NextSentencePred(nn.Block):
    """BERT의 다음 문장 예측 작업."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` 모양: (배치 크기, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """BERT의 다음 문장 예측 작업."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` 모양: (배치 크기, `num_hiddens`)
        return self.output(X)
```

우리는 [**`NextSentencePred` 인스턴스의 순방향 추론**]이
각 BERT 입력 시퀀스에 대해 이진 예측을 반환함을 알 수 있습니다.

```{.python .input}
#@tab mxnet
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch는 기본적으로 텐서를 평탄화하지 않지만, mxnet에서는 flatten=True인 경우
# 입력 데이터의 첫 번째 축을 제외한 모든 축이 함께 축소됩니다
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP의 input_shape: (배치 크기, `num_hiddens`)
nsp = NextSentencePred()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

2개의 이진 분류에 대한 크로스 엔트로피 손실도 계산할 수 있습니다.

```{.python .input}
#@tab mxnet
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

앞서 언급한 두 사전 훈련 작업의 모든 레이블은
수동 라벨링 노력 없이 사전 훈련 코퍼스에서 사소하게 얻을 수 있다는 점이 주목할 만합니다.
원래 BERT는 BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015`와 영문 위키피디아의 연결에 대해 사전 훈련되었습니다.
이 두 텍스트 코퍼스는 거대합니다:
각각 8억 단어와 25억 단어를 가지고 있습니다.


## [**종합하기 (Putting It All Together)**]

BERT를 사전 훈련할 때, 최종 손실 함수는
마스킹된 언어 모델링과 다음 문장 예측에 대한 손실 함수의 선형 결합입니다.
이제 우리는 `BERTEncoder`, `MaskLM`, `NextSentencePred`의 세 가지 클래스를 인스턴스화하여 `BERTModel` 클래스를 정의할 수 있습니다.
순방향 추론은 인코딩된 BERT 표현 `encoded_X`,
마스킹된 언어 모델링의 예측 `mlm_Y_hat`,
그리고 다음 문장 예측 `nsp_Y_hat`을 반환합니다.

```{.python .input}
#@tab mxnet
#@save
class BERTModel(nn.Block):
    """BERT 모델."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 다음 문장 예측을 위한 MLP 분류기의 은닉층.
        # 0은 '<cls>' 토큰의 인덱스입니다
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """BERT 모델."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 다음 문장 예측을 위한 MLP 분류기의 은닉층.
        # 0은 '<cls>' 토큰의 인덱스입니다
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 요약 (Summary)

* word2vec 및 GloVe와 같은 단어 임베딩 모델은 문맥 독립적입니다. 단어의 문맥에 관계없이(있는 경우) 동일한 단어에 동일한 사전 훈련된 벡터를 할당합니다. 자연어의 다의어나 복잡한 의미론을 잘 처리하기 어렵습니다.
* ELMo 및 GPT와 같은 문맥 의존적 단어 표현의 경우, 단어의 표현은 문맥에 따라 달라집니다.
* ELMo는 문맥을 양방향으로 인코딩하지만 작업별 아키텍처를 사용합니다(그러나 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 실제로 쉽지 않습니다). 반면 GPT는 작업 불가지론적이지만 문맥을 왼쪽에서 오른쪽으로 인코딩합니다.
* BERT는 두 세계의 장점을 결합합니다: 문맥을 양방향으로 인코딩하고 광범위한 자연어 처리 작업에 대해 최소한의 아키텍처 변경만 요구합니다.
* BERT 입력 시퀀스의 임베딩은 토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합입니다.
* BERT 사전 훈련은 마스킹된 언어 모델링과 다음 문장 예측이라는 두 가지 작업으로 구성됩니다. 전자는 단어를 표현하기 위해 양방향 문맥을 인코딩할 수 있으며, 후자는 텍스트 쌍 간의 논리적 관계를 명시적으로 모델링합니다.


## 연습 문제 (Exercises)

1. 다른 모든 조건이 동일하다면, 마스킹된 언어 모델은 왼쪽에서 오른쪽으로 진행하는 언어 모델보다 수렴하는 데 더 많은 사전 훈련 단계가 필요합니까 아니면 더 적게 필요합니까? 그 이유는 무엇입니까?
2. BERT의 원래 구현에서, `BERTEncoder`의 포지션와이즈 피드 포워드 네트워크(`d2l.TransformerEncoderBlock`를 통해)와 `MaskLM`의 완전 연결 레이어는 모두 활성화 함수로 GELU(Gaussian error linear unit) :cite:`Hendrycks.Gimpel.2016`를 사용합니다. GELU와 ReLU의 차이점을 조사하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab: