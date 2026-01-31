# 단어 유사성과 유추 (Word Similarity and Analogy)
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining`에서,
우리는 작은 데이터셋에서 word2vec 모델을 훈련하고,
입력 단어에 대해 의미적으로 유사한 단어를 찾는 데 적용했습니다.
실제로,
대규모 코퍼스에서 사전 훈련된 단어 벡터는
:numref:`chap_nlp_app`에서 나중에 다룰
다운스트림 자연어 처리 작업에 적용될 수 있습니다.
대규모 코퍼스에서 사전 훈련된 단어 벡터의 의미를
직관적인 방식으로 입증하기 위해,
단어 유사성 및 유추 작업에 적용해 봅시다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## 사전 훈련된 단어 벡터 로드 (Loading Pretrained Word Vectors)

아래에는 [GloVe 웹사이트](https://nlp.stanford.edu/projects/glove/)에서 다운로드할 수 있는 50, 100, 300차원의 사전 훈련된 GloVe 임베딩이 나열되어 있습니다.
사전 훈련된 fastText 임베딩은 여러 언어로 제공됩니다.
여기서는 [fastText 웹사이트](https://fasttext.cc/)에서 다운로드할 수 있는 영어 버전(300차원 "wiki.en") 하나를 고려합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

이러한 사전 훈련된 GloVe 및 fastText 임베딩을 로드하기 위해, 다음 `TokenEmbedding` 클래스를 정의합니다.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe 웹사이트: https://nlp.stanford.edu/projects/glove/
        # fastText 웹사이트: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 헤더 정보 건너뛰기 (예: fastText의 첫 번째 행)
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

아래에서는 (위키피디아 하위 집합에서 사전 훈련된)
50차원 GloVe 임베딩을 로드합니다.
`TokenEmbedding` 인스턴스를 생성할 때,
지정된 임베딩 파일이 아직 다운로드되지 않았다면 다운로드해야 합니다.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

어휘 크기를 출력합니다. 어휘에는 400,000개의 단어(토큰)와 특수 알 수 없는 토큰이 포함되어 있습니다.

```{.python .input}
#@tab all
len(glove_6b50d)
```

어휘에서 단어의 인덱스를 얻거나 그 반대로 할 수 있습니다.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 사전 훈련된 단어 벡터 적용 (Applying Pretrained Word Vectors)

로드된 GloVe 벡터를 사용하여,
다음 단어 유사성 및 유추 작업에 적용하여 그 의미를 보여줄 것입니다.


### 단어 유사성 (Word Similarity)

:numref:`subsec_apply-word-embed`와 유사하게,
단어 벡터 간의 코사인 유사도를 기반으로
입력 단어에 대해 의미적으로 유사한 단어를 찾기 위해,
다음 `knn` ($k$-최근접 이웃) 함수를 구현합니다.

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # 수치적 안정성을 위해 1e-9를 더합니다
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # 수치적 안정성을 위해 1e-9를 더합니다
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

그런 다음 `TokenEmbedding` 인스턴스 `embed`에서
사전 훈련된 단어 벡터를 사용하여
유사한 단어를 검색합니다.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 입력 단어 제외
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`의 사전 훈련된 단어 벡터 어휘에는 400,000개의 단어와 특수 알 수 없는 토큰이 포함되어 있습니다.
입력 단어와 알 수 없는 토큰을 제외하고,
이 어휘 중에서
"chip"이라는 단어와 의미적으로 가장 유사한
세 단어를 찾아봅시다.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

아래는 "baby"와 "beautiful"에 유사한 단어를 출력합니다.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 단어 유추 (Word Analogy)

유사한 단어를 찾는 것 외에도,
단어 벡터를 단어 유추 작업에 적용할 수도 있습니다.
예를 들어,
“man”:“woman”::“son”:“daughter”는
단어 유추의 형태입니다:
“man”이 “woman”에 해당하는 것은 “son”이 “daughter”에 해당하는 것과 같습니다.
구체적으로,
단어 유추 완성 작업은 다음과 같이 정의할 수 있습니다:
단어 유추 $a : b :: c : d$에 대해, 처음 세 단어 $a$, $b$, $c$가 주어졌을 때 $d$를 찾습니다.
단어 $w$의 벡터를 $	extrm{vec}(w)$라고 표시합시다.
유추를 완성하기 위해,
우리는 벡터가 $	extrm{vec}(c)+	extrm{vec}(b)-	extrm{vec}(a)$의 결과와
가장 유사한 단어를 찾을 것입니다.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 알 수 없는 단어 제거
```

로드된 단어 벡터를 사용하여 "male-female" 유추를 확인해 봅시다.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

아래는 “capital-country” 유추를 완성합니다:
“beijing”:“china”::“tokyo”:“japan”.
이것은 사전 훈련된 단어 벡터의 의미론을 보여줍니다.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

“bad”:“worst”::“big”:“biggest”와 같은
“adjective-superlative adjective” 유추의 경우,
사전 훈련된 단어 벡터가 구문 정보를 캡처할 수 있음을 알 수 있습니다.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

사전 훈련된 단어 벡터에서 캡처된 과거 시제 개념을 보여주기 위해,
"present tense-past tense" 유추를 사용하여 구문을 테스트할 수 있습니다: “do”:“did”::“go”:“went”.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 요약 (Summary)

* 실제로 대규모 코퍼스에서 사전 훈련된 단어 벡터는 다운스트림 자연어 처리 작업에 적용될 수 있습니다.
* 사전 훈련된 단어 벡터는 단어 유사성 및 유추 작업에 적용될 수 있습니다.


## 연습 문제 (Exercises)

1. `TokenEmbedding('wiki.en')`을 사용하여 fastText 결과를 테스트하십시오.
2. 어휘가 매우 클 때, 유사한 단어를 찾거나 단어 유추를 더 빨리 완료하려면 어떻게 해야 합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab: