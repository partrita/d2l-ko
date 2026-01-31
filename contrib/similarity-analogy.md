# 단어 유사성과 유추 찾기 (Finding Synonyms and Analogies)
:label:`sec_synonyms`

:numref:`sec_word2vec_gluon`에서 우리는 소규모 데이터셋에서 word2vec 단어 임베딩 모델을 훈련하고 단어 벡터의 코사인 유사도를 사용하여 유의어를 검색했습니다. 실제로 대규모 코퍼스에서 사전 훈련된 단어 벡터는 다운스트림 자연어 처리 작업에 종종 적용될 수 있습니다. 이 섹션에서는 이러한 사전 훈련된 단어 벡터를 사용하여 유의어와 유추를 찾는 방법을 보여줄 것입니다. 우리는 후속 섹션에서 사전 훈련된 단어 벡터를 계속해서 적용할 것입니다.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
import matplotlib.pyplot as plt
from mxnet import np, npx
import numpy
import os
from sklearn.decomposition import PCA

npx.set_np()
```

## 사전 훈련된 단어 벡터 사용하기 (Using Pretrained Word Vectors)

아래에는 [GloVe 웹사이트](https://nlp.stanford.edu/projects/glove/)에서 다운로드할 수 있는 50, 100, 300차원의 사전 훈련된 GloVe 임베딩이 나열되어 있습니다. 사전 훈련된 fastText 임베딩은 여러 언어로 제공됩니다. 여기서는 [fastText 웹사이트](https://fasttext.cc/)에서 다운로드할 수 있는 한 가지 영어 버전(300차원 "wiki.en")을 고려합니다.

```{.python .input  n=2}
#@tab mxnet
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

위의 사전 훈련된 GloVe 및 fastText 임베딩을 로드하기 위해 다음 `TokenEmbedding` 클래스를 정의합니다.

```{.python .input  n=3}
#@tab mxnet
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
                # fastText의 상단 행과 같은 헤더 정보 건너뛰기
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, np.array(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[np.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

다음으로, 위키피디아의 하위 집합에서 사전 훈련된 50차원 GloVe 임베딩을 사용합니다. 해당 단어 임베딩은 사전 훈련된 단어 임베딩 인스턴스를 처음 생성할 때 자동으로 다운로드됩니다.

```{.python .input  n=4}
#@tab mxnet
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

사전 크기를 출력합니다. 사전에는 400,000개의 단어와 특수한 알 수 없는 토큰이 포함되어 있습니다.

```{.python .input  n=5}
#@tab mxnet
len(glove_6b50d)
```

단어를 사용하여 사전에서의 인덱스를 얻거나, 인덱스로부터 단어를 얻을 수 있습니다.

```{.python .input  n=6}
#@tab mxnet
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 사전 훈련된 단어 벡터 적용하기 (Applying Pretrained Word Vectors)

아래에서는 GloVe를 예로 들어 사전 훈련된 단어 벡터의 응용을 보여줍니다.

### 유의어 찾기 (Finding Synonyms)

여기서는 :numref:`sec_word2vec`에서 소개된 코사인 유사도에 의한 유의어 검색 알고리즘을 다시 구현합니다.

유추를 찾을 때 $k$-최근접 이웃을 찾는 로직을 재사용하기 위해, 이 로직의 일부를 `knn` ($k$-최근접 이웃) 함수에 별도로 캡슐화합니다.

```{.python .input  n=7}
#@tab mxnet
def knn(W, x, k):
    # 수치 안정성을 위해 1e-9를 더합니다
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

그런 다음, 사전 훈련된 단어 벡터 인스턴스 `embed`를 통해 유의어를 검색합니다.

```{.python .input  n=8}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 입력 단어 제거
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

이미 생성된 사전 훈련된 단어 벡터 인스턴스 `glove_6b50d`의 사전에는 400,000개의 단어와 특수한 알 수 없는 토큰이 포함되어 있습니다. 입력 단어와 알 수 없는 단어를 제외하고, "chip"과 의미가 가장 유사한 세 단어를 검색합니다.

```{.python .input  n=9}
#@tab mxnet
get_similar_tokens('chip', 3, glove_6b50d)
```

다음으로, "baby"와 "beautiful"의 유의어를 검색합니다.

```{.python .input  n=10}
#@tab mxnet
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input  n=11}
#@tab mxnet
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 유추 찾기 (Finding Analogies)

유의어를 찾는 것 외에도, 사전 훈련된 단어 벡터를 사용하여 단어 간의 유추를 찾을 수도 있습니다. 예를 들어, “man”:“woman”::“son”:“daughter”는 유추의 한 예로, “man”에 대한 “woman”의 관계는 “son”에 대한 “daughter”의 관계와 같습니다. 유추를 찾는 문제는 다음과 같이 정의될 수 있습니다: 유추 관계 $a : b :: c : d$에 있는 네 단어에 대해, 처음 세 단어 $a$, $b$, $c$가 주어졌을 때 $d$를 찾고자 합니다. 단어 $w$에 대한 단어 벡터를 $	ext{vec}(w)$라고 가정합시다. 유추 문제를 해결하기 위해, $	ext{vec}(c)+	ext{vec}(b)-	ext{vec}(a)$의 결과 벡터와 가장 유사한 단어 벡터를 찾아야 합니다.

```{.python .input  n=12}
#@tab mxnet
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 알 수 없는 단어 제거
```

"male-female" 유추를 확인합니다.

```{.python .input  n=13}
#@tab mxnet
get_analogy('man', 'woman', 'son', glove_6b50d)
```

“수도-국가” 유추: "beijing"과 "china"의 관계는 "tokyo"와 무엇의 관계와 같을까요? 답은 "japan"이어야 합니다.

```{.python .input  n=14}
#@tab mxnet
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"형용사-최상급 형용사" 유추: "bad"와 "worst"의 관계는 "big"과 무엇의 관계와 같을까요? 답은 "biggest"여야 합니다.

```{.python .input  n=15}
#@tab mxnet
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

"현재 시제 동사-과거 시제 동사" 유추: "do"와 "did"의 관계는 "go"와 무엇의 관계와 같을까요? 답은 "went"여야 합니다.

```{.python .input  n=16}
#@tab mxnet
get_analogy('do', 'did', 'go', glove_6b50d)
```

```{.python .input  n=51}
#@tab mxnet
def visualization(token_pairs, embed):
    plt.figure(figsize=(7, 5))
    vecs = np.concatenate([embed[pair] for pair in token_pairs])
    vecs_pca = PCA(n_components=2).fit_transform(numpy.array(vecs))
    for i, pair in enumerate(token_pairs):
        x1, y1 = vecs_pca[2 * i]
        x2, y2 = vecs_pca[2 * i + 1]
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.annotate(pair[0], xy=(x1, y1))
        plt.annotate(pair[1], xy=(x2, y2))
        plt.plot([x1, x2], [y1, y2])
    plt.show()
```

```{.python .input  n=57}
#@tab mxnet
token_pairs = [['man', 'woman'], ['son', 'daughter'], ['king', 'queen'],
              ['uncle', 'aunt'], ['sir', 'madam'], ['sister', 'brother']]
visualization(token_pairs, glove_6b50d)
```

## 요약 (Summary)

* 대규모 코퍼스에서 사전 훈련된 단어 벡터는 종종 다운스트림 자연어 처리 작업에 적용될 수 있습니다.
* 사전 훈련된 단어 벡터를 사용하여 유의어와 유추를 찾을 수 있습니다.


## 연습 문제 (Exercises)

1. `TokenEmbedding('wiki.en')`을 사용하여 fastText 결과를 테스트하십시오.
2. 사전이 매우 클 때, 유의어와 유추를 찾는 것을 어떻게 가속화할 수 있을까요?


## [토론](https://discuss.mxnet.io/t/2390)

![](../img/qr_similarity-analogy.svg)