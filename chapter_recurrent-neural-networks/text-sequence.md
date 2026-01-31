# 원시 텍스트를 시퀀스 데이터로 변환하기 (Converting Raw Text into Sequence Data)
:label:`sec_text-sequence`

이 책 전체에서 우리는 종종 단어, 문자 또는 단어 조각의 시퀀스로 표현되는 텍스트 데이터로 작업할 것입니다. 
시작하려면 원시 텍스트를 적절한 형식의 시퀀스로 변환하기 위한 몇 가지 기본 도구가 필요합니다. 
일반적인 전처리 파이프라인은 다음 단계를 실행합니다:

1. 텍스트를 문자열로 메모리에 로드합니다.
2. 문자열을 토큰(예: 단어 또는 문자)으로 분할합니다.
3. 각 어휘 요소를 수치 인덱스와 연관시키는 어휘 딕셔너리를 구축합니다.
4. 텍스트를 수치 인덱스 시퀀스로 변환합니다.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## 데이터셋 읽기 (Reading the Dataset)

여기서는 30,000개 이상의 단어가 포함된 H. G. Wells의 [타임 머신(The Time Machine)](http://www.gutenberg.org/ebooks/35) 책으로 작업할 것입니다. 
실제 응용 프로그램은 일반적으로 훨씬 더 큰 데이터셋을 포함하지만, 이것은 전처리 파이프라인을 보여주기에 충분합니다. 
다음 `_download` 메서드는 (**원시 텍스트를 문자열로 읽습니다**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """타임 머신 데이터셋."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

단순함을 위해 원시 텍스트를 전처리할 때 구두점과 대문자를 무시합니다.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## 토큰화 (Tokenization)

*토큰*은 텍스트의 원자적(나눌 수 없는) 단위입니다. 
각 타임 스텝은 1개의 토큰에 해당하지만, 정확히 무엇이 토큰을 구성하는지는 설계 선택입니다. 
예를 들어 "Baby needs a new pair of shoes"라는 문장을 7개의 단어 시퀀스로 표현할 수 있으며, 여기서 모든 단어의 집합은 큰 어휘(일반적으로 수만 또는 수십만 단어)를 구성합니다. 
또는 훨씬 더 작은 어휘(고유한 ASCII 문자는 256개뿐임)를 사용하여 동일한 문장을 훨씬 더 긴 30개의 문자 시퀀스로 표현할 수 있습니다. 
아래에서는 전처리된 텍스트를 문자 시퀀스로 토큰화합니다.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## 어휘 (Vocabulary)

이 토큰들은 여전히 문자열입니다. 
그러나 우리 모델에 대한 입력은 궁극적으로 수치 입력으로 구성되어야 합니다. 
[**다음으로 *어휘(vocabularies)*, 즉 각 고유한 토큰 값을 고유한 인덱스와 연관시키는 객체를 구축하기 위한 클래스를 소개합니다.**] 
먼저 훈련 *말뭉치(corpus)*에서 고유한 토큰 집합을 결정합니다. 
그런 다음 각 고유 토큰에 수치 인덱스를 할당합니다. 
드문 어휘 요소는 편의를 위해 종종 삭제됩니다. 
훈련 또는 테스트 시 이전에 보지 못했거나 어휘에서 삭제된 토큰을 만날 때마다, 
이것이 *알 수 없는* 값임을 나타내는 특수 "<unk>" 토큰으로 표현합니다.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """텍스트용 어휘."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # 필요한 경우 2D 리스트를 평탄화
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # 토큰 빈도 계산
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # 고유 토큰 리스트
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # 알 수 없는 토큰의 인덱스
        return self.token_to_idx['<unk>']
```

이제 데이터셋에 대한 [**어휘를 구축**]하여 문자열 시퀀스를 수치 인덱스 리스트로 변환합니다. 
정보를 잃지 않았으며 데이터셋을 원래(문자열) 표현으로 쉽게 다시 변환할 수 있다는 점에 유의하십시오.

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## 모두 합치기 (Putting It All Together)

위의 클래스와 메서드를 사용하여, 
우리는 [**모든 것을 `TimeMachine` 클래스의 다음 `build` 메서드에 패키징**]합니다. 이 메서드는 토큰 인덱스 리스트인 `corpus`와 *타임 머신* 말뭉치의 어휘인 `vocab`을 반환합니다. 
여기서 수행한 수정 사항은 다음과 같습니다: 
(i) 이후 섹션의 훈련을 단순화하기 위해 텍스트를 단어가 아닌 문자로 토큰화합니다; 
(ii) *타임 머신* 데이터셋의 각 텍스트 라인이 반드시 문장이나 단락인 것은 아니므로 `corpus`는 토큰 리스트의 리스트가 아니라 단일 리스트입니다.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## 탐색적 언어 통계 (Exploratory Language Statistics)
:label:`subsec_natural-lang-stat`

실제 말뭉치와 단어에 대해 정의된 `Vocab` 클래스를 사용하여 말뭉치에서의 단어 사용에 관한 기본 통계를 검사할 수 있습니다. 
아래에서는 *타임 머신*에서 사용된 단어로 어휘를 구축하고 가장 자주 발생하는 10개 단어를 인쇄합니다.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

(**가장 자주 사용되는 10개 단어**)가 그리 설명적이지 않다는 점에 유의하십시오. 
우리가 무작위로 아무 책이나 선택했더라도 매우 유사한 목록을 보게 될 것이라고 상상할 수도 있습니다. 
"the"와 "a" 같은 관사, "i"와 "my" 같은 대명사, "of", "to", "in" 같은 전치사는 일반적인 구문론적 역할을 수행하기 때문에 자주 발생합니다. 
흔하지만 특별히 설명적이지 않은 이러한 단어들을 종종 (***불용어(stop words)***)라고 하며, 
소위 BoW(bag-of-words) 표현에 기반한 이전 세대의 텍스트 분류기에서는 가장 자주 필터링되었습니다. 
그러나 현대의 RNN 및 Transformer 기반 신경망 모델로 작업할 때는 의미를 전달하므로 필터링할 필요가 없습니다. 
목록을 더 아래로 내려다보면 단어 빈도가 빠르게 감소한다는 것을 알 수 있습니다. 
$10^{\textrm{th}}$로 가장 빈번한 단어는 가장 인기 있는 단어보다 $1/5$ 미만으로 흔합니다. 
단어 빈도는 순위가 내려갈수록 거듭제곱 법칙 분포(구체적으로는 Zipfian)를 따르는 경향이 있습니다. 
더 나은 아이디어를 얻기 위해 [**단어 빈도 그림을 그립니다**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

처음 몇 단어를 예외로 처리하면, 나머지 모든 단어는 로그-로그 플롯에서 대략 직선을 따릅니다. 
이 현상은 *Zipf의 법칙*으로 포착됩니다. 
$i^\textrm{th}$로 가장 빈번한 단어의 빈도 $n_i$는 다음과 같습니다:

$$n_i \propto \frac{1}{i^\alpha},$$ 
:eqlabel:`eq_zipf_law`

이는 다음과 같습니다:

$$\log n_i = - \alpha \log i + c,$$ 

여기서 $\alpha$는 분포를 특징짓는 지수이고 $c$는 상수입니다. 
통계를 세어 단어를 모델링하고 싶다면 이것은 이미 우리에게 생각할 거리를 줍니다. 
결국 우리는 드문 단어라고도 알려진 꼬리 부분의 빈도를 상당히 과대평가하게 될 것입니다. 하지만 [**두 개의 연속 단어(bigrams), 세 개의 연속 단어(trigrams)와 같은 다른 단어 조합은 어떻습니까?**], 그리고 그 이상은요? 
bigram 빈도가 단일 단어(unigram) 빈도와 같은 방식으로 동작하는지 봅시다.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

한 가지 주목할 점이 있습니다. 가장 빈번한 10개의 단어 쌍 중 9개는 불용어로 구성되어 있고 실제 책과 관련된 것은 단 하나, "the time"뿐입니다. 게다가 trigram 빈도가 같은 방식으로 동작하는지 봅시다.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:]))]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

이제 unigrams, bigrams, trigrams 이 세 모델 간의 [**토큰 빈도를 시각화**]해 봅시다.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

이 그림은 꽤 흥미진진합니다. 
첫째, unigram 단어를 넘어 단어 시퀀스도 시퀀스 길이에 따라 :eqref:`eq_zipf_law`에서 더 작은 지수 $\alpha$를 갖지만 Zipf의 법칙을 따르는 것으로 보입니다. 
둘째, 고유한 $n$-gram의 수가 그리 크지 않습니다. 
이것은 언어에 꽤 많은 구조가 있다는 희망을 줍니다. 
셋째, 많은 $n$-gram이 매우 드물게 발생합니다. 
이것은 특정 방법을 언어 모델링에 부적합하게 만들고 딥러닝 모델의 사용에 동기를 부여합니다. 
우리는 다음 섹션에서 이에 대해 논의할 것입니다.


## 요약 (Summary)

텍스트는 딥러닝에서 접하는 가장 일반적인 형태의 시퀀스 데이터 중 하나입니다. 
토큰을 구성하는 일반적인 선택은 문자, 단어, 단어 조각입니다. 
텍스트를 전처리하기 위해 우리는 보통 (i) 텍스트를 토큰으로 분할하고; (ii) 토큰 문자열을 수치 인덱스로 매핑하기 위한 어휘를 구축하고; (iii) 모델이 조작할 수 있도록 텍스트 데이터를 토큰 인덱스로 변환합니다. 
실제로 단어의 빈도는 Zipf의 법칙을 따르는 경향이 있습니다. 이것은 개별 단어(unigrams)뿐만 아니라 $n$-grams에도 해당됩니다.


## 연습 문제 (Exercises)

1. 이 섹션의 실험에서 텍스트를 단어로 토큰화하고 `Vocab` 인스턴스의 `min_freq` 인수 값을 변경하십시오. `min_freq`의 변경이 결과 어휘의 크기에 미치는 영향을 정성적으로 설명하십시오.
2. 이 말뭉치에서 unigrams, bigrams, trigrams에 대한 Zipfian 분포의 지수를 추정하십시오.
3. 다른 데이터 소스를 찾으십시오(표준 머신러닝 데이터셋 다운로드, 다른 퍼블릭 도메인 책 선택, 웹사이트 스크래핑 등). 각각에 대해 단어 및 문자 수준 모두에서 데이터를 토큰화하십시오. `min_freq`의 동등한 값에서 어휘 크기가 *타임 머신* 말뭉치와 어떻게 비교됩니까. 이 말뭉치에 대한 unigram 및 bigram 분포에 해당하는 Zipfian 분포의 지수를 추정하십시오. *타임 머신* 말뭉치에서 관찰한 값과 어떻게 비교됩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18011)
:end_tab: