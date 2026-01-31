```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 기계 번역과 데이터셋 (Machine Translation and the Dataset)
:label:`sec_machine_translation`

현대 RNN에 대한 광범위한 관심을 불러일으킨 주요 돌파구 중 하나는 통계적 *기계 번역* 응용 분야에서의 큰 발전이었습니다. 
여기서 모델은 한 언어의 문장을 제시받고 다른 언어의 해당 문장을 예측해야 합니다. 
여기서 문장의 길이는 서로 다를 수 있으며, 두 언어의 문법 구조 차이로 인해 두 문장의 해당 단어가 동일한 순서로 나타나지 않을 수 있음에 유의하십시오.


많은 문제들이 이러한 두 "정렬되지 않은" 시퀀스 간의 매핑 성격을 가지고 있습니다.
예를 들어 대화 프롬프트에서 응답으로, 또는 질문에서 답변으로의 매핑이 그 예입니다.
광범위하게 이러한 문제를 *시퀀스-투-시퀀스(sequence-to-sequence, seq2seq)* 문제라고 하며, 이 장의 나머지 부분과 :numref:`chap_attention-and-transformers`의 상당 부분에서 우리의 초점이 됩니다.

이 섹션에서는 기계 번역 문제와 후속 예제에서 사용할 예제 데이터셋을 소개합니다.
수십 년 동안 언어 간 번역의 통계적 정식화가 인기 있었으며 :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`,
연구자들이 신경망 접근 방식을 작동시키기 전에도 마찬가지였습니다(방법들은 종종 *신경 기계 번역*이라는 용어 아래 하나로 묶였습니다).


먼저 데이터를 처리하기 위한 새로운 코드가 필요합니다.
:numref:`sec_language-model`에서 본 언어 모델링과 달리,
여기서 각 예제는 두 개의 별도 텍스트 시퀀스, 즉 소스 언어의 시퀀스와 타겟 언어의 시퀀스(번역)로 구성됩니다.
다음 코드 스니펫은 훈련을 위해 전처리된 데이터를 미니배치로 로드하는 방법을 보여줍니다.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**데이터셋 다운로드 및 전처리**]

시작하기 위해, [Tatoeba 프로젝트의 이국어 문장 쌍](http://www.manythings.org/anki/)으로 구성된 영어-프랑스어 데이터셋을 다운로드합니다.
데이터셋의 각 줄은 영어 텍스트 시퀀스(*소스*)와 번역된 프랑스어 텍스트 시퀀스(*타겟*)로 구성된 탭 구분 쌍입니다.
각 텍스트 시퀀스는 단 한 문장일 수도 있고, 여러 문장으로 된 단락일 수도 있음에 유의하십시오.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """영어-프랑스어 데이터셋."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

데이터셋을 다운로드한 후,
원시 텍스트 데이터에 대해 [**여러 전처리 단계를 진행**]합니다.
예를 들어 줄 바꿈 없는 공백(non-breaking space)을 일반 공백으로 바꾸고,
대문자를 소문자로 변환하며,
단어와 구두점 사이에 공백을 삽입합니다.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # 줄 바꿈 없는 공백을 일반 공백으로 교체
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # 단어와 구두점 사이에 공백 삽입
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**토큰화 (Tokenization)**]

:numref:`sec_language-model`의 문자 수준 토큰화와 달리,
기계 번역을 위해 여기서는 단어 수준 토큰화를 선호합니다
(오늘날의 최첨단 모델은 더 복잡한 토큰화 기술을 사용합니다). 
다음 `_tokenize` 메서드는 처음 `max_examples`개의 텍스트 시퀀스 쌍을 토큰화하며,
여기서 각 토큰은 단어이거나 구두점입니다. 
시퀀스의 끝을 나타내기 위해 모든 시퀀스 끝에 특수 “&lt;eos&gt;” 토큰을 추가합니다. 
모델이 토큰별로 시퀀스를 생성하여 예측할 때, “&lt;eos&gt;” 토큰의 생성은 출력 시퀀스가 완료되었음을 시사할 수 있습니다. 
마지막으로 아래 메서드는 `src`와 `tgt`라는 두 개의 토큰 리스트의 리스트를 반환합니다.
구체적으로 `src[i]`는 소스 언어(여기서는 영어)의 $i^\textrm{th}$번째 텍스트 시퀀스의 토큰 리스트이고, `tgt[i]`는 타겟 언어(여기서는 프랑스어)의 토큰 리스트입니다.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # 빈 토큰 건너뛰기
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

[**텍스트 시퀀스당 토큰 수의 히스토그램을 그려봅시다.**]
이 간단한 영어-프랑스어 데이터셋에서 대부분의 텍스트 시퀀스는 20개 미만의 토큰을 가집니다.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """리스트 길이 쌍에 대한 히스토그램을 그립니다."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## 고정 길이 시퀀스 로드하기 (Loading Sequences of Fixed Length)
:label:`subsec_loading-seq-fixed-len`

언어 모델링에서 [**각 예제 시퀀스**](한 문장의 세그먼트이거나 여러 문장에 걸친 범위)가 (**고정된 길이를 가졌음**)을 상기하십시오.
이는 :numref:`sec_language-model`의 `num_steps`(타임 스텝 또는 토큰의 수) 인수에 의해 지정되었습니다. 
기계 번역에서 각 예제는 소스 및 타겟 텍스트 시퀀스의 쌍이며, 여기서 두 텍스트 시퀀스는 길이가 다를 수 있습니다.

계산 효율성을 위해,
우리는 여전히 *자르기(truncation)*와 *패딩(padding)*을 통해 한 번에 텍스트 시퀀스의 미니배치를 처리할 수 있습니다. 
동일한 미니배치의 모든 시퀀스가 동일한 길이 `num_steps`를 가져야 한다고 가정합시다. 
텍스트 시퀀스에 `num_steps`보다 적은 토큰이 있는 경우, 길이가 `num_steps`에 도달할 때까지 끝에 특수 "&lt;pad&gt;" 토큰을 계속 추가합니다. 
그렇지 않으면 처음 `num_steps`개의 토큰만 취하고 나머지는 버림으로써 텍스트 시퀀스를 자릅니다. 
이런 식으로 모든 텍스트 시퀀스는 동일한 모양의 미니배치로 로드될 수 있도록 동일한 길이를 갖게 됩니다. 
더욱이 패딩 토큰을 제외한 소스 시퀀스의 길이도 기록합니다. 
이 정보는 나중에 다룰 일부 모델에서 필요할 것입니다.


기계 번역 데이터셋은 언어 쌍으로 구성되므로,
소스 언어와 타겟 언어 각각에 대해 두 개의 어휘를 별도로 구축할 수 있습니다. 
단어 수준 토큰화를 사용하면 어휘 크기가 문자 수준 토큰화를 사용할 때보다 훨씬 더 커집니다. 
이를 완화하기 위해 여기서는 두 번 미만으로 나타나는 드문 토큰을 동일한 알 수 없는("&lt;unk&gt;") 토큰으로 취급합니다. 
나중에 설명하겠지만(:numref:`fig_seq2seq`), 
타겟 시퀀스로 훈련할 때 디코더 출력(레이블 토큰)은 한 토큰만큼 이동된 동일한 디코더 입력(타겟 토큰)일 수 있으며,
특수 문장 시작("&lt;bos&gt;") 토큰이 타겟 시퀀스를 예측하기 위한 첫 번째 입력 토큰으로 사용될 것입니다(:numref:`fig_seq2seq_predict`).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**데이터셋 읽기**]

마지막으로 데이터 반복자를 반환하기 위해 `get_dataloader` 메서드를 정의합니다.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

영어-프랑스어 데이터셋에서 [**첫 번째 미니배치를 읽어봅시다.**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

위의 `_build_arrays` 메서드에 의해 처리된 소스 및 타겟 시퀀스 쌍을 보여줍니다(문자열 형식).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## 요약 (Summary)

자연어 처리에서 *기계 번역*이란 *소스* 언어의 텍스트 문자열을 나타내는 시퀀스에서 *타겟* 언어의 그럴듯한 번역을 나타내는 문자열로 자동으로 매핑하는 작업을 말합니다. 단어 수준 토큰화를 사용하면 어휘 크기가 문자 수준 토큰화를 사용할 때보다 훨씬 커지지만 시퀀스 길이는 훨씬 짧아집니다. 큰 어휘 크기를 완화하기 위해 드문 토큰을 어떤 "알 수 없는" 토큰으로 취급할 수 있습니다. 텍스트 시퀀스를 자르고 패딩하여 모든 시퀀스가 미니배치로 로드될 수 있도록 동일한 길이를 갖게 할 수 있습니다. 현대적인 구현에서는 종종 패딩에 대한 과도한 계산 낭비를 피하기 위해 비슷한 길이의 시퀀스를 버킷팅(bucket)합니다. 


## 연습 문제 (Exercises)

1. `_tokenize` 메서드에서 `max_examples` 인수의 다양한 값을 시도해 보십시오. 이것이 소스 언어와 타겟 언어의 어휘 크기에 어떤 영향을 미칩니까?
2. 중국어와 일본어 같은 일부 언어의 텍스트에는 단어 경계 표시(예: 공백)가 없습니다. 그러한 경우에도 단어 수준 토큰화가 여전히 좋은 아이디어일까요? 왜 그런가요 혹은 왜 아닌가요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18020)
:end_tab:

```