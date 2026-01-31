# 하위 단어 임베딩 (Subword Embedding)
:label:`sec_fasttext`

영어에서,
"helps", "helped", "helping"과 같은 단어들은
동일한 단어 "help"의 굴절 형태입니다.
"dog"와 "dogs" 사이의 관계는
"cat"과 "cats" 사이의 관계와 같고,
"boy"와 "boyfriend" 사이의 관계는
"girl"과 "girlfriend" 사이의 관계와 같습니다.
프랑스어나 스페인어와 같은 다른 언어에서는,
많은 동사가 40개 이상의 굴절 형태를 가지며,
핀란드어에서는 명사가 최대 15개의 격(case)을 가질 수 있습니다.
언어학에서 형태론(morphology)은 단어 형성과 단어 관계를 연구합니다.
그러나
단어의 내부 구조는
word2vec에서도 GloVe에서도 탐구되지 않았습니다.

## fastText 모델 (The fastText Model)

word2vec에서 단어가 어떻게 표현되는지 상기해 보십시오.
스킵-그램 모델과 CBOW 모델 모두에서,
동일한 단어의 다른 굴절 형태는
공유 파라미터 없이
서로 다른 벡터로 직접 표현됩니다.
형태학적 정보를 사용하기 위해,
*fastText* 모델은
*하위 단어 임베딩(subword embedding)* 접근 방식을 제안했습니다.
여기서 하위 단어는 문자 $n$-그램입니다 :cite:`Bojanowski.Grave.Joulin.ea.2017`.
단어 수준 벡터 표현을 학습하는 대신,
fastText는 하위 단어 수준 스킵-그램으로 간주될 수 있으며,
여기서 각 *중심 단어*는
그 하위 단어 벡터들의 합으로 표현됩니다.

"where"라는 단어를 사용하여
fastText에서 각 중심 단어에 대한 하위 단어를 얻는 방법을 설명해 보겠습니다.
먼저, 접두사와 접미사를 다른 하위 단어와 구별하기 위해
단어의 시작과 끝에 특수 문자 “&lt;”와 “&gt;”를 추가합니다.
그런 다음 단어에서 문자 $n$-그램을 추출합니다.
예를 들어 $n=3$일 때,
우리는 길이 3의 모든 하위 단어"&lt;wh", "whe", "her", "ere", "re&gt;"와 특수 하위 단어"&lt;where&gt;"를 얻습니다.


fastText에서, 임의의 단어 $w$에 대해,
길이 3에서 6 사이의 모든 하위 단어와
그 특수 하위 단어의 합집합을 $\mathcal{G}_w$라고 표시합니다.
어휘(vocabulary)는 모든 단어의 하위 단어들의 합집합입니다.
$\\mathbf{z}_g$를 사전에 있는 하위 단어 $g$의 벡터라고 할 때,
스킵-그램 모델에서 중심 단어로서의 단어 $w$에 대한 벡터 $\\mathbf{v}_w$는
그 하위 단어 벡터들의 합입니다:

$$\\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText의 나머지 부분은 스킵-그램 모델과 동일합니다. 스킵-그램 모델과 비교할 때,
fastText의 어휘는 더 크며,
결과적으로 더 많은 모델 파라미터를 갖습니다.
게다가,
단어의 표현을 계산하기 위해,
모든 하위 단어 벡터를 합산해야 하므로,
계산 복잡도가 더 높습니다.
그러나
유사한 구조를 가진 단어들 간의 하위 단어 파라미터 공유 덕분에,
희귀 단어(rare words)와 심지어 어휘 밖의 단어(out-of-vocabulary words)도
fastText에서 더 나은 벡터 표현을 얻을 수 있습니다.



## 바이트 페어 인코딩 (Byte Pair Encoding)
:label:`subsec_Byte_Pair_Encoding`

fastText에서 추출된 모든 하위 단어는 $3$에서 $6$과 같이 지정된 길이를 가져야 하므로, 어휘 크기를 미리 정의할 수 없습니다.
고정 크기 어휘에서 가변 길이 하위 단어를 허용하기 위해,
우리는 하위 단어를 추출하기 위해 *바이트 페어 인코딩(byte pair encoding, BPE)*이라는 압축 알고리즘을 적용할 수 있습니다 :cite:`Sennrich.Haddow.Birch.2015`.

바이트 페어 인코딩은 훈련 데이터셋의 통계적 분석을 수행하여 단어 내의 공통 기호(예: 임의 길이의 연속 문자)를 발견합니다.
길이 1인 기호부터 시작하여,
바이트 페어 인코딩은 가장 빈번한 연속 기호 쌍을 반복적으로 병합하여 새로운 더 긴 기호를 생성합니다.
효율성을 위해 단어 경계를 넘는 쌍은 고려하지 않는다는 점에 유의하십시오.
결국, 우리는 이러한 기호를 하위 단어로 사용하여 단어를 분할할 수 있습니다.
바이트 페어 인코딩과 그 변형은 GPT-2 :cite:`Radford.Wu.Child.ea.2019` 및 RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`와 같은 인기 있는 자연어 처리 사전 훈련 모델의 입력 표현에 사용되었습니다.
다음에서는 바이트 페어 인코딩이 어떻게 작동하는지 설명합니다.

먼저, 기호 어휘를 모든 영어 소문자, 특수 단어 끝 기호 `'_'`, 특수 알 수 없는 기호 `'[UNK]'`로 초기화합니다.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

우리는 단어 경계를 넘는 기호 쌍을 고려하지 않으므로,
우리는 단어를 데이터셋 내의 빈도(발생 횟수)에 매핑하는 딕셔너리 `raw_token_freqs`만 있으면 됩니다.
출력 기호 시퀀스(예: "a_ tall er_ man")에서
단어 시퀀스(예: "a taller man")를 쉽게 복구할 수 있도록
각 단어에 특수 기호 `'_'`가 추가된다는 점에 유의하십시오.
단일 문자와 특수 기호만 있는 어휘에서 병합 프로세스를 시작하므로, 각 단어 내의 모든 연속 문자 쌍 사이에 공백이 삽입됩니다(딕셔너리 `token_freqs`의 키).
즉, 공백은 단어 내 기호 간의 구분자입니다.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

우리는 다음 `get_max_freq_pair` 함수를 정의합니다. 이 함수는 단어 내에서 가장 빈번한 연속 기호 쌍을 반환하며, 여기서 단어는 입력 딕셔너리 `token_freqs`의 키에서 나옵니다.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # `pairs`의 키는 두 연속 기호의 튜플입니다
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # 최댓값을 가진 `pairs`의 키
```

연속 기호의 빈도에 기반한 탐욕적 접근 방식으로서,
바이트 페어 인코딩은 다음 `merge_symbols` 함수를 사용하여 가장 빈번한 연속 기호 쌍을 병합하여 새로운 기호를 생성합니다.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

이제 딕셔너리 `token_freqs`의 키에 대해 바이트 페어 인코딩 알고리즘을 반복적으로 수행합니다. 첫 번째 반복에서 가장 빈번한 연속 기호 쌍은 `'t'`와 `'a'`이므로, 바이트 페어 인코딩은 이들을 병합하여 새로운 기호 `'ta'`를 생성합니다. 두 번째 반복에서 바이트 페어 인코딩은 `'ta'`와 `'l'`을 계속 병합하여 또 다른 새로운 기호 `'tal'`을 생성합니다.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

바이트 페어 인코딩을 10회 반복한 후, 리스트 `symbols`에 다른 기호에서 반복적으로 병합된 10개의 기호가 더 포함되어 있음을 알 수 있습니다.

```{.python .input}
#@tab all
print(symbols)
```

딕셔너리 `raw_token_freqs`의 키에 지정된 동일한 데이터셋에 대해,
바이트 페어 인코딩 알고리즘의 결과로
데이터셋의 각 단어는 이제 하위 단어 "fast_", "fast", "er_", "tall_", "tall"로 분할됩니다.
예를 들어, 단어 "faster_"와 "taller_"는 각각 "fast er_"와 "tall er_"로 분할됩니다.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

바이트 페어 인코딩의 결과는 사용되는 데이터셋에 따라 달라진다는 점에 유의하십시오.
우리는 한 데이터셋에서 학습한 하위 단어를 사용하여
다른 데이터셋의 단어를 분할할 수도 있습니다.
탐욕적 접근 방식으로서, 다음 `segment_BPE` 함수는 입력 인수 `symbols`에서 가능한 가장 긴 하위 단어로 단어를 분할하려고 시도합니다.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # symbols에서 가능한 가장 긴 하위 단어로 토큰을 분할합니다
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

다음에서는 앞서 언급한 데이터셋에서 학습한 리스트 `symbols`의 하위 단어를 사용하여
다른 데이터셋을 나타내는 `tokens`를 분할합니다.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 요약 (Summary)

* fastText 모델은 하위 단어 임베딩 접근 방식을 제안합니다. word2vec의 스킵-그램 모델을 기반으로, 중심 단어를 하위 단어 벡터의 합으로 표현합니다.
* 바이트 페어 인코딩은 훈련 데이터셋의 통계적 분석을 수행하여 단어 내의 공통 기호를 발견합니다. 탐욕적 접근 방식으로서, 바이트 페어 인코딩은 가장 빈번한 연속 기호 쌍을 반복적으로 병합합니다.
* 하위 단어 임베딩은 희귀 단어와 사전 밖 단어의 표현 품질을 향상시킬 수 있습니다.

## 연습 문제 (Exercises)

1. 예로서, 영어에는 약 $3\times 10^8$개의 가능한 $6$-그램이 있습니다. 하위 단어가 너무 많으면 어떤 문제가 있습니까? 이 문제를 어떻게 해결합니까? 힌트: fastText 논문 :cite:`Bojanowski.Grave.Joulin.ea.2017`의 섹션 3.2 끝부분을 참조하십시오.
2. CBOW 모델을 기반으로 하위 단어 임베딩 모델을 어떻게 설계합니까?
3. 초기 기호 어휘 크기가 $n$일 때, 크기 $m$의 어휘를 얻으려면 몇 번의 병합 작업이 필요합니까?
4. 구(phrases)를 추출하기 위해 바이트 페어 인코딩 아이디어를 어떻게 확장할 수 있습니까?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4587)
:end_tab: