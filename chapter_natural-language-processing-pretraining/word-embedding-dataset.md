# 단어 임베딩 사전 훈련을 위한 데이터셋 (The Dataset for Pretraining Word Embeddings
:label:`sec_word2vec_data`

이제 word2vec 모델과 근사 훈련 방법의 기술적 세부 사항을 알았으니, 그 구현 과정을 살펴보겠습니다. 구체적으로, :numref:`sec_word2vec`의 스킵-그램(skip-gram) 모델과 :numref:`sec_approx_train`의 네거티브 샘플링(negative sampling)을 예로 들어 보겠습니다. 이 섹션에서는 단어 임베딩 모델 사전 훈련을 위한 데이터셋부터 시작합니다. 데이터의 원래 형식이 훈련 중에 반복해서 불러올 수 있는 미니배치(minibatches)로 변환될 것입니다.

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## 데이터셋 읽기 (Reading the Dataset)

여기서 사용할 데이터셋은 [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42)입니다. 이 코퍼스(corpus)는 Wall Street Journal 기사에서 샘플링되었으며, 훈련, 검증, 테스트 세트로 나뉩니다. 원래 형식에서 텍스트 파일의 각 줄은 공백으로 구분된 단어들로 구성된 문장을 나타냅니다. 여기서는 각 단어를 하나의 토큰(token)으로 취급합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """PTB 데이터셋을 텍스트 줄의 리스트로 로드합니다."""
    data_dir = d2l.download_extract('ptb')
    # 훈련 세트 읽기
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# 문장 수: {len(sentences)}'
```

훈련 세트를 읽은 후 코퍼스에 대한 어휘 사전(vocabulary)을 구축합니다. 여기서 10번 미만으로 나타나는 단어는 "<unk>" 토큰으로 대체됩니다. 원래 데이터셋에도 드문(알 수 없는) 단어를 나타내는 "<unk>" 토큰이 포함되어 있음에 유의하십시오.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'어휘 사전 크기: {len(vocab)}'
```

## 서브샘플링 (Subsampling)

텍스트 데이터에는 일반적으로 "the", "a", "in"과 같은 고빈도 단어가 있습니다. 이러한 단어들은 매우 큰 코퍼스에서 수십억 번 나타날 수도 있습니다. 그러나 이러한 단어들은 종종 문맥 윈도우(context windows) 내에서 많은 다른 단어들과 함께 나타나며, 유용한 신호를 거의 제공하지 않습니다. 예를 들어, 문맥 윈도우에서 "chip"이라는 단어를 고려해 보십시오. 직관적으로 고빈도 단어인 "a"와 함께 나타나는 것보다 저빈도 단어인 "intel"과 함께 나타나는 것이 훈련에 더 유용합니다. 게다가 엄청난 양의 (고빈도) 단어로 훈련하는 것은 느립니다. 따라서 단어 임베딩 모델을 훈련할 때 고빈도 단어는 *서브샘플링(subsampled)*될 수 있습니다 :cite:`Mikolov.Sutskever.Chen.ea.2013`. 구체적으로, 데이터셋의 각 인덱싱된 단어 $w_i$는 다음 확률로 폐기됩니다.


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

여기서 $f(w_i)$는 데이터셋의 총 단어 수에 대한 단어 $w_i$ 수의 비율이고, 상수 $t$는 하이퍼파라미터입니다(실험에서는 $10^{-4}$). 상대 빈도 $f(w_i) > t$일 때만 (고빈도) 단어 $w_i$가 폐기될 수 있으며, 단어의 상대 빈도가 높을수록 폐기될 확률이 커짐을 알 수 있습니다.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """고빈도 단어를 서브샘플링합니다."""
    # 알 수 없는 토큰('<unk>') 제외
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # 서브샘플링 중에 `token`이 유지되면 True 반환
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

다음 코드 스니펫은 서브샘플링 전후의 문장당 토큰 수 히스토그램을 그립니다. 예상대로 서브샘플링은 고빈도 단어를 제거함으로써 문장을 크게 단축시키며, 이는 훈련 속도 향상으로 이어집니다.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['원래 데이터', '서브샘플링됨'], '# 문장당 토큰 수',
                            '개수', sentences, subsampled);
```

개별 토큰의 경우, 고빈도 단어 "the"의 샘플링 속도는 1/20 미만입니다.

```{.python .input}
#@tab all
def compare_counts(token):
    return (
            f'"{{token}}"의 수: '
            f'이전={sum([l.count(token) for l in sentences])}, '
            f'이후={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

반면, 저빈도 단어 "join"은 완전히 유지됩니다.

```{.python .input}
#@tab all
compare_counts('join')
```

서브샘플링 후, 토큰을 코퍼스의 인덱스로 매핑합니다.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 중심 단어와 문맥 단어 추출 (Extracting Center Words and Context Words)


다음 `get_centers_and_contexts` 함수는 `corpus`에서 모든 중심 단어와 그 문맥 단어들을 추출합니다. 이 함수는 1과 `max_window_size` 사이의 정수를 문맥 윈도우 크기로 균일하게 무작위 샘플링합니다. 임의의 중심 단어에 대해, 샘플링된 문맥 윈도우 크기를 초과하지 않는 거리에 있는 단어들이 문맥 단어가 됩니다.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """스킵-그램에서의 중심 단어와 문맥 단어를 반환합니다."""
    centers, contexts = [], []
    for line in corpus:
        # "중심 단어-문맥 단어" 쌍을 형성하려면 각 문장에 최소 2개의 단어가 있어야 합니다
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # `i`를 중심으로 하는 문맥 윈도우
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 문맥 단어에서 중심 단어 제외
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

다음으로, 각각 7단어와 3단어인 두 문장을 포함하는 인공 데이터셋을 만듭니다. 최대 문맥 윈도우 크기를 2로 설정하고 모든 중심 단어와 그 문맥 단어들을 출력해 봅니다.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('데이터셋', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('중심 단어', center, '의 문맥 단어:', context)
```

PTB 데이터셋에서 훈련할 때는 최대 문맥 윈도우 크기를 5로 설정합니다. 다음은 데이터셋의 모든 중심 단어와 그 문맥 단어들을 추출합니다.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# 중심 단어-문맥 단어 쌍의 수: {sum([len(contexts) for contexts in all_contexts])}'
```

## 네거티브 샘플링 (Negative Sampling)

근사 훈련을 위해 네거티브 샘플링을 사용합니다. 미리 정의된 분포에 따라 노이즈 단어(noise words)를 샘플링하기 위해 다음 `RandomGenerator` 클래스를 정의합니다. 여기서 (아마도 정규화되지 않은) 샘플링 분포는 `sampling_weights` 인수를 통해 전달됩니다.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """n개의 샘플링 가중치에 따라 {1, ..., n} 중에서 무작위로 추출합니다."""
    def __init__(self, sampling_weights):
        # 제외 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # `k`개의 무작위 샘플링 결과 캐시
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

예를 들어, 샘플링 확률 $P(X=1)=2/9, P(X=2)=3/9, P(X=3)=4/9$인 인덱스 1, 2, 3 중에서 10개의 확률 변수 $X$를 다음과 같이 추출할 수 있습니다.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

중심 단어와 문맥 단어 쌍에 대해, `K`(실험에서는 5)개의 노이즈 단어를 무작위로 샘플링합니다. word2vec 논문의 제안에 따라, 노이즈 단어 $w$의 샘플링 확률 $P(w)$는 사전에서의 상대 빈도에 0.75승을 한 값으로 설정됩니다 :cite:`Mikolov.Sutskever.Chen.ea.2013`.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """네거티브 샘플링에서의 노이즈 단어를 반환합니다."""
    # 어휘 사전에서 인덱스 1, 2, ... (인덱스 0은 제외된 알 수 없는 토큰)인 단어들에 대한 샘플링 가중치
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 노이즈 단어는 문맥 단어가 될 수 없음
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 미니배치로 훈련 예제 로드하기 (Loading Training Examples in Minibatches)
:label:`subsec_word2vec-minibatch-loading`

모든 중심 단어와 그 문맥 단어 및 샘플링된 노이즈 단어가 추출된 후, 이들은 훈련 중에 반복해서 로드될 수 있는 예제의 미니배치로 변환됩니다.



미니배치에서 $i^\textrm{th}$번째 예제는 중심 단어와 그 $n_i$개의 문맥 단어 및 $m_i$개의 노이즈 단어를 포함합니다. 문맥 윈도우 크기가 다르기 때문에 $n_i+m_i$는 $i$마다 다릅니다. 따라서 각 예제에 대해 `contexts_negatives` 변수에서 문맥 단어와 노이즈 단어를 연결하고, 연결된 길이가 $\max_i n_i+m_i$ (`max_len`)에 도달할 때까지 0을 추가(패딩)합니다. 손실 계산에서 패딩을 제외하기 위해 마스크 변수 `masks`를 정의합니다. `masks`의 요소와 `contexts_negatives`의 요소 사이에는 일대일 대응 관계가 있으며, `masks`의 0(그 외에는 1)은 `contexts_negatives`의 패딩에 대응합니다.


긍정 예제와 부정 예제를 구별하기 위해 `labels` 변수를 통해 `contexts_negatives`에서 문맥 단어와 노이즈 단어를 분리합니다. `masks`와 유사하게 `labels`의 요소와 `contexts_negatives`의 요소 사이에도 일대일 대응 관계가 있으며, `labels`의 1(그 외에는 0)은 `contexts_negatives`의 문맥 단어(긍정 예제)에 대응합니다.


위의 아이디어는 다음 `batchify` 함수에서 구현됩니다. 입력 `data`는 배치 크기와 동일한 길이를 가진 리스트이며, 각 요소는 중심 단어 `center`, 그 문맥 단어 `context`, 그리고 노이즈 단어 `negative`로 구성된 예제입니다. 이 함수는 마스크 변수를 포함하여 훈련 중에 계산을 위해 로드될 수 있는 미니배치를 반환합니다.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """네거티브 샘플링을 사용한 스킵-그램을 위한 예제 미니배치를 반환합니다."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

두 개의 예제로 구성된 미니배치를 사용하여 이 함수를 테스트해 봅시다.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['중심 단어', '문맥_부정_단어', '마스크', '레이블']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 종합하기 (Putting It All Together)

마지막으로, PTB 데이터셋을 읽고 데이터 반복자와 어휘 사전을 반환하는 `load_data_ptb` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """PTB 데이터셋을 다운로드한 후 메모리로 로드합니다."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """PTB 데이터셋을 다운로드한 후 메모리로 로드합니다."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

데이터 반복자의 첫 번째 미니배치를 출력해 봅니다.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, '모양:', data.shape)
    break
```

## 요약 (Summary)

* 고빈도 단어는 훈련에 그렇게 유용하지 않을 수 있습니다. 훈련 속도를 높이기 위해 이를 서브샘플링할 수 있습니다.
* 계산 효율성을 위해 예제를 미니배치로 로드합니다. 패딩과 패딩이 아닌 것을 구별하고, 긍정 예제와 부정 예제를 구별하기 위해 다른 변수들을 정의할 수 있습니다.



## 연습 문제 (Exercises)

1. 서브샘플링을 사용하지 않을 경우 이 섹션의 코드 실행 시간이 어떻게 변합니까?
2. `RandomGenerator` 클래스는 `k`개의 무작위 샘플링 결과를 캐시합니다. `k`를 다른 값으로 설정해 보고 데이터 로딩 속도에 어떤 영향을 미치는지 확인해 보십시오.
3. 이 섹션의 코드에서 데이터 로딩 속도에 영향을 줄 수 있는 다른 하이퍼파라미터는 무엇입니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1330)
:end_tab: