# 감정 분석과 데이터셋 (Sentiment Analysis and the Dataset)
:label:`sec_sentiment`


온라인 소셜 미디어와 리뷰 플랫폼의 확산으로,
방대한 양의
의견 데이터가 기록되었으며,
의사 결정 과정을 지원할 수 있는 큰 잠재력을 가지고 있습니다.
*감정 분석(Sentiment analysis)*은
제품 리뷰,
블로그 댓글,
포럼 토론과 같이
사람들이 생산한 텍스트에서 사람들의 감정을 연구합니다.
이는 정치(예: 정책에 대한 대중의 정서 분석),
금융(예: 시장의 정서 분석),
마케팅(예: 제품 연구 및 브랜드 관리)과 같이
다양한 분야에 널리 응용되고 있습니다.

감정은
이산적인 극성이나 척도(예: 긍정 및 부정)로 분류될 수 있으므로,
우리는 감정 분석을
가변 길이 텍스트 시퀀스를
고정 길이 텍스트 범주로 변환하는
텍스트 분류 작업으로 간주할 수 있습니다.
이 장에서는
감정 분석을 위해 스탠포드의 [대규모 영화 리뷰 데이터셋](https://ai.stanford.edu/%7Eamaas/data/sentiment/)을 사용할 것입니다.
이것은 훈련 세트와 테스트 세트로 구성되며,
각각 IMDb에서 다운로드한 25,000개의 영화 리뷰를 포함합니다.
두 데이터셋 모두
"긍정(positive)"과 "부정(negative)" 레이블이 동일한 수로 있어
서로 다른 감정 극성을 나타냅니다.

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

## 데이터셋 읽기 (Reading the Dataset)

먼저, 이 IMDb 리뷰 데이터셋을 다운로드하고
`../data/aclImdb` 경로에 압축을 풉니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz',
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

다음으로, 훈련 및 테스트 데이터셋을 읽습니다. 각 예제는 리뷰와 그 레이블입니다: "긍정"은 1, "부정"은 0입니다.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """IMDb 리뷰 데이터셋 텍스트 시퀀스와 레이블을 읽습니다."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## 데이터셋 전처리 (Preprocessing the Dataset)

각 단어를 토큰으로 취급하고
5회 미만으로 나타나는 단어를 필터링하여,
훈련 데이터셋에서 어휘(vocabulary)를 생성합니다.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

토큰화 후,
토큰 단위의 리뷰 길이 히스토그램을 그려봅시다.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

예상대로,
리뷰의 길이는 다양합니다. 
매번 이러한 리뷰의 미니배치를 처리하기 위해,
우리는 :numref:`sec_machine_translation`의
기계 번역 데이터셋에 대한 전처리 단계와 유사하게
자르기(truncation) 및 패딩(padding)을 사용하여 각 리뷰의 길이를 500으로 설정합니다.

```{.python .input}
#@tab all
num_steps = 500  # 시퀀스 길이
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## 데이터 반복자 생성 (Creating Data Iterators)

이제 데이터 반복자를 생성할 수 있습니다.
각 반복마다 예제의 미니배치가 반환됩니다.

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## 종합하기 (Putting It All Together)

마지막으로, 위 단계를 `load_data_imdb` 함수로 래핑합니다.
이 함수는 훈련 및 테스트 데이터 반복자와 IMDb 리뷰 데이터셋의 어휘를 반환합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """데이터 반복자와 IMDb 리뷰 데이터셋의 어휘를 반환합니다."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """데이터 반복자와 IMDb 리뷰 데이터셋의 어휘를 반환합니다."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## 요약 (Summary)

* 감정 분석은 생성된 텍스트에서 사람들의 감정을 연구하며, 이는 가변 길이 텍스트 시퀀스를 고정 길이 텍스트 범주로 변환하는 텍스트 분류 문제로 간주됩니다.
* 전처리 후, 스탠포드의 대규모 영화 리뷰 데이터셋(IMDb 리뷰 데이터셋)을 어휘와 함께 데이터 반복자로 로드할 수 있습니다.


## 연습 문제 (Exercises)


1. 감정 분석 모델 훈련을 가속화하기 위해 이 섹션의 어떤 하이퍼파라미터를 수정할 수 있습니까?
1. [Amazon 리뷰](https://snap.stanford.edu/data/web-Amazon.html) 데이터셋을 감정 분석을 위한 데이터 반복자와 레이블로 로드하는 함수를 구현할 수 있습니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab: