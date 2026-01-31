# 자연어 추론과 데이터셋 (Natural Language Inference and the Dataset)
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment`에서, 우리는 감정 분석 문제에 대해 논의했습니다.
이 작업은 단일 텍스트 시퀀스를 미리 정의된 범주(예: 감정 극성 집합)로 분류하는 것을 목표로 합니다.
그러나 한 문장이 다른 문장에서 유추될 수 있는지 결정해야 하거나,
의미적으로 동등한 문장을 식별하여 중복을 제거해야 할 때,
하나의 텍스트 시퀀스를 분류하는 방법을 아는 것만으로는 충분하지 않습니다.
대신, 우리는 텍스트 시퀀스 쌍에 대해 추론할 수 있어야 합니다.


## 자연어 추론 (Natural Language Inference)

*자연어 추론(Natural language inference)*은 *가설(hypothesis)*이
*전제(premise)*로부터 유추될 수 있는지를 연구합니다. 여기서 둘 다 텍스트 시퀀스입니다.
즉, 자연어 추론은 텍스트 시퀀스 쌍 간의 논리적 관계를 결정합니다.
이러한 관계는 일반적으로 다음 세 가지 유형으로 나뉩니다:

* *함의(Entailment)*: 가설이 전제로부터 유추될 수 있습니다.
* *모순(Contradiction)*: 가설의 부정이 전제로부터 유추될 수 있습니다.
* *중립(Neutral)*: 다른 모든 경우입니다.

자연어 추론은 텍스트 함의 인식(recognizing textual entailment) 작업이라고도 합니다.
예를 들어, 다음 쌍은 *함의*로 레이블이 지정됩니다. 가설의 "showing affection(애정 표현)"이 전제의 "hugging one another(서로 포옹)"에서 유추될 수 있기 때문입니다.

> 전제: Two women are hugging each other. (두 여자가 서로 포옹하고 있다.)

> 가설: Two women are showing affection. (두 여자가 애정을 표현하고 있다.)

다음은 *모순*의 예입니다. "running the coding example(코딩 예제 실행)"은 "sleeping(수면)"이 아니라 "not sleeping(자지 않음)"을 나타내기 때문입니다.

> 전제: A man is running the coding example from Dive into Deep Learning. (한 남자가 Dive into Deep Learning의 코딩 예제를 실행하고 있다.)

> 가설: The man is sleeping. (남자가 자고 있다.)

세 번째 예는 *중립* 관계를 보여줍니다. "famous(유명함)"나 "not famous(유명하지 않음)" 모두 "are performing for us(우리를 위해 공연하고 있다)"라는 사실에서 유추될 수 없기 때문입니다.

> 전제: The musicians are performing for us. (음악가들이 우리를 위해 공연하고 있다.)

> 가설: The musicians are famous. (음악가들은 유명하다.)

자연어 추론은 자연어 이해의 핵심 주제였습니다.
정보 검색에서 개방형 질문 응답에 이르기까지 광범위한 응용 분야를 가지고 있습니다.
이 문제를 연구하기 위해, 인기 있는 자연어 추론 벤치마크 데이터셋을 조사하는 것으로 시작하겠습니다.


## 스탠포드 자연어 추론 (SNLI) 데이터셋

[**Stanford Natural Language Inference (SNLI) Corpus**]는 500,000개 이상의 레이블이 지정된 영어 문장 쌍 모음입니다 :cite:`Bowman.Angeli.Potts.ea.2015`.
우리는 추출된 SNLI 데이터셋을 다운로드하여 `../data/snli_1.0` 경로에 저장합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**데이터셋 읽기 (Reading the Dataset)**]

원본 SNLI 데이터셋에는 실험에 실제로 필요한 것보다 훨씬 더 풍부한 정보가 포함되어 있습니다. 따라서 데이터셋의 일부만 추출한 다음 전제, 가설 및 레이블 목록을 반환하는 `read_snli` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """SNLI 데이터셋을 전제, 가설, 레이블로 읽습니다."""
    def extract_text(s):
        # 사용하지 않을 정보 제거
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # 두 개 이상의 연속 공백을 공백으로 대체
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

이제 전제와 가설의 [**처음 3개 쌍과 레이블을 인쇄해 봅시다**] ("0", "1", "2"는 각각 "함의", "모순", "중립"에 해당합니다).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

훈련 세트에는 약 550,000개의 쌍이 있고,
테스트 세트에는 약 10,000개의 쌍이 있습니다.
다음은 훈련 세트와 테스트 세트 모두에서
[**"함의", "모순", "중립"의 세 가지 레이블이 균형을 이루고 있음**]을 보여줍니다.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**데이터셋 로드를 위한 클래스 정의 (Defining a Class for Loading the Dataset)**]

아래에서는 Gluon의 `Dataset` 클래스를 상속하여 SNLI 데이터셋을 로드하는 클래스를 정의합니다. 클래스 생성자의 `num_steps` 인수는 텍스트 시퀀스의 길이를 지정하여 시퀀스의 각 미니배치가 동일한 모양을 갖도록 합니다.
즉, 더 긴 시퀀스의 처음 `num_steps` 이후 토큰은 잘리고, 더 짧은 시퀀스에는 길이가 `num_steps`가 될 때까지 특수 토큰 “&lt;pad&gt;”가 추가됩니다.
`__getitem__` 함수를 구현함으로써, 인덱스 `idx`로 전제, 가설 및 레이블에 임의로 접근할 수 있습니다.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """SNLI 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """SNLI 데이터셋을 로드하기 위한 사용자 정의 데이터셋."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**종합하기 (Putting It All Together)**]

이제 `read_snli` 함수와 `SNLIDataset` 클래스를 호출하여 SNLI 데이터셋을 다운로드하고 훈련 및 테스트 세트 모두에 대한 `DataLoader` 인스턴스와 훈련 세트의 어휘를 반환할 수 있습니다.
훈련 세트에서 구성된 어휘를 테스트 세트의 어휘로 사용해야 한다는 점에 유의해야 합니다.
결과적으로 테스트 세트의 새로운 토큰은 훈련 세트에서 훈련된 모델에 대해 알 수 없는(unknown) 토큰이 됩니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """SNLI 데이터셋을 다운로드하고 데이터 반복자와 어휘를 반환합니다."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """SNLI 데이터셋을 다운로드하고 데이터 반복자와 어휘를 반환합니다."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

여기서 배치 크기를 128로, 시퀀스 길이를 50으로 설정하고,
`load_data_snli` 함수를 호출하여 데이터 반복자와 어휘를 얻습니다.
그런 다음 어휘 크기를 인쇄합니다.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

이제 첫 번째 미니배치의 모양을 인쇄합니다.
감정 분석과 달리,
전제와 가설의 쌍을 나타내는 두 개의 입력 `X[0]`과 `X[1]`이 있습니다.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 요약 (Summary)

* 자연어 추론은 가설이 전제로부터 유추될 수 있는지를 연구합니다. 여기서 둘 다 텍스트 시퀀스입니다.
* 자연어 추론에서, 전제와 가설 간의 관계에는 함의, 모순, 중립이 포함됩니다.
* Stanford Natural Language Inference (SNLI) Corpus는 인기 있는 자연어 추론 벤치마크 데이터셋입니다.


## 연습 문제 (Exercises)

1. 기계 번역은 오랫동안 출력 번역과 정답 번역 간의 피상적인 $n$-그램 매칭을 기반으로 평가되어 왔습니다. 자연어 추론을 사용하여 기계 번역 결과를 평가하는 척도를 설계할 수 있습니까?
2. 어휘 크기를 줄이기 위해 하이퍼파라미터를 어떻게 변경할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab: