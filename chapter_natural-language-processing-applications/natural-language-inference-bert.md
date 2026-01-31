# 자연어 추론: BERT 미세 조정 (Natural Language Inference: Fine-Tuning BERT)
:label:`sec_natural-language-inference-bert`

이 장의 이전 섹션들에서,
우리는 SNLI 데이터셋(:numref:`sec_natural-language-inference-and-dataset`에 설명됨)의
자연어 추론 작업을 위한
어텐션 기반 아키텍처(:numref:`sec_natural-language-inference-attention`)를 설계했습니다.
이제 BERT를 미세 조정하여 이 작업을 다시 살펴보겠습니다.
:numref:`sec_finetuning-bert`에서 논의했듯이,
자연어 추론은 시퀀스 수준의 텍스트 쌍 분류 문제이며,
BERT를 미세 조정하는 데는 :numref:`fig_nlp-map-nli-bert`에 설명된 대로
추가적인 MLP 기반 아키텍처만 필요합니다.

![이 섹션에서는 자연어 추론을 위해 사전 훈련된 BERT를 MLP 기반 아키텍처에 공급합니다.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

이 섹션에서는
사전 훈련된 BERT의 작은 버전을 다운로드한 다음,
SNLI 데이터셋에서 자연어 추론을 위해
미세 조정할 것입니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**사전 훈련된 BERT 로드 (Loading Pretrained BERT)**]

우리는 :numref:`sec_bert-dataset`과 :numref:`sec_bert-pretraining`에서
WikiText-2 데이터셋에 대해 BERT를 사전 훈련하는 방법을 설명했습니다
(원래 BERT 모델은 훨씬 더 큰 코퍼스에서 사전 훈련되었음에 유의하십시오).
:numref:`sec_bert-pretraining`에서 논의했듯이,
원래 BERT 모델에는 수억 개의 파라미터가 있습니다.
다음에서,
우리는 두 가지 버전의 사전 훈련된 BERT를 제공합니다:
"bert.base"는 원래 BERT 기본 모델만큼 커서 미세 조정에 많은 계산 리소스가 필요하고,
"bert.small"은 시연을 용이하게 하기 위한 작은 버전입니다.

```{.python .input}
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

두 사전 훈련된 BERT 모델 모두 어휘 집합을 정의하는 "vocab.json" 파일과
사전 훈련된 파라미터의 "pretrained.params" 파일을 포함합니다.
우리는 [**사전 훈련된 BERT 파라미터를 로드**]하기 위해 다음 `load_pretrained_model` 함수를 구현합니다.

```{.python .input}
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 미리 정의된 어휘를 로드하기 위해 빈 어휘 정의
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # 사전 훈련된 BERT 파라미터 로드
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 미리 정의된 어휘를 로드하기 위해 빈 어휘 정의
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # 사전 훈련된 BERT 파라미터 로드
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

대부분의 기계에서 시연을 용이하게 하기 위해,
이 섹션에서는 사전 훈련된 BERT의 작은 버전("bert.small")을 로드하고 미세 조정할 것입니다.
연습 문제에서는 훨씬 더 큰 "bert.base"를 미세 조정하여 테스트 정확도를 크게 향상시키는 방법을 보여줄 것입니다.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**BERT 미세 조정을 위한 데이터셋 (The Dataset for Fine-Tuning BERT)**]

SNLI 데이터셋의 다운스트림 작업 자연어 추론을 위해,
사용자 정의 데이터셋 클래스 `SNLIBERTDataset`을 정의합니다.
각 예제에서,
전제와 가설은 텍스트 시퀀스 쌍을 형성하고
:numref:`fig_bert-two-seqs`에 묘사된 대로 하나의 BERT 입력 시퀀스로 압축됩니다.
:numref:`subsec_bert_input_rep`를 상기하면, 세그먼트 ID는
BERT 입력 시퀀스에서 전제와 가설을 구별하는 데 사용됩니다.
BERT 입력 시퀀스의 미리 정의된 최대 길이(`max_len`)로,
입력 텍스트 쌍 중 더 긴 것의 마지막 토큰은
`max_len`이 충족될 때까지 계속 제거됩니다.
BERT 미세 조정을 위한 SNLI 데이터셋 생성을 가속화하기 위해,
우리는 4개의 작업자 프로세스를 사용하여 훈련 또는 테스트 예제를 병렬로 생성합니다.

```{.python .input}
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 4개의 작업자 프로세스 사용
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # BERT 입력을 위해 '<CLS>', '<SEP>', '<SEP>' 토큰을 위한 슬롯 예약
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 4개의 작업자 프로세스 사용
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # BERT 입력을 위해 '<CLS>', '<SEP>', '<SEP>' 토큰을 위한 슬롯 예약
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

SNLI 데이터셋을 다운로드한 후,
`SNLIBERTDataset` 클래스를 인스턴스화하여 [**훈련 및 테스트 예제를 생성**]합니다.
이러한 예제들은 자연어 추론의 훈련 및 테스트 중에 미니배치로 읽힐 것입니다.

```{.python .input}
#@tab mxnet
# 메모리 부족 오류가 발생하면 `batch_size`를 줄이십시오. 원래 BERT 모델에서 `max_len` = 512입니다
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# 메모리 부족 오류가 발생하면 `batch_size`를 줄이십시오. 원래 BERT 모델에서 `max_len` = 512입니다
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## BERT 미세 조정 (Fine-Tuning BERT)

:numref:`fig_bert-two-seqs`가 나타내듯이,
자연어 추론을 위한 BERT 미세 조정에는
두 개의 완전 연결 레이어로 구성된 추가 MLP만 필요합니다
(다음 `BERTClassifier` 클래스의 `self.hidden` 및 `self.output` 참조).
[**이 MLP는 전제와 가설 모두의 정보를 인코딩하는
특수 “&lt;cls&gt;” 토큰의 BERT 표현을
자연어 추론의 세 가지 출력(함의, 모순, 중립)으로 변환합니다**].

```{.python .input}
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

다음에서,
사전 훈련된 BERT 모델 `bert`는 다운스트림 응용 프로그램을 위해
`BERTClassifier` 인스턴스 `net`에 공급됩니다.
BERT 미세 조정의 일반적인 구현에서,
추가 MLP의 출력 레이어 파라미터(`net.output`)만 처음부터 학습됩니다.
사전 훈련된 BERT 인코더의 모든 파라미터(`net.encoder`)와 추가 MLP의 은닉층(`net.hidden`)은 미세 조정됩니다.

```{.python .input}
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

:numref:`sec_bert`에서
`MaskLM` 클래스와 `NextSentencePred` 클래스 모두
사용된 MLP에 파라미터가 있음을 상기하십시오.
이 파라미터들은 사전 훈련된 BERT 모델 `bert`의 파라미터의 일부이므로 `net`의 파라미터의 일부입니다.
그러나 이러한 파라미터는
사전 훈련 중에 마스킹된 언어 모델링 손실과
다음 문장 예측 손실을 계산하기 위한 것일 뿐입니다.
이 두 손실 함수는 다운스트림 응용 프로그램의 미세 조정과 관련이 없으므로,
BERT가 미세 조정될 때 `MaskLM` 및 `NextSentencePred`에서 사용된 MLP의 파라미터는 업데이트되지 않습니다(staled).

오래된 기울기를 가진 파라미터를 허용하기 위해,
`d2l.train_batch_ch13`의 `step` 함수에서 `ignore_stale_grad=True` 플래그가 설정됩니다.
우리는 이 함수를 사용하여 SNLI의 훈련 세트(`train_iter`)와 테스트 세트(`test_iter`)를 사용하여 모델 `net`을 훈련하고 평가합니다.
제한된 계산 리소스로 인해, [**훈련**] 및 테스트 정확도는
더 향상될 수 있습니다. 이에 대한 논의는 연습 문제로 남겨둡니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 요약 (Summary)

* SNLI 데이터셋의 자연어 추론과 같은 다운스트림 응용 프로그램을 위해 사전 훈련된 BERT 모델을 미세 조정할 수 있습니다.
* 미세 조정 중에, BERT 모델은 다운스트림 응용 프로그램을 위한 모델의 일부가 됩니다. 사전 훈련 손실과만 관련된 파라미터는 미세 조정 중에 업데이트되지 않습니다.



## 연습 문제 (Exercises)

1. 계산 리소스가 허용하는 경우 원래 BERT 기본 모델만큼 큰 훨씬 더 큰 사전 훈련된 BERT 모델을 미세 조정하십시오. `load_pretrained_model` 함수의 인수를 다음과 같이 설정하십시오: 'bert.small'을 'bert.base'로 교체하고, `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4`, `num_blks=2`의 값을 각각 768, 3072, 12, 12로 늘립니다. 미세 조정 에포크를 늘리고(가능하면 다른 하이퍼파라미터를 조정하여), 0.86보다 높은 테스트 정확도를 얻을 수 있습니까?
2. 길이 비율에 따라 시퀀스 쌍을 어떻게 자를 수 있습니까? 이 쌍 자르기 방법과 `SNLIBERTDataset` 클래스에서 사용된 방법을 비교하십시오. 장단점은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1526)
:end_tab: