# BERT 사전 훈련을 위한 데이터셋 (The Dataset for Pretraining BERT)
:label:`sec_bert-dataset`

:numref:`sec_bert`에서 구현된 BERT 모델을 사전 훈련하기 위해, 두 가지 사전 훈련 작업인 마스킹된 언어 모델링(masked language modeling)과 다음 문장 예측(next sentence prediction)을 용이하게 할 수 있는 이상적인 형식의 데이터셋을 생성해야 합니다. 한편으로 원래 BERT 모델은 BookCorpus와 영어 위키피디아라는 두 거대한 코퍼스의 연결에 대해 사전 훈련되어(:numref:`subsec_bert_pretraining_tasks` 참조), 이 책의 대부분 독자들이 실행하기 어렵습니다. 다른 한편으로 기성 사전 훈련된 BERT 모델은 의학과 같은 특정 도메인의 응용 프로그램에는 적합하지 않을 수 있습니다. 따라서 사용자 정의 데이터셋에서 BERT를 사전 훈련하는 것이 인기를 얻고 있습니다. BERT 사전 훈련 시연을 용이하게 하기 위해, 우리는 더 작은 코퍼스인 WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`를 사용합니다.

:numref:`sec_word2vec_data`에서 word2vec 사전 훈련에 사용된 PTB 데이터셋과 비교할 때, WikiText-2는 (i) 원래의 구두점을 유지하여 다음 문장 예측에 적합하고, (ii) 원래의 대소문자와 숫자를 유지하며, (iii) 두 배 이상 큽니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

[**WikiText-2 데이터셋**]에서 각 줄은 단락을 나타내며, 구두점과 그 앞의 토큰 사이에 공백이 삽입되어 있습니다. 최소 두 문장이 있는 단락만 유지됩니다. 문장을 분할하기 위해 단순함을 위해 마침표만 구분자로 사용합니다. 더 복잡한 문장 분할 기술에 대한 논의는 이 섹션 끝의 연습 문제로 남겨둡니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 대문자를 소문자로 변환
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 사전 훈련 작업을 위한 도우미 함수 정의하기

다음에서는 두 가지 BERT 사전 훈련 작업인 다음 문장 예측과 마스킹된 언어 모델링을 위한 도우미 함수를 구현하는 것부터 시작합니다. 이러한 도우미 함수는 나중에 원시 텍스트 코퍼스를 BERT 사전 훈련을 위한 이상적인 형식의 데이터셋으로 변환할 때 호출될 것입니다.

### [**다음 문장 예측 작업 생성하기**]

:numref:`subsec_nsp`의 설명에 따라, `_get_next_sentence` 함수는 이진 분류 작업을 위한 훈련 예제를 생성합니다.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs`는 리스트의 리스트의 리스트입니다
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

다음 함수는 `_get_next_sentence` 함수를 호출하여 입력 `paragraph`로부터 다음 문장 예측을 위한 훈련 예제를 생성합니다. 여기서 `paragraph`는 문장들의 리스트이고 각 문장은 토큰들의 리스트입니다. `max_len` 인수는 사전 훈련 중 BERT 입력 시퀀스의 최대 길이를 지정합니다.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 1개의 '<cls>' 토큰과 2개의 '<sep>' 토큰 고려
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**마스킹된 언어 모델링 작업 생성하기**]
:label:`subsec_prepare_mlm_data`

BERT 입력 시퀀스로부터 마스킹된 언어 모델링 작업을 위한 훈련 예제를 생성하기 위해, 다음 `_replace_mlm_tokens` 함수를 정의합니다. 입력에서 `tokens`는 BERT 입력 시퀀스를 나타내는 토큰 리스트이고, `candidate_pred_positions`는 특수 토큰을 제외한 BERT 입력 시퀀스의 토큰 인덱스 리스트(마스킹된 언어 모델링 작업에서는 특수 토큰을 예측하지 않음)이며, `num_mlm_preds`는 예측할 수(예측할 무작위 토큰 15% 상기)를 나타냅니다. :numref:`subsec_mlm`의 마스킹된 언어 모델링 작업 정의에 따라, 각 예측 위치에서 입력은 특수 “<mask>” 토큰이나 무작위 토큰으로 대체되거나 변경되지 않은 상태로 유지될 수 있습니다. 마지막으로 함수는 가능한 대체 후의 입력 토큰, 예측이 일어나는 토큰 인덱스 및 이러한 예측에 대한 레이블을 반환합니다.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 마스킹된 언어 모델의 입력을 위해, 토큰의 새로운 복사본을 만들고 
    # 그 중 일부를 '<mask>' 또는 무작위 토큰으로 대체합니다
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 마스킹된 언어 모델링 작업에서 예측을 위해 15%의 무작위 토큰을 얻기 위해 섞음
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%의 경우: 단어를 '<mask>' 토큰으로 교체
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%의 경우: 단어를 변경하지 않고 유지
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%의 경우: 단어를 무작위 단어로 교체
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

앞서 언급한 `_replace_mlm_tokens` 함수를 호출하여, 다음 함수는 BERT 입력 시퀀스(`tokens`)를 입력으로 받아 (:numref:`subsec_mlm`에서 설명한 가능한 토큰 대체 후의) 입력 토큰 인덱스, 예측이 일어나는 토큰 인덱스 및 이러한 예측에 대한 레이블 인덱스를 반환합니다.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens`는 문자열 리스트입니다
    for i, token in enumerate(tokens):
        # 마스킹된 언어 모델링 작업에서 특수 토큰은 예측하지 않음
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 마스킹된 언어 모델링 작업에서 15%의 무작위 토큰을 예측함
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## 텍스트를 사전 훈련 데이터셋으로 변환하기

이제 BERT 사전 훈련을 위한 `Dataset` 클래스를 사용자 정의할 준비가 거의 다 되었습니다. 그 전에, [**입력에 특수 “<pad>” 토큰을 추가하기 위한**] 도우미 함수 `_pad_bert_inputs`를 정의해야 합니다. 이 함수의 `examples` 인수는 두 가지 사전 훈련 작업을 위한 도우미 함수 `_get_nsp_data_from_paragraph` 및 `_get_mlm_data_from_tokens`의 출력을 포함합니다.

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens`는 '<pad>' 토큰 수를 제외함
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # 패딩된 토큰의 예측은 0 가중치 곱셈을 통해 손실에서 필터링됨
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens`는 '<pad>' 토큰 수를 제외함
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 패딩된 토큰의 예측은 0 가중치 곱셈을 통해 손실에서 필터링됨
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

두 가지 사전 훈련 작업의 훈련 예제를 생성하기 위한 도우미 함수와 입력을 패딩하기 위한 도우미 함수를 결합하여, [**BERT 사전 훈련을 위한 WikiText-2 데이터셋**]으로서 다음 `_WikiTextDataset` 클래스를 사용자 정의합니다. `__getitem__` 함수를 구현함으로써 WikiText-2 코퍼스의 문장 쌍으로부터 생성된 사전 훈련(마스킹된 언어 모델링 및 다음 문장 예측) 예제에 임의로 접근할 수 있습니다.

원래 BERT 모델은 어휘 크기가 30,000인 WordPiece 임베딩을 사용합니다 :cite:`Wu.Schuster.Chen.ea.2016`. WordPiece의 토큰화 방법은 :numref:`subsec_Byte_Pair_Encoding`의 원래 바이트 페어 인코딩 알고리즘을 약간 수정한 것입니다. 단순함을 위해 여기서는 `d2l.tokenize` 함수를 사용하여 토큰화합니다. 5번 미만으로 나타나는 드문 토큰은 필터링됩니다.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 입력 `paragraphs[i]`는 단락을 나타내는 문장 문자열들의 리스트입니다;
        # 출력 `paragraphs[i]`는 단락을 나타내는 문장들의 리스트이며 각 문장은 토큰들의 리스트입니다
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 다음 문장 예측 작업을 위한 데이터 가져오기
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 마스킹된 언어 모델 작업을 위한 데이터 가져오기
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 입력 패딩
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 입력 `paragraphs[i]`는 단락을 나타내는 문장 문자열들의 리스트입니다;
        # 출력 `paragraphs[i]`는 단락을 나타내는 문장들의 리스트이며 각 문장은 토큰들의 리스트입니다
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 다음 문장 예측 작업을 위한 데이터 가져오기
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 마스킹된 언어 모델 작업을 위한 데이터 가져오기
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 입력 패딩
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

`_read_wiki` 함수와 `_WikiTextDataset` 클래스를 사용하여, [**WikiText-2 데이터셋을 다운로드하고 그로부터 사전 훈련 예제를 생성하는**] 다음 `load_data_wiki`를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """WikiText-2 데이터셋을 로드합니다."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """WikiText-2 데이터셋을 로드합니다."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

배치 크기를 512로, BERT 입력 시퀀스의 최대 길이를 64로 설정하여, [**BERT 사전 훈련 예제의 미니배치 모양을 인쇄해 봅니다.**] 각 BERT 입력 시퀀스에서 마스킹된 언어 모델링 작업을 위해 $10$ ($64 \times 0.15$)개의 위치가 예측됨에 유의하십시오.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

마지막으로 어휘 크기를 살펴봅시다. 드문 토큰을 필터링한 후에도 PTB 데이터셋보다 두 배 이상 큽니다.

```{.python .input}
#@tab all
len(vocab)
```

## 요약 (Summary)

* PTB 데이터셋과 비교할 때, WikiText-2 데이터셋은 원래의 구두점, 대소문자 및 숫자를 유지하며 두 배 이상 큽니다.
* WikiText-2 코퍼스의 문장 쌍으로부터 생성된 사전 훈련(마스킹된 언어 모델링 및 다음 문장 예측) 예제에 임의로 접근할 수 있습니다.


## 연습 문제 (Exercises)

1. 단순함을 위해 문장 분할을 위한 유일한 구분자로 마침표를 사용했습니다. spaCy나 NLTK와 같은 다른 문장 분할 기술을 시도해 보십시오. NLTK를 예로 들면, 먼저 NLTK를 설치해야 합니다: `pip install nltk`. 코드에서 먼저 `import nltk`를 합니다. 그런 다음 Punkt 문장 토크나이저를 다운로드합니다: `nltk.download('punkt')`. `sentences = 'This is great ! Why not ?'`와 같은 문장을 분할하려면, `nltk.tokenize.sent_tokenize(sentences)`를 호출하여 두 문장 문자열의 리스트인 `['This is great !', 'Why not ?']`을 반환합니다.
2. 드문 토큰을 하나도 필터링하지 않는다면 어휘 크기는 얼마입니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1496)
:end_tab: