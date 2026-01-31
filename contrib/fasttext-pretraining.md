# fastText 사전 훈련 (Pretraining fastText)
:label:`sec_word2vec_gluon`

이 섹션에서는 :numref:`sec_word2vec`에서 정의된 skip-gram 모델을 훈련할 것입니다.

먼저 실험에 필요한 패키지와 모듈을 임포트하고, PTB 데이터셋을 로드합니다.

```{.python .input  n=1}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from functools import partial
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

## 데이터 로드 (Loading the Dataset)

PTB 데이터셋을 로드합니다.

```{.python .input  n=2}
#@tab mxnet
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## 스킵 그램 모델 (The Skip-gram Model)

우리는 임베딩 레이어와 미니배치 행렬 곱셈을 통해 skip-gram 모델을 구현할 것입니다.

### 임베딩 레이어 (Embedding Layer)

임베딩 레이어의 가중치는 각 행이 단어 벡터인 행렬입니다.

```{.python .input  n=3}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

임베딩 레이어의 입력은 단어의 인덱스입니다. 입력이 인덱스 $i$인 경우, 임베딩 레이어는 가중치 행렬의 $i$번째 행을 반환합니다.

```{.python .input  n=4}
#@tab mxnet
x = np.array([1, 2, 3])
embed(x)
```

### 미니배치 행렬 곱셈 (Minibatch Matrix Multiplication)

미니배치 행렬 곱셈을 사용하여 중심 단어 벡터와 컨텍스트 단어 벡터의 내적을 계산할 수 있습니다.

```{.python .input  n=5}
#@tab mxnet
X = np.ones((2, 1, 4))
Y = np.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

### Skip-gram 모델 구현 (Skip-gram Model Implementation)

임베딩 레이어를 사용하여 skip-gram 모델을 구현합니다.

```{.python .input  n=6}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

## 훈련 루프 (Training Loop)

이제 훈련 루프를 정의합니다.

### 바이너리 크로스 엔트로피 손실 (Binary Cross-Entropy Loss)

네거티브 샘플링을 사용하여 skip-gram 모델을 훈련하기 위해 바이너리 크로스 엔트로피 손실 함수를 사용합니다.

```{.python .input  n=7}
#@tab mxnet
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

### 모델 초기화 (Initializing Model Parameters)

두 개의 임베딩 레이어를 초기화합니다.

```{.python .input  n=8}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
net.initialize(init.Xavier(), ctx=d2l.try_all_gpus())
```

### 훈련 정의 (Defining Training)

훈련 함수를 정의합니다.

```{.python .input  n=9}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        metric = d2l.Accumulator(2)  # loss의 합, 샘플 수
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) /
                     mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs, d2l.try_gpu())
```

## 단어 임베딩 적용 (Applying Word Embeddings)

훈련된 모델을 사용하여 유사한 단어를 찾을 수 있습니다.

```{.python .input  n=10}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # 코사인 유사도 계산
    cos = np.dot(W, x) / (np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9))
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 입력어 제외
        print(f'cosine sim={cos[i].item():.3f}: {vocab.idx_to_token[i]}')

get_similar_tokens('chip', 3, net[0])
```

## 요약 (Summary)

* 임베딩 레이어와 미니배치 행렬 곱셈을 사용하여 skip-gram 모델을 효율적으로 구현할 수 있습니다.
* 네거티브 샘플링은 바이너리 크로스 엔트로피 손실을 사용하여 모델을 훈련할 수 있게 해줍니다.

## 연습 문제 (Exercises)

1. 임베딩 크기를 변경해 보십시오. 결과에 어떤 영향을 줍니까?
2. 다른 최적화 알고리즘을 사용해 보십시오.
3. 네거티브 샘플의 수를 변경해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/400)
:end_tab: