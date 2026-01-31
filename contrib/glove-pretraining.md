# GloVe 사전 훈련 (Pretraining GloVe)
:label:`sec_GloVe_gluon`

이 섹션에서는 :numref:`sec_glove`에서 정의된 GloVe 모델을 훈련할 것입니다.

먼저 실험에 필요한 패키지와 모듈을 가져옵니다.

```{.python .input  n=1}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

## 데이터셋 전처리 (Preprocessing Dataset)
우리는 PTB 데이터셋에서 GloVe 모델을 훈련할 것입니다.

먼저 PTB 데이터셋을 읽고, 단어들로 어휘 사전을 구축하고, 각 토큰을 인덱스로 매핑하여 코퍼스를 구성합니다.

```{.python .input  n=2}
#@tab mxnet
sentences = d2l.read_ptb()
vocab = d2l.Vocab(sentences, min_freq=10)
corpus = [vocab[line] for line in sentences]
```

### 공생 횟수 구성 (Construct Cooccurrence Counts)
단어-단어 공생 횟수를 $X$라고 표시합시다. 여기서 $x_{ij}$는 단어 $i$의 문맥에서 단어 $j$가 나타나는 횟수를 표로 나타낸 것입니다.

다음으로, 모든 중심 타겟 단어와 그 문맥 단어들을 추출하는 함수를 정의합니다. 이 함수는 $d$만큼 떨어진 단어 쌍이 전체 횟수에 $1/d$만큼 기여하도록 감소하는 가중치 함수를 사용합니다. 이는 매우 멀리 떨어진 단어 쌍은 서로의 관계에 대해 관련 정보가 덜 포함될 것으로 예상된다는 사실을 설명하는 한 가지 방법입니다.

```{.python .input  n=3}
#@tab mxnet
def get_coocurrence_counts(corpus, window_size):
    centers, contexts = [], []
    cooccurence_counts = defaultdict(float)
    for line in corpus:
        # "중심 타겟 단어 - 문맥 단어" 쌍을 형성하려면 각 문장에 최소 2개의 단어가 필요합니다
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # i를 중심으로 하는 문맥 윈도우
            left_indices = list(range(max(0, i - window_size), i))
            right_indices = list(range(i + 1,
                                       min(len(line), i + 1 + window_size)))
            left_context = [line[idx] for idx in left_indices]
            right_context = [line[idx] for idx in right_indices]
            for distance, word in enumerate(left_context[::-1]):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
            for distance, word in enumerate(right_context):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
    cooccurence_counts = [(word[0], word[1], count)
                          for word, count in cooccurence_counts.items()]
    return cooccurence_counts
```

각각 5단어와 2단어인 두 문장을 포함하는 인공 데이터셋을 만듭니다. 최대 문맥 윈도우가 4라고 가정합니다. 그런 다음 모든 중심 타겟 단어와 문맥 단어의 공생 횟수를 출력합니다.

```{.python .input  n=4}
#@tab mxnet
tiny_dataset = [list(range(5)), list(range(5, 7))]
print('데이터셋', tiny_dataset)
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4):
        print('중심: %s, 문맥: %s, 공생: %.2f' %
          (center, context, coocurrence))
```

최대 문맥 윈도우 크기를 5로 설정합니다. 다음은 데이터셋의 모든 중심 타겟 단어와 그 문맥 단어들을 추출하고 공생 횟수를 계산합니다.

```{.python .input  n=5}
#@tab mxnet
coocurrence_matrix = get_coocurrence_counts(corpus, 5)
'# 중심-문맥 쌍의 수: %d' % len(coocurrence_matrix)
```

### 종합하기 (Putting All Things Together)

마지막으로, PTB 데이터셋을 읽고 데이터 로더를 반환하는 `load_data_ptb_glove` 함수를 정의합니다.

```{.python .input  n=16}
#@tab mxnet
def load_data_ptb_glove(batch_size, window_size):
    num_workers = d2l.get_dataloader_workers()
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=5)
    corpus = [vocab[line] for line in sentences]
    coocurrence_matrix = get_coocurrence_counts(corpus, window_size)
    dataset = gluon.data.ArrayDataset(coocurrence_matrix)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      num_workers=num_workers)
    return data_iter, vocab

batch_size, window_size = 1024, 10
data_iter, vocab = load_data_ptb_glove(batch_size, window_size)
```

데이터 반복자의 첫 번째 미니배치를 출력해 봅시다.

```{.python .input  n=17}
#@tab mxnet
names = ['중심', '문맥', '공생']
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, '모양:', data.shape)
    break
```

## GloVe 모델 (The GloVe Model)

섹션 15.1에서 GloVe의 목표는 손실 함수를 최소화하는 것이라고 소개했습니다.

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}}
 h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - 
\log\,x_{ij}\right)^2.$$ 

우리는 손실 함수의 각 부분을 구현하여 GloVe 모델을 구현할 것입니다.

### 가중치 함수 (Weight function)

Glove는 손실 함수에 가중치 함수 $h(x_{ij})$를 도입했습니다.

$$h(x_{ij})=\begin{cases}
(\frac{x}{x_{max}})^
\alpha & x_{ij}<x_{max}\
1 & \text{그 외}
\end{cases}$$ 


가중치 함수 $h(x_{ij})$를 구현합니다. $x_{ij}<x_{max}$는 $(\frac{x}{x_{max}})^
\alpha < 1$과 동등하므로 다음과 같이 구현할 수 있습니다.

```{.python .input  n=18}
#@tab mxnet
def compute_weight(x, x_max = 30, alpha = 0.75):
    w = (x / x_max) ** alpha
    return np.minimum(w, 1)
```

다음은 $x_{max}$를 2로, $\alpha$를 0.75로 설정했을 때 모든 중심 타겟 단어와 문맥 단어의 공생 횟수에 대한 가중치를 출력합니다.

```{.python .input  n=19}
#@tab mxnet
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4)[:5]:
    print('중심: %s, 문맥: %s, 공생: %.2f, 가중치: %.2f' %
          (center, context, coocurrence, compute_weight(coocurrence, x_max = 2, alpha = 0.75)))
```

### 편향 항 (Bias Term)

Glove는 각 단어 $w_i$에 대해 두 개의 스칼라 모델 파라미터인 편향 항 $b_i$(중심 타겟 단어용)와 $c_i$(문맥 단어용)를 갖습니다. 편향 항은 임베딩 레이어로 실현될 수 있습니다. 임베딩 레이어의 가중치는 행 수가 사전 크기(input_dim)이고 열 수가 1인 행렬입니다.

사전 크기를 20으로 설정합니다.

```{.python .input}
#@tab mxnet
embed_bias = nn.Embedding(input_dim=20, output_dim=1)
embed_bias.initialize()
embed_bias.weight
```

임베딩 레이어의 입력은 단어의 인덱스입니다. 단어의 인덱스 $i$를 입력하면, 임베딩 레이어는 가중치 값의 $i$번째 행을 편향 항으로 반환합니다.

```{.python .input}
#@tab mxnet
x = np.array([1, 2, 3])
embed_bias(x)
```

### GloVe 모델 순방향 계산 (GloVe Model Forward Calculation)

순방향 계산에서 GloVe 모델의 입력은 중심 타겟 단어 인덱스 `center`와 문맥 단어 인덱스 `context`를 포함합니다. 여기서 `center` 변수는 (배치 크기, 1) 모양을 갖고, `context` 변수도 (배치 크기, 1) 모양을 갖습니다. 이 두 변수는 먼저 단어 임베딩 레이어에 의해 단어 인덱스에서 단어 벡터로 변환됩니다.

```{.python .input  n=20}
#@tab mxnet
def GloVe(center, context, coocurrence, embed_v, embed_u,
          bias_v, bias_u, x_max, alpha):
    # v의 모양: (batch_size, embed_size)
    v = embed_v(center)
    # u의 모양: (batch_size, embed_size)
    u = embed_u(context)
    # b의 모양: (batch_size, )
    b = bias_v(center).squeeze()
    # c의 모양: (batch_size, )
    c = bias_u(context).squeeze()
    # embed_products의 모양: (batch_size,)
    embed_products = npx.batch_dot(np.expand_dims(v, 1),
                                   np.expand_dims(u, 2)).squeeze()
    # distance_expr의 모양: (batch_size,)
    distance_expr = np.power(embed_products + b +
                     c - np.log(coocurrence), 2)
    # weight의 모양: (batch_size,)
    weight = compute_weight(coocurrence)
    return weight * distance_expr
```

출력 모양이 (배치 크기, )여야 함을 확인합니다.

```{.python .input  n=21}
#@tab mxnet
embed_word = nn.Embedding(input_dim=20, output_dim=4)
embed_word.initialize()
GloVe(np.ones((2)), np.ones((2)), np.ones((2)), embed_word, embed_word,
      embed_bias, embed_bias, x_max = 2, alpha = 0.75).shape
```

## 훈련 (Training)

단어 임베딩 모델을 훈련하기 전에 모델의 손실 함수를 정의해야 합니다.

### 모델 파라미터 초기화 (Initializing Model Parameters)
단어의 임베딩 레이어와 추가 편향을 구성하고, 하이퍼파라미터 단어 벡터 차원 `embed_size`를 100으로 설정합니다.

```{.python .input  n=22}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=1),
        nn.Embedding(input_dim=len(vocab), output_dim=1))
```

### 훈련 (Training)

훈련 함수는 아래에 정의되어 있습니다.

```{.python .input  n=23}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, x_max, alpha, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'AdaGrad',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context, coocurrence = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                l = GloVe(center, context, coocurrence.astype('float32'),
                          net[0], net[1], net[2], net[3], x_max, alpha)
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

이제 GloVe 모델을 훈련할 수 있습니다.

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs = 0.1, 5
x_max, alpha = 100, 0.75
train(net, data_iter, lr, num_epochs, x_max, alpha)
```

## GloVe 모델 적용 (Applying the GloVe Model)

Glove 모델은 두 세트의 단어 벡터 `embed_v`와 `embed_u`를 생성합니다. `embed_v`와 `embed_u`는 동등하며 무작위 초기화 결과로만 차이가 납니다. 두 세트의 벡터는 동등하게 수행되어야 합니다. 일반적으로 우리는 `embed_v`+`embed_u`의 합을 단어 벡터로 사용하기로 선택합니다.


Glove 모델을 훈련한 후에도 여전히 두 단어 벡터의 코사인 유사도를 기반으로 단어 간의 의미 유사성을 나타낼 수 있습니다.

```{.python .input  n=13}
#@tab mxnet
def get_similar_tokens(query_token, k, embed_v, embed_u):
    W = embed_v.weight.data() + embed_u.weight.data()
    x = W[vocab[query_token]]
    # 코사인 유사도 계산. 수치 안정성을 위해 1e-9를 더합니다
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 입력 단어 제거
        print('cosine sim=%.3f: %s' % (cos[i], (vocab.idx_to_token[i])))

get_similar_tokens('chip', 3, net[0], net[1])
```

## 요약 (Summary)

* GloVe 모델을 사전 훈련할 수 있습니다.


## 연습 문제 (Exercises)



## 토론 (Discussions)