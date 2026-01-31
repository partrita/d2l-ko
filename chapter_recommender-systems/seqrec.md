# 시퀀스 인식 추천 시스템 (Sequence-Aware Recommender Systems)

이전 섹션에서는 사용자의 단기 행동을 고려하지 않고 추천 작업을 행렬 완성 문제로 추상화했습니다. 이 섹션에서는 순차적으로 정렬된 사용자 상호 작용 로그를 고려하는 추천 모델을 소개합니다. 이것은 입력이 과거 사용자 행동의 정렬되고 종종 타임스탬프가 있는 목록인 시퀀스 인식 추천기 :cite:`Quadrana.Cremonesi.Jannach.2018`입니다. 최근의 많은 문헌은 사용자의 시간적 행동 패턴을 모델링하고 관심 변화를 발견하는 데 이러한 정보를 통합하는 것의 유용성을 입증했습니다.

우리가 소개할 모델인 Caser :cite:`Tang.Wang.2018`는 합성곱 시퀀스 임베딩 추천 모델(convolutional sequence embedding recommendation model)의 약자로, 합성곱 신경망을 채택하여 사용자의 최근 활동의 동적 패턴 영향을 포착합니다. Caser의 주요 구성 요소는 수평 합성곱 네트워크와 수직 합성곱 네트워크로 구성되며, 각각 결합 수준(union-level) 및 포인트 수준(point-level) 시퀀스 패턴을 밝히는 것을 목표로 합니다. 포인트 수준 패턴은 과거 시퀀스의 단일 항목이 대상 항목에 미치는 영향을 나타내는 반면, 결합 수준 패턴은 여러 이전 행동이 후속 대상에 미치는 영향을 나타냅니다. 예를 들어, 우유와 버터를 함께 구매하면 둘 중 하나만 구매하는 것보다 밀가루를 구매할 확률이 더 높습니다. 또한 사용자의 일반적인 관심사 또는 장기적인 선호도도 마지막 완전 연결 레이어에서 모델링되어 사용자 관심사를 보다 포괄적으로 모델링할 수 있습니다. 모델의 세부 사항은 다음과 같습니다.

## 모델 아키텍처 (Model Architectures)

시퀀스 인식 추천 시스템에서, 각 사용자는 항목 집합의 일부 항목 시퀀스와 연관됩니다. $S^u = (S_1^u, ... S_{|S_u|}^u)$를 정렬된 시퀀스라고 합시다. Caser의 목표는 사용자의 일반적인 취향과 단기 의도를 고려하여 항목을 추천하는 것입니다. 이전 $L$개 항목을 고려한다고 가정하면 타임 스텝 $t$에 대한 이전 상호 작용을 나타내는 임베딩 행렬을 구성할 수 있습니다.

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

여기서 $\mathbf{Q} \in \mathbb{R}^{n \times k}$는 항목 임베딩을 나타내고 $\mathbf{q}_i$는 $i^\textrm{th}$ 행을 나타냅니다. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$는 타임 스텝 $t$에서 사용자 $u$의 일시적 관심을 추론하는 데 사용할 수 있습니다. 입력 행렬 $\mathbf{E}^{(u, t)}$를 후속 두 합성곱 구성 요소의 입력인 이미지로 볼 수 있습니다.

수평 합성곱 레이어에는 $d$개의 수평 필터 $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$가 있고, 수직 합성곱 레이어에는 $d'$개의 수직 필터 $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$가 있습니다. 일련의 합성곱 및 풀링 연산 후 두 가지 출력을 얻습니다.

$$
\mathbf{o} = \textrm{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \
\mathbf{o}'= \textrm{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

여기서 $\mathbf{o} \in \mathbb{R}^d$는 수평 합성곱 네트워크의 출력이고 $\mathbf{o}' \in \mathbb{R}^{kd'}$는 수직 합성곱 네트워크의 출력입니다. 단순화를 위해 합성곱 및 풀링 연산의 세부 사항은 생략합니다. 이들은 연결되어 완전 연결 신경망 레이어에 공급되어 더 높은 수준의 표현을 얻습니다.

$$ 
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

여기서 $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$는 가중치 행렬이고 $\mathbf{b} \in \mathbb{R}^k$는 편향입니다. 학습된 벡터 $\mathbf{z} \in \mathbb{R}^k$는 사용자의 단기 의도를 나타냅니다.

마지막으로, 예측 함수는 사용자의 단기 및 일반적인 취향을 결합하며 다음과 같이 정의됩니다.

$$ 
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

여기서 $\mathbf{V} \in \mathbb{R}^{n \times 2k}$는 또 다른 항목 임베딩 행렬입니다. $\mathbf{b}' \in \mathbb{R}^n$은 항목별 편향입니다. $\mathbf{P} \in \mathbb{R}^{m \times k}$는 사용자의 일반적인 취향을 위한 사용자 임베딩 행렬입니다. $\mathbf{p}_u \in \mathbb{R}^{ k}$는 $P$의 $u^\textrm{th}$ 행이고 $\mathbf{v}_i \in \mathbb{R}^{2k}$는 $\mathbf{V}$의 $i^\textrm{th}$ 행입니다.

모델은 BPR 또는 힌지 손실로 학습할 수 있습니다. Caser의 아키텍처는 다음과 같습니다.

![Caser 모델 그림](../img/rec-caser.svg)

먼저 필요한 라이브러리를 가져옵니다.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## 모델 구현 (Model Implementation)
다음 코드는 Caser 모델을 구현합니다. 수직 합성곱 레이어, 수평 합성곱 레이어 및 완전 연결 레이어로 구성됩니다.

```{.python .input  n=4}
#@tab mxnet
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # 수직 합성곱 레이어
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # 수평 합성곱 레이어
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # 완전 연결 레이어
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## 네거티브 샘플링을 사용한 순차적 데이터셋 (Sequential Dataset with Negative Sampling)
순차적 상호 작용 데이터를 처리하려면 `Dataset` 클래스를 재구현해야 합니다. 다음 코드는 `SeqDataset`이라는 새 데이터셋 클래스를 생성합니다. 각 샘플에서 사용자 ID, 이전 $L$개의 상호 작용 항목을 시퀀스로, 그리고 다음 상호 작용 항목을 대상으로 출력합니다. 다음 그림은 한 사용자의 데이터 로딩 프로세스를 보여줍니다. 이 사용자가 9개의 영화를 좋아했다고 가정하면, 이 9개의 영화를 시간순으로 정리합니다. 최신 영화는 테스트 항목으로 남겨둡니다. 나머지 8개의 영화에 대해 3개의 훈련 샘플을 얻을 수 있으며, 각 샘플에는 5개($L=5$) 영화 시퀀스와 후속 항목이 대상 항목으로 포함됩니다. 네거티브 샘플도 사용자 정의 데이터셋에 포함됩니다.

![데이터 생성 프로세스 그림](../img/rec-seq-data.svg)

```{.python .input  n=5}
#@tab mxnet
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## MovieLens 100K 데이터셋 로드 (Load the MovieLens 100K dataset)

그 후, MovieLens 100K 데이터셋을 시퀀스 인식 모드로 읽고 분할하고 위에서 구현된 순차적 데이터 로더를 사용하여 훈련 데이터를 로드합니다.

```{.python .input  n=6}
#@tab mxnet
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

훈련 데이터 구조는 위에 나와 있습니다. 첫 번째 요소는 사용자 ID이고, 다음 목록은 이 사용자가 좋아한 지난 5개의 항목을 나타내며, 마지막 요소는 사용자가 5개 항목 이후에 좋아한 항목입니다.

## 모델 훈련 (Train the Model)
이제 모델을 훈련해 봅시다. 결과를 비교할 수 있도록 지난 섹션의 NeuMF와 동일한 설정(학습률, 최적화 도구 및 $k$ 포함)을 사용합니다.

```{.python .input  n=7}
#@tab mxnet
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# 실행에 1시간 이상 소요됨 (MXNet 수정 대기 중)
# d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter, num_users, num_items, num_epochs, devices, d2l.evaluate_ranking, candidates, eval_step=1)
```

## 요약 (Summary)
* 사용자의 단기 및 장기 관심사를 추론하면 사용자가 선호하는 다음 항목을 더 효과적으로 예측할 수 있습니다.
* 합성곱 신경망을 활용하여 순차적 상호 작용에서 사용자의 단기 관심사를 포착할 수 있습니다.

## 연습 문제 (Exercises)

* 수평 및 수직 합성곱 네트워크 중 하나를 제거하여 절제 연구를 수행하십시오. 어떤 구성 요소가 더 중요합니까?
* 하이퍼파라미터 $L$을 변경해 보십시오. 과거 상호 작용이 더 길수록 정확도가 높아집니까?
* 위에서 소개한 시퀀스 인식 추천 작업 외에도 세션 기반 추천 :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`이라는 또 다른 유형의 시퀀스 인식 추천 작업이 있습니다. 이 두 작업의 차이점을 설명할 수 있습니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/404)
:end_tab: