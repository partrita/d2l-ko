# 인수 분해 머신 (Factorization Machines)

:citet:`Rendle.2010`에 의해 제안된 인수 분해 머신(Factorization Machines, FM)은 분류, 회귀 및 순위 지정 작업에 사용할 수 있는 지도 학습 알고리즘입니다. 이는 빠르게 주목을 받았으며 예측 및 추천을 위한 인기 있고 영향력 있는 방법이 되었습니다. 특히, 이는 선형 회귀 모델과 행렬 분해 모델의 일반화입니다. 더욱이, 이는 다항식 커널을 가진 서포트 벡터 머신을 연상시킵니다. 선형 회귀 및 행렬 분해에 비해 인수 분해 머신의 강점은 다음과 같습니다: (1) $\chi$-way 변수 상호 작용을 모델링할 수 있습니다. 여기서 $\chi$는 다항식 차수이며 일반적으로 2로 설정됩니다. (2) 인수 분해 머신과 관련된 빠른 최적화 알고리즘은 다항식 계산 시간을 선형 복잡도로 줄여줄 수 있어, 특히 고차원 희소 입력에 대해 매우 효율적입니다. 이러한 이유로 인수 분해 머신은 현대적인 광고 및 제품 추천에 널리 사용됩니다. 기술적인 세부 사항과 구현은 아래에 설명되어 있습니다.


## 2-Way 인수 분해 머신

공식적으로, $x \in \mathbb{R}^d$를 한 샘플의 특성 벡터라고 하고, $y$를 그에 대응하는 레이블이라고 합시다. 레이블은 실수 값 레이블이거나 이진 클래스 "클릭/비클릭"과 같은 클래스 레이블일 수 있습니다. 차수가 2인 인수 분해 머신의 모델은 다음과 같이 정의됩니다:

$$ 
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j 
$$ 

여기서 $\mathbf{w}_0 \in \mathbb{R}$은 전역 편향(global bias)이고; $\mathbf{w} \in \mathbb{R}^d$는 $i$번째 변수의 가중치를 나타내며; $\mathbf{V} \in \mathbb{R}^{d\times k}$는 특성 임베딩을 나타냅니다. $\mathbf{v}_i$는 $\mathbf{V}$의 $i^{\textrm{th}}$번째 행을 나타내고; $k$는 잠재 요인의 차원이며; $\langle\cdot, \cdot \rangle$은 두 벡터의 내적입니다. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$은 $i^{\textrm{th}}$번째와 $j^{\textrm{th}}$번째 특성 간의 상호 작용을 모델링합니다. 일부 특성 상호 작용은 쉽게 이해될 수 있으므로 전문가에 의해 설계될 수 있습니다. 그러나 대부분의 다른 특성 상호 작용은 데이터 속에 숨겨져 있어 식별하기 어렵습니다. 따라서 특성 상호 작용을 자동으로 모델링하면 특성 엔지니어링의 노력을 크게 줄일 수 있습니다. 처음 두 항이 선형 회귀 모델에 대응하고 마지막 항이 행렬 분해 모델의 확장임은 명백합니다. 특성 $i$가 항목을 나타내고 특성 $j$가 사용자를 나타낸다면, 세 번째 항은 정확히 사용자 임베딩과 항목 임베딩 간의 내적입니다. FM이 더 높은 차수(차수 > 2)로 일반화될 수도 있다는 점은 주목할 가치가 있습니다. 그럼에도 불구하고 수치적 안정성이 일반화 능력을 약화시킬 수 있습니다.


## 효율적인 최적화 기준 (An Efficient Optimization Criterion)

인수 분해 머신을 단순한 방식으로 최적화하면 모든 쌍별 상호 작용을 계산해야 하므로 $\mathcal{O}(kd^2)$의 복잡도가 발생합니다. 이 비효율성 문제를 해결하기 위해, 우리는 FM의 세 번째 항을 재구성하여 계산 비용을 크게 줄여 선형 시간 복잡도($\mathcal{O}(kd)$)로 이끌 수 있습니다. 쌍별 상호 작용 항의 재구성은 다음과 같습니다:

$$ 
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \ 
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \ 
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \ 
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$ 

이 재구성을 통해 모델 복잡도가 크게 감소합니다. 더욱이, 희소 특성의 경우 0이 아닌 요소만 계산하면 되므로 전체 복잡도는 0이 아닌 특성의 수에 선형적입니다.

FM 모델을 학습하기 위해, 회귀 작업에는 MSE 손실을, 분류 작업에는 크로스 엔트로피 손실을, 순위 지정 작업에는 BPR 손실을 사용할 수 있습니다. 확률적 경사 하강법 및 Adam과 같은 표준 최적화기를 최적화에 사용할 수 있습니다.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## 모델 구현 (Model Implementation)
다음 코드는 인수 분해 머신을 구현합니다. FM이 선형 회귀 블록과 효율적인 특성 상호 작용 블록으로 구성되어 있음을 명확히 알 수 있습니다. 우리는 CTR 예측을 분류 작업으로 취급하므로 최종 점수에 시그모이드 함수를 적용합니다.

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## 광고 데이터셋 로드 (Load the Advertising Dataset)
우리는 이전 섹션의 CTR 데이터 래퍼를 사용하여 온라인 광고 데이터셋을 로드합니다.

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 모델 훈련 (Train the Model)
그 후, 모델을 훈련합니다. 학습률은 0.02로 설정하고 임베딩 크기는 기본적으로 20으로 설정합니다. 모델 훈련을 위해 `Adam` 최적화기와 `SigmoidBinaryCrossEntropyLoss` 손실을 사용합니다.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 요약 (Summary)

* FM은 회귀, 분류 및 순위 지정과 같은 다양한 작업에 적용될 수 있는 일반적인 프레임워크입니다.
* 특성 상호 작용/교차(crossing)는 예측 작업에 중요하며, 2-way 상호 작용은 FM으로 효율적으로 모델링될 수 있습니다.

## 연습 문제 (Exercises)

* Avazu, MovieLens, Criteo 데이터셋과 같은 다른 데이터셋에서 FM을 테스트할 수 있습니까?
* 임베딩 크기를 변경하여 성능에 미치는 영향을 확인해 보십시오. 행렬 분해와 유사한 패턴을 관찰할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/406)
:end_tab:

```