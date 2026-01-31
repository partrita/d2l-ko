# 심층 인수분해 머신 (Deep Factorization Machines)

효과적인 특성 조합을 학습하는 것은 클릭률 예측 작업의 성공에 중요합니다. 인수분해 머신은 선형 패러다임(예: 이중 선형 상호 작용)으로 특성 상호 작용을 모델링합니다. 이는 내재된 특성 교차 구조가 일반적으로 매우 복잡하고 비선형적인 실제 데이터에는 종종 불충분합니다. 설상가상으로 실제로는 인수분해 머신에서 2차 특성 상호 작용이 일반적으로 사용됩니다. 인수분해 머신으로 더 높은 차수의 특성 조합을 모델링하는 것은 이론적으로 가능하지만 수치적 불안정성과 높은 계산 복잡성으로 인해 일반적으로 채택되지 않습니다.

한 가지 효과적인 해결책은 심층 신경망을 사용하는 것입니다. 심층 신경망은 특성 표현 학습에 강력하며 정교한 특성 상호 작용을 학습할 수 있는 잠재력이 있습니다. 따라서 심층 신경망을 인수분해 머신에 통합하는 것은 자연스러운 일입니다. 인수분해 머신에 비선형 변환 레이어를 추가하면 저차 특성 조합과 고차 특성 조합을 모두 모델링할 수 있는 기능이 제공됩니다. 또한 입력의 비선형 고유 구조도 심층 신경망으로 포착할 수 있습니다. 이 섹션에서는 FM과 심층 신경망을 결합한 심층 인수분해 머신(DeepFM) :cite:`Guo.Tang.Ye.ea.2017`이라는 대표적인 모델을 소개합니다.


## 모델 아키텍처 (Model Architectures)

DeepFM은 병렬 구조로 통합된 FM 구성 요소와 딥 구성 요소로 구성됩니다. FM 구성 요소는 저차 특성 상호 작용을 모델링하는 데 사용되는 2-way 인수분해 머신과 동일합니다. 딥 구성 요소는 고차 특성 상호 작용과 비선형성을 포착하는 데 사용되는 MLP입니다. 이 두 구성 요소는 동일한 입력/임베딩을 공유하고 그 출력은 최종 예측으로 합산됩니다. DeepFM의 정신은 암기와 일반화를 모두 포착할 수 있는 Wide & Deep 아키텍처와 유사하다는 점을 지적할 가치가 있습니다. Wide & Deep 모델에 비해 DeepFM의 장점은 특성 조합을 자동으로 식별하여 수작업 특성 엔지니어링 노력을 줄여준다는 것입니다.

간결함을 위해 FM 구성 요소에 대한 설명은 생략하고 출력을 $\hat{y}^{(FM)}$으로 표시합니다. 자세한 내용은 지난 섹션을 참조하십시오. $\mathbf{e}_i \in \mathbb{R}^{k}$를 $i^\textrm{th}$ 필드의 잠재 특성 벡터라고 합시다. 딥 구성 요소의 입력은 희소 범주형 특성 입력으로 조회된 모든 필드의 밀집 임베딩의 연결이며 다음과 같이 표시됩니다.

$$ 
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$ 

여기서 $f$는 필드 수입니다. 그런 다음 다음 신경망에 공급됩니다.

$$ 
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$ 

여기서 $\alpha$는 활성화 함수입니다. $\mathbf{W}_{l}$과 $\mathbf{b}_{l}$은 $l^\textrm{th}$ 레이어의 가중치와 편향입니다. $y_{DNN}$을 예측의 출력이라고 합시다. DeepFM의 최종 예측은 FM과 DNN의 출력의 합입니다. 따라서 다음을 얻습니다.

$$ 
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$ 

여기서 $\sigma$는 시그모이드 함수입니다. DeepFM의 아키텍처는 아래와 같습니다.
![DeepFM 모델 그림](../img/rec-deepfm.svg)

DeepFM만이 심층 신경망과 FM을 결합하는 유일한 방법은 아니라는 점에 유의해야 합니다. 특성 상호 작용에 비선형 레이어를 추가할 수도 있습니다 :cite:`He.Chua.2017`.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## DeepFM 구현 (Implementation of DeepFM)

DeepFM의 구현은 FM의 구현과 유사합니다. FM 부분은 변경하지 않고 활성화 함수로 `relu`가 있는 MLP 블록을 사용합니다. 드롭아웃도 모델을 정규화하는 데 사용됩니다. MLP의 뉴런 수는 `mlp_dims` 하이퍼파라미터로 조정할 수 있습니다.

```{.python .input  n=2}
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## 모델 훈련 및 평가 (Training and Evaluating the Model)

데이터 로딩 프로세스는 FM과 동일합니다. DeepFM의 MLP 구성 요소를 피라미드 구조(30-20-10)가 있는 3레이어 밀집 네트워크로 설정합니다. 다른 모든 하이퍼파라미터는 FM과 동일하게 유지됩니다.

```{.python .input  n=4}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

FM과 비교할 때 DeepFM은 더 빨리 수렴하고 더 나은 성능을 달성합니다.

## 요약 (Summary)

* 신경망을 FM에 통합하면 복잡한 고차 상호 작용을 모델링할 수 있습니다.
* DeepFM은 광고 데이터셋에서 원래 FM보다 성능이 뛰어납니다.

## 연습 문제 (Exercises)

* MLP의 구조를 변경하여 모델 성능에 미치는 영향을 확인하십시오.
* 데이터셋을 Criteo로 변경하고 원래 FM 모델과 비교하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:

```