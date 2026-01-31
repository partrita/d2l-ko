```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Kaggle에서 주택 가격 예측하기 (Predicting House Prices on Kaggle)
:label:`sec_kaggle_house`

심층 네트워크를 구축 및 훈련하고 가중치 감쇠 및 드롭아웃과 같은 기술로 정규화하는 몇 가지 기본 도구를 소개했으므로, 
이제 Kaggle 대회에 참가하여 이 모든 지식을 실습에 적용할 준비가 되었습니다. 
주택 가격 예측 대회는 시작하기에 좋은 곳입니다. 
데이터가 상당히 일반적이며 전문 모델(오디오나 비디오처럼)이 필요할 수 있는 이국적인 구조를 보이지 않습니다. 
:citet:`De-Cock.2011`이 수집한 이 데이터셋은 2006년부터 2010년까지 아이오와주 에임스(Ames)의 주택 가격을 다룹니다. 
Harrison과 Rubinfeld(1978)의 유명한 [보스턴 주택 데이터셋](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)보다 훨씬 크며 더 많은 예제와 특성을 자랑합니다.


이 섹션에서는 데이터 전처리, 모델 설계 및 하이퍼파라미터 선택에 대한 세부 사항을 안내해 드립니다. 
실습 접근 방식을 통해 데이터 과학자로서의 경력에 지침이 될 몇 가지 직관을 얻으시길 바랍니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## 데이터 다운로드하기 (Downloading Data)

책 전반에 걸쳐 다양한 다운로드된 데이터셋에서 모델을 훈련하고 테스트할 것입니다. 
여기서는 zip 또는 tar 파일을 다운로드하고 압축을 풀기 위한 (**두 가지 유틸리티 함수를 구현**)합니다. 
다시 한번, 이러한 유틸리티 함수의 구현 세부 사항은 생략합니다.

```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """파일을 폴더로 다운로드하고 로컬 파일 경로를 반환합니다."""

def extract(filename, folder):
    """폴더에 zip/tar 파일을 풉니다."""
```

## Kaggle

[Kaggle](https://www.kaggle.com)은 머신러닝 대회를 주최하는 인기 있는 플랫폼입니다. 
각 대회는 데이터셋을 중심으로 하며, 많은 대회가 우승 솔루션에 상금을 제공하는 이해 관계자들에 의해 후원됩니다. 
플랫폼은 사용자가 포럼과 공유 코드를 통해 상호 작용하도록 도와 협업과 경쟁을 모두 촉진합니다. 
리더보드 추격이 종종 통제 불능 상태가 되어 연구자들이 근본적인 질문을 던지기보다는 전처리 단계에 근시안적으로 집중하게 되기도 하지만, 
경쟁하는 접근 방식 간의 직접적인 정량적 비교뿐만 아니라 코드 공유를 용이하게 하여 모든 사람이 무엇이 효과가 있었고 무엇이 효과가 없었는지 배울 수 있게 하는 플랫폼의 객관성에는 엄청난 가치가 있습니다. 
Kaggle 대회에 참가하려면 먼저 계정을 등록해야 합니다(:numref:`fig_kaggle` 참조).

![Kaggle 웹사이트.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing`에 설명된 주택 가격 예측 대회 페이지에서 데이터셋("Data" 탭 아래)을 찾고 예측을 제출하고 순위를 볼 수 있습니다. URL은 바로 여기에 있습니다:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![주택 가격 예측 대회 페이지.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 데이터셋 액세스 및 읽기 (Accessing and Reading the Dataset)

대회 데이터가 훈련 세트와 테스트 세트로 나뉘어 있다는 점에 유의하십시오. 
각 레코드에는 주택의 자산 가치와 거리 유형, 건축 연도, 지붕 유형, 지하실 상태 등의 속성이 포함되어 있습니다. 
특성은 다양한 데이터 유형으로 구성됩니다. 
예를 들어 건축 연도는 정수로, 지붕 유형은 이산형 범주 할당으로, 기타 특성은 부동 소수점 숫자로 표현됩니다. 
그리고 여기서 현실이 상황을 복잡하게 만듭니다: 일부 예제의 경우 일부 데이터가 완전히 누락되어 있으며 결측값은 단순히 "na"로 표시됩니다. 
각 주택의 가격은 훈련 세트에만 포함되어 있습니다(어쨌든 대회니까요). 
훈련 세트를 분할하여 검증 세트를 만들고 싶겠지만, Kaggle에 예측을 업로드한 후에만 공식 테스트 세트에서 모델을 평가할 수 있습니다. 
:numref:`fig_house_pricing`의 대회 탭에 있는 "Data" 탭에는 데이터를 다운로드할 수 있는 링크가 있습니다.

시작하기 위해, :numref:`sec_pandas`에서 소개한 [**`pandas`를 사용하여 데이터를 읽고 처리**]할 것입니다. 
편의를 위해 Kaggle 주택 데이터셋을 다운로드하고 캐시할 수 있습니다. 
이 데이터셋에 해당하는 파일이 이미 캐시 디렉토리에 존재하고 SHA-1이 `sha1_hash`와 일치하면, 우리 코드는 중복 다운로드로 인터넷을 막는 것을 방지하기 위해 캐시된 파일을 사용합니다.

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

훈련 데이터셋에는 1460개의 예제, 80개의 특성, 1개의 레이블이 포함되어 있으며, 검증 데이터에는 1459개의 예제와 80개의 특성이 포함되어 있습니다.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## 데이터 전처리 (Data Preprocessing)

처음 4개 예제에서 [**처음 4개와 마지막 2개의 특성뿐만 아니라 레이블(SalePrice)을 살펴봅시다**].

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

각 예제에서 첫 번째 특성이 식별자임을 알 수 있습니다. 
이는 모델이 각 훈련 예제를 결정하는 데 도움이 됩니다. 
편리하기는 하지만 예측 목적으로는 어떤 정보도 전달하지 않습니다. 
따라서 모델에 데이터를 공급하기 전에 데이터셋에서 제거할 것입니다. 
또한 다양한 데이터 유형을 감안할 때 모델링을 시작하기 전에 데이터를 전처리해야 합니다.


수치형 특성부터 시작해 봅시다. 
먼저, [**모든 결측값을 해당 특성의 평균으로 대체**]하는 휴리스틱을 적용합니다. 
그런 다음 모든 특성을 공통 스케일에 두기 위해, (***데이터를 표준화*하여 특성을 평균 0 및 단위 분산으로 재조정**)합니다:

$$x \leftarrow \frac{x - \mu}{\sigma},
$$

여기서 $\mu$와 $\sigma$는 각각 평균과 표준 편차를 나타냅니다. 
이것이 실제로 우리 특성(변수)을 평균 0 및 단위 분산을 갖도록 변환하는지 확인하기 위해, 
$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$이고 
$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$임을 유의하십시오. 
직관적으로 우리는 두 가지 이유로 데이터를 표준화합니다. 
첫째, 최적화에 편리함이 입증되었습니다. 
둘째, 어떤 특성이 관련성이 있을지 *사전적으로* 알지 못하므로, 한 특성에 할당된 계수에 다른 특성보다 더 많은 페널티를 주고 싶지 않기 때문입니다.

[**다음으로 이산 값을 다룹니다.**] 
여기에는 "MSZoning"과 같은 특성이 포함됩니다. 
앞서 다중 클래스 레이블을 벡터로 변환했던 것과 같은 방식으로 (**원-핫 인코딩으로 대체**)합니다(:numref:`subsec_classification-problem` 참조). 
예를 들어 "MSZoning"은 "RL"과 "RM" 값을 가정합니다. 
"MSZoning" 특성을 삭제하고, 값이 0 또는 1인 두 개의 새로운 지시 특성 "MSZoning_RL"과 "MSZoning_RM"이 생성됩니다. 
원-핫 인코딩에 따르면, "MSZoning"의 원래 값이 "RL"이면 "MSZoning_RL"은 1이고 "MSZoning_RM"은 0입니다. 
`pandas` 패키지가 이를 자동으로 수행해 줍니다.

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # ID와 레이블 열 제거
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # 수치형 열 표준화
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # NAN 수치형 특성을 0으로 대체
    features[numeric_features] = features[numeric_features].fillna(0)
    # 이산형 특성을 원-핫 인코딩으로 대체
    features = pd.get_dummies(features, dummy_na=True)
    # 전처리된 특성 저장
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

이 변환으로 특성 수가 79개에서 331개로 늘어난 것을 볼 수 있습니다(ID 및 레이블 열 제외).

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## 오차 측정 (Error Measure)

시작하기 위해 제곱 손실을 사용하여 선형 모델을 훈련할 것입니다. 놀랍지 않게도 우리 선형 모델은 대회 우승 제출물로 이어지지는 않겠지만, 데이터에 의미 있는 정보가 있는지 확인하는 정상성 확인을 제공합니다. 여기서 무작위 추측보다 더 잘할 수 없다면 데이터 처리 버그가 있을 가능성이 큽니다. 그리고 일이 잘 풀린다면, 선형 모델은 단순한 모델이 가장 잘 보고된 모델들에 얼마나 근접하는지에 대한 직관을 제공하는 베이스라인 역할을 하는 것으로, 더 화려한 모델에서 얼마나 많은 이득을 기대해야 하는지에 대한 감각을 줄 것입니다.

주식 가격과 마찬가지로 주택 가격의 경우, 우리는 절대적인 양보다 상대적인 양에 관심을 갖습니다. 
따라서 [**우리는 절대 오차 $y - \hat{y}$보다 상대 오차 $\frac{y - \hat{y}}{y}$에 더 관심을 갖는 경향이 있습니다.**] 
예를 들어, 일반적인 주택 가치가 125,000달러인 오하이오 시골 지역의 주택 가격을 추정할 때 예측이 100,000달러 빗나간다면 아마도 끔찍한 일을 하고 있는 것입니다. 
반면에 캘리포니아 로스 알토스 힐스(중위 주택 가격이 400만 달러를 초과함)에서 이 금액만큼 틀린다면, 이는 놀랍도록 정확한 예측을 나타낼 수 있습니다.

(**이 문제를 해결하는 한 가지 방법은 가격 추정치의 로그 불일치를 측정하는 것입니다.**) 
사실 이는 제출물의 품질을 평가하기 위해 대회에서 사용하는 공식적인 오차 측정치이기도 합니다. 
결국 $|\log y - \log \hat{y}| \leq \delta$에 대한 작은 값 $\delta$는 $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^{\delta}$로 변환됩니다. 
이것은 예측 가격의 로그와 레이블 가격의 로그 사이의 다음과 같은 제곱근 평균 제곱 오차(root-mean-squared-error)로 이어집니다:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$ 

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # 가격의 로그
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-겹 교차 검증 ($K$-Fold Cross-Validation)

모델 선택을 다루는 방법을 논의한 :numref:`subsec_generalization-model-selection`에서 [**교차 검증**]을 소개했던 것을 기억하실 것입니다. 
우리는 이를 모델 설계를 선택하고 하이퍼파라미터를 조정하는 데 유용하게 사용할 것입니다. 
먼저 $K$-겹 교차 검증 절차에서 데이터의 $i$번째 겹(fold)을 반환하는 함수가 필요합니다. 
이 함수는 $i$번째 세그먼트를 검증 데이터로 잘라내고 나머지를 훈련 데이터로 반환하는 방식으로 진행됩니다. 
이것이 데이터를 처리하는 가장 효율적인 방법은 아니며 데이터셋이 상당히 더 크다면 분명히 훨씬 더 똑똑한 방법을 사용할 것입니다. 
하지만 이 추가된 복잡성은 불필요하게 코드를 난독화할 수 있으므로 문제의 단순성 덕분에 여기서는 안전하게 생략할 수 있습니다.

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

$K$-겹 교차 검증에서 $K$번 훈련할 때 [**평균 검증 오차가 반환됩니다.**]

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'평균 검증 로그 mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**모델 선택 (Model Selection)**]

이 예제에서는 조정되지 않은 하이퍼파라미터 세트를 선택하고 모델을 개선하는 것은 독자에게 맡깁니다. 
좋은 선택을 찾는 것은 최적화하는 변수의 수에 따라 시간이 걸릴 수 있습니다. 
충분히 큰 데이터셋과 일반적인 종류의 하이퍼파라미터가 있다면, $K$-겹 교차 검증은 다중 테스트에 대해 상당히 탄력적인 경향이 있습니다. 
그러나 비합리적으로 많은 수의 옵션을 시도하면 검증 성능이 더 이상 실제 오차를 대표하지 않음을 알게 될 수도 있습니다.

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

때로는 하이퍼파라미터 세트에 대한 훈련 오차 수가 매우 낮을 수 있지만, $K$-겹 교차 검증의 오차 수는 상당히 높아질 수 있음에 유의하십시오. 
이는 과대적합되고 있음을 나타냅니다. 
훈련 내내 두 숫자를 모두 모니터링하고 싶을 것입니다. 
과대적합이 적다는 것은 데이터가 더 강력한 모델을 지원할 수 있음을 나타낼 수 있습니다. 
엄청난 과대적합은 정규화 기술을 통합하여 이득을 얻을 수 있음을 시사할 수 있습니다.

## [**Kaggle에 예측 제출하기**]

이제 하이퍼파라미터의 좋은 선택이 무엇인지 알았으므로, 모든 $K$개 모델에 의한 테스트 세트에서의 평균 예측을 계산할 수 있습니다. 
예측을 csv 파일에 저장하면 결과를 Kaggle에 업로드하는 것이 간편해집니다. 
다음 코드는 `submission.csv`라는 파일을 생성합니다.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
# 로그 스케일에서의 예측을 지수화
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

다음으로 :numref:`fig_kaggle_submit2`에 설명된 대로 Kaggle에 예측을 제출하고 테스트 세트의 실제 주택 가격(레이블)과 비교해 볼 수 있습니다. 
단계는 꽤 간단합니다:

* Kaggle 웹사이트에 로그인하고 주택 가격 예측 대회 페이지를 방문합니다.
* "Submit Predictions" 또는 "Late Submission" 버튼을 클릭합니다.
* 페이지 하단의 점선 상자에 있는 "Upload Submission File" 버튼을 클릭하고 업로드할 예측 파일을 선택합니다.
* 페이지 하단의 "Make Submission" 버튼을 클릭하여 결과를 확인합니다.

![Kaggle에 데이터 제출하기.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 요약 및 토론 (Summary and Discussion)

실제 데이터는 종종 다양한 데이터 유형의 혼합을 포함하며 전처리가 필요합니다. 
실수 값 데이터를 평균 0 및 단위 분산으로 재조정하는 것은 좋은 기본값입니다. 결측값을 평균으로 대체하는 것도 마찬가지입니다. 
또한 범주형 특성을 지시 특성으로 변환하면 원-핫 벡터처럼 취급할 수 있습니다. 
절대 오차보다 상대 오차에 더 신경 쓸 때는 예측의 로그 불일치를 측정할 수 있습니다. 
모델을 선택하고 하이퍼파라미터를 조정하기 위해 $K$-겹 교차 검증을 사용할 수 있습니다.



## 연습 문제 (Exercises)

1. 이 섹션에 대한 예측을 Kaggle에 제출하십시오. 얼마나 좋습니까?
2. 결측값을 평균으로 대체하는 것이 항상 좋은 아이디어일까요? 힌트: 값이 무작위로 누락되지 않는 상황을 구성할 수 있습니까?
3. $K$-겹 교차 검증을 통해 하이퍼파라미터를 튜닝하여 점수를 개선하십시오.
4. 모델(예: 레이어, 가중치 감쇠, 드롭아웃)을 개선하여 점수를 개선하십시오.
5. 이 섹션에서 한 것처럼 연속 수치형 특성을 표준화하지 않으면 어떻게 됩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17988)
:end_tab: