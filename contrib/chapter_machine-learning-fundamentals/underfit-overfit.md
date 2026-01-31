```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 과소적합과 과대적합 (Underfitting and Overfitting)
:label:`sec_polynomial`

이 섹션에서는 이전에 보았던 몇 가지 개념을 테스트해 봅니다. 문제를 단순하게 유지하기 위해, 다항식 회귀를 장난감 예제로 사용합니다.

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import math
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

### 데이터셋 생성하기 (Generating the Dataset)

먼저 데이터가 필요합니다. $x$가 주어졌을 때, 훈련 및 테스트 데이터에 대해 레이블을 생성하기 위해 [**다음과 같은 3차 다항식을 사용합니다**]:

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ 여기서 }\n\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

노이즈 항 $\epsilon$은 평균 0, 표준 편차 0.1인 정규 분포를 따릅니다. 최적화를 위해 일반적으로 기울기나 손실의 값이 매우 커지는 것을 피하고 싶어 합니다. 이것이 *특성(features)*이 $x^i$에서 $\frac{x^i}{i!}$로 재조정된 이유입니다. 이는 큰 지수 $i$에 대해 매우 큰 값을 피할 수 있게 해 줍니다. 훈련 세트와 테스트 세트에 대해 각각 100개의 샘플을 합성할 것입니다.

```{.python .input  n=6}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()        
        p, n = max(3, self.num_inputs), num_train + num_val
        w = d2l.tensor([1.2, -3.4, 5.6] + [0]*(p-3))
        if tab.selected('mxnet') or tab.selected('pytorch'):
            x = d2l.randn(n, 1)
            noise = d2l.randn(n, 1) * 0.1
        if tab.selected('tensorflow'):
            x = d2l.normal((n, 1))
            noise = d2l.normal((n, 1)) * 0.1
        X = d2l.concat([x ** (i+1) / math.gamma(i+2) for i in range(p)], 1)
        self.y = d2l.matmul(X, d2l.reshape(w, (-1, 1))) + noise
        self.X = X[:,:num_inputs]
        
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

다시 한번, `poly_features`에 저장된 단항식(monomials)은 감마 함수에 의해 재조정되며, 여기서 $\Gamma(n)=(n-1)!$입니다. 생성된 데이터셋에서 [**처음 2개의 샘플을 살펴봅시다**]. 값 1은 기술적으로 특성, 즉 편향(bias)에 대응하는 상수 특성입니다.

### [**3차 다항식 함수 피팅 (정상)**]

먼저 데이터 생성 함수와 동일한 차수인 3차 다항식 함수를 사용하는 것으로 시작하겠습니다. 결과는 이 모델의 훈련 및 테스트 손실이 모두 효과적으로 줄어들 수 있음을 보여줍니다. 학습된 모델 파라미터 또한 실제 값인 $w = [1.2, -3.4, 5.6], b=5$에 가깝습니다.

```{.python .input  n=7}
%%tab all
def train(p):
    if tab.selected('mxnet') or tab.selected('tensorflow'):
        model = d2l.LinearRegression(lr=0.01)
    if tab.selected('pytorch'):
        model = d2l.LinearRegression(p, lr=0.01)
    model.board.ylim = [1, 1e2]
    data = Data(200, 200, p, 20)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    print(model.get_w_b())
    
train(p=3)
```

### [**선형 함수 피팅 (과소적합)**]

선형 함수 피팅을 다시 한번 살펴봅시다. 초기 에포크에서의 하락 이후, 이 모델의 훈련 손실을 더 줄이는 것은 어려워집니다. 마지막 에포크 반복이 완료된 후에도 훈련 손실은 여전히 높습니다. 비선형 패턴(여기서는 3차 다항식 함수)을 피팅하는 데 사용될 때 선형 모델은 과소적합(underfit)되기 쉽습니다.

```{.python .input  n=8}
%%tab all
train(p=1)
```

### [**고차 다항식 함수 피팅 (과대적합)**]

이제 너무 높은 차수의 다항식을 사용하여 모델을 훈련해 봅시다. 여기서는 고차 항의 계수 값이 0에 가까워야 한다는 것을 배우기에 데이터가 충분하지 않습니다. 그 결과, 우리의 과하게 복잡한 모델은 너무 민감해져서 훈련 데이터의 노이즈에 영향을 받게 됩니다. 훈련 손실은 효과적으로 줄어들 수 있지만, 테스트 손실은 여전히 훨씬 더 높습니다. 이는 복잡한 모델이 데이터에 과대적합(overfit)되었음을 보여줍니다.

```{.python .input  n=9}
%%tab all
train(p=10)
```

후속 섹션들에서 우리는 과대적합 문제와 가중치 감쇠(weight decay) 및 드롭아웃(dropout)과 같은 과대적합 처리 방법들에 대해 계속 논의할 것입니다.


## 요약 (Summary)

* 일반화 오차는 훈련 오차를 기반으로 추정될 수 없으므로, 단순히 훈련 오차를 최소화하는 것이 반드시 일반화 오차의 감소를 의미하지는 않습니다. 머신러닝 모델은 일반화 오차를 최소화하기 위해 과대적합을 방지하도록 주의해야 합니다.
* 검증 세트는 너무 방만하게 사용되지 않는 한 모델 선택을 위해 사용될 수 있습니다.
* 과소적합은 모델이 훈련 오차를 줄일 수 없음을 의미합니다. 훈련 오차가 검증 오차보다 훨씬 낮을 때 과대적합이 발생합니다.
* 우리는 적절하게 복잡한 모델을 선택해야 하며 불충분한 훈련 샘플을 사용하는 것을 피해야 합니다.


## 연습 문제 (Exercises)

1. 다항식 회귀 문제를 정확하게 풀 수 있습니까? 힌트: 선형 대수를 사용하십시오.
2. 다항식에 대한 모델 선택을 고려해 보십시오:
    1. 훈련 손실 대 모델 복잡도(다항식의 차수)를 플롯하십시오. 무엇을 관찰하셨습니까? 훈련 손실을 0으로 줄이려면 몇 차 다항식이 필요합니까?
    2. 이 경우의 테스트 손실을 플롯하십시오.
    3. 데이터 양의 함수로서 동일한 플롯을 생성하십시오.
3. 다항식 특성 $x^i$의 정규화($1/i!$)를 생략하면 어떻게 됩니까? 다른 방법으로 이를 고칠 수 있습니까?
4. 일반화 오차가 0이 되는 것을 볼 수 있을 것으로 기대할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/234)
:end_tab: