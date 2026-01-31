# 추천 시스템을 위한 개인화 순위 (Personalized Ranking for Recommender Systems)

이전 섹션에서는 명시적 피드백만 고려되었으며 모델은 관찰된 평점으로 훈련 및 테스트되었습니다. 이러한 방법에는 두 가지 단점이 있습니다. 첫째, 대부분의 피드백은 실제 시나리오에서 명시적이지 않고 암시적이며, 명시적 피드백은 수집하는 데 비용이 더 많이 들 수 있습니다. 둘째, 사용자 관심사를 예측할 수 있는 관찰되지 않은 사용자-항목 쌍이 완전히 무시되어, 평점이 무작위가 아니라 사용자 선호도 때문에 누락된 경우에 이러한 방법이 부적합합니다. 관찰되지 않은 사용자-항목 쌍은 실제 부정적 피드백(사용자가 항목에 관심이 없음)과 누락된 값(사용자가 나중에 항목과 상호 작용할 수 있음)의 혼합입니다. 행렬 분해 및 AutoRec에서는 관찰되지 않은 쌍을 단순히 무시합니다. 분명히 이러한 모델은 관찰된 쌍과 관찰되지 않은 쌍을 구별할 수 없으며 일반적으로 개인화 순위 작업에 적합하지 않습니다.

이를 위해 암시적 피드백에서 순위가 매겨진 추천 목록을 생성하는 것을 목표로 하는 추천 모델 클래스가 인기를 얻었습니다. 일반적으로 개인화 순위 모델은 포인트와이즈(pointwise), 페어와이즈(pairwise) 또는 리스트와이즈(listwise) 접근 방식으로 최적화할 수 있습니다. 포인트와이즈 접근 방식은 한 번에 단일 상호 작용을 고려하고 분류기 또는 회귀 분석기를 훈련하여 개별 선호도를 예측합니다. 행렬 분해 및 AutoRec은 포인트와이즈 목표로 최적화됩니다. 페어와이즈 접근 방식은 각 사용자에 대해 항목 쌍을 고려하고 해당 쌍에 대한 최적의 순서를 근사하는 것을 목표로 합니다. 일반적으로 페어와이즈 접근 방식은 상대적 순서를 예측하는 것이 순위의 본질을 상기시키기 때문에 순위 작업에 더 적합합니다. 리스트와이즈 접근 방식은 전체 항목 목록의 순서를 근사합니다. 예를 들어 Normalized Discounted Cumulative Gain ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain))과 같은 순위 척도를 직접 최적화합니다. 그러나 리스트와이즈 접근 방식은 포인트와이즈 또는 페어와이즈 접근 방식보다 더 복잡하고 계산 집약적입니다. 이 섹션에서는 두 가지 페어와이즈 목표/손실인 베이지안 개인화 순위 손실과 힌지 손실 및 그 구현을 소개합니다.

## 베이지안 개인화 순위 손실 및 구현 (Bayesian Personalized Ranking Loss and its Implementation)

베이지안 개인화 순위(BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009`는 최대 사후 추정기(maximum posterior estimator)에서 파생된 페어와이즈 개인화 순위 손실입니다. 많은 기존 추천 모델에서 널리 사용되었습니다. BPR의 훈련 데이터는 긍정적 쌍과 부정적 쌍(누락된 값)으로 구성됩니다. 사용자가 관찰되지 않은 다른 모든 항목보다 긍정적인 항목을 선호한다고 가정합니다.

공식적으로 훈련 데이터는 $(u, i, j)$ 형식의 튜플로 구성되며, 이는 사용자 $u$가 항목 $j$보다 항목 $i$를 선호함을 나타냅니다. 사후 확률을 최대화하는 것을 목표로 하는 BPR의 베이지안 공식은 다음과 같습니다.

$$
p(Θ t >_u )  r c a b c p(>_u t t Θ) p(Θ)
$$ 

여기서 $Θ$는 임의의 추천 모델의 파라미터를 나타내고, $>_u$는 사용자 $u$에 대한 모든 항목의 원하는 개인화된 전체 순위를 나타냅니다. 최대 사후 추정기를 공식화하여 개인화 순위 작업에 대한 일반적인 최적화 기준을 도출할 수 있습니다.

$$ 
\beginredived
\textbf{BPR-OPT} : & \ln p(Θ t >_u) \ \b \c \ln p(>_u t t Θ) p(Θ) \ &= \ln \product_{(u, i, j \in D)} \sigma(\hty}_{ui} - \hty}_{uj}) p(Θ) \ &= \sum_{(u, i, j \in D)} \ln \sigma(\hty}_{ui} - \hty}_{uj}) + \ln p(Θ) \ &= \sum_{(u, i, j \in D)} \ln \sigma(\hty}_{ui} - \hty}_{uj}) - \lambda_Θ ||Θ ||^2
endredived
$$ 


여기서 $D \def = {(u, i, j) | i \in I^+_u \^ j \in I \\I I^+_u }$는 훈련 세트이며, $I^+_u$는 사용자 $u$가 좋아한 항목을 나타내고, $I$는 모든 항목을 나타내며, $I \I I^+_u$는 사용자가 좋아한 항목을 제외한 다른 모든 항목을 나타냅니다. $\hty}_{ui}$와 $\hty}_{uj}$는 각각 사용자 $u$가 항목 $i$와 $j$에 대해 예측한 점수입니다. 사전 $p(Θ)$는 0 평균과 분산-공분산 행렬 $Σ}_Θ$를 갖는 정규 분포입니다. 여기서 $Σ}_Θ = \lambda_Θ I$로 둡니다.

![베이지안 개인화 순위 그림](../img/rec-ranking.svg)
우리는 기본 클래스 `mxnet.gluon.loss.Loss`를 구현하고 `forward` 메서드를 재정의하여 베이지안 개인화 순위 손실을 구성할 것입니다. Loss 클래스와 np 모듈을 가져오는 것으로 시작합니다.

```{.python .input  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

BPR 손실의 구현은 다음과 같습니다.

```{.python .input  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## 힌지 손실 및 구현 (Hinge Loss and its Implementation)

순위를 위한 힌지 손실은 SVM과 같은 분류기에서 종종 사용되는 gluon 라이브러리 내에서 제공되는 [힌지 손실](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss)과 다른 형태를 가집니다. 추천 시스템의 순위에 사용되는 손실은 다음과 같은 형태를 가집니다.

$$ 
\sum_{(u, i, j \in D)} \max( m - \hty}_{ui} + \hty}_{uj}, 0)
$$ 

여기서 $m$은 안전 마진 크기입니다. 긍정적인 항목에서 부정적인 항목을 밀어내는 것을 목표로 합니다. BPR과 유사하게 절대 출력이 아닌 긍정적인 샘플과 부정적인 샘플 간의 관련 거리를 최적화하는 것을 목표로 하므로 추천 시스템에 적합합니다.

```{.python .input  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

이 두 가지 손실은 추천의 개인화 순위에 대해 상호 교환 가능합니다.

## 요약 (Summary)

- 추천 시스템의 개인화 순위 작업에는 포인트와이즈, 페어와이즈 및 리스트와이즈 방법이라는 세 가지 유형의 순위 손실을 사용할 수 있습니다.
- 두 가지 페어와이즈 손실인 베이지안 개인화 순위 손실과 힌지 손실은 상호 교환적으로 사용할 수 있습니다.

## 연습 문제 (Exercises)

- BPR 및 힌지 손실의 변형이 있습니까?
- BPR 또는 힌지 손실을 사용하는 추천 모델을 찾을 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab: