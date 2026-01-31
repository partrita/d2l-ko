```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 하이퍼파라미터 최적화란 무엇인가? (What Is Hyperparameter Optimization?)
:label:`sec_what_is_hpo`

이전 장에서 보았듯이 심층 신경망에는 훈련 중에 학습되는 많은 수의 파라미터 또는 가중치가 있습니다. 이 외에도 모든 신경망에는 사용자가 구성해야 하는 추가적인 *하이퍼파라미터*가 있습니다. 예를 들어, 확률적 경사 하강법이 훈련 손실의 국소 최적값으로 수렴하도록 하려면(:numref:`chap_optimization` 참조), 학습률과 배치 크기를 조정해야 합니다. 훈련 데이터셋에 대한 과대적합을 피하기 위해, 가중치 감소(:numref:`sec_weight_decay` 참조)나 드롭아웃(:numref:`sec_dropout` 참조)과 같은 정규화 파라미터를 설정해야 할 수도 있습니다. 레이어 수와 레이어당 유닛 또는 필터 수(즉, 유효 가중치 수)를 설정하여 모델의 용량과 귀납적 편향을 정의할 수 있습니다.

불행히도 훈련 손실을 최소화하여 이러한 하이퍼파라미터를 단순히 조정할 수는 없습니다. 그렇게 하면 훈련 데이터에 과대적합되기 때문입니다. 예를 들어 드롭아웃이나 가중치 감소와 같은 정규화 파라미터를 0으로 설정하면 훈련 손실은 작아지지만 일반화 성능은 저하될 수 있습니다.

![다양한 하이퍼파라미터로 모델을 여러 번 훈련하는 것으로 구성된 머신러닝의 일반적인 워크플로우.](../img/ml_workflow.svg)
:label:`ml_workflow`

다른 형태의 자동화 없이는 하이퍼파라미터를 시행착오 방식으로 수동으로 설정해야 하며, 이는 머신러닝 워크플로우에서 시간이 많이 걸리고 어려운 부분입니다. 예를 들어 CIFAR-10에서 ResNet(:numref:`sec_resnet` 참조)을 훈련하는 것을 고려해 보십시오. Amazon Elastic Cloud Compute (EC2) `g4dn.xlarge` 인스턴스에서 2시간 이상 걸립니다. 10개의 하이퍼파라미터 구성을 순서대로 시도하는 것만으로도 대략 하루가 걸립니다. 설상가상으로 하이퍼파라미터는 일반적으로 아키텍처와 데이터셋 간에 직접 전이되지 않으며 :cite:`feurer-arxiv22,wistuba-ml18,bardenet-icml13a`, 모든 새로운 작업에 대해 다시 최적화해야 합니다. 또한 대부분의 하이퍼파라미터에 대한 경험 법칙이 없으며 합리적인 값을 찾으려면 전문가 지식이 필요합니다.

*하이퍼파라미터 최적화(Hyperparameter Optimization, HPO)* 알고리즘은 이 문제를 전역 최적화 문제로 프레임화하여 원칙적이고 자동화된 방식으로 해결하도록 설계되었습니다 :cite:`feurer-automlbook18a`. 기본 목표는 보류된 검증 데이터셋에 대한 오류이지만 원칙적으로 다른 비즈니스 메트릭일 수 있습니다. 훈련 시간, 추론 시간 또는 모델 복잡성과 같은 보조 목표와 결합되거나 제한될 수 있습니다.

최근 하이퍼파라미터 최적화는 *신경망 아키텍처 검색(Neural Architecture Search, NAS)* :cite:`elsken-arxiv18a,wistuba-arxiv19`으로 확장되었습니다. 여기서 목표는 완전히 새로운 신경망 아키텍처를 찾는 것입니다. 고전적인 HPO에 비해 NAS는 계산 측면에서 훨씬 더 비싸고 실제로 실행 가능하려면 추가적인 노력이 필요합니다. HPO와 NAS는 모두 전체 ML 파이프라인을 자동화하는 것을 목표로 하는 AutoML :cite:`hutter-book19a`의 하위 분야로 간주될 수 있습니다.

이 섹션에서는 HPO를 소개하고 :numref:`sec_softmax_concise`에서 소개된 로지스틱 회귀 예제의 최적 하이퍼파라미터를 자동으로 찾는 방법을 보여줍니다.

##  최적화 문제 (The Optimization Problem)
:label:`sec_definition_hpo`

간단한 장난감 문제로 시작하겠습니다. Fashion MNIST 데이터셋에 대한 검증 오류를 최소화하기 위해 :numref:`sec_softmax_concise`의 다중 클래스 로지스틱 회귀 모델 `SoftmaxRegression`의 학습률을 검색합니다. 배치 크기나 에포크 수와 같은 다른 하이퍼파라미터도 튜닝할 가치가 있지만 간단히 하기 위해 학습률에만 집중합니다.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from scipy import stats
```

HPO를 실행하기 전에 먼저 목적 함수와 구성 공간이라는 두 가지 요소를 정의해야 합니다.

### 목적 함수 (The Objective Function)

학습 알고리즘의 성능은 하이퍼파라미터 공간 $\mathbf{x} \in \mathcal{X}$에서 검증 손실로 매핑하는 함수 $f: \mathcal{X} \rightarrow \mathbb{R}$로 볼 수 있습니다. $f(\mathbf{x})$를 평가할 때마다 머신러닝 모델을 훈련하고 검증해야 하는데, 대규모 데이터셋에서 훈련된 심층 신경망의 경우 시간과 계산이 많이 소요될 수 있습니다. 기준 $f(\mathbf{x})$가 주어지면 우리의 목표는 $\mathbf{x}_{\star} \in \mathrm{argmin}_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$를 찾는 것입니다.

$\\mathbf{x}$에 대한 $f$의 기울기를 계산하는 간단한 방법은 없습니다. 전체 훈련 과정을 통해 기울기를 전파해야 하기 때문입니다. 근사적인 "하이퍼그라디언트(hypergradients)"로 HPO를 구동하려는 최근 연구 :cite:`maclaurin-icml15,franceschi-icml17a`가 있지만, 기존 접근 방식 중 어느 것도 아직 최첨단 기술과 경쟁력이 없으므로 여기서는 논의하지 않겠습니다. 또한 $f$를 평가하는 계산 부담으로 인해 HPO 알고리즘은 가능한 한 적은 샘플로 전역 최적값에 접근해야 합니다.

신경망의 훈련은 확률적입니다(예: 가중치가 무작위로 초기화되고 미니배치가 무작위로 샘플링됨). 따라서 관찰은 노이즈가 있습니다: $y \sim f(\mathbf{x}) + \epsilon$, 여기서 우리는 보통 $\\epsilon \sim N(0, \sigma)$ 관찰 노이즈가 가우시안 분포를 따른다고 가정합니다.

이러한 모든 과제에 직면하여 우리는 일반적으로 전역 최적값을 정확하게 맞추는 대신 성능이 좋은 하이퍼파라미터 구성의 작은 집합을 빠르게 식별하려고 합니다. 그러나 대부분의 신경망 모델의 큰 계산 요구로 인해 이것조차 며칠 또는 몇 주가 걸릴 수 있습니다. 우리는 :numref:`sec_mf_hpo`에서 검색을 분산시키거나 목적 함수의 평가 비용이 더 저렴한 근사를 사용하여 최적화 프로세스를 가속화하는 방법을 탐구할 것입니다.

모델의 검증 오류를 계산하는 메서드로 시작합니다.

```{.python .input  n=8}
%%tab pytorch
class HPOTrainer(d2l.Trainer):  #@save
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_idx = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_idx += 1
        return 1 -  accuracy / val_batch_idx
```

`learning_rate`로 구성된 하이퍼파라미터 구성 `config`에 대해 검증 오류를 최적화합니다. 각 평가에 대해 `max_epochs` 동안 모델을 훈련한 다음 검증 오류를 계산하고 반환합니다.

```{.python .input  n=5}
%%tab pytorch
def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config["learning_rate"]
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return d2l.numpy(trainer.validation_error())
```

### 구성 공간 (The Configuration Space)
:label:`sec_intro_config_spaces`

목적 함수 $f(\mathbf{x})$와 함께 최적화할 실행 가능 집합 $\\mathbf{x} \in \mathcal{X}$도 정의해야 하며, 이를 *구성 공간(configuration space)* 또는 *검색 공간(search space)*이라고 합니다. 로지스틱 회귀 예제의 경우 다음을 사용합니다.

```{.python .input  n=6}
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

여기서는 SciPy의 `loguniform` 객체를 사용합니다. 이는 로그 공간에서 -4와 -1 사이의 균일 분포를 나타냅니다. 이 객체를 사용하여 이 분포에서 확률 변수를 샘플링할 수 있습니다.

각 하이퍼파라미터에는 `learning_rate`의 `float`와 같은 데이터 유형과 닫힌 경계 범위(즉, 하한 및 상한)가 있습니다. 우리는 일반적으로 각 하이퍼파라미터에 샘플링할 사전 분포(예: 균일 또는 로그 균일)를 할당합니다. `learning_rate`와 같은 일부 양수 파라미터는 최적값이 몇 자리 수만큼 다를 수 있으므로 로그 스케일로 가장 잘 표현되는 반면, 모멘텀과 같은 다른 파라미터는 선형 스케일로 제공됩니다.

아래에서는 유형 및 표준 범위를 포함하여 다층 퍼셉트론의 일반적인 하이퍼파라미터로 구성된 구성 공간의 간단한 예를 보여줍니다.

: 다층 퍼셉트론의 구성 공간 예시
:label:`tab_example_configspace`

| 이름 (Name) | 유형 (Type) | 하이퍼파라미터 범위 (Hyperparameter Ranges) | 로그 스케일 (log-scale) |
| :----: | :----: |:------------------------------:|:---------:|
| 학습률 (learning rate) | 실수 (float) | $[10^{-6},10^{-1}]$ | 예 (yes) |
| 배치 크기 (batch size) | 정수 (integer) | $[8,256]$ | 예 (yes) |
| 모멘텀 (momentum) | 실수 (float) | $[0,0.99]$ | 아니오 (no) |
| 활성화 함수 (activation function) | 범주형 (categorical) | {$\textrm{tanh}, \textrm{relu}$} | - |
| 유닛 수 (number of units) | 정수 (integer) | $[32, 1024]$ | 예 (yes) |
| 레이어 수 (number of layers) | 정수 (integer) | $[1, 6]$ | 아니오 (no) |



일반적으로 구성 공간 $\\mathcal{X}$의 구조는 복잡할 수 있으며 $\\mathbb{R}^d$와 상당히 다를 수 있습니다. 실제로 일부 하이퍼파라미터는 다른 하이퍼파라미터의 값에 따라 달라질 수 있습니다. 예를 들어 다층 퍼셉트론의 레이어 수와 각 레이어의 유닛 수를 조정하려고 한다고 가정해 봅시다. $l\textrm{-번째}$ 레이어의 유닛 수는 네트워크에 적어도 $l+1$개의 레이어가 있는 경우에만 관련이 있습니다. 이러한 고급 HPO 문제는 이 장의 범위를 벗어납니다. 관심 있는 독자는 :cite:`hutter-lion11a,jenatton-icml17a,baptista-icml18a`를 참조하십시오.

구성 공간은 하이퍼파라미터 최적화에서 중요한 역할을 합니다. 어떤 알고리즘도 구성 공간에 포함되지 않은 것을 찾을 수 없기 때문입니다. 반면에 범위가 너무 크면 성능이 좋은 구성을 찾는 데 드는 계산 예산이 실행 불가능할 수 있습니다.

## 무작위 검색 (Random Search)
:label:`sec_rs`

*무작위 검색*은 우리가 고려할 첫 번째 하이퍼파라미터 최적화 알고리즘입니다. 무작위 검색의 주요 아이디어는 미리 정의된 예산(예: 최대 반복 횟수)이 소진될 때까지 구성 공간에서 독립적으로 샘플링하고 관찰된 최적의 구성을 반환하는 것입니다. 모든 평가는 병렬로 독립적으로 실행될 수 있지만(:numref:`sec_rs_async` 참조), 여기서는 간단히 순차 루프를 사용합니다.

```{.python .input  n=7}
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
    learning_rate = config_space["learning_rate"].rvs()
    print(f"Trial {i}: learning_rate = {learning_rate}")
    y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
    print(f"    validation_error = {y}")
    values.append(learning_rate)
    errors.append(y)
```

그러면 최적의 학습률은 단순히 검증 오류가 가장 낮은 학습률입니다.

```{.python .input  n=7}
best_idx = np.argmin(errors)
print(f"optimal learning rate = {values[best_idx]}")
```

단순성과 일반성으로 인해 무작위 검색은 가장 자주 사용되는 HPO 알고리즘 중 하나입니다. 정교한 구현이 필요하지 않으며 각 하이퍼파라미터에 대한 확률 분포를 정의할 수 있는 한 모든 구성 공간에 적용할 수 있습니다.

불행히도 무작위 검색에는 몇 가지 단점도 있습니다. 첫째, 지금까지 수집한 이전 관찰을 기반으로 샘플링 분포를 조정하지 않습니다. 따라서 성능이 좋은 구성보다 성능이 나쁜 구성을 샘플링할 가능성이 동일합니다. 둘째, 일부 구성은 초기에 나쁜 성능을 보여 이전에 본 구성보다 성능이 뛰어날 가능성이 낮음에도 불구하고 모든 구성에 동일한 리소스가 소비됩니다.

다음 섹션에서는 모델을 사용하여 검색을 안내함으로써 무작위 검색의 단점을 극복하는 더 샘플 효율적인 하이퍼파라미터 최적화 알고리즘을 살펴볼 것입니다. 또한 성능이 좋지 않은 구성의 평가 프로세스를 자동으로 중지하여 최적화 프로세스를 가속화하는 알고리즘도 살펴볼 것입니다.

## 요약 (Summary)

이 섹션에서는 하이퍼파라미터 최적화(HPO)를 소개하고 구성 공간과 목적 함수를 정의하여 이를 전역 최적화로 표현하는 방법을 설명했습니다. 또한 첫 번째 HPO 알고리즘인 무작위 검색을 구현하고 간단한 소프트맥스 분류 문제에 적용했습니다.

무작위 검색은 매우 간단하지만 고정된 하이퍼파라미터 세트만 평가하는 그리드 검색보다 더 나은 대안입니다. 무작위 검색은 차원의 저주를 어느 정도 완화하며 :cite:`bellman-science66`, 기준이 하이퍼파라미터의 작은 하위 집합에 가장 강하게 의존하는 경우 그리드 검색보다 훨씬 효율적일 수 있습니다.

## 연습 문제 (Exercises)

1. 이 장에서는 분리된 훈련 세트에서 훈련한 후 모델의 검증 오류를 최적화합니다. 간단히 하기 위해 우리 코드는 `FashionMNIST.val` 주변의 로더에 매핑되는 `Trainer.val_dataloader`를 사용합니다.
    1. 이것이 우리가 훈련을 위해 원본 FashionMNIST 훈련 세트(60,000개 예제)를 사용하고 검증을 위해 원본 *테스트 세트*(10,000개 예제)를 사용한다는 의미임을 코드(코드를 살펴봄으로써)를 통해 확신하십시오.
    2. 이 관행이 문제가 될 수 있는 이유는 무엇입니까? 힌트: :numref:`sec_generalization_basics`를 다시 읽어보십시오. 특히 *모델 선택*에 대해 읽어보십시오.
    3. 대신 무엇을 했어야 합니까?
2. 위에서 경사 하강법에 의한 하이퍼파라미터 최적화는 수행하기 매우 어렵다고 말했습니다. 배치 크기가 256인 FashionMNIST 데이터셋(:numref:`sec_mlp-implementation`)에서 2레이어 퍼셉트론을 훈련하는 것과 같은 작은 문제를 생각해 보십시오. 훈련 1에포크 후 검증 메트릭을 최소화하기 위해 SGD의 학습률을 튜닝하고 싶습니다.
    1. 이 목적을 위해 검증 *오류*를 사용할 수 없는 이유는 무엇입니까? 검증 세트에서 어떤 메트릭을 사용하시겠습니까?
    2. 1에포크 훈련 후 검증 메트릭의 계산 그래프를 (대략적으로) 스케치하십시오. 초기 가중치와 하이퍼파라미터(예: 학습률)가 이 그래프의 입력 노드라고 가정할 수 있습니다. 힌트: :numref:`sec_backprop`에서 계산 그래프에 대해 다시 읽어보십시오.
    3. 이 그래프에서 순방향 패스 동안 저장해야 하는 부동 소수점 값의 수를 대략적으로 추정하십시오. 힌트: FashionMNIST에는 60,000개의 케이스가 있습니다. 필요한 메모리가 각 레이어 이후의 활성화에 의해 지배된다고 가정하고 :numref:`sec_mlp-implementation`에서 레이어 너비를 찾아보십시오.
    5. 필요한 엄청난 양의 계산 및 저장 공간 외에도 기울기 기반 하이퍼파라미터 최적화가 겪게 될 다른 문제는 무엇입니까? 힌트: :numref:`sec_numerical_stability`에서 기울기 소실 및 폭발에 대해 다시 읽어보십시오.
    6. *고급*: 기울기 기반 HPO에 대한 우아한(아직 다소 비실용적인) 접근 방식에 대해서는 :cite:`maclaurin-icml15`를 읽어보십시오.
3. 그리드 검색은 또 다른 HPO 기준선으로, 각 하이퍼파라미터에 대해 등간격 그리드를 정의한 다음 구성을 제안하기 위해 (조합) 데카르트 곱을 반복합니다.
    1. 위에서 우리는 기준이 하이퍼파라미터의 작은 하위 집합에 가장 강하게 의존하는 경우 상당한 수의 하이퍼파라미터에 대한 HPO에 대해 무작위 검색이 그리드 검색보다 훨씬 효율적일 수 있다고 말했습니다. 그 이유는 무엇입니까? 힌트: :cite:`bergstra2011algorithms`를 읽어보십시오.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12090)
:end_tab:
```