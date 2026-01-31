```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 다중 충실도 하이퍼파라미터 최적화 (Multi-Fidelity Hyperparameter Optimization)
:label:`sec_mf_hpo`

신경망 훈련은 중간 규모의 데이터셋에서도 비용이 많이 들 수 있습니다. 구성 공간(:numref:`sec_intro_config_spaces` 참조)에 따라, 하이퍼파라미터 최적화는 잘 작동하는 하이퍼파라미터 구성을 찾기 위해 수십 번에서 수백 번의 함수 평가를 필요로 합니다. :numref:`sec_rs_async`에서 보았듯이, 병렬 리소스를 활용하여 HPO의 전체 벽시계 시간(wall-clock time)을 크게 단축할 수 있지만, 이는 필요한 총 계산량을 줄이지는 못합니다.

이 섹션에서는 하이퍼파라미터 구성 평가를 어떻게 가속화할 수 있는지 보여줄 것입니다. 랜덤 서치와 같은 방법은 각 하이퍼파라미터 평가에 동일한 양의 리소스(예: 에폭 수, 훈련 데이터 포인트 수)를 할당합니다. :numref:`img_samples_lc`는 신경망 하이퍼파라미터 구성 집합의 학습 곡선을 보여줍니다.

![여러 하이퍼파라미터 구성의 에폭에 따른 검증 오차(학습 곡선). 좋은 구성과 나쁜 구성을 비교적 빠르게 구별할 수 있다는 것을 알 수 있습니다.](../img/learning_curves.svg)
:label:`img_samples_lc`

많은 나쁜 구성들이 훈련 초기에 이미 성능이 좋지 않음을 알 수 있습니다. *다중 충실도(multi-fidelity)* HPO는 이 관찰을 활용합니다. 이는 저충실도 평가(예: 적은 수의 에폭에 대한 훈련)를 사용하여 나쁜 구성을 조기에 식별하고 성능이 좋은 것으로 보이는 구성에만 더 많은 리소스를 할당하려고 시도합니다.

이 섹션에서는 가장 인기 있는 다중 충실도 HPO 알고리즘 중 하나인 Successive Halving(:cite:`jamieson-aistats16,karnin-icml13`)을 살펴볼 것입니다.


## Successive Halving

Successive Halving(SH) 알고리즘은 성능이 낮은 구성을 리소스 $R$의 고정된 집합에서 조기에 중단하는 아이디어에 기반합니다. 리소스 $r$은 에폭 수, 훈련 데이터 포인트 수 또는 하위 샘플링된 이미지 해상도일 수 있습니다.

입력으로 $n$개의 하이퍼파라미터 구성 집합과 최소 리소스 $r_{min}$ 및 최대 리소스 $r_{max}$를 받습니다. $r_{min}$에서 시작하여 모든 구성을 평가합니다. 그런 다음 $1/\eta$ 비율의 최상의 구성을 유지하고 다음 스테이지로 이동하여 이 구성들을 더 많은 리소스로 평가합니다. 이 과정은 구성이 하나만 남을 때까지 반복됩니다.

공식적으로, $\eta$를 축소 인자(reduction factor)라고 합시다. 스테이지 $i$에서 우리는 $n_i$개의 구성을 각각 $r_i$ 리소스를 사용하여 평가합니다. 다음 스테이지 $i+1$로 넘어가기 위해 상위 $n_{i+1} = \lfloor n_i / \eta \rfloor$개의 구성을 선택하고 리소스를 $r_{i+1} = r_i \cdot \eta$로 늘립니다.

리소스 당 평가되는 구성의 수 $n_i r_i$는 모든 스테이지에서 거의 일정하게 유지됩니다. $\eta=2$라고 가정하면, 매 스테이지마다 구성의 절반을 탈락시키고 리소스를 두 배로 늘립니다. 따라서 전체 비용은 모든 $n$개 구성을 최대 리소스 $r_{max}$로 평가하는 비용보다 훨씬 작습니다. 구체적으로 SH의 총 리소스 비용은 대략 $n \cdot r_{min} \cdot \log_\eta(r_{max}/r_{min})$입니다.


## 구현 (Implementation)

이제 Successive Halving을 구현해 보겠습니다. 리소스로 에폭 수를 사용합니다.

```{.python .input}
#@tab pytorch
import numpy as np
from d2l import torch as d2l

class SuccessiveHalving(d2l.HPOScheduler):
    def __init__(self, searcher, eta, r_min, r_max):
        self.searcher = searcher
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.s_max = int(np.log(r_max / r_min) / np.log(eta))
        self.stage = 0
        self.configs = [self.searcher.sample_config() for _ in range(self.get_n_i(0))]
        self.observed_error = []

    def get_n_i(self, stage):
        return int(len(self.configs) * (self.eta ** -stage))

    def get_r_i(self, stage):
        return self.r_min * (self.eta ** stage)

    def suggest(self):
        if len(self.configs) == 0:
            return None
        config = self.configs.pop(0)
        config['epochs'] = self.get_r_i(self.stage)
        return config

    def update(self, config, error):
        self.observed_error.append((config, error))
        if len(self.configs) == 0:  # 스테이지 종료
            self.stage += 1
            if self.stage <= self.s_max:
                # 오차를 기준으로 정렬하고 상위 1/eta 유지
                self.observed_error.sort(key=lambda x: x[1])
                n_next = self.get_n_i(self.stage)
                self.configs = [x[0] for x in self.observed_error[:n_next]]
                self.observed_error = []
```

## 요약 (Summary)

* 다중 충실도 HPO는 저충실도 평가를 사용하여 나쁜 하이퍼파라미터 구성을 조기에 식별함으로써 검색 프로세스를 가속화합니다.
* Successive Halving은 반복적으로 최상의 구성 집합을 유지하고 리소스를 늘리는 간단하지만 강력한 알고리즘입니다.
* 에폭 수를 리소스로 사용함으로써 훈련 프로세스를 조기에 중단할 수 있어 계산 효율성을 높일 수 있습니다.

## 연습 문제 (Exercises)

1. SH의 축소 인자 $\eta$가 성능에 어떤 영향을 미치는지 논의해 보십시오. $\eta$가 매우 크거나 작을 때의 장단점은 무엇입니까?
2. SH의 한 가지 잠재적인 단점은 초기 스테이지($r_{min}$)에서 성능이 좋지 않지만 나중에($r_{max}$) 성능이 좋아질 구성을 탈락시킬 수 있다는 것입니다. 이를 "나쁜 초기 단계(bad start)" 문제라고 합니다. 이를 완화할 수 있는 방법이 있을까요?
3. Successive Halving을 Hyperband(:cite:`li-jmlr17`)로 확장하는 방법을 찾아보십시오. Hyperband는 SH의 어떤 한계를 해결합니까?

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/12104)
:end_tab:

```