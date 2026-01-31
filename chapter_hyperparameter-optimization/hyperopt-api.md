```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 하이퍼파라미터 최적화 API (Hyperparameter Optimization API)
:label:`sec_api_hpo`

방법론을 살펴보기 전에, 먼저 다양한 HPO 알고리즘을 효율적으로 구현할 수 있는 기본 코드 구조에 대해 논의하겠습니다. 일반적으로 여기서 고려되는 모든 HPO 알고리즘은 *검색(searching)*과 *스케줄링(scheduling)*이라는 두 가지 의사 결정 프리미티브를 구현해야 합니다. 첫째, 새로운 하이퍼파라미터 구성을 샘플링해야 하는데, 여기에는 종종 구성 공간에 대한 일종의 검색이 포함됩니다. 둘째, 각 구성에 대해 HPO 알고리즘은 평가를 스케줄링하고 얼마나 많은 리소스를 할당할지 결정해야 합니다. 구성 평가를 시작하면 이를 *시험(trial)*이라고 합니다. 이러한 결정을 `HPOSearcher`와 `HPOScheduler`라는 두 클래스에 매핑합니다. 그 위에 최적화 프로세스를 실행하는 `HPOTuner` 클래스도 제공합니다.

스케줄러와 검색자의 개념은 Syne Tune :cite:`salinas-automl22`, Ray Tune :cite:`liaw-arxiv18` 또는 Optuna :cite:`akiba-sigkdd19`와 같은 인기 있는 HPO 라이브러리에도 구현되어 있습니다.

```{.python .input  n=2}
%%tab pytorch
import time
from d2l import torch as d2l
from scipy import stats
```

## 검색자 (Searcher)

아래에서는 `sample_configuration` 함수를 통해 새로운 후보 구성을 제공하는 검색자의 기본 클래스를 정의합니다. 이 함수를 구현하는 간단한 방법은 :numref:`sec_what_is_hpo`에서 무작위 검색을 위해 했던 것처럼 구성을 균일하게 무작위로 샘플링하는 것입니다. 베이지안 최적화와 같은 더 정교한 알고리즘은 이전 시험의 성능을 기반으로 이러한 결정을 내립니다. 결과적으로 이러한 알고리즘은 시간이 지남에 따라 더 유망한 후보를 샘플링할 수 있습니다. `update` 함수를 추가하여 이전 시험의 기록을 업데이트하며, 이는 샘플링 분포를 개선하는 데 활용될 수 있습니다.

```{.python .input  n=3}
%%tab pytorch
class HPOSearcher(d2l.HyperParameters):  #@save
    def sample_configuration() -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass
```

다음 코드는 이전 섹션의 무작위 검색 최적화 도구를 이 API에서 구현하는 방법을 보여줍니다. 약간의 확장으로 사용자가 `initial_config`를 통해 평가할 첫 번째 구성을 지정할 수 있도록 하고, 후속 구성은 무작위로 추출됩니다.

```{.python .input  n=4}
%%tab pytorch
class RandomSearcher(HPOSearcher):  #@save
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

## 스케줄러 (Scheduler)

새로운 시험을 위한 구성을 샘플링하는 것 외에도 시험을 언제 얼마나 오래 실행할지 결정해야 합니다. 실제로 이러한 모든 결정은 `HPOScheduler`에 의해 수행되며, 이는 새로운 구성의 선택을 `HPOSearcher`에 위임합니다. `suggest` 메서드는 훈련을 위한 리소스가 사용 가능해질 때마다 호출됩니다. 검색자의 `sample_configuration`을 호출하는 것 외에도 `max_epochs`(즉, 모델을 훈련할 기간)와 같은 파라미터를 결정할 수도 있습니다. `update` 메서드는 시험이 새로운 관찰을 반환할 때마다 호출됩니다.

```{.python .input  n=5}
%%tab pytorch
class HPOScheduler(d2l.HyperParameters):  #@save
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

무작위 검색뿐만 아니라 다른 HPO 알고리즘을 구현하기 위해, 새로운 리소스가 사용 가능해질 때마다 새로운 구성을 스케줄링하는 기본 스케줄러만 있으면 됩니다.

```{.python .input  n=6}
%%tab pytorch
class BasicScheduler(HPOScheduler):  #@save
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

## 튜너 (Tuner)

마지막으로 스케줄러/검색자를 실행하고 결과의 부기(book-keeping)를 수행하는 구성 요소가 필요합니다. 다음 코드는 훈련 작업을 차례로 평가하는 HPO 시험의 순차적 실행을 구현하며 기본 예제로 사용됩니다. 나중에 더 확장 가능한 분산 HPO 사례를 위해 *Syne Tune*을 사용할 것입니다.

```{.python .input  n=7}
%%tab pytorch
class HPOTuner(d2l.HyperParameters):  #@save
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        # 플로팅을 위한 결과 부기
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

## HPO 알고리즘의 성능 부기 (Bookkeeping the Performance of HPO Algorithms)

어떤 HPO 알고리즘이든 우리는 주로 주어진 벽시계 시간(wall-clock time) 이후의 최고 성능 구성(*incumbent*라고 함)과 검증 오류에 관심이 있습니다. 이것이 반복당 `runtime`을 추적하는 이유이며, 여기에는 평가를 실행하는 시간(`objective` 호출)과 결정을 내리는 시간(`scheduler.suggest` 호출)이 모두 포함됩니다. 뒤이어 `scheduler`(및 `searcher`) 측면에서 정의된 HPO 알고리즘의 *any-time 성능*을 시각화하기 위해 `cumulative_runtime` 대 `incumbent_trajectory`를 플롯할 것입니다. 이를 통해 최적화 도구가 찾은 구성이 얼마나 잘 작동하는지뿐만 아니라 최적화 도구가 얼마나 빨리 찾을 수 있는지도 정량화할 수 있습니다.

```{.python .input  n=8}
%%tab pytorch
@d2l.add_to_class(HPOTuner)  #@save
def bookkeeping(self, config: dict, error: float, runtime: float):
    self.records.append({"config": config, "error": error, "runtime": runtime})
    # 마지막 하이퍼파라미터 구성이 incumbent보다 성능이 좋은지 확인
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
    # 현재 관찰된 최고 성능을 최적화 궤적에 추가
    self.incumbent_trajectory.append(self.incumbent_error)
    # 런타임 업데이트
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## 예제: 합성곱 신경망의 하이퍼파라미터 최적화

이제 무작위 검색의 새로운 구현을 사용하여 :numref:`sec_lenet`의 `LeNet` 합성곱 신경망의 *배치 크기*와 *학습률*을 최적화합니다. 목적 함수를 정의하는 것으로 시작하며, 이번에도 검증 오류가 될 것입니다.

```{.python .input  n=9}
%%tab pytorch
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):  #@save
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error
```

구성 공간도 정의해야 합니다. 또한 평가할 첫 번째 구성은 :numref:`sec_lenet`에서 사용된 기본 설정입니다.

```{.python .input  n=10}
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

이제 무작위 검색을 시작할 수 있습니다.

```{.python .input}
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)
tuner.run(number_of_trials=5)
```

아래에서는 무작위 검색의 any-time 성능을 얻기 위해 incumbent의 최적화 궤적을 그립니다.

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(
    tuner.cumulative_runtime, tuner.incumbent_trajectory
):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## HPO 알고리즘 비교

훈련 알고리즘이나 모델 아키텍처와 마찬가지로, 서로 다른 HPO 알고리즘을 가장 잘 비교하는 방법을 이해하는 것이 중요합니다. 각 HPO 실행은 무작위 가중치 초기화나 미니배치 순서와 같은 훈련 과정의 무작위 효과와 무작위 검색의 무작위 샘플링과 같은 HPO 알고리즘 자체의 본질적인 무작위성이라는 두 가지 주요 무작위성 소스에 따라 달라집니다. 따라서 다른 알고리즘을 비교할 때 각 실험을 여러 번 실행하고 난수 생성기의 서로 다른 시드를 기반으로 한 알고리즘의 여러 반복 모집단에 대한 평균 또는 중앙값과 같은 통계를 보고하는 것이 중요합니다.

이를 설명하기 위해 피드 포워드 신경망의 하이퍼파라미터 튜닝에서 무작위 검색(:numref:`sec_rs` 참조)과 베이지안 최적화 :cite:`snoek-nips12`를 비교합니다. 각 알고리즘은 다른 무작위 시드로 $50$번 평가되었습니다. 실선은 이 $50$번의 반복에 걸친 incumbent의 평균 성능을 나타내고 점선은 표준 편차를 나타냅니다. 무작위 검색과 베이지안 최적화는 ~1000초까지 거의 동일하게 수행되지만, 베이지안 최적화는 과거 관찰을 사용하여 더 나은 구성을 식별할 수 있으므로 그 이후에는 무작위 검색보다 빠르게 성능이 뛰어납니다.


![두 알고리즘 A와 B를 비교하기 위한 예시 any-time 성능 플롯.](../img/example_anytime_performance.svg)
:label:`example_anytime_performance`

## 요약

이 섹션에서는 이 장에서 살펴볼 다양한 HPO 알고리즘을 구현할 수 있는 간단하면서도 유연한 인터페이스를 제시했습니다. 유사한 인터페이스는 인기 있는 오픈 소스 HPO 프레임워크에서 찾을 수 있습니다. 또한 HPO 알고리즘을 비교하는 방법과 주의해야 할 잠재적인 함정을 살펴보았습니다.

## 연습 문제

1. 이 연습의 목표는 약간 더 도전적인 HPO 문제에 대한 목적 함수를 구현하고 더 현실적인 실험을 실행하는 것입니다. :numref:`sec_dropout`에서 구현된 두 은닉층 MLP `DropoutMLP`를 사용할 것입니다.
    1. 모델의 모든 하이퍼파라미터와 `batch_size`에 의존해야 하는 목적 함수를 코딩하십시오. `max_epochs=50`을 사용하십시오. 여기서는 GPU가 도움이 되지 않으므로 `num_gpus=0`입니다. 힌트: `hpo_objective_lenet`을 수정하십시오.
    2. `num_hiddens_1`, `num_hiddens_2`는 $[8, 1024]$의 정수이고, 드롭아웃 값은 $[0, 0.95]$에 있으며, `batch_size`는 $[16, 384]$에 있는 합리적인 검색 공간을 선택하십시오. `scipy.stats`의 합리적인 분포를 사용하여 `config_space`에 대한 코드를 제공하십시오.
    3. `number_of_trials=20`으로 이 예제에서 무작위 검색을 실행하고 결과를 플로팅하십시오. :numref:`sec_dropout`의 기본 구성인 `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`을 먼저 평가해야 합니다.
2. 이 연습에서는 과거 데이터를 기반으로 결정을 내리는 새로운 검색자(`HPOSearcher`의 서브클래스)를 구현할 것입니다. 파라미터 `probab_local`, `num_init_random`에 따라 달라집니다. `sample_configuration` 메서드는 다음과 같이 작동합니다. 처음 `num_init_random` 호출의 경우 `RandomSearcher.sample_configuration`과 동일하게 수행합니다. 그렇지 않으면 확률 `1 - probab_local`로 `RandomSearcher.sample_configuration`과 동일하게 수행합니다. 그렇지 않으면 지금까지 가장 작은 검증 오류를 달성한 구성을 선택하고 하이퍼파라미터 중 하나를 무작위로 선택한 다음 `RandomSearcher.sample_configuration`과 같이 값을 무작위로 샘플링하지만 다른 모든 값은 그대로 둡니다. 이 하나의 하이퍼파라미터를 제외하고 지금까지 가장 좋은 구성과 동일한 이 구성을 반환합니다.
    1. 이 새로운 `LocalSearcher`를 코딩하십시오. 힌트: 검색자는 구성 시 인수로 `config_space`가 필요합니다. `RandomSearcher` 유형의 멤버를 자유롭게 사용하십시오. `update` 메서드도 구현해야 합니다.
    2. 이전 연습의 실험을 다시 실행하되 `RandomSearcher` 대신 새 검색자를 사용하십시오. `probab_local`, `num_init_random`에 대해 다른 값으로 실험해 보십시오. 그러나 다른 HPO 방법을 적절하게 비교하려면 실험을 여러 번 반복하고 이상적으로는 여러 벤치마크 작업을 고려해야 한다는 점에 유의하십시오.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12092)
:end_tab: