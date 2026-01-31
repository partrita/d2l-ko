```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 비동기 무작위 검색 (Asynchronous Random Search)
:label:`sec_rs_async`

이전 :numref:`sec_api_hpo`에서 보았듯이 하이퍼파라미터 구성의 평가 비용이 많이 들기 때문에 무작위 검색이 좋은 하이퍼파라미터 구성을 반환하기까지 몇 시간 또는 며칠을 기다려야 할 수 있습니다. 실제로 우리는 종종 동일한 머신에 있는 여러 GPU 또는 단일 GPU가 있는 여러 머신과 같은 리소스 풀에 액세스할 수 있습니다. 이는 다음과 같은 질문을 제기합니다: *무작위 검색을 어떻게 효율적으로 분산할 수 있을까?*

일반적으로 동기식 병렬 하이퍼파라미터 최적화와 비동기식 병렬 하이퍼파라미터 최적화를 구별합니다(:numref:`distributed_scheduling` 참조). 동기식 설정에서는 다음 배치를 시작하기 전에 동시에 실행 중인 모든 시험이 끝날 때까지 기다립니다. 심층 신경망의 필터 수 또는 레이어 수와 같은 하이퍼파라미터를 포함하는 구성 공간을 고려해 보십시오. 더 많은 레이어나 필터를 포함하는 하이퍼파라미터 구성은 당연히 완료하는 데 더 많은 시간이 걸리며, 동일한 배치의 다른 모든 시험은 최적화 프로세스를 계속하기 전에 동기화 지점(:numref:`distributed_scheduling`의 회색 영역)에서 기다려야 합니다.

비동기 설정에서는 리소스가 사용 가능해지는 즉시 새로운 시험을 스케줄링합니다. 동기화 오버헤드를 피할 수 있으므로 리소스를 최적으로 활용할 수 있습니다. 무작위 검색의 경우, 각각의 새로운 하이퍼파라미터 구성은 다른 모든 구성과 독립적으로, 특히 이전 평가의 관찰을 활용하지 않고 선택됩니다. 이는 무작위 검색을 비동기식으로 쉽게 병렬화할 수 있음을 의미합니다. 이전 관찰을 기반으로 결정을 내리는 더 정교한 방법의 경우 이는 간단하지 않습니다(:numref:`sec_sh_async` 참조). 순차적 설정보다 더 많은 리소스에 액세스해야 하지만 비동기 무작위 검색은 선형 속도 향상을 보여줍니다. 즉, $K$개의 시험을 병렬로 실행할 수 있으면 특정 성능에 $K$배 더 빨리 도달합니다.


![하이퍼파라미터 최적화 프로세스를 동기식 또는 비동기식으로 분산합니다. 순차적 설정에 비해 전체 계산을 일정하게 유지하면서 전체 벽시계 시간을 줄일 수 있습니다. 동기식 스케줄링은 낙오자가 있는 경우 작업자가 유휴 상태가 될 수 있습니다.](../img/distributed_scheduling.svg)
:label:`distributed_scheduling`

이 노트북에서는 동일한 머신의 여러 파이썬 프로세스에서 시험이 실행되는 비동기 무작위 검색을 살펴볼 것입니다. 분산 작업 스케줄링 및 실행은 처음부터 구현하기 어렵습니다. 비동기 HPO를 위한 간단한 인터페이스를 제공하는 *Syne Tune* :cite:`salinas-automl22`을 사용할 것입니다. Syne Tune은 다양한 실행 백엔드로 실행되도록 설계되었으며, 관심 있는 독자는 분산 HPO에 대해 자세히 알아보기 위해 간단한 API를 연구하도록 초대됩니다.

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## 목적 함수

먼저 `report` 콜백을 통해 성능을 Syne Tune에 반환하도록 새로운 목적 함수를 정의해야 합니다.

```{.python .input  n=34}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Trainer 상태 초기화
            trainer.fit(model=model, data=data) 
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

Syne Tune의 `PythonBackend`는 함수 정의 내에서 종속성을 가져와야 한다는 점에 유의하십시오.

## 비동기 스케줄러

먼저 시험을 동시에 평가하는 작업자 수를 정의합니다. 또한 전체 벽시계 시간에 대한 상한을 정의하여 무작위 검색을 실행할 기간을 지정해야 합니다.

```{.python .input  n=37}
n_workers = 2  # 사용 가능한 GPU 수보다 작거나 같아야 함

max_wallclock_time = 12 * 60  # 12분
```

다음으로 최적화하려는 메트릭과 이 메트릭을 최소화할지 최대화할지 명시합니다. 즉, `metric`은 `report` 콜백에 전달된 인수 이름과 일치해야 합니다.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

이전 예제의 구성 공간을 사용합니다. Syne Tune에서 이 딕셔너리는 훈련 스크립트에 상수 속성을 전달하는 데에도 사용할 수 있습니다. `max_epochs`를 전달하기 위해 이 기능을 활용합니다. 또한 `initial_config`에 평가할 첫 번째 구성을 지정합니다.

```{.python .input  n=39}
config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": 10,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

다음으로 작업 실행을 위한 백엔드를 지정해야 합니다. 여기서는 병렬 작업이 하위 프로세스로 실행되는 로컬 머신의 배포만 고려합니다. 그러나 대규모 HPO의 경우 각 시험이 전체 인스턴스를 소비하는 클러스터 또는 클라우드 환경에서도 실행할 수 있습니다.

```{.python .input  n=40}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

이제 :numref:`sec_api_hpo`의 `BasicScheduler`와 동작이 유사한 비동기 무작위 검색을 위한 스케줄러를 만들 수 있습니다.

```{.python .input  n=41}
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
)
```

Syne Tune은 또한 `Tuner`를 제공합니다. 여기서 주요 실험 루프와 부기가 중앙 집중화되고 스케줄러와 백엔드 간의 상호 작용이 중재됩니다.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler, 
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
```

분산 HPO 실험을 실행해 봅시다. 중지 기준에 따르면 약 12분 동안 실행됩니다.

```{.python .input  n=43}
tuner.run()
```

평가된 모든 하이퍼파라미터 구성의 로그는 추가 분석을 위해 저장됩니다. 튜닝 작업 중 언제든지 지금까지 얻은 결과를 쉽게 가져와서 incumbent 궤적을 그릴 수 있습니다.

```{.python .input  n=46}
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## 비동기 최적화 프로세스 시각화

아래에서는 비동기 최적화 프로세스 중에 모든 시험의 학습 곡선(플롯의 각 색상은 시험을 나타냄)이 어떻게 진화하는지 시각화합니다. 어느 시점에서든 작업자 수만큼의 시험이 동시에 실행됩니다. 시험이 끝나면 다른 시험이 끝나기를 기다리지 않고 즉시 다음 시험을 시작합니다. 비동기 스케줄링으로 작업자의 유휴 시간이 최소화됩니다.

```{.python .input  n=45}
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
    
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("objective function")
```

## 요약

병렬 리소스에 시험을 분산하여 무작위 검색의 대기 시간을 상당히 줄일 수 있습니다. 일반적으로 동기식 스케줄링과 비동기식 스케줄링을 구별합니다. 동기식 스케줄링은 이전 배치가 완료되면 새로운 하이퍼파라미터 구성 배치를 샘플링하는 것을 의미합니다. 다른 시험보다 완료하는 데 시간이 더 오래 걸리는 시험인 낙오자가 있는 경우 작업자는 동기화 지점에서 기다려야 합니다. 비동기식 스케줄링은 리소스가 사용 가능해지는 즉시 새로운 하이퍼파라미터 구성을 평가하므로 모든 작업자가 어느 시점에서든 바쁘게 움직이도록 보장합니다. 무작위 검색은 비동기식으로 분산하기 쉽고 실제 알고리즘의 변경이 필요하지 않지만 다른 방법은 약간의 추가 수정이 필요합니다.

## 연습 문제

1. :numref:`sec_dropout`에서 구현되고 :numref:`sec_api_hpo`의 연습 문제 1에서 사용된 `DropoutMLP` 모델을 고려하십시오.
    1. Syne Tune과 함께 사용할 목적 함수 `hpo_objective_dropoutmlp_synetune`을 구현하십시오. 함수가 매 에포크 후 검증 오류를 보고하는지 확인하십시오.
    2. :numref:`sec_api_hpo`의 연습 문제 1 설정을 사용하여 무작위 검색과 베이지안 최적화를 비교하십시오. SageMaker를 사용하는 경우 실험을 병렬로 실행하기 위해 Syne Tune의 벤치마킹 기능을 자유롭게 사용하십시오. 힌트: 베이지안 최적화는 `syne_tune.optimizer.baselines.BayesianOptimization`으로 제공됩니다.
    3. 이 연습을 위해서는 최소 4개의 CPU 코어가 있는 인스턴스에서 실행해야 합니다. 위에서 사용된 방법 중 하나(무작위 검색, 베이지안 최적화)에 대해 `n_workers=1`, `n_workers=2`, `n_workers=4`로 실험을 실행하고 결과(incumbent 궤적)를 비교하십시오. 적어도 무작위 검색의 경우 작업자 수에 따른 선형 확장을 관찰해야 합니다. 힌트: 강력한 결과를 얻으려면 각각 여러 번 반복하여 평균을 내야 할 수 있습니다.
2. *고급*. 이 연습의 목표는 Syne Tune에서 새로운 스케줄러를 구현하는 것입니다.
    1. [d2lbook](https://github.com/d2l-ai/d2l-en/blob/master/INFO.md#installation-for-developers) 및 [syne-tune](https://syne-tune.readthedocs.io/en/latest/getting_started.html) 소스를 모두 포함하는 가상 환경을 만듭니다.
    2. :numref:`sec_api_hpo`의 연습 문제 2에 있는 `LocalSearcher`를 Syne Tune의 새 검색자로 구현하십시오. 힌트: [이 튜토리얼](https://syne-tune.readthedocs.io/en/latest/tutorials/developer/README.html)을 읽어보십시오. 대안으로 [이 예제](https://syne-tune.readthedocs.io/en/latest/examples.html#launch-hpo-experiment-with-home-made-scheduler)를 따를 수 있습니다.
    3. `DropoutMLP` 벤치마크에서 새로운 `LocalSearcher`와 `RandomSearch`를 비교하십시오.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12093)
:end_tab: