```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 비동기 연속 반감 (Asynchronous Successive Halving)

:label:`sec_sh_async`

:numref:`sec_rs_async`에서 보았듯이 여러 인스턴스 또는 단일 인스턴스의 여러 CPU/GPU에 하이퍼파라미터 구성 평가를 분산하여 HPO를 가속화할 수 있습니다. 그러나 무작위 검색에 비해 분산 설정에서 연속 반감(SH)을 비동기적으로 실행하는 것은 간단하지 않습니다. 다음에 실행할 구성을 결정하기 전에 먼저 현재 렁 수준에서 모든 관찰을 수집해야 합니다. 이를 위해서는 각 렁 수준에서 작업자를 동기화해야 합니다. 예를 들어 가장 낮은 렁 수준 $r_{\mathrm{min}}$의 경우, 그중 $\frac{1}{\eta}$를 다음 렁 수준으로 승격시키기 전에 먼저 모든 $N = \eta^K$ 구성을 평가해야 합니다.

어떤 분산 시스템에서든 동기화는 일반적으로 작업자의 유휴 시간을 의미합니다. 첫째, 하이퍼파라미터 구성 전반에 걸쳐 훈련 시간에 큰 차이가 종종 관찰됩니다. 예를 들어 레이어당 필터 수가 하이퍼파라미터라고 가정하면 필터가 적은 네트워크는 필터가 많은 네트워크보다 훈련이 더 빨리 끝나므로 낙오자로 인해 유휴 작업자 시간이 발생합니다. 또한 렁 수준의 슬롯 수가 항상 작업자 수의 배수는 아니며, 이 경우 일부 작업자는 전체 배치 동안 유휴 상태일 수도 있습니다.

그림 :numref:`synchronous_sh`는 두 작업자가 있는 4개의 서로 다른 시험에 대해 $\eta=2$인 동기식 SH의 스케줄링을 보여줍니다. Trial-0과 Trial-1을 1 에포크 동안 평가하는 것으로 시작하고 완료되면 즉시 다음 두 시험을 계속합니다. 다른 시험보다 상당히 많은 시간이 걸리는 Trial-2가 완료될 때까지 기다려야 상위 두 시험, 즉 Trial-0과 Trial-3을 다음 렁 수준으로 승격시킬 수 있습니다. 이로 인해 Worker-1에 유휴 시간이 발생합니다. 그런 다음 렁 1을 계속합니다. 여기에서도 Trial-3이 Trial-0보다 오래 걸리므로 Worker-0의 추가 유휴 시간이 발생합니다. 렁 2에 도달하면 가장 좋은 시험인 Trial-0만 남게 되어 한 명의 작업자만 점유합니다. 그 시간 동안 Worker-1이 유휴 상태가 되는 것을 피하기 위해 대부분의 SH 구현은 이미 다음 라운드를 계속하고 첫 번째 렁에서 새로운 시험(예: Trial-4)을 평가하기 시작합니다.

![두 작업자가 있는 동기식 연속 반감.](../img/sync_sh.svg)
:label:`synchronous_sh`

비동기 연속 반감(ASHA) :cite:`li-arxiv18`은 SH를 비동기 병렬 시나리오에 적응시킵니다. ASHA의 주요 아이디어는 현재 렁 수준에서 최소 $\eta$개의 관찰을 수집하는 즉시 구성을 다음 렁 수준으로 승격시키는 것입니다. 이 결정 규칙은 차선책으로 승격될 수 있습니다. 구성이 다음 렁 수준으로 승격될 수 있지만, 나중에 돌이켜보면 같은 렁 수준의 다른 대부분의 구성과 비교하여 유리하지 않을 수 있습니다. 반면에 우리는 이런 식으로 모든 동기화 지점을 제거합니다. 실제로 이러한 차선책 초기 승격은 성능에 미미한 영향을 미칩니다. 하이퍼파라미터 구성의 순위가 렁 수준 전체에서 상당히 일관될 뿐만 아니라 렁이 시간이 지남에 따라 커지고 이 수준의 메트릭 값 분포를 점점 더 잘 반영하기 때문입니다. 작업자가 비어 있지만 승격할 수 있는 구성이 없는 경우 $r = r_{\mathrm{min}}$, 즉 첫 번째 렁 수준에서 새 구성을 시작합니다.

:numref:`asha`는 ASHA에 대한 동일한 구성의 스케줄링을 보여줍니다. Trial-1이 완료되면 두 시험(즉, Trial-0과 Trial-1)의 결과를 수집하고 그중 더 나은 것(Trial-0)을 즉시 다음 렁 수준으로 승격시킵니다. Trial-0이 렁 1에서 완료된 후에는 추가 승격을 지원하기에는 그곳에 시험이 너무 적습니다. 따라서 렁 0을 계속하고 Trial-3을 평가합니다. Trial-3이 완료되면 Trial-2는 아직 보류 중입니다. 이 시점에서 우리는 렁 0에서 3개의 시험을 평가했고 렁 1에서 하나의 시험을 이미 평가했습니다. Trial-3이 렁 0에서 Trial-0보다 성능이 나쁘고 $\eta=2$이므로 아직 새로운 시험을 승격시킬 수 없으며 Worker-1은 대신 처음부터 Trial-4를 시작합니다. 그러나 Trial-2가 완료되고 Trial-3보다 점수가 나쁘면 후자는 렁 1로 승격됩니다. 그 후 렁 1에서 2개의 평가를 수집했으므로 이제 Trial-0을 렁 2로 승격시킬 수 있습니다. 동시에 Worker-1은 렁 0에서 새로운 시험(즉, Trial-5)을 계속 평가합니다.


![두 작업자가 있는 비동기 연속 반감(ASHA).](../img/asha.svg)
:label:`asha`

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## 목적 함수

우리는 :numref:`sec_rs_async`와 동일한 목적 함수로 *Syne Tune*을 사용할 것입니다.

```{.python .input  n=54}
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

이전과 동일한 구성 공간을 사용합니다.

```{.python .input  n=55}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## 비동기 스케줄러

먼저 시험을 동시에 평가하는 작업자 수를 정의합니다. 또한 전체 벽시계 시간에 대한 상한을 정의하여 무작위 검색을 실행할 기간을 지정해야 합니다.

```{.python .input  n=56}
n_workers = 2  # 사용 가능한 GPU 수보다 작거나 같아야 함
max_wallclock_time = 12 * 60  # 12분
```

ASHA를 실행하는 코드는 비동기 무작위 검색에 대해 했던 것의 간단한 변형입니다.

```{.python .input  n=56}
mode = "min"
metric = "validation_error"
resource_attr = "epoch"

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,
    reduction_factor=eta,
)
```

여기서 `metric`과 `resource_attr`은 `report` 콜백과 함께 사용되는 키 이름을 지정하고 `max_resource_attr`은 목적 함수에 대한 어떤 입력이 $r_{\mathrm{max}}$에 해당하는지 나타냅니다. 또한 `grace_period`는 $r_{\mathrm{min}}$을 제공하고 `reduction_factor`는 $\eta$입니다. 이전과 같이 Syne Tune을 실행할 수 있습니다(약 12분 소요).

```{.python .input  n=57}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
tuner.run()
```

우리는 실적이 저조한 시험을 조기에 중지하는 ASHA 변형을 실행하고 있다는 점에 유의하십시오. 이는 각 훈련 작업이 고정된 `max_epochs`로 시작되는 :numref:`sec_mf_hpo_sh`의 구현과 다릅니다. 후자의 경우 전체 10 에포크에 도달하는 성능이 좋은 시험은 먼저 1, 그 다음 2, 4, 8 에포크를 훈련해야 하며 매번 처음부터 시작해야 합니다. 이러한 유형의 일시 중지 및 재개 스케줄링은 각 에포크 후 훈련 상태를 체크포인팅하여 효율적으로 구현할 수 있지만 여기서는 이 추가 복잡성을 피합니다. 실험이 완료된 후 결과를 검색하고 플로팅할 수 있습니다.

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## 최적화 프로세스 시각화

다시 한 번 모든 시험의 학습 곡선(플롯의 각 색상은 시험을 나타냄)을 시각화합니다. 이를 :numref:`sec_rs_async`의 비동기 무작위 검색과 비교해 보십시오. :numref:`sec_mf_hpo`의 연속 반감에서 보았듯이 대부분의 시험은 1 또는 2 에포크($r_{\mathrm{min}}$ 또는 $\eta * r_{\mathrm{min}}$)에서 중지됩니다. 그러나 에포크당 다른 시간이 필요하기 때문에 시험이 같은 지점에서 중지되지는 않습니다. ASHA 대신 표준 연속 반감을 실행했다면 구성을 다음 렁 수준으로 승격시키기 전에 작업자를 동기화해야 했을 것입니다.

```{.python .input  n=60}
d2l.set_figsize([6, 2.5])
results = e.results
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

무작위 검색에 비해 연속 반감은 비동기 분산 설정에서 실행하기가 그리 간단하지 않습니다. 동기화 지점을 피하기 위해 구성을 가능한 한 빨리 다음 렁 수준으로 승격시키며, 이는 일부 잘못된 구성을 승격시키는 것을 의미하더라도 마찬가지입니다. 실제로 이것은 대개 큰 해가 되지 않으며, 비동기식 대 동기식 스케줄링의 이점은 일반적으로 차선책 의사 결정의 손실보다 훨씬 큽니다.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12101)
:end_tab:
