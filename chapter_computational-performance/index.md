# 계산 성능 (Computational Performance)
:label:`chap_performance`

딥러닝에서,
데이터셋과 모델은 일반적으로 크며,
이는 많은 계산을 수반합니다.
따라서 계산 성능이 매우 중요합니다.
이 장에서는 계산 성능에 영향을 미치는 주요 요인인
명령형 프로그래밍(imperative programming), 기호 프로그래밍(symbolic programming), 비동기 컴퓨팅(asynchronous computing), 자동 병렬 처리(automatic parallelism), 다중 GPU 계산(multi-GPU computation)에 초점을 맞출 것입니다.
이 장을 공부함으로써, 여러분은 이전 장에서 구현된 모델들의 계산 성능을 더욱 향상시킬 수 있습니다.
예를 들어, 정확도에 영향을 주지 않으면서 훈련 시간을 단축할 수 있습니다.

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-concise
parameterserver
```