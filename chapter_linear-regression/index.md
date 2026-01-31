# 회귀를 위한 선형 신경망 (Linear Neural Networks for Regression)
:label:`chap_regression`

신경망을 깊게 만드는 것을 걱정하기 전에, 입력이 출력에 직접 연결되는 몇 가지 얕은 신경망을 구현해 보는 것이 도움이 될 것입니다. 이는 몇 가지 이유로 중요합니다.
첫째, 복잡한 아키텍처에 주의를 빼앗기기보다 출력 레이어 파라미터화, 데이터 처리, 손실 함수 지정, 모델 훈련을 포함한 신경망 훈련의 기초에 집중할 수 있습니다.
둘째, 이 클래스의 얕은 네트워크는 선형 및 소프트맥스 회귀를 포함하여 통계적 예측의 많은 고전적인 방법을 포괄하는 선형 모델 세트를 구성합니다.
이러한 고전적인 도구들을 이해하는 것은 많은 맥락에서 널리 사용되고, 더 화려한 아키텍처의 사용을 정당화할 때 베이스라인으로 자주 사용해야 하기 때문에 매우 중요합니다.
이 장에서는 선형 회귀에 좁게 초점을 맞출 것이며, 다음 장에서는 분류를 위한 선형 신경망을 개발하여 우리의 모델링 레퍼토리를 확장할 것입니다.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```