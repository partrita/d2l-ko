# 분류를 위한 선형 신경망 (Linear Neural Networks for Classification)
:label:`chap_classification`

이제 모든 메커니즘을 살펴봤으므로, 배운 기술을 더 넓은 종류의 작업에 적용할 준비가 되었습니다. 
분류로 방향을 전환하더라도 데이터 로딩, 모델 통과, 출력 생성, 손실 계산, 가중치에 대한 기울기 계산, 모델 업데이트와 같은 대부분의 배관 작업은 동일하게 유지됩니다. 
하지만 타겟의 정확한 형태, 출력 레이어의 파라미터화, 손실 함수의 선택은 *분류(classification)* 설정에 맞게 조정될 것입니다.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```