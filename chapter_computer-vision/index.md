# 컴퓨터 비전 (Computer Vision)
:label:`chap_cv`

의료 진단, 자율 주행 차량, 카메라 모니터링, 스마트 필터 등 컴퓨터 비전 분야의 많은 응용 프로그램은 현재와 미래의 우리 삶과 밀접한 관련이 있습니다.
최근 몇 년 동안 딥러닝은 컴퓨터 비전 시스템의 성능을 발전시키는 혁신적인 힘이었습니다.
가장 진보된 컴퓨터 비전 애플리케이션은 딥러닝과 거의 분리할 수 없다고 말할 수 있습니다.
이러한 관점에서 이 장에서는 컴퓨터 비전 분야에 초점을 맞추고 최근 학계와 산업계에서 영향력 있는 방법과 응용 프로그램을 조사할 것입니다.


:numref:`chap_cnn` 및 :numref:`chap_modern_cnn`에서 우리는 컴퓨터 비전에서 일반적으로 사용되는 다양한 합성곱 신경망을 연구하고 이를 간단한 이미지 분류 작업에 적용했습니다.
이 장의 시작 부분에서는 모델 일반화를 개선할 수 있는 두 가지 방법, 즉 *이미지 증강(image augmentation)*과 *미세 조정(fine-tuning)*을 설명하고 이를 이미지 분류에 적용할 것입니다.
심층 신경망은 이미지를 여러 수준에서 효과적으로 표현할 수 있기 때문에, 이러한 계층별 표현은 *객체 감지(object detection)*, *시맨틱 분할(semantic segmentation)*, *스타일 전송(style transfer)*과 같은 다양한 컴퓨터 비전 작업에서 성공적으로 사용되었습니다.
컴퓨터 비전에서 계층별 표현을 활용하는 핵심 아이디어에 따라, 우리는 객체 감지를 위한 주요 구성 요소와 기술부터 시작할 것입니다.
다음으로 이미지의 시맨틱 분할을 위해 *완전 합성곱 네트워크(fully convolutional networks)*를 사용하는 방법을 보여줄 것입니다.
그런 다음 스타일 전송 기술을 사용하여 이 책의 표지와 같은 이미지를 생성하는 방법을 설명할 것입니다.
마지막으로, 널리 사용되는 두 가지 컴퓨터 비전 벤치마크 데이터셋에 이 장과 이전 몇 장의 자료를 적용하여 이 장을 마무리합니다.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```