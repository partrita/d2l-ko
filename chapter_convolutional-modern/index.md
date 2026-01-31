# 현대 합성곱 신경망 (Modern Convolutional Neural Networks)
:label:`chap_modern_cnn`

이제 CNN을 함께 연결하는 기본 사항을 이해했으므로 현대 CNN 아키텍처를 둘러보겠습니다. 
흥미진진한 새로운 디자인이 엄청나게 추가되고 있기 때문에 이 둘러보기는 필연적으로 불완전합니다. 
이들의 중요성은 비전 작업에 직접 사용될 수 있을 뿐만 아니라 추적 :cite:`Zhang.Sun.Jiang.ea.2021`, 분할 :cite:`Long.Shelhamer.Darrell.2015`, 객체 감지 :cite:`Redmon.Farhadi.2018`, 스타일 변환 :cite:`Gatys.Ecker.Bethge.2016`과 같은 고급 작업을 위한 기본 특성 생성기 역할을 한다는 사실에서 비롯됩니다. 
이 장의 대부분 섹션은 한때(또는 현재) 많은 연구 프로젝트와 배포된 시스템이 구축된 기본 모델이었던 중요한 CNN 아키텍처에 해당합니다. 
이러한 네트워크 각각은 잠시 동안 지배적인 아키텍처였으며 많은 네트워크가 2010년 이후 컴퓨터 비전의 지도 학습 진행 상황을 가늠하는 척도 역할을 해온 [ImageNet 대회](https://www.image-net.org/challenges/LSVRC/)의 우승자 또는 준우승자였습니다. 
최근에야 :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`를 시작으로 Swin Transformer :cite:`liu2021swin`가 그 뒤를 이으면서 Transformer가 CNN을 대체하기 시작했습니다. 
이 개발 내용은 나중에 :numref:`chap_attention-and-transformers`에서 다룰 것입니다.

*심층(deep)* 신경망의 아이디어는 매우 간단하지만(여러 레이어를 함께 쌓음), 성능은 아키텍처와 하이퍼파라미터 선택에 따라 크게 달라질 수 있습니다. 
이 장에서 설명하는 신경망은 직관, 몇 가지 수학적 통찰력, 그리고 많은 시행착오의 산물입니다. 
우리는 이 모델들을 시간 순서대로 제시하는데, 부분적으로는 역사의 흐름을 전달하여 여러분이 이 분야가 어디로 향하고 있는지에 대한 직관을 형성하고 아마도 여러분만의 아키텍처를 개발할 수 있도록 하기 위함입니다. 
예를 들어, 이 장에서 설명하는 배치 정규화와 잔차 연결은 심층 모델을 훈련하고 설계하기 위한 두 가지 인기 있는 아이디어를 제공했으며, 두 가지 모두 이후 컴퓨터 비전을 넘어선 아키텍처에도 적용되었습니다.

대규모 비전 챌린지에서 기존 컴퓨터 비전 방법을 이기기 위해 배포된 최초의 대규모 네트워크인 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`으로 현대 CNN 둘러보기를 시작합니다; 
반복되는 요소 블록을 사용하는 VGG 네트워크 :cite:`Simonyan.Zisserman.2014`; 
입력에 대해 전체 신경망을 패치 단위로 합성곱하는 네트워크 인 네트워크(NiN) :cite:`Lin.Chen.Yan.2013`; 
다중 분기 합성곱을 사용하는 네트워크를 사용하는 GoogLeNet :cite:`Szegedy.Liu.Jia.ea.2015`; 
컴퓨터 비전에서 가장 인기 있는 기성 아키텍처 중 하나로 남아 있는 잔차 네트워크(ResNet) :cite:`He.Zhang.Ren.ea.2016`; 
더 희소한 연결을 위한 ResNeXt 블록 :cite:`Xie.Girshick.Dollar.ea.2017`; 
그리고 잔차 아키텍처의 일반화를 위한 DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`이 있습니다. 
시간이 지남에 따라 좌표 이동(ShiftNet) :cite:`wu2018shift`과 같은 효율적인 네트워크를 위한 많은 특수 최적화가 개발되었습니다. 
이는 MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`와 같은 효율적인 아키텍처에 대한 자동 검색으로 정점을 찍었습니다. 
또한 이 장의 뒷부분에서 논의할 RegNetX/Y로 이어진 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 반자동 설계 탐색도 포함됩니다. 
이 작업은 효율적인 설계 공간을 탐색하는 데 있어 무차별 대입 계산과 실험자의 독창성을 결합하는 경로를 제공한다는 점에서 교훈적입니다. 
주목할 만한 것은 훈련 기술(예: 최적화기, 데이터 증강, 정규화)이 정확도 향상에 중추적인 역할을 한다는 것을 보여주는 :citet:`liu2022convnet`의 연구입니다. 
또한 계산과 데이터의 증가를 감안할 때 합성곱 윈도우 크기와 같은 오래된 가정을 재검토해야 할 수도 있음을 보여줍니다. 
이 장 전체에서 적절한 때에 이 문제와 더 많은 질문을 다룰 것입니다.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```