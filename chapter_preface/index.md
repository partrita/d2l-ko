# 서문 (Preface)

불과 몇 년 전만 해도, 주요 기업과 스타트업에서
지능형 제품과 서비스를 개발하는 딥러닝 과학자 군단은 없었습니다.
우리가 이 분야에 들어왔을 때, 머신러닝은
일간지 헤드라인을 장식하지 않았습니다.
우리 부모님은 머신러닝이 무엇인지 전혀 몰랐고,
의학이나 법학 경력보다 우리가 왜 이것을 선호하는지는 말할 것도 없었습니다.
머신러닝은 산업적 중요성이 음성 인식 및 컴퓨터 비전을 포함한
좁은 범위의 실제 응용 분야에 국한된
비현실적인 학문 분야였습니다.
게다가 이러한 응용 분야 중 다수는
너무 많은 도메인 지식을 필요로 하여
종종 머신러닝이 하나의 작은 구성 요소인 완전히 별개의 영역으로 간주되었습니다.
당시에는 신경망(이 책에서 우리가 초점을 맞추는 딥러닝 방법의 전신)이
일반적으로 구식으로 간주되었습니다.


그러나 불과 몇 년 만에 딥러닝은 세상을 놀라게 했으며,
컴퓨터 비전, 자연어 처리, 자동 음성 인식, 강화 학습, 생물의학 정보학 등
다양한 분야에서 빠른 발전을 주도했습니다.
게다가 실용적인 관심이 있는 수많은 작업에서 딥러닝의 성공은
이론적 머신러닝과 통계학의 발전까지 촉매했습니다.
이러한 발전을 바탕으로, 이제 우리는 그 어느 때보다 더 높은 자율성으로 스스로 운전하는 자동차
(일부 회사에서 믿게 만드는 것보다는 덜 자율적이지만),
명확한 질문을 함으로써 코드를 디버깅하는 대화 시스템,
그리고 수십 년은 걸릴 것이라고 생각했던 바둑과 같은 보드게임에서 세계 최고의 인간 플레이어를 이기는 소프트웨어 에이전트를 구축할 수 있습니다.
이미 이러한 도구들은 산업과 사회에 점점 더 광범위한 영향을 미치고 있으며,
영화 제작 방식, 질병 진단 방식을 바꾸고,
천체 물리학에서 기후 모델링, 기상 예측, 생물의학에 이르기까지 기초 과학에서 점점 더 큰 역할을 하고 있습니다.



## 이 책에 대하여

이 책은 여러분에게 *개념(concepts)*, *맥락(context)*, *코드(code)*를 가르쳐
딥러닝에 쉽게 접근할 수 있도록 하기 위한 저희의 시도입니다.

### 코드, 수학, HTML을 결합한 하나의 매체

어떤 컴퓨팅 기술이든 완전한 영향을 미치려면,
잘 이해되고, 잘 문서화되고,
성숙하고 잘 유지 관리되는 도구의 지원을 받아야 합니다.
핵심 아이디어는 명확하게 증류되어야 하며,
새로운 실무자를 최신 상태로 만드는 데 필요한 온보딩 시간을 최소화해야 합니다.
성숙한 라이브러리는 일반적인 작업을 자동화해야 하며,
모범 코드는 실무자가 자신의 필요에 맞게
공통 애플리케이션을 쉽게 수정, 적용 및 확장할 수 있도록 해야 합니다.


예를 들어 동적 웹 애플리케이션을 생각해 봅시다.
아마존과 같은 많은 기업들이 1990년대에 성공적인 데이터베이스 기반 웹 애플리케이션을 개발했음에도 불구하고,
이 기술이 창의적인 기업가를 도울 수 있는 잠재력은
강력하고 잘 문서화된 프레임워크의 개발 덕분에
지난 10년 동안 훨씬 더 큰 정도로 실현되었습니다.


딥러닝의 잠재력을 테스트하는 것은 독특한 도전 과제를 제시합니다.
단일 애플리케이션이 다양한 분야를 하나로 모으기 때문입니다.
딥러닝을 적용하려면 동시에 다음을 이해해야 합니다:
(i) 문제를 특정 방식으로 캐스팅하는 동기;
(ii) 주어진 모델의 수학적 형태;
(iii) 모델을 데이터에 적합시키는 최적화 알고리즘;
(iv) 모델이 보지 못한 데이터로 일반화될 것으로 예상해야 하는 시기를 알려주는 통계적 원리와
실제로 일반화되었음을 증명하는 실용적인 방법;
그리고 (v) 모델을 효율적으로 훈련하고,
수치 컴퓨팅의 함정을 탐색하고,
사용 가능한 하드웨어를 최대한 활용하는 데 필요한 엔지니어링 기술.
문제를 공식화하는 데 필요한 비판적 사고 기술,
문제를 해결하기 위한 수학,
그리고 솔루션을 구현하기 위한 소프트웨어 도구를
한곳에서 가르치는 것은 엄청난 도전 과제를 제시합니다.
이 책에서 우리의 목표는 예비 실무자들을 빠르게 적응시키기 위한
통합된 리소스를 제공하는 것입니다.

우리가 이 책 프로젝트를 시작했을 때,
동시에 다음을 만족하는 리소스는 없었습니다:
(i) 최신 상태를 유지함;
(ii) 충분한 기술적 깊이로 현대 머신러닝 관행의 폭을 다룸;
(iii) 교과서에서 기대하는 품질의 설명과
실습 튜토리얼에서 기대하는 깔끔한 실행 가능한 코드를 교차시킴.
우리는 주어진 딥러닝 프레임워크를 사용하는 방법(예: TensorFlow에서 행렬로 기본 수치 계산을 수행하는 방법)이나
특정 기술을 구현하는 방법(예: LeNet, AlexNet, ResNet 등의 코드 조각)을 보여주는
많은 코드 예제가 다양한 블로그 게시물과 GitHub 저장소에 흩어져 있는 것을 발견했습니다.
그러나 이러한 예제는 일반적으로 주어진 접근 방식을 *어떻게* 구현하는지에 초점을 맞추었지만,
*왜* 특정 알고리즘 결정이 내려졌는지에 대한 논의는 빠져 있었습니다.
일부 대화형 리소스가 특정 주제를 다루기 위해 간헐적으로 나타났지만,
예를 들어 웹사이트 [Distill](http://distill.pub)이나 개인 블로그에 게시된 매력적인 블로그 게시물,
그들은 딥러닝의 선택된 주제만 다루었으며 종종 관련 코드가 부족했습니다.
반면, 딥러닝 기초에 대한 포괄적인 조사를 제공하는
:citet:`Goodfellow.Bengio.Courville.2016`와 같은 여러 딥러닝 교과서가 등장했지만,
이러한 리소스는 설명을 코드의 개념 실현과 결합하지 않아,
때로는 독자가 구현 방법에 대해 전혀 알 수 없게 만듭니다.
게다가 너무 많은 리소스가 상업용 강의 제공업체의 유료 벽 뒤에 숨겨져 있습니다.

우리는 다음과 같은 리소스를 만들기 시작했습니다:
(i) 누구나 무료로 이용할 수 있을 것;
(ii) 실제로 응용 머신러닝 과학자가 되는 길의 출발점을 제공할 수 있을 만큼
충분한 기술적 깊이를 제공할 것;
(iii) 실행 가능한 코드를 포함하여 독자들에게 실제로 문제를 *어떻게* 해결하는지 보여줄 것;
(iv) 저희뿐만 아니라 커뮤니티 전체에 의해 빠르게 업데이트될 수 있을 것;
(v) 기술적인 세부 사항에 대한 대화형 토론과 질문 답변을 위한 [포럼](https://discuss.d2l.ai/c/5)으로 보완될 것.

이러한 목표들은 종종 충돌했습니다.
방정식, 정리 및 인용은 LaTeX에서 가장 잘 관리되고 배치됩니다.
코드는 Python으로 가장 잘 설명됩니다.
그리고 웹페이지는 HTML과 JavaScript가 기본입니다.
또한 우리는 콘텐츠가 실행 가능한 코드, 실제 책, 다운로드 가능한 PDF,
그리고 인터넷상의 웹사이트로 모두 액세스 가능하기를 원했습니다.
이러한 요구 사항에 맞는 워크플로우가 없어 보여서,
우리는 자체적으로 조립하기로 결정했습니다 (:numref:`sec_how_to_contribute`).
우리는 소스를 공유하고 커뮤니티 기여를 촉진하기 위해 GitHub를;
코드, 방정식 및 텍스트를 혼합하기 위해 주피터 노트북을;
렌더링 엔진으로 Sphinx를;
토론 플랫폼으로 Discourse를 선택했습니다.
우리 시스템이 완벽하지는 않지만,
이러한 선택은 경쟁하는 우려 사항들 사이에서 타협점을 찾습니다.
우리는 *Dive into Deep Learning*이
이러한 통합 워크플로우를 사용하여 출판된 첫 번째 책일 수 있다고 믿습니다.


### 실천을 통한 학습 (Learning by Doing)

많은 교과서가 개념을 연달아 제시하며,
각각을 철저하게 상세히 다룹니다.
예를 들어, :citet:`Bishop.2006`의 훌륭한 교과서는
각 주제를 너무 철저하게 가르쳐서
선형 회귀 챕터에 도달하기까지
상당한 양의 작업이 필요합니다.
전문가들은 바로 그 철저함 때문에 이 책을 좋아하지만,
진정한 초보자에게는 이 속성이 입문서로서의 유용성을 제한합니다.

이 책에서 우리는 대부분의 개념을 *적시에(just in time)* 가르칩니다.
즉, 어떤 실용적인 목적을 달성하는 데 필요한 바로 그 순간에 개념을 배우게 됩니다.
처음에 선형 대수와 확률 같은 기본적인 예비 지식을 가르치는 데 시간을 할애하지만,
우리는 여러분이 더 난해한 개념에 대해 걱정하기 전에
첫 번째 모델을 훈련하는 만족감을 맛보기를 원합니다.

기본적인 수학적 배경에 대한 집중 코스를 제공하는 몇 개의 예비 노트북을 제외하고,
각 후속 챕터는 합리적인 수의 새로운 개념을 소개하고
실제 데이터셋을 사용하는 여러 독립적인 작업 예제를 제공합니다.
이것은 조직적인 도전 과제를 제시했습니다.
일부 모델은 논리적으로 단일 노트북에 함께 그룹화될 수 있습니다.
그리고 일부 아이디어는 여러 모델을 연속으로 실행하여 가장 잘 가르칠 수 있습니다.
반면에, *하나의 작업 예제, 하나의 노트북* 정책을 고수하는 것에는 큰 장점이 있습니다:
이것은 여러분이 우리 코드를 활용하여 자신의 연구 프로젝트를 시작하는 것을
가능한 한 쉽게 만듭니다.
노트북을 복사하고 수정하기만 하면 됩니다.

전체적으로 우리는 실행 가능한 코드를 필요에 따라 배경 자료와 교차시킵니다.
일반적으로 우리는 도구를 완전히 설명하기 전에
도구를 사용할 수 있게 하는 쪽을 택합니다(종종 나중에 배경을 채웁니다).
예를 들어, 우리는 *확률적 경사 하강법*이 왜 유용한지 설명하거나
왜 작동하는지에 대한 직관을 제공하기 전에 사용할 수 있습니다.
이는 실무자에게 문제를 빠르게 해결하는 데 필요한 탄약을 제공하는 데 도움이 되지만,
독자가 우리의 일부 큐레이터 결정을 신뢰해야 한다는 대가가 따릅니다.

이 책은 딥러닝 개념을 처음부터(from scratch) 가르칩니다.
때로는 최신 딥러닝 프레임워크에 의해 사용자에게 숨겨져 있는
모델에 대한 세부 사항을 깊이 파고듭니다.
이는 특히 기본 튜토리얼에서 나타나는데,
우리는 여러분이 주어진 레이어(layer)나 최적화기(optimizer)에서 일어나는
모든 일을 이해하기를 원하기 때문입니다.
이러한 경우, 우리는 종종 예제의 두 가지 버전을 제시합니다:
하나는 NumPy와 유사한 기능과 자동 미분에만 의존하여 모든 것을 처음부터 구현하는 것이고,
다른 하나는 딥러닝 프레임워크의 고수준 API를 사용하여
간결한 코드를 작성하는 더 실용적인 예제입니다.
일부 구성 요소가 어떻게 작동하는지 설명한 후,
후속 튜토리얼에서는 고수준 API에 의존합니다.


### 콘텐츠 및 구조

이 책은 대략 세 부분으로 나눌 수 있으며,
예비 지식,
딥러닝 기술,
그리고 실제 시스템과 응용에 초점을 맞춘
고급 주제를 다룹니다 (:numref:`fig_book_org`).

![책 구조.](../img/book-org.svg)
:label:`fig_book_org`


* **1부: 기초 및 예비 지식**.
:numref:`chap_introduction`은 딥러닝에 대한 소개입니다.
그 다음 :numref:`chap_preliminaries`에서는
데이터 저장 및 조작 방법,
선형 대수, 미적분, 확률의 기본 개념을 기반으로
다양한 수치 연산을 적용하는 방법 등
실습 딥러닝에 필요한 필수 조건을 빠르게 알려드립니다.
:numref:`chap_regression` 및 :numref:`chap_perceptrons`는
회귀 및 분류; 선형 모델; 다층 퍼셉트론;
과대적합 및 정규화를 포함한
딥러닝의 가장 기본적인 개념과 기술을 다룹니다.

* **2부: 현대 딥러닝 기술**.
:numref:`chap_computation`은 딥러닝 시스템의 핵심 계산 구성 요소를 설명하고
더 복잡한 모델의 후속 구현을 위한 토대를 마련합니다.
다음으로 :numref:`chap_cnn` 및 :numref:`chap_modern_cnn`은
대부분의 현대 컴퓨터 비전 시스템의 중추를 형성하는 강력한 도구인
합성곱 신경망(CNN)을 제시합니다.
마찬가지로 :numref:`chap_rnn` 및 :numref:`chap_modern_rnn`은
데이터의 순차적(예: 시간적) 구조를 활용하고
자연어 처리 및 시계열 예측에 일반적으로 사용되는 모델인
순환 신경망(RNN)을 소개합니다.
:numref:`chap_attention-and-transformers`에서는
대부분의 자연어 처리 작업에서 지배적인 아키텍처로 RNN을 대체한
소위 *어텐션 메커니즘*에 기반한 비교적 새로운 종류의 모델을 설명합니다.
이 섹션들은 딥러닝 실무자들이 널리 사용하는
가장 강력하고 일반적인 도구에 대해 빠르게 알려드릴 것입니다.

* **3부: 확장성, 효율성 및 응용** ([온라인](https://d2l.ai)에서 이용 가능).
12장에서는 딥러닝 모델을 훈련하는 데 사용되는
몇 가지 일반적인 최적화 알고리즘에 대해 논의합니다.
다음으로 13장에서는 딥러닝 코드의 계산 성능에 영향을 미치는
몇 가지 주요 요소를 살펴봅니다.
그런 다음 14장에서는 컴퓨터 비전에서 딥러닝의 주요 응용 사례를 보여줍니다.
마지막으로 15장과 16장에서는 언어 표현 모델을 사전 훈련하고
자연어 처리 작업에 적용하는 방법을 시연합니다.


### 코드
:label:`sec_code`

이 책의 대부분의 섹션에는 실행 가능한 코드가 있습니다.
우리는 일부 직관이 시행착오를 통해,
코드를 조금씩 조정하고 결과를 관찰함으로써 가장 잘 개발된다고 믿습니다.
이상적으로는 우아한 수학적 이론이 원하는 결과를 얻기 위해
코드를 어떻게 조정해야 하는지 정확하게 알려줄 수 있습니다.
그러나 오늘날 딥러닝 실무자들은 종종 확실한 이론적 지침이 없는 곳을 밟아야 합니다.
우리의 최선의 노력에도 불구하고, 다양한 기술의 효능에 대한 공식적인 설명은
여러 가지 이유로 여전히 부족합니다: 이러한 모델을 특징짓는 수학은 매우 어려울 수 있습니다;
설명은 현재 명확한 정의가 부족한 데이터의 속성에 따라 달라질 가능성이 높습니다;
그리고 이러한 주제에 대한 진지한 탐구는 최근에야 본격화되었습니다.
우리는 딥러닝 이론이 발전함에 따라,
이 책의 미래 판이 현재 이용 가능한 것보다 더 뛰어난 통찰력을 제공하기를 희망합니다.

불필요한 반복을 피하기 위해, 우리는 가장 자주 가져오고 사용하는
함수와 클래스 중 일부를 `d2l` 패키지에 캡처합니다.
전체적으로 우리는 코드 블록(함수, 클래스,
또는 import 문 모음 등)을 `#@save`로 표시하여
나중에 `d2l` 패키지를 통해 액세스될 것임을 나타냅니다.
우리는 :numref:`sec_d2l`에서 이러한 클래스와 함수에 대한 자세한 개요를 제공합니다.
`d2l` 패키지는 가볍고 다음 종속성만 필요합니다:

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
이 책의 대부분의 코드는 AWS(Amazon Web Services)뿐만 아니라
많은 대학과 회사에서 선호하는 오픈 소스 딥러닝 프레임워크인
Apache MXNet을 기반으로 합니다.
이 책의 모든 코드는 최신 MXNet 버전에서 테스트를 통과했습니다.
그러나 딥러닝의 빠른 발전으로 인해 *인쇄판*의 일부 코드는
향후 버전의 MXNet에서 제대로 작동하지 않을 수 있습니다.
우리는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참조하여
코드와 런타임 환경을 업데이트하십시오.
아래는 MXNet 구현의 종속성을 나열합니다.
:end_tab:

:begin_tab:`pytorch`
이 책의 대부분의 코드는 딥러닝 연구 커뮤니티에서
열광적으로 받아들여진 인기 있는 오픈 소스 프레임워크인
PyTorch를 기반으로 합니다.
이 책의 모든 코드는 최신 안정 버전의 PyTorch에서 테스트를 통과했습니다.
그러나 딥러닝의 빠른 발전으로 인해 *인쇄판*의 일부 코드는
향후 버전의 PyTorch에서 제대로 작동하지 않을 수 있습니다.
우리는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참조하여
코드와 런타임 환경을 업데이트하십시오.
아래는 PyTorch 구현의 종속성을 나열합니다.
:end_tab:

:begin_tab:`tensorflow`
이 책의 대부분의 코드는 업계에서 널리 채택되고
연구자들 사이에서 인기 있는 오픈 소스 딥러닝 프레임워크인
TensorFlow를 기반으로 합니다.
이 책의 모든 코드는 최신 안정 버전의 TensorFlow에서 테스트를 통과했습니다.
그러나 딥러닝의 빠른 발전으로 인해 *인쇄판*의 일부 코드는
향후 버전의 TensorFlow에서 제대로 작동하지 않을 수 있습니다.
우리는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참조하여
코드와 런타임 환경을 업데이트하십시오.
아래는 TensorFlow 구현의 종속성을 나열합니다.
:end_tab:

:begin_tab:`jax`
이 책의 대부분의 코드는 임의의 Python 및 NumPy 함수의 미분,
JIT 컴파일, 벡터화 등과 같은 구성 가능한 함수 변환을 가능하게 하는
오픈 소스 프레임워크인 Jax를 기반으로 합니다!
머신러닝 연구 공간에서 인기를 얻고 있으며
배우기 쉬운 NumPy와 유사한 API를 가지고 있습니다.
실제로 JAX는 NumPy와 1:1 동등성을 달성하려고 노력하므로,
코드를 전환하는 것은 단일 import 문을 변경하는 것만큼 간단할 수 있습니다!
그러나 딥러닝의 빠른 발전으로 인해 *인쇄판*의 일부 코드는
향후 버전의 Jax에서 제대로 작동하지 않을 수 있습니다.
우리는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참조하여
코드와 런타임 환경을 업데이트하십시오.
아래는 JAX 구현의 종속성을 나열합니다.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### 대상 독자

이 책은 딥러닝의 실용적인 기술을 확실하게 파악하고자 하는
학생(학부 또는 대학원), 엔지니어 및 연구자를 위한 것입니다.
우리는 모든 개념을 처음부터 설명하므로
딥러닝이나 머신러닝에 대한 이전 배경 지식은 필요하지 않습니다.
딥러닝의 방법을 완전히 설명하려면 약간의 수학과 프로그래밍이 필요하지만,
우리는 여러분이 약간의 선형 대수, 미적분, 확률 및 Python 프로그래밍을 포함한
몇 가지 기본 지식을 가지고 들어온다고 가정할 것입니다.
혹시 잊어버렸을 경우를 대비하여,
[온라인 부록](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)은
이 책에서 찾을 수 있는 대부분의 수학에 대한 복습을 제공합니다.
일반적으로 우리는 수학적 엄격함보다 직관과 아이디어를 우선시할 것입니다.
이 책을 이해하기 위한 필수 조건을 넘어 이러한 기초를 확장하고 싶다면,
몇 가지 다른 훌륭한 리소스를 기쁘게 추천합니다:
:citet:`Bollobas.1999`의 *Linear Analysis*는
선형 대수와 함수 해석학을 깊이 있게 다룹니다.
*All of Statistics* :cite:`Wasserman.2013`는
통계에 대한 놀라운 소개를 제공합니다.
확률과 추론에 대한 Joe Blitzstein의 [책](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)과
[강의](https://projects.iq.harvard.edu/stat110/home)는 교육적 보석입니다.
그리고 Python을 사용해 본 적이 없다면,
이 [Python 튜토리얼](http://learnpython.org/)을 정독하고 싶을 수도 있습니다.


### 노트북, 웹사이트, GitHub 및 포럼

모든 노트북은 [D2L.ai 웹사이트](https://d2l.ai)와
[GitHub](https://github.com/d2l-ai/d2l-en)에서 다운로드할 수 있습니다.
이 책과 관련하여 우리는 [discuss.d2l.ai](https://discuss.d2l.ai/c/5)에서 토론 포럼을 시작했습니다.
책의 어느 섹션에 대해서든 질문이 있을 때마다,
각 노트북 끝에서 관련 토론 페이지로 연결되는 링크를 찾을 수 있습니다.



## 감사의 말

우리는 영어와 중국어 초안 모두에 대해 수백 명의 기여자들에게 빚을 지고 있습니다.
그들은 콘텐츠를 개선하는 데 도움을 주었고 귀중한 피드백을 제공했습니다.
이 책은 원래 기본 프레임워크로 MXNet을 사용하여 구현되었습니다.
이전 MXNet 코드의 대다수 부분을 각각 PyTorch와 TensorFlow 구현으로 수정해 준 Anirudh Dagar와 Yuan Tang에게 감사드립니다.
2021년 7월부터 우리는 이 책을 PyTorch, MXNet, TensorFlow로 재설계하고 재구현했으며, PyTorch를 기본 프레임워크로 선택했습니다.
최신 PyTorch 코드의 대다수 부분을 JAX 구현으로 수정해 준 Anirudh Dagar에게 감사드립니다.
중국어 초안에서 최신 PyTorch 코드의 대다수 부분을 PaddlePaddle 구현으로 수정해 준 Baidu의 Gaosheng Wu, Liujun Hu, Ge Zhang, Jiehang Xie에게 감사드립니다.
언론사의 LaTeX 스타일을 PDF 빌드에 통합해 준 Shuai Zhang에게 감사드립니다.

GitHub에서 모두를 위해 이 영어 초안을 더 좋게 만들어 준 모든 기여자에게 감사드립니다.
그들의 GitHub ID 또는 이름은 다음과 같습니다(순서 없음):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

이 책을 집필하는 데 아낌없는 지원을 해준 Amazon Web Services, 특히 Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky, Andrew Jassy에게 감사드립니다.
사용 가능한 시간, 자원, 동료와의 토론, 지속적인 격려가 없었다면 이 책은 탄생하지 못했을 것입니다.
출판을 위해 책을 준비하는 동안 Cambridge University Press는 훌륭한 지원을 제공했습니다.
도움과 전문성을 보여준 커미셔닝 편집자 David Tranah에게 감사드립니다.


## 요약

딥러닝은 패턴 인식을 혁신하여
컴퓨터 비전, 자연어 처리, 자동 음성 인식과 같은 다양한 분야에서
현재 광범위한 기술을 구동하는 기술을 도입했습니다.
딥러닝을 성공적으로 적용하려면,
문제를 캐스팅하는 방법, 모델링의 기본 수학,
모델을 데이터에 적합시키는 알고리즘,
그리고 이 모든 것을 구현하는 엔지니어링 기술을 이해해야 합니다.
이 책은 산문, 그림, 수학, 코드를 모두 한곳에 포함하는 포괄적인 리소스를 제공합니다.



## 연습 문제

1. 이 책의 토론 포럼 [discuss.d2l.ai](https://discuss.d2l.ai/)에 계정을 등록하십시오.
2. 컴퓨터에 Python을 설치하십시오.
3. 섹션 하단에 있는 링크를 따라 포럼으로 이동하여 도움을 구하고 책에 대해 토론하며 저자 및 광범위한 커뮤니티와 교류하여 질문에 대한 답변을 찾을 수 있습니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17963)
:end_tab: