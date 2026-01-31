# 시퀀스로 작업하기 (Working with Sequences)
:label:`sec_sequence`

지금까지 우리는 입력이 단일 특성 벡터 $\mathbf{x} \in \mathbb{R}^d$로 구성된 모델에 초점을 맞추었습니다. 
시퀀스를 처리할 수 있는 모델을 개발할 때 관점의 주요 변화는 이제 특성 벡터의 순서가 있는 리스트 $\mathbf{x}_1, \dots, \mathbf{x}_T$로 구성된 입력에 초점을 맞춘다는 것입니다. 
여기서 각 특성 벡터 $\mathbf{x}_t$는 $\mathbb{R}^d$에 있는 타임 스텝 $t \in \mathbb{Z}^+$로 인덱싱됩니다.

일부 데이터셋은 단일 거대한 시퀀스로 구성됩니다. 
예를 들어 기후 과학자들이 사용할 수 있는 센서 판독값의 매우 긴 스트림을 고려하십시오. 
이러한 경우, 미리 결정된 길이의 하위 시퀀스를 무작위로 샘플링하여 훈련 데이터셋을 만들 수 있습니다. 
더 자주 데이터는 시퀀스 모음으로 도착합니다. 
다음 예를 고려하십시오: 
(i) 문서 모음, 각 문서는 고유한 단어 시퀀스로 표현되고 고유한 길이 $T_i$를 가짐; 
(ii) 병원 환자 입원의 시퀀스 표현, 각 입원은 여러 사건으로 구성되고 시퀀스 길이는 대략 입원 기간에 따라 다름.


이전에는 개별 입력을 다룰 때 동일한 기저 분포 $P(X)$에서 독립적으로 샘플링되었다고 가정했습니다. 
우리는 여전히 전체 시퀀스(예: 전체 문서 또는 환자 궤적)가 독립적으로 샘플링된다고 가정하지만, 
각 타임 스텝에 도착하는 데이터가 서로 독립적이라고 가정할 수는 없습니다. 
예를 들어 문서 뒷부분에 나타날 가능성이 높은 단어는 문서 앞부분에 나타나는 단어에 크게 의존합니다. 
병원 방문 10일째에 환자가 받을 가능성이 있는 약은 이전 9일 동안 일어난 일에 크게 의존합니다.

이것은 놀라운 일이 아닙니다. 
시퀀스의 요소가 관련이 없다고 믿었다면 애초에 시퀀스로 모델링하지 않았을 것입니다. 
검색 도구와 최신 이메일 클라이언트에서 널리 사용되는 자동 완성 기능의 유용성을 고려하십시오. 
초기 접두사가 주어졌을 때 시퀀스의 가능한 연속이 무엇일지 예측하는 것이(불완전하지만 무작위 추측보다는 낫게) 종종 가능하기 때문에 유용합니다. 
대부분의 시퀀스 모델의 경우, 시퀀스의 독립성이나 심지어 정상성(stationarity)도 요구하지 않습니다. 
대신 시퀀스 자체가 전체 시퀀스에 대한 고정된 기저 분포에서 샘플링되어야 한다는 것만 요구합니다.

이 유연한 접근 방식은 (i) 문서가 처음과 끝에서 상당히 다르게 보이거나; 
(ii) 병원 입원 기간 동안 환자 상태가 회복 또는 사망 쪽으로 진화하거나; 
(iii) 추천 시스템과의 지속적인 상호 작용 과정에서 고객 취향이 예측 가능한 방식으로 진화하는 것과 같은 현상을 허용합니다.


때로는 순차적으로 구조화된 입력이 주어졌을 때 고정된 타겟 $y$를 예측하고 싶을 때가 있습니다(예: 영화 리뷰 기반 감성 분석). 
다른 때는 고정된 입력이 주어졌을 때 순차적으로 구조화된 타겟($y_1, \ldots, y_T$)을 예측하고 싶을 때가 있습니다(예: 이미지 캡션). 
또 다른 경우에는 순차적으로 구조화된 입력을 기반으로 순차적으로 구조화된 타겟을 예측하는 것이 목표입니다(예: 기계 번역 또는 비디오 캡션). 
이러한 시퀀스-투-시퀀스 작업은 두 가지 형태를 취합니다: 
(i) *정렬된(aligned)*: 각 타임 스텝의 입력이 해당 타겟과 정렬되는 경우(예: 품사 태깅); 
(ii) *정렬되지 않은(unaligned)*: 입력과 타겟이 반드시 단계별 대응을 보이지 않는 경우(예: 기계 번역).

어떤 종류의 타겟을 처리하는 것에 대해 걱정하기 전에 가장 간단한 문제인 비지도 밀도 모델링(또는 *시퀀스 모델링*)을 다룰 수 있습니다. 
여기서 시퀀스 모음이 주어지면 우리의 목표는 주어진 시퀀스를 볼 가능성이 얼마나 되는지, 즉 $p(\mathbf{x}_1, \ldots, \mathbf{x}_T)$를 알려주는 확률 질량 함수를 추정하는 것입니다.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=9}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

## 자기회귀 모델 (Autoregressive Models)


순차적으로 구조화된 데이터를 처리하도록 설계된 특수 신경망을 소개하기 전에, 실제 시퀀스 데이터를 살펴보고 몇 가지 기본적인 직관과 통계 도구를 구축해 보겠습니다. 
특히 FTSE 100 지수의 주가 데이터에 초점을 맞출 것입니다 (:numref:`fig_ftse100`). 
각 *타임 스텝* $t \in \mathbb{Z}^+$에서 우리는 그 시점의 지수 가격 $x_t$를 관찰합니다.


![약 30년 동안의 FTSE 100 지수.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`


이제 트레이더가 다음 타임 스텝에서 지수가 상승할지 하락할지 믿는 것에 따라 전략적으로 지수에 진입하거나 빠져나오면서 단기 거래를 하고 싶다고 가정해 봅시다. 
다른 특성(뉴스, 재무 보고 데이터 등)이 없는 경우, 후속 값을 예측하는 데 사용할 수 있는 유일한 신호는 현재까지의 가격 역사입니다. 
따라서 트레이더는 다음 타임 스텝에서 지수가 취할 수 있는 가격에 대한 확률 분포

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

를 아는 데 관심이 있습니다. 
연속적인 값을 갖는 확률 변수에 대한 전체 분포를 추정하는 것은 어려울 수 있지만, 트레이더는 분포의 몇 가지 주요 통계, 특히 기댓값과 분산에 집중하는 것으로 만족할 것입니다. 
조건부 기댓값

$$\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

을 추정하기 위한 한 가지 간단한 전략은 선형 회귀 모델을 적용하는 것일 수 있습니다(:numref:`sec_linear_regression` 상기). 
신호 값을 동일한 신호의 이전 값으로 회귀하는 이러한 모델을 자연스럽게 *자기회귀 모델(autoregressive models)*이라고 합니다. 
한 가지 큰 문제가 있습니다: 입력 수 $x_{t-1}, \ldots, x_1$이 $t$에 따라 달라집니다. 
즉, 우리가 마주치는 데이터의 양에 따라 입력 수가 증가합니다. 
따라서 과거 데이터를 훈련 세트로 취급하려면 각 예제마다 특성 수가 다르다는 문제에 봉착하게 됩니다. 
이 장의 나머지 내용 대부분은 관심 대상이 $P(x_t \mid x_{t-1}, \ldots, x_1)$ 또는 이 분포의 일부 통계인 *자기회귀* 모델링 문제에 참여할 때 이러한 문제를 극복하기 위한 기술을 중심으로 전개될 것입니다.

몇 가지 전략이 자주 반복됩니다. 
우선, 긴 시퀀스 $x_{t-1}, \ldots, x_1$을 사용할 수 있지만 가까운 미래를 예측할 때 역사적으로 그렇게 멀리 되돌아볼 필요는 없을 수 있다고 믿을 수 있습니다. 
이 경우 우리는 길이 $\tau$의 어떤 윈도우에 조건을 걸고 $x_{t-1}, \ldots, x_{t-\tau}$ 관찰만 사용하는 것으로 만족할 수 있습니다. 
즉각적인 이점은 이제 적어도 $t > \tau$에 대해 인수 수가 항상 동일하다는 것입니다. 
이를 통해 고정 길이 벡터를 입력으로 필요로 하는 모든 선형 모델 또는 심층 네트워크를 훈련할 수 있습니다. 
둘째, 과거 관찰의 요약 $h_t$를 유지하는 모델을 개발할 수 있습니다(:numref:`fig_sequence-model` 참조). 동시에 예측 $\hat{x}_t$ 외에 $h_t$를 업데이트합니다. 
이것은 $\hat{x}_t = P(x_t \mid h_{t})$로 $x_t$를 추정할 뿐만 아니라 $h_t = g(h_{t-1}, x_{t-1})$ 형태의 업데이트도 수행하는 모델로 이어집니다. 
$h_t$는 결코 관찰되지 않으므로 이러한 모델을 *잠재 자기회귀 모델(latent autoregressive models)*이라고도 합니다.

![잠재 자기회귀 모델.](../img/sequence-model.svg)
:label:`fig_sequence-model`

과거 데이터로부터 훈련 데이터를 구성하기 위해 일반적으로 윈도우를 무작위로 샘플링하여 예제를 생성합니다. 
일반적으로 우리는 시간이 멈출 것이라고 기대하지 않습니다. 
그러나 $x_t$의 특정 값은 바뀔 수 있지만, 이전 관찰이 주어졌을 때 각 후속 관찰이 생성되는 역학은 변하지 않는다고 종종 가정합니다. 
통계학자들은 변하지 않는 역학을 *정상적(stationary)*이라고 부릅니다.



## 시퀀스 모델 (Sequence Models)

때로는 특히 언어 작업을 할 때 전체 시퀀스의 결합 확률을 추정하고 싶을 때가 있습니다. 
이것은 단어와 같은 이산 *토큰*으로 구성된 시퀀스로 작업할 때 일반적인 작업입니다. 
일반적으로 이렇게 추정된 함수를 *시퀀스 모델*이라고 하며 자연어 데이터의 경우 *언어 모델*이라고 합니다. 
시퀀스 모델링 분야는 자연어 처리에 의해 너무 많이 주도되어 비언어 데이터를 다룰 때조차 시퀀스 모델을 종종 "언어 모델"이라고 설명합니다. 
언어 모델은 온갖 이유로 유용함이 입증되었습니다. 
때로는 문장의 우도(likelihood)를 평가하고 싶을 때가 있습니다. 
예를 들어 기계 번역 시스템이나 음성 인식 시스템이 생성한 두 후보 출력의 자연스러움을 비교하고 싶을 수 있습니다. 
하지만 언어 모델링은 우도를 *평가*할 수 있는 능력뿐만 아니라 시퀀스를 *샘플링*하고 가장 가능성 있는 시퀀스에 대해 최적화할 수 있는 능력도 제공합니다.

언어 모델링이 언뜻 보기에는 자기회귀 문제처럼 보이지 않을 수 있지만, 
확률의 연쇄 법칙을 적용하여 시퀀스의 결합 밀도 $p(x_1, \ldots, x_T)$를 왼쪽에서 오른쪽 방식으로 조건부 밀도의 곱으로 분해함으로써 언어 모델링을 자기회귀 예측으로 줄일 수 있습니다:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$ 

단어와 같은 이산 신호로 작업하는 경우, 
자기회귀 모델은 확률적 분류기여야 하며, 
왼쪽 문맥이 주어졌을 때 다음에 올 단어에 대해 어휘 전체에 걸친 전체 확률 분포를 출력해야 한다는 점에 유의하십시오.



### 마르코프 모델 (Markov Models)
:label:`subsec_markov-models`


이제 전체 시퀀스 역사 $x_{t-1}, \ldots, x_1$보다는 $\tau$개의 이전 타임 스텝, 즉 $x_{t-1}, \ldots, x_{t-\tau}$에만 조건을 거는 위에서 언급한 전략을 채택하고 싶다고 가정해 봅시다. 
예측력의 손실 없이 이전 $\tau$ 단계 너머의 역사를 버릴 수 있을 때마다, 
우리는 시퀀스가 *마르코프 조건*을 만족한다고 말합니다. 즉, *미래는 최근 역사가 주어졌을 때 과거와 조건부 독립*입니다. 
$\tau = 1$일 때 데이터는 *1차 마르코프 모델*로 특징지어진다고 말하고, $\tau = k$일 때 데이터는 *$k$차 마르코프 모델*로 특징지어진다고 말합니다. 
1차 마르코프 조건이 성립할 때($\tau = 1$), 결합 확률의 인수분해는 이전 *단어*가 주어졌을 때 각 단어의 확률의 곱이 됩니다:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$ 

우리는 마르코프 조건이 *대략적으로*만 사실이라는 것을 알고 있을 때조차도 마치 마르코프 조건이 충족된 것처럼 진행하는 모델로 작업하는 것이 유용하다는 것을 종종 발견합니다. 
실제 텍스트 문서의 경우 왼쪽 문맥을 더 많이 포함할수록 정보를 계속 얻습니다. 
하지만 이러한 이득은 빠르게 줄어듭니다. 
따라서 때로는 타협하여 유효성이 *$k$차* 마르코프 조건에 의존하는 모델을 훈련함으로써 계산적 및 통계적 어려움을 제거합니다. 
오늘날의 거대한 RNN 및 Transformer 기반 언어 모델조차도 수천 단어 이상의 문맥을 통합하는 경우는 거의 없습니다.


이산 데이터의 경우, 실제 마르코프 모델은 단순히 각 문맥에서 각 단어가 발생한 횟수를 세어 $P(x_t \mid x_{t-1})$의 상대 빈도 추정치를 생성합니다. 
데이터가 이산 값만 가정할 때마다(언어와 같이), 
가장 가능성 있는 단어 시퀀스는 동적 프로그래밍을 사용하여 효율적으로 계산할 수 있습니다.


### 디코딩 순서 (The Order of Decoding)

텍스트 시퀀스 $P(x_1, \ldots, x_T)$의 인수분해를 왜 왼쪽에서 오른쪽으로 가는 조건부 확률 체인으로 표현했는지 궁금할 수 있습니다. 
왜 오른쪽에서 왼쪽이나 겉보기에 무작위인 순서가 아닐까요? 
원칙적으로 $P(x_1, \ldots, x_T)$를 역순으로 펼치는 데는 아무런 문제가 없습니다. 
결과는 유효한 인수분해입니다:

$$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$ 


그러나 우리가 읽는 것과 동일한 방향(대부분의 언어에서는 왼쪽에서 오른쪽, 아랍어와 히브리어에서는 오른쪽에서 왼쪽)으로 텍스트를 인수분해하는 것이 언어 모델링 작업에 선호되는 데는 여러 가지 이유가 있습니다. 
첫째, 이것은 우리가 생각하기에 더 자연스러운 방향입니다. 
결국 우리는 모두 매일 텍스트를 읽으며, 이 과정은 어떤 단어와 구문이 다음에 올 가능성이 높은지 예상하는 우리의 능력에 의해 안내됩니다. 
다른 사람의 문장을 얼마나 자주 완성했는지 생각해 보십시오. 
따라서 이러한 순서대로의 디코딩을 선호할 다른 이유가 없더라도, 이 순서로 예측할 때 무엇이 가능성 있어야 하는지에 대한 더 나은 직관을 가지고 있기 때문에 유용할 것입니다.

둘째, 순서대로 인수분해함으로써 동일한 언어 모델을 사용하여 임의로 긴 시퀀스에 확률을 할당할 수 있습니다. 
1단계에서 $t$까지의 확률을 단어 $t+1$까지 확장되는 확률로 변환하려면 단순히 이전 토큰이 주어졌을 때 추가 토큰의 조건부 확률을 곱하면 됩니다:
$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)$. 

셋째, 우리는 임의의 다른 위치에 있는 단어보다 인접한 단어를 예측하는 데 더 강력한 예측 모델을 가지고 있습니다. 
모든 순서의 인수분해가 유효하지만, 모두 똑같이 쉬운 예측 모델링 문제를 나타내는 것은 아닙니다. 
이것은 언어뿐만 아니라 다른 종류의 데이터에도 해당됩니다. 예를 들어 데이터가 인과적으로 구조화된 경우입니다. 
예를 들어 우리는 미래의 사건이 과거에 영향을 줄 수 없다고 믿습니다. 
따라서 $x_t$를 변경하면 앞으로 $x_{t+1}$에 대해 일어나는 일에 영향을 줄 수 있지만 그 반대는 아닙니다. 
즉, $x_t$를 변경해도 과거 사건에 대한 분포는 변경되지 않습니다. 
일부 맥락에서 이것은 $P(x_t \mid x_{t+1})$를 예측하는 것보다 $P(x_{t+1} \mid x_t)$를 예측하는 것을 더 쉽게 만듭니다. 
예를 들어 어떤 경우에는 어떤 가산적 노이즈 $\epsilon$에 대해 $x_{t+1} = f(x_t) + \epsilon$을 찾을 수 있지만, 그 역은 참이 아닙니다 :cite:`Hoyer.Janzing.Mooij.ea.2009`. 
이것은 좋은 소식입니다. 우리가 추정하는 데 관심이 있는 것은 일반적으로 순방향이기 때문입니다. 
:citet:`Peters.Janzing.Scholkopf.2017`의 책에는 이 주제에 대한 더 많은 내용이 포함되어 있습니다. 
우리는 겉핥기만 하고 있습니다.


## 훈련 (Training)

텍스트 데이터에 관심을 집중하기 전에 먼저 연속 값 합성 데이터로 이것을 시도해 봅시다.

(**여기서 1000개의 합성 데이터는 타임 스텝의 0.01배에 적용된 삼각 함수 `sin`을 따릅니다. 
문제를 좀 더 흥미롭게 만들기 위해 각 샘플에 가산적 노이즈를 섞습니다.**) 
이 시퀀스에서 각각 특성과 레이블로 구성된 훈련 예제를 추출합니다.

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        if tab.selected('jax'):
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key,
                                                                   [T]) * 0.2
```

```{.python .input}
%%tab all
data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

시작하기 위해, 우리는 데이터가 $\tau^{\textrm{th}}$차 마르코프 조건을 만족하는 것처럼 행동하는 모델을 시도합니다. 
따라서 과거 $\tau$개의 관찰만 사용하여 $x_t$를 예측합니다. 
[**따라서 각 타임 스텝마다 레이블 $y  = x_t$와 특성 $\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$을 가진 예제가 있습니다.**] 
예리한 독자라면 $y_1, \ldots, y_\tau$에 대한 충분한 역사가 부족하기 때문에 이것이 $1000-\tau$개의 예제를 낳는다는 것을 눈치챘을 것입니다. 
처음 $\tau$ 시퀀스를 0으로 채울 수도 있지만, 일을 단순하게 유지하기 위해 지금은 삭제합니다. 
결과 데이터셋에는 $T - \tau$개의 예제가 포함되어 있으며, 모델에 대한 각 입력은 시퀀스 길이가 $\tau$입니다. 
우리는 sin 함수의 주기를 포함하는 (**처음 600개 예제에 대한 데이터 반복자를 생성**)합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

이 예제에서 우리 모델은 표준 선형 회귀가 될 것입니다.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## 예측 (Prediction)

[**모델을 평가하기 위해 먼저 1단계 앞선 예측에서 얼마나 잘 수행되는지 확인합니다**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

```{.python .input}
%%tab jax
onestep_preds = model.apply({'params': trainer.state.params}, data.features)
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

이 예측들은 $t=1000$인 끝부분 근처에서도 좋아 보입니다.

하지만 타임 스텝 604(`n_train + tau`)까지만 시퀀스 데이터를 관찰했고 미래로 몇 단계 예측하고 싶다면 어떨까요? 
불행히도 타임 스텝 609에 대한 1단계 앞선 예측을 직접 계산할 수 없습니다. $x_{604}$까지만 보았기 때문에 해당 입력을 모르기 때문입니다. 
우리는 이전 예측을 후속 예측을 위한 모델의 입력으로 꽂아 넣고, 원하는 타임 스텝에 도달할 때까지 한 번에 한 단계씩 앞으로 투영함으로써 이 문제를 해결할 수 있습니다:

$$\begin{aligned}
\hat{x}_{605} &= f(x_{601}, x_{602}, x_{603}, x_{604}), \\n
\hat{x}_{606} &= f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\n
\hat{x}_{607} &= f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\n
\hat{x}_{608} &= f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\n
\hat{x}_{609} &= f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\n
&\vdots
\end{aligned}$$

일반적으로 관찰된 시퀀스 $x_1, \ldots, x_t$에 대해, 타임 스텝 $t+k$에서의 예측 출력 $\hat{x}_{t+k}$를 $k$*-단계 앞선 예측*이라고 합니다. 
우리는 $x_{604}$까지 관찰했으므로 $k$-단계 앞선 예측은 $\hat{x}_{604+k}$입니다. 
다시 말해, 다단계 앞선 예측을 하려면 우리 자신의 예측을 계속 사용해야 합니다. 
이것이 어떻게 진행되는지 봅시다.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab jax
multistep_preds = d2l.zeros(data.T)
multistep_preds = multistep_preds.at[:].set(data.x)
for i in range(data.num_train + data.tau, data.T):
    pred = model.apply({'params': trainer.state.params},
                       d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
    multistep_preds = multistep_preds.at[i].set(pred.item())
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

불행히도 이 경우 우리는 장엄하게 실패합니다. 
예측은 몇 단계 후에 꽤 빨리 상수로 감쇠합니다. 
미래로 더 멀리 예측할 때 알고리즘이 왜 그렇게 훨씬 나쁘게 수행되었을까요? 
궁극적으로 이것은 오류가 쌓인다는 사실 때문입니다. 
1단계 후에 오류 $\epsilon_1 = \bar\epsilon$이 있다고 가정해 봅시다. 
이제 2단계에 대한 *입력*이 $\epsilon_1$에 의해 섭동되므로, 어떤 상수 $c$에 대해 $\epsilon_2 = \bar\epsilon + c \epsilon_1$ 정도의 오류를 겪게 되고, 이런 식으로 계속됩니다. 
예측은 실제 관찰에서 빠르게 발산할 수 있습니다. 
여러분은 이미 이 일반적인 현상에 익숙할 수 있습니다. 
예를 들어 향후 24시간 동안의 일기 예보는 꽤 정확한 경향이 있지만 그 이상은 정확도가 급격히 떨어집니다. 
우리는 이 장과 그 이후에 걸쳐 이를 개선하기 위한 방법을 논의할 것입니다.

$k = 1, 4, 16, 64$에 대해 전체 시퀀스에서 예측을 계산하여 [$k$-단계 앞선 예측의 어려움을 자세히 살펴봅시다].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # (i+tau)번째 요소는 (i+1)단계 앞선 예측을 저장합니다
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab jax
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # (i+tau)번째 요소는 (i+1)단계 앞선 예측을 저장합니다
    for i in range(k):
        preds = model.apply({'params': trainer.state.params},
                            d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

이것은 미래로 더 멀리 예측하려고 할 때 예측의 품질이 어떻게 변하는지 명확하게 보여줍니다. 
4단계 앞선 예측은 여전히 좋아 보이지만 그 이상은 거의 쓸모가 없습니다.

## 요약 (Summary)

보간과 외삽 사이에는 꽤 큰 난이도 차이가 있습니다. 
결과적으로 시퀀스가 있는 경우 훈련할 때 항상 데이터의 시간 순서를 존중하십시오. 즉, 미래 데이터에 대해 훈련하지 마십시오. 
이런 종류의 데이터가 주어지면 시퀀스 모델은 추정을 위한 전문 통계 도구가 필요합니다. 
두 가지 인기 있는 선택은 자기회귀 모델과 잠재 변수 자기회귀 모델입니다. 
인과 모델(예: 시간이 앞으로 진행됨)의 경우 순방향을 추정하는 것이 일반적으로 역방향보다 훨씬 쉽습니다. 
타임 스텝 $t$까지의 관찰된 시퀀스에 대해, 타임 스텝 $t+k$에서의 예측 출력은 $k$*-단계 앞선 예측*입니다. 
$k$를 늘려 시간적으로 더 멀리 예측할수록 오류가 누적되고 예측 품질이 종종 극적으로 저하됩니다.

## 연습 문제 (Exercises)

1. 이 섹션의 실험에서 모델을 개선하십시오.
    1. 과거 4개 이상의 관찰을 통합합니까? 실제로 몇 개가 필요합니까?
    1. 노이즈가 없다면 과거 관찰이 몇 개나 필요합니까? 힌트: $\sin$과 $\cos$를 미분 방정식으로 쓸 수 있습니다.
    1. 총 특성 수를 일정하게 유지하면서 더 오래된 관찰을 통합할 수 있습니까? 이것이 정확도를 향상시킵니까? 그 이유는 무엇입니까?
    1. 신경망 아키텍처를 변경하고 성능을 평가하십시오. 새 모델을 더 많은 에폭으로 훈련할 수 있습니다. 무엇을 관찰합니까?
2. 투자자가 매수할 좋은 증권을 찾고 싶어 합니다. 
   그들은 과거 수익률을 보고 어떤 것이 잘 될지 결정합니다. 
   이 전략에서 무엇이 잘못될 수 있습니까?
3. 인과 관계가 텍스트에도 적용됩니까? 어느 정도까지입니까?
4. 데이터의 역학을 포착하기 위해 잠재 자기회귀 모델이 필요할 수 있는 예를 드십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1048)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18010)
:end_tab: