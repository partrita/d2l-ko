# 정보 이론 (Information Theory)
:label:`sec_information_theory`

우주는 정보로 넘쳐나고 있습니다. 정보는 학문적 차이를 가로지르는 공통 언어를 제공합니다: 셰익스피어의 소네트에서 코넬 ArXiv의 연구자 논문에 이르기까지, 반 고흐의 별이 빛나는 밤에서 베토벤의 교향곡 5번에 이르기까지, 최초의 프로그래밍 언어 Plankalkül에서 최첨단 머신러닝 알고리즘에 이르기까지 말입니다. 모든 것은 형식이 무엇이든 정보 이론의 규칙을 따라야 합니다. 정보 이론을 통해 우리는 서로 다른 신호에 얼마나 많은 정보가 존재하는지 측정하고 비교할 수 있습니다. 이 섹션에서는 정보 이론의 기본 개념과 머신러닝에서의 정보 이론 응용을 조사할 것입니다.

시작하기 전에 머신러닝과 정보 이론 사이의 관계를 개략적으로 설명해 봅시다. 머신러닝은 데이터에서 흥미로운 신호를 추출하고 중요한 예측을 하는 것을 목표로 합니다. 반면에 정보 이론은 정보를 인코딩, 디코딩, 전송 및 조작하는 것을 연구합니다. 결과적으로 정보 이론은 머신러닝 시스템에서의 정보 처리를 논의하기 위한 근본적인 언어를 제공합니다. 예를 들어, 많은 머신러닝 응용 프로그램은 :numref:`sec_softmax`에서 설명한 대로 크로스 엔트로피 손실을 사용합니다. 이 손실은 정보 이론적 고려 사항에서 직접 도출될 수 있습니다.


## 정보 (Information)

정보 이론의 "영혼"인 정보부터 시작하겠습니다. *정보(Information)*는 하나 이상의 인코딩 형식의 특정 시퀀스를 가진 무엇이든 인코딩될 수 있습니다. 우리가 정보라는 개념을 정의하려고 노력하는 과제를 스스로에게 맡겼다고 가정해 봅시다. 우리의 시작점은 무엇이 될 수 있을까요?

다음 사고 실험을 고려해 보십시오. 카드 한 덱을 가진 친구가 있습니다. 그들은 덱을 섞고, 카드 몇 장을 뒤집고, 카드에 대한 진술을 우리에게 말해줄 것입니다. 우리는 각 진술의 정보 내용을 평가하려고 노력할 것입니다.

먼저, 그들은 카드를 한 장 뒤집고 "카드가 보여."라고 말합니다. 이것은 우리에게 아무런 정보도 제공하지 않습니다. 우리는 이미 그것이 사실이라는 것을 확신했으므로 정보가 0이 되기를 바랍니다.

다음으로, 그들은 카드를 한 장 뒤집고 "하트가 보여."라고 말합니다. 이것은 우리에게 약간의 정보를 제공하지만, 실제로는 가능한 4가지 다른 무늬가 있고 각각의 가능성이 동일했으므로 우리는 이 결과에 놀라지 않습니다. 정보의 척도가 무엇이든 이 이벤트는 낮은 정보 내용을 가져야 할 것입니다.

다음으로, 그들은 카드를 한 장 뒤집고 "이것은 스페이드 3이야."라고 말합니다. 이것은 더 많은 정보입니다. 실제로 52가지의 동등하게 가능성 있는 결과가 있었고, 우리 친구는 그중 어느 것인지 알려주었습니다. 이것은 중간 정도의 정보량이어야 합니다.

이것을 논리적 극단으로 가져가 봅시다. 마지막으로 그들이 덱의 모든 카드를 뒤집고 섞인 덱의 전체 시퀀스를 읽어준다고 가정해 봅시다. 덱에는 $52!$가지의 다른 순서가 있고, 다시 모두 동등하게 가능성이 높으므로 그것이 어느 것인지 알기 위해 많은 정보가 필요합니다.

우리가 개발하는 정보의 개념은 이 직관에 부합해야 합니다. 실제로 다음 섹션에서 우리는 이러한 이벤트들이 각각 $0\textrm{ 비트}$, $2\textrm{ 비트}$, $~5.7\textrm{ 비트}$, $~225.6\textrm{ 비트}$의 정보를 가지고 있음을 계산하는 방법을 배울 것입니다.

이러한 사고 실험을 읽어보면 자연스러운 아이디어가 보입니다. 시작점으로서 지식에 신경 쓰기보다는, 정보가 놀라움의 정도나 이벤트의 추상적인 가능성을 나타낸다는 아이디어를 기반으로 구축할 수 있습니다. 예를 들어, 특이한 이벤트를 설명하려면 많은 정보가 필요합니다. 흔한 이벤트의 경우 많은 정보가 필요하지 않을 수 있습니다.

1948년, 클로드 E. 섀넌(Claude E. Shannon)은 정보 이론을 정립한 *통신의 수학적 이론(A Mathematical Theory of Communication)* :cite:`Shannon.1948`을 발표했습니다. 그의 기사에서 섀넌은 정보 엔트로피라는 개념을 처음으로 도입했습니다. 여기서 우리의 여정을 시작하겠습니다.


### 자기 정보 (Self-information)

정보가 이벤트의 추상적인 가능성을 구체화한다면, 그 가능성을 비트 수로 어떻게 매핑할까요? 섀넌은 존 투키(John Tukey)가 처음 만든 정보의 단위로 *비트(bit)*라는 용어를 도입했습니다. 그렇다면 "비트"란 무엇이며 왜 정보를 측정하는 데 그것을 사용할까요? 역사적으로 오래된 송신기는 $0$과 $1$이라는 두 가지 유형의 코드만 보내거나 받을 수 있었습니다. 실제로 이진 인코딩은 모든 현대 디지털 컴퓨터에서 여전히 공통적으로 사용됩니다. 이런 식으로 모든 정보는 일련의 $0$과 $1$로 인코딩됩니다. 따라서 길이가 $n$인 일련의 이진 숫자는 $n$비트의 정보를 포함합니다.

이제 임의의 일련의 코드에 대해 각 $0$ 또는 $1$이 $rac{1}{2}$의 확률로 발생한다고 가정합시다. 따라서 길이가 $n$인 일련의 코드를 가진 이벤트 $X$는 $rac{1}{2^n}$의 확률로 발생합니다. 동시에 앞서 언급했듯이 이 시리즈는 $n$비트의 정보를 포함합니다. 그렇다면 확률 $p$를 비트 수로 변환할 수 있는 수학적 함수로 일반화할 수 있을까요? 섀넌은 *자기 정보(self-information)*를 다음과 같이 정의하여 답을 주었습니다.

$$I(X) = - \log_2 (p),$$ 

이 이벤트 $X$에 대해 우리가 받은 정보의 *비트*입니다. 이 섹션에서는 항상 밑이 2인 로그를 사용할 것임에 유의하십시오. 단순함을 위해 이 섹션의 나머지 부분에서는 로그 표기법에서 아래 첨자 2를 생략할 것입니다. 즉, $\log(.)$은 항상 $\log_2(.)$을 지칭합니다. 예를 들어 코드 "0010"은 다음과 같은 자기 정보를 갖습니다.

$$I(\textrm{"0010"}) = - \log (p(\textrm{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \textrm{ 비트}.$$ 

아래와 같이 자기 정보를 계산할 수 있습니다. 그 전에 먼저 이 섹션에 필요한 모든 패키지를 가져옵시다.

```{.python .input}
#@tab mxnet
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # pytorch에는 nansum이 내장되어 있지 않으므로 정의합니다.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## 엔트로피 (Entropy)

자기 정보는 단일 이산 이벤트의 정보만 측정하므로, 이산 또는 연속 분포의 임의의 확률 변수에 대해 더 일반화된 척도가 필요합니다.


### 엔트로피의 동기 (Motivating Entropy)

우리가 원하는 것을 구체화해 봅시다. 이것은 *섀넌 엔트로피의 공리*라고 알려진 것들에 대한 비공식적인 진술이 될 것입니다. 다음의 상식적인 진술 모음이 우리를 정보의 고유한 정의로 이끌 것이라는 점이 밝혀질 것입니다. 이러한 공리들의 공식적인 버전은 다른 여러 공리들과 함께 :citet:`Csiszar.2008`에서 찾을 수 있습니다.

1. 확률 변수를 관찰함으로써 얻는 정보는 우리가 요소를 무엇이라고 부르는지, 또는 확률이 0인 추가 요소의 존재 여부에 의존하지 않습니다.
2. 두 확률 변수를 관찰함으로써 얻는 정보는 그것들을 따로 관찰함으로써 얻는 정보의 합보다 크지 않습니다. 만약 그것들이 독립적이라면 정확히 그 합과 같습니다.
3. (거의) 확실한 이벤트를 관찰할 때 얻는 정보는 (거의) 0입니다.

이 사실을 증명하는 것은 우리 텍스트의 범위를 벗어나지만, 이것이 엔트로피가 취해야 할 형태를 고유하게 결정한다는 것을 아는 것이 중요합니다. 이것들이 허용하는 유일한 모호함은 기본 단위의 선택에 있으며, 이는 단일 공정한 동전 던지기에 의해 제공되는 정보가 1비트라는 우리가 이전에 보았던 선택을 함으로써 가장 자주 정규화됩니다.

### 정의 (Definition)

확률 밀도 함수(p.d.f.) 또는 확률 질량 함수(p.m.f.) $p(x)$를 갖는 확률 분포 $P$를 따르는 임의의 확률 변수 $X$에 대해, 우리는 *엔트로피(entropy)* (또는 *섀넌 엔트로피*)를 통해 예상 정보량을 측정합니다.

$$H(X) = - E_{x \sim P} [\log p(x)].$$ 
:eqlabel:`eq_ent_def`

구체적으로, $X$가 이산형인 경우, $$H(X) = - \sum_i p_i \log p_i \textrm{, 여기서 } p_i = P(X_i).$$ 

그렇지 않고 $X$가 연속형인 경우 엔트로피를 *미분 엔트로피(differential entropy)*라고도 부릅니다.

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$ 

아래와 같이 엔트로피를 정의할 수 있습니다.

```{.python .input}
#@tab mxnet
def entropy(p):
    entropy = - p * np.log2(p)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### 해석 (Interpretations)

궁금하실 수 있습니다: 엔트로피 정의 :eqref:`eq_ent_def`에서 왜 음의 로그의 기대값을 사용할까요? 여기 몇 가지 직관이 있습니다.

먼저, 왜 *로그* 함수 $\log$를 사용할까요? $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$라고 가정해 봅시다. 여기서 각 성분 함수 $f_i(x)$는 서로 독립적입니다. 이는 각 $f_i(x)$가 $p(x)$로부터 얻은 총 정보에 독립적으로 기여함을 의미합니다. 위에서 논의한 대로, 우리는 엔트로피 공식이 독립 확률 변수들에 대해 가산적(additive)이기를 원합니다. 다행히 $\log$는 자연스럽게 확률 분포의 곱을 개별 항들의 합으로 바꿀 수 있습니다.

다음으로, 왜 *음의* $\log$를 사용할까요? 직관적으로, 더 빈번한 이벤트는 덜 흔한 이벤트보다 적은 정보를 포함해야 합니다. 평범한 경우보다 특이한 경우에서 종종 더 많은 정보를 얻기 때문입니다. 그러나 $\log$는 확률에 따라 단조 증가하며, 실제로 $[0, 1]$의 모든 값에 대해 음수입니다. 우리는 이벤트의 확률과 그들의 엔트로피 사이에 단조 감소 관계를 구축해야 하며, 이는 이상적으로 항상 양수여야 합니다(우리가 관찰하는 어떤 것도 우리가 알고 있는 것을 잊도록 강제해서는 안 되기 때문입니다). 따라서 $\log$ 함수 앞에 음수 부호를 붙입니다.

마지막으로, *기대값* 함수는 어디에서 왔을까요? 확률 변수 $X$를 고려해 보십시오. 자기 정보($-\log(p)$)를 특정 결과를 볼 때 우리가 느끼는 *놀라움*의 양으로 해석할 수 있습니다. 실제로 확률이 0에 가까워질수록 놀라움은 무한대가 됩니다. 마찬가지로, 엔트로피를 $X$를 관찰함으로써 얻는 평균 놀라움의 양으로 해석할 수 있습니다. 예를 들어, 슬롯머신 시스템이 각각 확률 ${p_1, \ldots, p_k}$로 기호 ${s_1, \ldots, s_k}$를 통계적으로 독립적으로 내보낸다고 상상해 보십시오. 그러면 이 시스템의 엔트로피는 각 출력을 관찰함으로써 얻는 평균 자기 정보와 같습니다. 즉,

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$ 



### 엔트로피의 속성 (Properties of Entropy)

위의 예제와 해석을 통해 엔트로피 :eqref:`eq_ent_def`의 다음과 같은 속성을 도출할 수 있습니다. 여기서 $X$를 이벤트로, $P$를 $X$의 확률 분포로 지칭합니다.

* 모든 이산 $X$에 대해 $H(X) \geq 0$입니다(연속 $X$의 경우 엔트로피가 음수일 수 있습니다).

* $X \sim P$가 p.d.f. 또는 p.m.f. $p(x)$를 갖고, 우리가 $P$를 p.d.f. 또는 p.m.f. $q(x)$를 갖는 새로운 확률 분포 $Q$로 추정하려고 한다면, $$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)] \textrm{이며, 등호는 } P = Q \textrm{일 때만 성립합니다.}}$$ 대안적으로, $H(X)$는 $P$에서 추출된 기호를 인코딩하는 데 필요한 평균 비트 수의 하한을 제공합니다.

* $X \sim P$인 경우, $x$가 가능한 모든 결과에 고르게 퍼져 있을 때 최대량의 정보를 전달합니다. 구체적으로, 확률 분포 $P$가 $k$-클래스 ${p_1, \ldots, p_k }$를 갖는 이산형인 경우, $$H(X) \leq \log(k) \textrm{이며, 등호는 모든 } i \textrm{에 대해 } p_i = \frac{1}{k} \textrm{일 때만 성립합니다.}}$$ $P$가 연속 확률 변수라면 이야기는 훨씬 더 복잡해집니다. 그러나 추가적으로 $P$가 유한한 구간(0과 1 사이의 모든 값)에서 지원된다고 부과하면, $P$가 해당 구간에서 균등 분포일 때 가장 높은 엔트로피를 갖습니다.


## 상호 정보량 (Mutual Information)

이전에 단일 확률 변수 $X$의 엔트로피를 정의했는데, 한 쌍의 확률 변수 $(X, Y)$의 엔트로피는 어떨까요? 우리는 이러한 기술을 다음과 같은 유형의 질문에 답하려는 시도로 생각할 수 있습니다. "X와 Y가 따로따로 있을 때와 비교하여 함께 있을 때 어떤 정보가 포함되어 있는가? 중복된 정보가 있는가, 아니면 모두 고유한가?"

다음 논의를 위해, 우리는 항상 $(X, Y)$를 결합 확률 분포 $P$(p.d.f. 또는 p.m.f. $p_{X, Y}(x, y)$ 가짐)를 따르는 확률 변수 쌍으로 사용하고, $X$와 $Y$는 각각 확률 분포 $p_X(x)$와 $p_Y(y)$를 따릅니다.


### 결합 엔트로피 (Joint Entropy)

단일 확률 변수의 엔트로피 :eqref:`eq_ent_def`와 유사하게, 우리는 확률 변수 쌍 $(X, Y)$의 *결합 엔트로피(joint entropy)* $H(X, Y)$를 다음과 같이 정의합니다.

$$H(X, Y) = -E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

정확하게 말하면, 한편으로 $(X, Y)$가 이산 확률 변수 쌍이라면 다음과 같습니다.

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$ 

다른 한편으로, $(X, Y)$가 연속 확률 변수 쌍이라면 *미분 결합 엔트로피*를 다음과 같이 정의합니다.

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$ 

우리는 :eqref:`eq_joint_ent_def`를 확률 변수 쌍의 총 무작위성을 알려주는 것으로 생각할 수 있습니다. 극단적인 한 쌍으로서, $X = Y$가 두 개의 동일한 확률 변수라면 쌍의 정보는 정확히 하나의 정보와 같으며 $H(X, Y) = H(X) = H(Y)$를 갖습니다. 다른 극단으로서, $X$와 $Y$가 독립적이라면 $H(X, Y) = H(X) + H(Y)$입니다. 실제로 한 쌍의 확률 변수에 포함된 정보는 어느 한 확률 변수의 엔트로피보다 작지 않고 두 엔트로피의 합보다 크지 않음을 항상 갖게 될 것입니다.

$$ 
 H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$ 

결합 엔트로피를 밑바닥부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

이것은 이전과 동일한 *코드*이지만, 이제는 두 확률 변수의 결합 분포에 대해 작동하는 것으로 다르게 해석한다는 점에 유의하십시오.


### 조건부 엔트로피 (Conditional Entropy)

위에서 정의된 결합 엔트로피는 확률 변수 쌍에 포함된 정보량입니다. 이것은 유용하지만, 우리가 관심을 갖는 것이 아닌 경우가 많습니다. 머신러닝의 설정을 고려해 보십시오. $X$를 이미지의 픽셀 값을 설명하는 확률 변수(또는 확률 변수 벡터)로, $Y$를 클래스 레이블인 확률 변수로 취해 봅시다. $X$는 상당한 정보를 포함해야 합니다 - 자연 이미지는 복잡한 것입니다. 그러나 이미지가 보여진 후 $Y$에 포함된 정보는 낮아야 합니다. 실제로 숫자의 이미지는 숫자가 읽을 수 없는 경우가 아니면 그것이 어떤 숫자인지에 대한 정보를 이미 포함하고 있어야 합니다. 따라서 정보 이론의 어휘를 계속 확장하려면 다른 확률 변수에 조건부인 확률 변수의 정보 내용을 추론할 수 있어야 합니다.

확률론에서 우리는 변수 간의 관계를 측정하기 위해 *조건부 확률*의 정의를 보았습니다. 우리는 이제 유사하게 *조건부 엔트로피(conditional entropy)* $H(Y \mid X)$를 정의하고자 합니다. 우리는 이를 다음과 같이 쓸 수 있습니다.

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

여기서 $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$는 조건부 확률입니다. 구체적으로, $(X, Y)$가 이산 확률 변수 쌍이라면 다음과 같습니다.

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$ 

$(X, Y)$가 연속 확률 변수 쌍이라면 *미분 조건부 엔트로피*는 다음과 같이 유사하게 정의됩니다.

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$ 


이제 *조건부 엔트로피* $H(Y \mid X)$가 엔트로피 $H(X)$ 및 결합 엔트로피 $H(X, Y)$와 어떤 관련이 있는지 묻는 것이 자연스럽습니다. 위의 정의를 사용하면 이를 다음과 같이 깔끔하게 표현할 수 있습니다.

$$H(Y \mid X) = H(X, Y) - H(X).$$ 

이것은 직관적인 해석을 갖습니다: $X$가 주어졌을 때 $Y$의 정보($H(Y \mid X)$)는 $X$와 $Y$가 함께 있을 때의 정보($H(X, Y)$)에서 이미 $X$에 포함된 정보($H(X)$)를 뺀 것과 같습니다. 이는 $X$에도 표현되지 않은 $Y$의 정보를 우리에게 제공합니다.

이제 조건부 엔트로피 :eqref:`eq_cond_ent_def`를 밑바닥부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### 상호 정보량 (Mutual Information)

확률 변수 $(X, Y)$의 이전 설정이 주어졌을 때, 여러분은 다음과 같이 궁금해하실 수 있습니다: "$Y$에는 포함되어 있지만 $X$에는 없는 정보가 얼마나 되는지 알았으니, 비슷하게 $X$와 $Y$ 사이에 공유되는 정보가 얼마나 되는지 물을 수 있을까?" 그 답은 $(X, Y)$의 *상호 정보량(mutual information)*이 될 것이며, 우리는 이를 $I(X, Y)$라고 쓸 것입니다.

공식적인 정의로 바로 뛰어드는 대신, 우리가 이전에 구성한 용어들에 전적으로 기반하여 상호 정보량에 대한 식을 먼저 유도해 봄으로써 우리의 직관을 연습해 봅시다. 우리는 두 확률 변수 사이에 공유되는 정보를 찾고 싶습니다. 이를 시도해 볼 수 있는 한 가지 방법은 $X$와 $Y$가 함께 포함하는 모든 정보에서 시작하여 공유되지 않는 부분을 제거하는 것입니다. $X$와 $Y$가 함께 포함하는 정보는 $H(X, Y)$로 쓰입니다. 우리는 여기서 $X$에는 포함되어 있지만 $Y$에는 없는 정보와, $Y$에는 포함되어 있지만 $X$에는 없는 정보를 빼고 싶습니다. 이전 섹션에서 보았듯이, 이는 각각 $H(X \mid Y)$와 $H(Y \mid X)$에 의해 주어집니다. 따라서 우리는 상호 정보량이 다음과 같아야 한다고 봅니다.

$$ 
I(X, Y) = H(X, Y) - H(Y \mid X) - H(X \mid Y).
$$ 

실제로 이것은 상호 정보량에 대한 유효한 정의입니다. 이러한 용어들의 정의를 확장하고 결합하면, 약간의 대수를 통해 이것이 다음과 같음을 보일 수 있습니다.

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`


그림 :numref:`fig_mutual_information`에서 이러한 모든 관계를 요약할 수 있습니다. 다음 진술들이 왜 모두 $I(X, Y)$와 동등한지 확인하는 것은 직관에 대한 훌륭한 테스트입니다.

* $H(X) - H(X \mid Y)$
* $H(Y) - H(Y \mid X)$
* $H(X) + H(Y) - H(X, Y)$

![결합 엔트로피 및 조건부 엔트로피와 상호 정보량의 관계.](../img/mutual-information.svg)
:label:`fig_mutual_information`


 여러 면에서 우리는 상호 정보량 :eqref:`eq_mut_ent_def`을 :numref:`sec_random_variables`에서 보았던 상관 계수의 원칙적인 확장으로 생각할 수 있습니다. 이를 통해 변수 간의 선형 관계뿐만 아니라 임의의 유형의 두 확률 변수 사이에 공유되는 최대 정보를 물을 수 있습니다.

이제 상호 정보량을 밑바닥부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # 연산자 `nansum`은 nan이 아닌 숫자를 합산합니다
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### 상호 정보량의 속성 (Properties of Mutual Information)

상호 정보량 :eqref:`eq_mut_ent_def`의 정의를 암기하기보다는 그 주목할 만한 속성들을 염두에 두기만 하면 됩니다.

* 상호 정보량은 대칭적입니다. 즉, $I(X, Y) = I(Y, X)$입니다.
* 상호 정보량은 비음수입니다. 즉, $I(X, Y) \geq 0$입니다.
* $I(X, Y) = 0$인 것은 $X$와 $Y$가 독립적인 것과 동등합니다. 예를 들어, $X$와 $Y$가 독립적이라면 $Y$를 아는 것이 $X$에 대한 어떠한 정보도 주지 않으며 그 반대도 마찬가지이므로, 그들의 상호 정보량은 0입니다.
* 대안적으로, $X$가 $Y$의 가역 함수라면 $Y$와 $X$는 모든 정보를 공유하며 $$I(X, Y) = H(Y) = H(X)$$를 갖습니다.

### 점별 상호 정보량 (Pointwise Mutual Information)

이 장의 시작 부분에서 엔트로피를 다룰 때,우리는 $-\log(p_X(x))$를 특정 결과에 대해 우리가 얼마나 *놀랐는지*로 해석할 수 있었습니다. 상호 정보량의 로그 항에 대해서도 유사한 해석을 할 수 있으며, 이는 종종 *점별 상호 정보량(pointwise mutual information)*이라고 불립니다.

$$\textrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$ 
:eqlabel:`eq_pmi_def`

우리는 :eqref:`eq_pmi_def`를 독립적인 무작위 결과에 대해 기대하는 것과 비교하여 결과 $x$와 $y$의 특정 조합이 얼마나 더 또는 덜 발생할 가능성이 있는지 측정하는 것으로 생각할 수 있습니다. 그것이 크고 양수라면, 이러한 두 특정 결과는 무작위 기회에 비해 훨씬 더 자주 발생하며(*참고*: 분모는 두 결과가 독립적이었을 때의 확률인 $p_X(x) p_Y(y)$입니다), 반면에 크고 음수라면 두 결과가 무작위 기회에 의해 기대하는 것보다 훨씬 덜 발생함을 나타냅니다.

이를 통해 상호 정보량 :eqref:`eq_mut_ent_def`을 두 결과가 독립적이었을 경우와 비교하여 두 결과가 함께 발생하는 것을 보았을 때 우리가 놀란 평균량으로 해석할 수 있습니다.

### 상호 정보량의 응용 (Applications of Mutual Information)

상호 정보량은 그 순수한 정의에서 약간 추상적일 수 있는데, 머신러닝과는 어떤 관련이 있을까요? 자연어 처리에서 가장 어려운 문제 중 하나는 *모호성 해소(ambiguity resolution)*, 즉 문맥상 단어의 의미가 불분명한 문제입니다. 예를 들어, 최근 뉴스 헤드라인에 "Amazon is on fire"라는 보도가 있었습니다. 여러분은 아마존 회사의 건물에 불이 났는지, 아니면 아마존 열대 우림에 불이 났는지 궁금하실 수 있습니다.

이 경우 상호 정보량은 이 모호성을 해결하는 데 도움이 될 수 있습니다. 우리는 먼저 전자 상거래, 기술, 온라인과 같이 아마존 회사와 상대적으로 큰 상호 정보량을 가진 단어 그룹을 찾습니다. 둘째, 비, 숲, 열대와 같이 아마존 열대 우림과 상대적으로 큰 상호 정보량을 가진 또 다른 단어 그룹을 찾습니다. "Amazon"의 모호성을 해소해야 할 때, 우리는 Amazon이라는 단어의 문맥에서 어느 그룹이 더 많이 발생하는지 비교할 수 있습니다. 이 경우 기사는 숲을 계속 설명하여 문맥을 명확히 할 것입니다.


## 쿨백-라이블러 발산 (Kullback–Leibler Divergence)

:numref:`sec_linear-algebra`에서 논의한 바와 같이, 우리는 모든 차원의 공간에서 두 점 사이의 거리를 측정하기 위해 노름(norms)을 사용할 수 있습니다. 우리는 확률 분포에 대해서도 유사한 작업을 수행하고 싶습니다. 이를 수행하는 방법은 많지만 정보 이론은 가장 좋은 방법 중 하나를 제공합니다. 우리는 이제 두 분포가 서로 가까운지 여부를 측정하는 방법을 제공하는 *쿨백-라이블러(KL) 발산(Kullback–Leibler (KL) divergence)*을 탐구합니다.


### 정의 (Definition)

확률 분포 $P$(p.d.f. 또는 p.m.f. $p(x)$ 가짐)를 따르는 확률 변수 $X$가 주어졌을 때, 우리가 $P$를 다른 확률 분포 $Q$(p.d.f. 또는 p.m.f. $q(x)$ 가짐)로 추정한다고 합시다. 그러면 $P$와 $Q$ 사이의 *쿨백-라이블러(KL) 발산* (또는 *상대 엔트로피*)은 다음과 같습니다.

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$ 
:eqlabel:`eq_kl_def`

점별 상호 정보량 :eqref:`eq_pmi_def`과 마찬가지로, 우리는 로그 항에 대한 해석을 다시 제공할 수 있습니다: $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$는 $Q$에 대해 기대하는 것보다 $P$ 하에서 $x$를 훨씬 더 자주 볼 때 크고 양수가 되고, 결과가 예상보다 훨씬 적게 보일 때 크고 음수가 될 것입니다. 이런 식으로, 우리는 이를 기준 분포(reference distribution)로부터 결과를 관찰했을 때 얼마나 놀랐을지와 비교하여 결과를 관찰했을 때 느끼는 우리의 *상대적인* 놀라움으로 해석할 수 있습니다.

KL 발산을 밑바닥부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### KL 발산의 속성 (KL Divergence Properties)

KL 발산 :eqref:`eq_kl_def`의 몇 가지 속성을 살펴봅시다.

* KL 발산은 비대칭입니다. 즉, $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$인 $P, Q$가 존재합니다.
* KL 발산은 비음수입니다. 즉, $$D_{\textrm{KL}}(P\|Q) \geq 0.$$ 등호는 $P = Q$일 때만 성립함에 유의하십시오.
* $p(x) > 0$이지만 $q(x) = 0$인 $x$가 존재한다면, $D_{\textrm{KL}}(P\|Q) = \infty$입니다.
* KL 발산과 상호 정보량 사이에는 밀접한 관계가 있습니다. :numref:`fig_mutual_information`에 표시된 관계 외에도 $I(X, Y)$는 다음과 같은 용어들과 수치적으로 동등합니다.
    1. $D_{\textrm{KL}}(P(X, Y)  \ | \ P(X)P(Y))$;
    1. $E_Y \{ D_{\textrm{KL}}(P(X \mid Y) \ | \ P(X)) \}$;
    1. $E_X \{ D_{\textrm{KL}}(P(Y \mid X) \ | \ P(Y)) \}$.

  첫 번째 항의 경우, 우리는 상호 정보량을 $P(X, Y)$와 $P(X)$ 및 $P(Y)$의 곱 사이의 KL 발산으로 해석하며, 따라서 결합 분포가 독립적이었을 경우의 분포와 얼마나 다른지에 대한 척도입니다. 두 번째 항의 경우, 상호 정보량은 $X$ 분포의 값을 학습함으로써 얻어지는 $Y$에 대한 불확실성의 평균 감소량을 알려줍니다. 세 번째 항에 대해서도 마찬가지입니다.


### 예제 (Example)

비대칭성을 명시적으로 확인하기 위해 간단한 예제를 살펴보겠습니다.

먼저, 길이가 $10,000$인 세 개의 텐서를 생성하고 정렬합니다: 정규 분포 $N(0, 1)$을 따르는 목표 텐서 $p$, 그리고 각각 정규 분포 $N(-1, 1)$ 및 $N(1, 1)$을 따르는 두 개의 후보 텐서 $q_1$과 $q_2$입니다.

```{.python .input}
#@tab mxnet
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

$q_1$과 $q_2$는 y축(즉, $x=0$)에 대해 대칭이므로, $D_{\textrm{KL}}(p\|q_1)$과 $D_{\textrm{KL}}(p\|q_2)$ 사이의 KL 발산 값이 비슷할 것으로 기대합니다. 아래에서 볼 수 있듯이 $D_{\textrm{KL}}(p\|q_1)$과 $D_{\textrm{KL}}(p\|q_2)$ 사이에는 3% 미만의 차이만 있습니다.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

대조적으로, $D_{\textrm{KL}}(q_2 \|p)$와 $D_{\textrm{KL}}(p \| q_2)$는 아래에 표시된 것처럼 약 40%의 큰 차이가 있음을 알 수 있습니다.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## 크로스 엔트로피 (Cross-Entropy)

딥러닝에서의 정보 이론 응용이 궁금하시다면 여기 간단한 예가 있습니다. 우리는 확률 분포 $p(x)$를 가진 실제 분포 $P$와, 확률 분포 $q(x)$를 가진 추정 분포 $Q$를 정의하고, 이 섹션의 나머지 부분에서 이를 사용할 것입니다.

주어진 $n$개의 데이터 예제 {$x_1, \ldots, x_n$}를 기반으로 이진 분류 문제를 풀어야 한다고 합시다. $1$과 $0$을 각각 양성 및 음성 클래스 레이블 $y_i$로 인코딩하고, 우리의 신경망이 $\theta$로 파라미터화되었다고 가정합시다. $\hat{y}_i= p_{\theta}(y_i \mid x_i)$가 되도록 하는 최상의 $\theta$를 찾는 것이 목표라면, :numref:`sec_softmax`에서 보았던 것처럼 최대 로그 우도 접근 방식을 적용하는 것이 자연스럽습니다. 구체적으로, 실제 레이블 $y_i$와 예측 $\hat{y}_i= p_{\theta}(y_i \mid x_i)$에 대해 양성으로 분류될 확률은 $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$입니다. 따라서 로그 우도 함수는 다음과 같습니다.

$$ 
\begin{aligned}
l(\theta) &= \log L(\theta) \\ 
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\ 
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$ 

로그 우도 함수 $l(\theta)$를 최대화하는 것은 $- l(\theta)$를 최소화하는 것과 동일하며, 따라서 여기서 최상의 $\theta$를 찾을 수 있습니다. 위의 손실을 임의의 분포로 일반화하기 위해, $-l(\theta)$를 *크로스 엔트로피 손실(cross-entropy loss)* $\textrm{CE}(y, \hat{y})$라고도 불렀습니다. 여기서 $y$는 실제 분포 $P$를 따르고 $\hat{y}$는 추정 분포 $Q$를 따릅니다.

이 모든 것은 최대 우도 관점에서 작업하여 도출되었습니다. 그러나 자세히 살펴보면 $\log(\pi_i)$와 같은 항들이 우리 계산에 들어왔음을 알 수 있으며, 이는 우리가 정보 이론적 관점에서 이 식을 이해할 수 있다는 강력한 증거입니다.


### 공식 정의 (Formal Definition)

KL 발산과 마찬가지로, 확률 변수 $X$에 대해 추정 분포 $Q$와 실제 분포 $P$ 사이의 발산을 *크로스 엔트로피(cross-entropy)*를 통해 측정할 수도 있습니다.

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$ 
:eqlabel:`eq_ce_def`

위에서 논의한 엔트로피의 속성을 사용하면, 이를 엔트로피 $H(P)$와 $P$ 및 $Q$ 사이의 KL 발산의 합으로 해석할 수도 있습니다. 즉,

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$ 


아래와 같이 크로스 엔트로피 손실을 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # `tf.gather_nd`는 텐서의 특정 인덱스를 선택하는 데 사용됩니다.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

이제 레이블과 예측을 위한 두 개의 텐서를 정의하고 그들의 크로스 엔트로피 손실을 계산해 봅시다.

```{.python .input}
#@tab mxnet
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### 속성 (Properties)

이 섹션의 시작 부분에서 암시했듯이, 크로스 엔트로피 :eqref:`eq_ce_def`는 최적화 문제에서 손실 함수를 정의하는 데 사용될 수 있습니다. 다음은 동등함이 밝혀졌습니다:

1. 분포 $P$에 대한 $Q$의 예측 확률 최대화 (즉, $E_{x \sim P} [\log (q(x))]$)
2. 크로스 엔트로피 $\textrm{CE} (P, Q)$ 최소화
3. KL 발산 $D_{\textrm{KL}}(P\|Q)$ 최소화

크로스 엔트로피의 정의는 실제 데이터의 엔트로피 $H(P)$가 상수인 한 목적 2와 목적 3 사이의 동등한 관계를 간접적으로 증명합니다.


### 다중 클래스 분류의 목적 함수로서의 크로스 엔트로피 (Cross-Entropy as An Objective Function of Multi-class Classification)

크로스 엔트로피 손실 $\textrm{CE}$를 사용한 분류 목적 함수를 깊이 파고들면, $\textrm{CE}$를 최소화하는 것이 로그 우도 함수 $L$을 최대화하는 것과 동등함을 알게 될 것입니다.

우선, $n$개의 예제가 있는 데이터셋이 주어지고 그것이 $k$-클래스로 분류될 수 있다고 가정합시다. 각 데이터 예제 $i$에 대해, 우리는 임의의 $k$-클래스 레이블 $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$를 *원-핫 인코딩(one-hot encoding)*으로 나타냅니다. 구체적으로, 예제 $i$가 클래스 $j$에 속하면 $j$번째 항목을 $1$로 설정하고 다른 모든 구성 요소를 $0$으로 설정합니다. 즉,

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\textrm{그렇지 않으면.}\\end{cases}\\
$$ 

예를 들어, 다중 클래스 분류 문제에 세 개의 클래스 $A, B, C$가 포함되어 있다면, 레이블 $\mathbf{y}_i$는 {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}로 인코딩될 수 있습니다.


우리 신경망이 $\theta$로 파라미터화되었다고 가정합시다. 실제 레이블 벡터 $\mathbf{y}_i$와 예측 $$\\hat{\\mathbf{y}}_i= p_{\\theta}(\\mathbf{y}_i \mid \\mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\\theta} (y_{ij} \mid \\mathbf{x}_i)$$에 대해,

*크로스 엔트로피 손실*은 다음과 같습니다.

$$ 
\\textrm{CE}(\\mathbf{y}, \\hat{\\mathbf{y}}) = - \sum_{i=1}^n \\mathbf{y}_i \log \\hat{\\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\\theta} (y_{ij} \mid \\mathbf{x}_i)}.\\
$$ 

다른 한편으로, 우리는 최대 우도 추정을 통해서도 문제에 접근할 수 있습니다. 우선, $k$-클래스 멀티누이 분포(multinoulli distribution)를 빠르게 소개합시다. 이것은 이진 클래스에서 다중 클래스로 확장된 베르누이 분포입니다. 확률 변수 $\mathbf{z} = (z_{1}, \ldots, z_{k})$가 확률 $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$)를 가진 $k$-클래스 *멀티누이 분포*를 따른다면, 즉 $$p(\\mathbf{z}) = p(z_1, \ldots, z_k) = \textrm{Multi} (p_1, \ldots, p_k) \textrm{이며, 여기서 } \sum_{i=1}^k p_i = 1$$이라면, $\mathbf{z}$의 결합 확률 질량 함수(p.m.f.)는 다음과 같습니다.
$$\\mathbf{p}^\\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$ 


각 데이터 예제의 레이블 $\mathbf{y}_i$가 확률 $\\boldsymbol{\\pi} =$ ($\\pi_{1}, \ldots, \\pi_{k}$)를 가진 $k$-클래스 멀티누이 분포를 따르고 있음을 알 수 있습니다. 따라서 각 데이터 예제 $\mathbf{y}_i$의 결합 p.m.f.는 $\\mathbf{\\pi}^{\\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}$입니다. 
따라서 로그 우도 함수는 다음과 같습니다.

$$ 
\\begin{aligned}
l(\\theta)
 = \log L(\\theta)
 = \log \prod_{i=1}^n \\boldsymbol{\\pi}^{\\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\\pi_{j}}.\\
\\end{aligned}
$$ 

최대 우도 추정에서는 $\\pi_{j} = p_{\\theta} (y_{ij} \mid \\mathbf{x}_i)$를 가짐으로써 목적 함수 $l(\theta)$를 최대화하기 때문입니다. 따라서 임의의 다중 클래스 분류에 대해, 위의 로그 우도 함수 $l(\theta)$를 최대화하는 것은 CE 손실 $\\textrm{CE}(y, \\hat{y})$를 최소화하는 것과 동등합니다.


위의 증명을 테스트하기 위해 내장 측정 척도인 `NegativeLogLikelihood`를 적용해 봅시다. 이전 예제와 동일한 `labels` 및 `preds`를 사용하면 소수점 5자리까지 이전 예제와 동일한 수치 손실을 얻을 것입니다.

```{.python .input}
#@tab mxnet
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# PyTorch의 크로스 엔트로피 손실 구현은 `nn.LogSoftmax()`와 `nn.NLLLoss()`를 결합합니다.
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # 레이블을 원-핫 벡터로 변환.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # 정의로부터 음의 로그 우도를 계산하지 않을 것입니다.
    # 대신 순환 논증을 따를 것입니다. NLL은 `cross_entropy`와 같으므로,
    # 만약 cross_entropy를 계산하면 그것이 우리에게 NLL을 줄 것입니다.
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## 요약 (Summary)

* 정보 이론은 정보를 인코딩, 디코딩, 전송 및 조작하는 것에 대한 연구 분야입니다.
* 엔트로피는 서로 다른 신호에 얼마나 많은 정보가 존재하는지 측정하는 단위입니다.
* KL 발산은 두 분포 사이의 발산을 측정할 수도 있습니다.
* 크로스 엔트로피는 다중 클래스 분류의 목적 함수로 간주될 수 있습니다. 크로스 엔트로피 손실을 최소화하는 것은 로그 우도 함수를 최대화하는 것과 동등합니다.


## 연습 문제 (Exercises)

1. 첫 번째 섹션의 카드 예제들이 정말로 주장된 엔트로피를 갖는지 확인하십시오.
2. 모든 분포 $p$와 $q$에 대해 KL 발산 $D(p\|q)$가 비음수임을 보이십시오. 힌트: 젠센 부등식을 사용하십시오. 즉, $-\log x$가 볼록 함수라는 사실을 사용하십시오.
3. 몇 가지 데이터 소스로부터 엔트로피를 계산해 봅시다:
    * 타자기를 치는 원숭이가 생성한 출력을 보고 있다고 가정해 봅시다. 원숭이는 타자기의 44개 키 중 하나를 무작위로 누릅니다(아직 특수 키나 쉬프트 키를 발견하지 못했다고 가정할 수 있습니다). 문자당 몇 비트의 무작위성을 관찰합니까?
    * 원숭이가 마음에 들지 않아 술 취한 식자공(typesetter)으로 교체했습니다. 그는 일관성은 없지만 단어를 생성할 수 있습니다. 대신 그는 2,000단어의 어휘 중에서 무작위로 단어를 선택합니다. 영어 단어의 평균 길이가 4.5자라고 가정해 봅시다. 이제 문자당 몇 비트의 무작위성을 관찰합니까?
    * 여전히 결과가 마음에 들지 않아 식자공을 고품질 언어 모델로 교체했습니다. 언어 모델은 현재 단어당 15포인트까지 낮은 퍼플렉서티를 얻을 수 있습니다. 언어 모델의 문자 *퍼플렉서티(perplexity)*는 확률 집합의 기하 평균의 역수로 정의되며, 각 확률은 단어의 문자에 대응합니다. 구체적으로 주어진 단어의 길이가 $l$이라면, $\textrm{PPL}(\textrm{word}) = \left[\prod_i p(\textrm{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\textrm{character}_i)} \right]$입니다. 테스트 단어의 길이가 4.5자라고 가정할 때, 이제 문자당 몇 비트의 무작위성을 관찰합니까?
4. $I(X, Y) = H(X) - H(X \mid Y)$인 이유를 직관적으로 설명하십시오. 그런 다음 양변을 결합 분포에 대한 기대값으로 표현하여 이것이 참임을 보이십시오.
5. 두 가우스 분포 $\mathcal{N}(\\mu_1, \\sigma_1^2)$과 $\\mathcal{N}(\\mu_2, \\sigma_2^2)$ 사이의 KL 발산은 얼마입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1105)
:end_tab: