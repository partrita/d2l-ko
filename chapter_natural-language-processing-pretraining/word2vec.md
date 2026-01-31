# 단어 임베딩 (word2vec)
:label:`sec_word2vec`


자연어는 의미를 표현하기 위해 사용되는 복잡한 시스템입니다. 이 시스템에서 단어는 의미의 기본 단위입니다. 이름에서 알 수 있듯이, *단어 벡터(word vectors)*는 단어를 나타내기 위해 사용되는 벡터이며, 단어의 특성 벡터 또는 표현으로 간주될 수도 있습니다. 단어를 실제 벡터에 매핑하는 기술을 *단어 임베딩(word embedding)*이라고 합니다. 최근 몇 년 동안 단어 임베딩은 점차 자연어 처리의 기본 지식이 되었습니다.


## 원-핫 벡터는 나쁜 선택입니다

우리는 :numref:`sec_rnn-scratch`에서 단어(문자가 단어임)를 나타내기 위해 원-핫 벡터를 사용했습니다. 사전에 있는 서로 다른 단어의 수(사전 크기)를 $N$이라고 하고, 각 단어가 $0$에서 $N-1$까지의 서로 다른 정수(인덱스)에 대응한다고 가정해 봅시다. 인덱스 $i$를 가진 임의의 단어에 대한 원-핫 벡터 표현을 얻기 위해, 우리는 모든 값이 0인 길이 $N$의 벡터를 생성하고 $i$번째 위치의 요소를 1로 설정합니다. 이런 식으로 각 단어는 길이 $N$의 벡터로 표현되며 신경망에서 직접 사용될 수 있습니다.


원-핫 단어 벡터는 구성하기 쉽지만, 일반적으로 좋은 선택은 아닙니다. 주된 이유는 원-핫 단어 벡터가 우리가 자주 사용하는 *코사인 유사도(cosine similarity)*와 같이 서로 다른 단어 간의 유사성을 정확하게 표현할 수 없기 때문입니다. 벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$에 대해, 그들의 코사인 유사도는 두 벡터 사이 각도의 코사인 값입니다:


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$ 

임의의 서로 다른 두 단어의 원-핫 벡터 간의 코사인 유사도는 0이므로, 원-핫 벡터는 단어 간의 유사성을 인코딩할 수 없습니다.


## 자기 지도 학습 word2vec

위의 문제를 해결하기 위해 [word2vec](https://code.google.com/archive/p/word2vec/) 도구가 제안되었습니다. 이는 각 단어를 고정 길이 벡터로 매핑하며, 이 벡터들은 서로 다른 단어 간의 유사성 및 유추 관계를 더 잘 표현할 수 있습니다. word2vec 도구에는 *스킵-그램(skip-gram)* :cite:`Mikolov.Sutskever.Chen.ea.2013`과 *CBOW(continuous bag of words)* :cite:`Mikolov.Chen.Corrado.ea.2013`이라는 두 가지 모델이 포함되어 있습니다. 의미론적으로 의미 있는 표현을 위해, 그들의 훈련은 코퍼스에서 주변 단어의 일부를 사용하여 일부 단어를 예측하는 것으로 볼 수 있는 조건부 확률에 의존합니다. 감독(supervision)이 레이블 없는 데이터에서 오기 때문에 스킵-그램과 CBOW는 모두 자기 지도(self-supervised) 모델입니다.

다음에서는 이 두 모델과 그 훈련 방법을 소개합니다.


## 스킵-그램(Skip-Gram) 모델
:label:`subsec_skip-gram`

*스킵-그램(skip-gram)* 모델은 텍스트 시퀀스에서 한 단어가 주변 단어들을 생성하는 데 사용될 수 있다고 가정합니다. "the", "man", "loves", "his", "son"이라는 텍스트 시퀀스를 예로 들어 보겠습니다. "loves"를 *중심 단어(center word)*로 선택하고 문맥 윈도우 크기를 2로 설정합시다. :numref:`fig_skip_gram`에 표시된 것처럼, 중심 단어 "loves"가 주어졌을 때 스킵-그램 모델은 중심 단어에서 2단어 이내에 있는 *문맥 단어(context words)*인 "the", "man", "his", "son"을 생성할 조건부 확률을 고려합니다:

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$ 

중심 단어가 주어졌을 때 문맥 단어들이 독립적으로 생성된다고 가정합니다(즉, 조건부 독립). 이 경우 위의 조건부 확률은 다음과 같이 다시 쓸 수 있습니다.

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$ 

![스킵-그램 모델은 중심 단어가 주어졌을 때 주변 문맥 단어를 생성할 조건부 확률을 고려합니다.](../img/skip-gram.svg)
:label:`fig_skip_gram`

스킵-그램 모델에서 각 단어는 조건부 확률을 계산하기 위해 두 개의 $d$차원 벡터 표현을 갖습니다. 더 구체적으로, 사전에 있는 인덱스 $i$를 가진 임의의 단어에 대해, 각각 *중심* 단어와 *문맥* 단어로 사용될 때의 두 벡터를 $\mathbf{v}_i\in\mathbb{R}^d$와 $\mathbf{u}_i\in\mathbb{R}^d$라고 표시합시다. 중심 단어 $w_c$(사전 인덱스 $c$)가 주어졌을 때 임의의 문맥 단어 $w_o$(사전 인덱스 $o$)를 생성할 조건부 확률은 벡터 내적에 대한 소프트맥스 연산으로 모델링될 수 있습니다:


$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)},$$ 
:eqlabel:`eq_skip-gram-softmax`

여기서 어휘 인덱스 집합 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$입니다. 길이 $T$인 텍스트 시퀀스가 주어지고 타임 스텝 $t$에서의 단어를 $w^{(t)}$라고 할 때, 임의의 중심 단어가 주어졌을 때 문맥 단어들이 독립적으로 생성된다고 가정합니다. 문맥 윈도우 크기가 $m$일 때, 스킵-그램 모델의 우도 함수(likelihood function)는 임의의 중심 단어가 주어졌을 때 모든 문맥 단어를 생성할 확률입니다:


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$ 

여기서 $1$보다 작거나 $T$보다 큰 타임 스텝은 생략될 수 있습니다.

### 훈련

스킵-그램 모델 파라미터는 어휘의 각 단어에 대한 중심 단어 벡터와 문맥 단어 벡터입니다. 훈련 시에는 우도 함수를 최대화(즉, 최대 우도 추정)하여 모델 파라미터를 학습합니다. 이는 다음 손실 함수를 최소화하는 것과 같습니다:

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \textrm{log}\, P(w^{(t+j)} \mid w^{(t)}).$$ 

손실을 최소화하기 위해 확률적 경사 하강법을 사용할 때, 각 반복에서 무작위로 더 짧은 하위 시퀀스를 샘플링하여 이 하위 시퀀스에 대한 (확률적) 기울기를 계산하여 모델 파라미터를 업데이트할 수 있습니다. 이 (확률적) 기울기를 계산하려면 중심 단어 벡터와 문맥 단어 벡터에 대한 로그 조건부 확률의 기울기를 얻어야 합니다. 일반적으로 :eqref:`eq_skip-gram-softmax`에 따라 임의의 중심 단어 $w_c$와 문맥 단어 $w_o$ 쌍을 포함하는 로그 조건부 확률은 다음과 같습니다.


$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$ 
:eqlabel:`eq_skip-gram-log`

미분을 통해 중심 단어 벡터 $\mathbf{v}_c$에 대한 기울기를 다음과 같이 얻을 수 있습니다.

$$\begin{aligned}\frac{\partial \textrm{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$ 
:eqlabel:`eq_skip-gram-grad`


:eqref:`eq_skip-gram-grad`의 계산에는 $w_c$를 중심 단어로 하는 사전의 모든 단어에 대한 조건부 확률이 필요하다는 점에 유의하십시오. 다른 단어 벡터에 대한 기울기도 같은 방식으로 얻을 수 있습니다.


훈련 후, 사전의 인덱스 $i$를 가진 임의의 단어에 대해 두 단어 벡터 $\,\mathbf{v}_i$(중심 단어로서)와 $\,\mathbf{u}_i$(문맥 단어로서)를 모두 얻습니다. 자연어 처리 응용 프로그램에서 스킵-그램 모델의 중심 단어 벡터는 일반적으로 단어 표현으로 사용됩니다.


## CBOW(Continuous Bag of Words) 모델


*CBOW(continuous bag of words)* 모델은 스킵-그램 모델과 유사합니다. 스킵-그램 모델과의 주요 차이점은 CBOW 모델은 텍스트 시퀀스에서 주변 문맥 단어들을 기반으로 중심 단어가 생성된다고 가정한다는 것입니다. 예를 들어 동일한 텍스트 시퀀스 "the", "man", "loves", "his", "son"에서 "loves"를 중심 단어로 하고 문맥 윈도우 크기를 2로 할 때, CBOW 모델은 문맥 단어 "the", "man", "his", "son"을 기반으로 중심 단어 "loves"를 생성할 조건부 확률을 고려합니다(:numref:`fig_cbow` 참조):

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$ 

![CBOW 모델은 주변 문맥 단어가 주어졌을 때 중심 단어를 생성할 조건부 확률을 고려합니다.](../img/cbow.svg)
:label:`fig_cbow`


CBOW 모델에는 여러 문맥 단어가 있으므로, 조건부 확률 계산 시 이러한 문맥 단어 벡터들의 평균을 냅니다. 구체적으로, 사전에 있는 인덱스 $i$를 가진 임의의 단어에 대해, 각각 *문맥* 단어와 *중심* 단어로 사용될 때의 두 벡터를 $\,\mathbf{v}_i\in\mathbb{R}^d$와 $\,\mathbf{u}_i\in\mathbb{R}^d$라고 표시합시다(스킵-그램 모델과는 의미가 반대임). 주변 문맥 단어 $w_{o_1}, \ldots, w_{o_{2m}}$(사전 인덱스 $o_1, \ldots, o_{2m}$)이 주어졌을 때 임의의 중심 단어 $w_c$(사전 인덱스 $c$)를 생성할 조건부 확률은 다음과 같이 모델링될 수 있습니다.


$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \exp\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}.$$ 
:eqlabel:`fig_cbow-full`


간결함을 위해 $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\ \}$ 및 $\,\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}} \right)/(2m)$라고 합시다. 그러면 :eqref:`fig_cbow-full`은 다음과 같이 단순화될 수 있습니다.

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$ 

길이 $T$인 텍스트 시퀀스가 주어지고 타임 스텝 $t$에서의 단어를 $w^{(t)}$라고 할 때, 문맥 윈도우 크기가 $m$인 경우 CBOW 모델의 우도 함수는 문맥 단어들이 주어졌을 때 모든 중심 단어를 생성할 확률입니다:


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$ 

### 훈련

CBOW 모델을 훈련하는 것은 스킵-그램 모델을 훈련하는 것과 거의 동일합니다. CBOW 모델의 최대 우도 추정은 다음 손실 함수를 최소화하는 것과 같습니다:



$$  -\sum_{t=1}^T  \textrm{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$ 

다음을 유의하십시오.

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$ 

미분을 통해 임의의 문맥 단어 벡터 $\,\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$)에 대한 기울기를 다음과 같이 얻을 수 있습니다.


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$ 
:eqlabel:`eq_cbow-gradient`


다른 단어 벡터에 대한 기울기도 같은 방식으로 얻을 수 있습니다. 스킵-그램 모델과 달리 CBOW 모델은 일반적으로 문맥 단어 벡터를 단어 표현으로 사용합니다.



## 요약 (Summary)

* 단어 벡터는 단어를 나타내기 위해 사용되는 벡터이며, 단어의 특성 벡터 또는 표현으로 간주될 수도 있습니다. 단어를 실제 벡터에 매핑하는 기술을 단어 임베딩이라고 합니다.
* word2vec 도구에는 스킵-그램 모델과 CBOW 모델이 모두 포함되어 있습니다.
* 스킵-그램 모델은 텍스트 시퀀스에서 한 단어가 주변 단어들을 생성하는 데 사용될 수 있다고 가정합니다. 반면 CBOW 모델은 주변 문맥 단어들을 기반으로 중심 단어가 생성된다고 가정합니다.



## 연습 문제 (Exercises)

1. 각 기울기를 계산하기 위한 계산 복잡도는 얼마입니까? 사전 크기가 매우 클 때 어떤 문제가 발생할 수 있습니까?
2. 영어의 일부 고정된 구문은 "new york"과 같이 여러 단어로 구성됩니다. 그들의 단어 벡터를 어떻게 훈련합니까? 힌트: word2vec 논문 :cite:`Mikolov.Sutskever.Chen.ea.2013`의 섹션 4를 참조하십시오.
3. 스킵-그램 모델을 예로 들어 word2vec 설계를 되짚어 봅시다. 스킵-그램 모델에서 두 단어 벡터의 내적과 코사인 유사도 사이의 관계는 무엇입니까? 의미론적으로 유사한 단어 쌍에 대해, 왜 그들의 단어 벡터(스킵-그램 모델로 훈련됨)의 코사인 유사도가 높을 수 있습니까?

[토론](https://discuss.d2l.ai/t/381)
