# GloVe를 이용한 단어 임베딩 (Word Embedding with Global Vectors (GloVe))
:label:`sec_glove`


문맥 윈도우 내의 단어-단어 공생(co-occurrences)은 풍부한 의미론적 정보를 담고 있을 수 있습니다. 예를 들어 대규모 코퍼스에서 "solid"라는 단어는 "steam"보다 "ice"와 함께 나타날 가능성이 높지만, "gas"라는 단어는 아마도 "ice"보다 "steam"과 더 자주 함께 나타날 것입니다. 게다가 이러한 공생에 대한 글로벌 코퍼스 통계는 미리 계산될 수 있으며, 이는 더 효율적인 훈련으로 이어질 수 있습니다. 단어 임베딩을 위해 전체 코퍼스의 통계 정보를 활용하기 위해, 먼저 :numref:`subsec_skip-gram`의 스킵-그램 모델을 공생 횟수와 같은 글로벌 코퍼스 통계를 사용하여 해석하는 것부터 다시 살펴보겠습니다.

## 글로벌 코퍼스 통계를 이용한 스킵-그램 (Skip-Gram with Global Corpus Statistics)
:label:`subsec_skipgram-global`

스킵-그램 모델에서 단어 $w_i$가 주어졌을 때 단어 $w_j$의 조건부 확률 $P(w_j
mid w_i)$를 $q_{ij}$라고 표시하면 다음과 같습니다.

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

여기서 임의의 인덱스 $i$에 대해 벡터 $\mathbf{v}_i$와 $\mathbf{u}_i$는 각각 단어 $w_i$를 중심 단어와 문맥 단어로 나타내며, $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$은 어휘의 인덱스 집합입니다.

코퍼스에서 여러 번 나타날 수 있는 단어 $w_i$를 고려해 봅시다. 전체 코퍼스에서 $w_i$가 중심 단어로 사용된 모든 문맥 윈도우에서 나타나는 문맥 단어들은 *동일한 요소의 여러 인스턴스를 허용하는* 단어 인덱스의 *멀티집합(multiset)* $\mathcal{C}_i$를 형성합니다. 임의의 요소에 대해 그 인스턴스 수를 *중복도(multiplicity)*라고 합니다. 예를 들어 설명하자면, 단어 $w_i$가 코퍼스에서 두 번 나타나고 두 문맥 윈도우에서 $w_i$를 중심 단어로 취하는 문맥 단어의 인덱스가 $k, j, m, k$ 및 $k, l, k, j$라고 가정해 봅시다. 그러면 멀티집합 $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$이며, 요소 $j, k, l, m$의 중복도는 각각 2, 4, 1, 1입니다.

이제 멀티집합 $\mathcal{C}_i$에서 요소 $j$의 중복도를 $x_{ij}$라고 표시합시다. 이는 전체 코퍼스에서 동일한 문맥 윈도우에 있는 단어 $w_j$(문맥 단어로)와 단어 $w_i$(중심 단어로)의 글로벌 공생 횟수입니다. 이러한 글로벌 코퍼스 통계를 사용하면 스킵-그램 모델의 손실 함수는 다음과 동등합니다.

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$ :eqlabel:`eq_skipgram-x_ij`

우리는 또한 $w_i$가 중심 단어로 나타나는 문맥 윈도우 내의 모든 문맥 단어의 수를 $x_i$로 표시하며, 이는 $|\mathcal{C}_i|$와 같습니다. 중심 단어 $w_i$가 주어졌을 때 문맥 단어 $w_j$를 생성할 조건부 확률 $x_{ij}/x_i$를 $p_{ij}$라고 하면, :eqref:`eq_skipgram-x_ij`는 다음과 같이 다시 쓸 수 있습니다.

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$ :eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij`에서 $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$는 글로벌 코퍼스 통계의 조건부 분포 $p_{ij}$와 모델 예측의 조건부 분포 $q_{ij}$ 사이의 크로스 엔트로피를 계산합니다. 이 손실은 위에서 설명한 대로 $x_i$에 의해 가중치가 부여됩니다. :eqref:`eq_skipgram-p_ij`의 손실 함수를 최소화하면 예측된 조건부 분포가 글로벌 코퍼스 통계의 조건부 분포에 가까워질 수 있습니다.


확률 분포 간의 거리를 측정하는 데 흔히 사용되지만, 크로스 엔트로피 손실 함수는 여기서 좋은 선택이 아닐 수 있습니다. 한편으로 :numref:`sec_approx_train`에서 언급했듯이, $q_{ij}$를 적절하게 정규화하는 비용은 전체 어휘에 대한 합산으로 이어져 계산 비용이 많이 들 수 있습니다. 다른 한편으로 대규모 코퍼스에서 발생하는 수많은 드문 사건들은 크로스 엔트로피 손실에 의해 종종 너무 많은 가중치가 할당되도록 모델링됩니다.

## GloVe 모델 (The GloVe Model)

이러한 점을 고려하여 *GloVe* 모델은 제곱 손실을 기반으로 스킵-그램 모델에 세 가지 변경을 가했습니다 :cite:`Pennington.Socher.Manning.2014`:

1. 확률 분포가 아닌 변수 $p'_{ij}=x_{ij}$와 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$를 사용하고 두 변수 모두에 로그를 취하므로, 제곱 손실 항은 $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$이 됩니다.
2. 각 단어 $w_i$에 대해 두 개의 스칼라 모델 파라미터인 중심 단어 편향 $b_i$와 문맥 단어 편향 $c_i$를 추가합니다.
3. 각 손실 항의 가중치를 가중치 함수 $h(x_{ij})$로 대체합니다. 여기서 $h(x)$는 구간 $[0, 1]$에서 증가하는 함수입니다.

모든 것을 종합하면, GloVe를 훈련하는 것은 다음 손실 함수를 최소화하는 것입니다:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$ :eqlabel:`eq_glove-loss`

가중치 함수에 대한 권장 선택은 다음과 같습니다: $x < c$ (예: $c = 100$)이면 $h(x) = (x/c) ^\alpha$ (예: $\alpha = 0.75$)이고, 그렇지 않으면 $h(x) = 1$입니다. 이 경우 $h(0)=0$이므로 $x_{ij}=0$인 임의의 제곱 손실 항은 계산 효율성을 위해 생략될 수 있습니다. 예를 들어 훈련을 위해 미니배치 확률적 경사 하강법을 사용할 때, 각 반복에서 *0이 아닌* $x_{ij}$의 미니배치를 무작위로 샘플링하여 기울기를 계산하고 모델 파라미터를 업데이트합니다. 이러한 0이 아닌 $x_{ij}$는 미리 계산된 글로벌 코퍼스 통계이므로, 이 모델을 글로벌 벡터를 의미하는 GloVe(Global Vectors)라고 부릅니다.

단어 $w_i$가 단어 $w_j$의 문맥 윈도우에 나타나면 그 반대도 마찬가지라는 점을 강조해야 합니다. 따라서 $x_{ij}=x_{ji}$입니다. 비대칭 조건부 확률 $p_{ij}$를 피팅하는 word2vec과 달리 GloVe는 대칭인 $\log \, x_{ij}$를 피팅합니다. 따라서 GloVe 모델에서 임의의 단어의 중심 단어 벡터와 문맥 단어 벡터는 수학적으로 동등합니다. 그러나 실제로는 초기화 값이 다르기 때문에 훈련 후에 동일한 단어라도 이 두 벡터에서 서로 다른 값을 가질 수 있습니다. GloVe는 이들을 더하여 출력 벡터로 사용합니다.



## 공생 확률의 비율을 통한 GloVe 해석 (Interpreting GloVe from the Ratio of Co-occurrence Probabilities)


우리는 또한 다른 관점에서 GloVe 모델을 해석할 수 있습니다. :numref:`subsec_skipgram-global`과 동일한 표기법을 사용하여, $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$를 코퍼스에서 중심 단어 $w_i$가 주어졌을 때 문맥 단어 $w_j$를 생성할 조건부 확률이라고 합시다. :numref:`tab_glove`는 대규모 코퍼스의 통계를 기반으로 단어 "ice"와 "steam"이 주어졌을 때의 몇 가지 공생 확률과 그 비율을 보여줍니다.


:대규모 코퍼스의 단어-단어 공생 확률 및 그 비율 (:citet:`Pennington.Socher.Manning.2014`의 표 1 수정)
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k
mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k
mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



:numref:`tab_glove`에서 다음과 같은 사실을 관찰할 수 있습니다:

* "ice"와 관련이 있지만 "steam"과는 관련이 없는 단어 $w_k$ (예: $w_k=\textrm{solid}$)의 경우, 8.9와 같이 더 큰 공생 확률 비율을 기대합니다.
* "steam"과 관련이 있지만 "ice"와는 관련이 없는 단어 $w_k$ (예: $w_k=\textrm{gas}$)의 경우, 0.085와 같이 더 작은 공생 확률 비율을 기대합니다.
* "ice"와 "steam" 모두와 관련이 있는 단어 $w_k$ (예: $w_k=\textrm{water}$)의 경우, 1.36과 같이 1에 가까운 공생 확률 비율을 기대합니다.
* "ice"와 "steam" 모두와 관련이 없는 단어 $w_k$ (예: $w_k=\textrm{fashion}$)의 경우, 0.96과 같이 1에 가까운 공생 확률 비율을 기대합니다.




공생 확률의 비율이 단어 간의 관계를 직관적으로 표현할 수 있음을 알 수 있습니다. 따라서 우리는 이 비율을 피팅하기 위해 세 단어 벡터의 함수를 설계할 수 있습니다. $w_i$가 중심 단어이고 $w_j$와 $w_k$가 문맥 단어일 때의 공생 확률 비율 ${p_{ij}}/{p_{ik}}$에 대해, 어떤 함수 $f$를 사용하여 이 비율을 피팅하고자 합니다:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$ :eqlabel:`eq_glove-f`

$f$에 대한 많은 가능한 설계 중에서 여기서는 합리적인 선택 하나만 고릅니다. 공생 확률의 비율이 스칼라이므로 $f$가 $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$와 같은 스칼라 함수여야 함을 요구합니다. :eqref:`eq_glove-f`에서 단어 인덱스 $j$와 $k$를 바꾸면 $f(x)f(-x)=1$이 성립해야 하므로, 한 가지 가능성은 $f(x)=\exp(x)$입니다. 즉,

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$ 

이제 $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$라고 합시다. 여기서 $\alpha$는 상수입니다. $p_{ij}=x_{ij}/x_i$이므로 양변에 로그를 취하면 $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$를 얻습니다. 우리는 중심 단어 편향 $b_i$와 문맥 단어 편향 $c_j$와 같은 추가적인 편향 항을 사용하여 $- \log\, \alpha + \log\, x_i$를 피팅할 수 있습니다:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$ :eqlabel:`eq_glove-square`

가중치를 부여하여 :eqref:`eq_glove-square`의 제곱 오차를 측정하면 :eqref:`eq_glove-loss`의 GloVe 손실 함수를 얻게 됩니다.



## 요약 (Summary)

* 스킵-그램 모델은 단어-단어 공생 횟수와 같은 글로벌 코퍼스 통계를 사용하여 해석될 수 있습니다.
* 크로스 엔트로피 손실은 두 확률 분포 사이의 차이를 측정하는 데, 특히 대규모 코퍼스에 대해서는 좋은 선택이 아닐 수 있습니다. GloVe는 미리 계산된 글로벌 코퍼스 통계를 피팅하기 위해 제곱 손실을 사용합니다.
* GloVe에서 임의의 단어의 중심 단어 벡터와 문맥 단어 벡터는 수학적으로 동등합니다.
* GloVe는 단어-단어 공생 확률의 비율로부터 해석될 수 있습니다.


## 연습 문제 (Exercises)

1. 단어 $w_i$와 $w_j$가 동일한 문맥 윈도우에서 공생하는 경우, 텍스트 시퀀스에서의 그들 사이의 거리를 사용하여 조건부 확률 $p_{ij}$를 계산하는 방법을 어떻게 다시 설계할 수 있을까요? 힌트: GloVe 논문 :cite:`Pennington.Socher.Manning.2014`의 섹션 4.2를 참조하십시오.
2. 임의의 단어에 대해 GloVe에서 중심 단어 편향과 문맥 단어 편향이 수학적으로 동등합니까? 그 이유는 무엇입니까?


[토론](https://discuss.d2l.ai/t/385)