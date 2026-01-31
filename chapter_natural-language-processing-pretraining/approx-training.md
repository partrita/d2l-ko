# 근사 훈련 (Approximate Training)
:label:`sec_approx_train`

:numref:`sec_word2vec`에서의 논의를 상기해 봅시다.
스킵-그램 모델의 주요 아이디어는
:eqref:`eq_skip-gram-softmax`에서
주어진 중심 단어 $w_c$를 기반으로
문맥 단어 $w_o$를 생성할 조건부 확률을 계산하기 위해
소프트맥스 연산을 사용하는 것이며,
이에 해당하는 로그 손실은
:eqref:`eq_skip-gram-log`의 반대 부호로 주어집니다.




소프트맥스 연산의 특성상,
문맥 단어는 사전 $\mathcal{V}$의 어느 것이든 될 수 있으므로,
:eqref:`eq_skip-gram-log`의 반대 부호는
전체 어휘 크기만큼의 항목 합계를 포함합니다.
결과적으로,
:eqref:`eq_skip-gram-grad`의 스킵-그램 모델에 대한 기울기 계산과
:eqref:`eq_cbow-gradient`의 CBOW 모델에 대한 기울기 계산 모두
합계를 포함합니다.
불행히도,
(종종 수십만 또는 수백만 단어를 포함하는) 큰 사전에 대해 합산하는
그러한 기울기에 대한 계산 비용은
엄청납니다!

앞서 언급한 계산 복잡성을 줄이기 위해, 이 섹션에서는 두 가지 근사 훈련 방법인
*네거티브 샘플링(negative sampling)*과 *계층적 소프트맥스(hierarchical softmax)*를 소개합니다.
스킵-그램 모델과 CBOW 모델 간의 유사성 때문에,
이 두 가지 근사 훈련 방법을 설명하기 위해
스킵-그램 모델을 예로 들 것입니다.

## 네거티브 샘플링 (Negative Sampling)
:label:`subsec_negative-sampling`


네거티브 샘플링은 원래 목적 함수를 수정합니다.
중심 단어 $w_c$의 문맥 윈도우가 주어졌을 때,
어떤 (문맥) 단어 $w_o$가
이 문맥 윈도우에서 나온다는 사실은
다음과 같이 모델링된 확률을 갖는 사건으로 간주됩니다.


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$ 

여기서 $\sigma$는 시그모이드 활성화 함수의 정의를 사용합니다:

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$ 
:eqlabel:`eq_sigma-f`

텍스트 시퀀스에서 이러한 모든 사건의 결합 확률을 최대화하여
단어 임베딩을 훈련하는 것으로 시작하겠습니다.
구체적으로,
길이 $T$인 텍스트 시퀀스가 주어졌을 때,
타임 스텝 $t$에서의 단어를 $w^{(t)}$로 표시하고
문맥 윈도우 크기를 $m$이라 할 때, 다음 결합 확률을 최대화하는 것을 고려하십시오.


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$ 
:eqlabel:`eq-negative-sample-pos`


그러나
:eqref:`eq-negative-sample-pos`는
긍정 예제(positive examples)와 관련된 사건만 고려합니다.
결과적으로,
:eqref:`eq-negative-sample-pos`의 결합 확률은
모든 단어 벡터가 무한대와 같을 때만
1로 최대화됩니다.
물론,
그러한 결과는 무의미합니다.
목적 함수를 더 의미 있게 만들기 위해,
*네거티브 샘플링*은
미리 정의된 분포에서 샘플링된 부정 예제(negative examples)를 추가합니다.

문맥 단어 $w_o$가 중심 단어 $w_c$의 문맥 윈도우에서 나온다는 사건을 $S$라고 표시합시다.
$w_o$와 관련된 이 사건에 대해,
미리 정의된 분포 $P(w)$에서
이 문맥 윈도우에 속하지 않는 $K$개의 *노이즈 단어(noise words)*를 샘플링합니다.
노이즈 단어 $w_k$ ($k=1, \ldots, K$)가
$w_c$의 문맥 윈도우에서 나오지 않는다는 사건을 $N_k$라고 표시합시다.
긍정 예제와 부정 예제 $S, N_1, \ldots, N_K$를 포함하는
이러한 사건들이 상호 독립적이라고 가정합니다.
네거티브 샘플링은
:eqref:`eq-negative-sample-pos`의 (긍정 예제만 포함하는) 결합 확률을
다음과 같이 다시 씁니다.

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$ 

여기서 조건부 확률은 사건 $S, N_1, \ldots, N_K$를 통해 근사됩니다:

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$ 
:eqlabel:`eq-negative-sample-conditional-prob`

텍스트 시퀀스의 타임 스텝 $t$에서의 단어 $w^{(t)}$와
노이즈 단어 $w_k$의 인덱스를
각각 $i_t$와 $h_k$라고 표시합시다.
:eqref:`eq-negative-sample-conditional-prob`의 조건부 확률에 대한 로그 손실은 다음과 같습니다.

$$ 
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$ 


이제 각 훈련 단계에서의 기울기 계산 비용은
사전 크기와 무관하며,
$K$에 선형적으로 의존한다는 것을 알 수 있습니다.
하이퍼파라미터 $K$를 더 작은 값으로 설정하면,
네거티브 샘플링을 사용한 각 훈련 단계에서의 기울기 계산 비용이 더 작아집니다.




## 계층적 소프트맥스 (Hierarchical Softmax) 

대안적인 근사 훈련 방법으로,
*계층적 소프트맥스(hierarchical softmax)*는
:numref:`fig_hi_softmax`에 묘사된 데이터 구조인
이진 트리를 사용합니다.
여기서 트리의 각 리프 노드(leaf node)는
사전 $\mathcal{V}$의 단어를 나타냅니다.

![근사 훈련을 위한 계층적 소프트맥스. 트리의 각 리프 노드는 사전에 있는 단어를 나타냅니다.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

이진 트리에서 루트 노드로부터 단어 $w$를 나타내는 리프 노드까지의 경로에 있는
노드(양 끝 포함)의 수를 $L(w)$라고 표시합시다.
이 경로상의 $j$번째 노드를 $n(w,j)$라고 하고,
그 문맥 단어 벡터를 $\mathbf{u}_{n(w, j)}$라고 합시다.
예를 들어, :numref:`fig_hi_softmax`에서 $L(w_3) = 4$입니다.
계층적 소프트맥스는 :eqref:`eq_skip-gram-softmax`의 조건부 확률을 다음과 같이 근사합니다.


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$ 

여기서 함수 $\sigma$는 :eqref:`eq_sigma-f`에 정의되어 있으며,
$\textrm{leftChild}(n)$은 노드 $n$의 왼쪽 자식 노드입니다. 만약 $x$가 참이면 $[\[x\]] = 1$; 그렇지 않으면 $[\[x\]] = -1$입니다.

설명하자면,
:numref:`fig_hi_softmax`에서 단어 $w_c$가 주어졌을 때
단어 $w_3$을 생성할 조건부 확률을 계산해 봅시다.
이를 위해서는 $w_c$의 단어 벡터 $\mathbf{v}_c$와
루트에서 $w_3$까지의 경로(:numref:`fig_hi_softmax`의 굵은 경로)에 있는 비-리프(non-leaf) 노드 벡터들 간의 내적이 필요하며,
이 경로는 왼쪽, 오른쪽, 그리고 왼쪽으로 이동합니다.


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$ 

$\\sigma(x)+\\sigma(-x) = 1$이므로,
임의의 단어 $w_c$를 기반으로
사전 $\mathcal{V}$의 모든 단어를 생성할
조건부 확률의 합은 1이 됩니다:

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$ 
:eqlabel:`eq_hi-softmax-sum-one`

다행히도 이진 트리 구조로 인해 $L(w_o)-1$은 $\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ 수준이므로,
사전 크기 $\mathcal{V}$가 매우 클 때,
계층적 소프트맥스를 사용하는 각 훈련 단계의 계산 비용은
근사 훈련을 사용하지 않는 경우에 비해
상당히 줄어듭니다.

## 요약 (Summary)

* 네거티브 샘플링은 긍정 예제와 부정 예제를 모두 포함하는 상호 독립적인 사건을 고려하여 손실 함수를 구성합니다. 훈련을 위한 계산 비용은 각 단계의 노이즈 단어 수에 선형적으로 의존합니다.
* 계층적 소프트맥스는 이진 트리에서 루트 노드로부터 리프 노드까지의 경로를 사용하여 손실 함수를 구성합니다. 훈련을 위한 계산 비용은 각 단계의 사전 크기의 로그에 의존합니다.

## 연습 문제 (Exercises)

1. 네거티브 샘플링에서 노이즈 단어를 어떻게 샘플링할 수 있습니까?
2. :eqref:`eq_hi-softmax-sum-one`이 성립함을 확인하십시오.
3. 네거티브 샘플링과 계층적 소프트맥스를 사용하여 CBOW 모델을 각각 어떻게 훈련합니까?

[Discussions](https://discuss.d2l.ai/t/382)