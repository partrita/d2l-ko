# 순환 신경망 (Recurrent Neural Networks)
:label:`sec_rnn`


:numref:`sec_language-model`에서 우리는 언어 모델링을 위한 마르코프 모델과 $n$-gram을 설명했습니다. 여기서 타임 스텝 $t$에서의 토큰 $x_t$의 조건부 확률은 이전 $n-1$ 토큰에만 의존합니다. 
타임 스텝 $t-(n-1)$ 이전의 토큰이 $x_t$에 미칠 수 있는 영향을 통합하려면 $n$을 늘려야 합니다. 
그러나 모델 파라미터의 수는 그에 따라 기하급수적으로 증가할 것입니다. 어휘 집합 $\mathcal{V}$에 대해 $|V|^n$개의 숫자를 저장해야 하기 때문입니다. 
따라서 $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$을 모델링하는 대신 잠재 변수 모델을 사용하는 것이 바람직합니다.

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$ 

여기서 $h_{t-1}$은 타임 스텝 $t-1$까지의 시퀀스 정보를 저장하는 *은닉 상태(hidden state)*입니다. 
일반적으로 
어떤 타임 스텝 $t$에서의 은닉 상태는 현재 입력 $x_{t}$와 이전 은닉 상태 $h_{t-1}$ 모두를 기반으로 계산될 수 있습니다:

$$h_t = f(x_{t}, h_{t-1}).$$ 
:eqlabel:`eq_ht_xt`

:eqref:`eq_ht_xt`의 함수 $f$가 충분히 강력하다면 잠재 변수 모델은 근사가 아닙니다. 결국 $h_t$는 지금까지 관찰한 모든 데이터를 단순히 저장할 수 있습니다. 
그러나 이는 잠재적으로 계산과 저장 모두를 비싸게 만들 수 있습니다.

:numref:`chap_perceptrons`에서 은닉 유닛이 있는 은닉층에 대해 논의했던 것을 상기해 보십시오. 
은닉층과 은닉 상태는 매우 다른 개념을 나타낸다는 점에 주목할 가치가 있습니다. 
설명했듯이 은닉층은 입력에서 출력으로 가는 경로에서 보이지 않게 숨겨진 레이어입니다. 
은닉 상태는 기술적으로 말해서 주어진 단계에서 우리가 하는 모든 일에 대한 *입력*이며, 이전 타임 스텝의 데이터를 봐야만 계산할 수 있습니다.

*순환 신경망(Recurrent neural networks)* (RNN)은 은닉 상태가 있는 신경망입니다. RNN 모델을 소개하기 전에 먼저 :numref:`sec_mlp`에서 소개한 MLP 모델을 다시 살펴보겠습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## 은닉 상태가 없는 신경망 (Neural Networks without Hidden States)

단일 은닉층이 있는 MLP를 살펴봅시다. 
은닉층의 활성화 함수를 $\phi$라고 합시다. 
배치 크기 $n$과 $d$개의 입력을 가진 예제 미니배치 $\mathbf{X} \in \mathbb{R}^{n \times d}$가 주어지면, 은닉층 출력 $\mathbf{H} \in \mathbb{R}^{n \times h}$는 다음과 같이 계산됩니다.

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

:eqref:`rnn_h_without_state`에서 우리는 가중치 파라미터 $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$, 편향 파라미터 $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$, 그리고 은닉층에 대한 은닉 유닛 수 $h$를 갖습니다. 
이렇게 준비된 상태에서 합산 중에 브로드캐스팅(:numref:`subsec_broadcasting` 참조)을 적용합니다. 
다음으로 은닉층 출력 $\mathbf{H}$는 출력 레이어의 입력으로 사용되며, 이는 다음과 같이 주어집니다.

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$ 

여기서 $\mathbf{O} \in \mathbb{R}^{n \times q}$는 출력 변수, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$는 가중치 파라미터, $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$는 출력 레이어의 편향 파라미터입니다. 분류 문제인 경우 $\mathrm{softmax}(\mathbf{O})$를 사용하여 출력 범주의 확률 분포를 계산할 수 있습니다.

이것은 이전에 :numref:`sec_sequence`에서 해결한 회귀 문제와 전적으로 유사하므로 세부 사항은 생략합니다. 
특성-레이블 쌍을 무작위로 선택하고 자동 미분 및 확률적 경사 하강법을 통해 네트워크의 파라미터를 학습할 수 있다고 말하는 것으로 충분합니다.

## 은닉 상태가 있는 순환 신경망 (Recurrent Neural Networks with Hidden States)
:label:`subsec_rnn_w_hidden_states`

은닉 상태가 있으면 상황이 완전히 다릅니다. 구조를 좀 더 자세히 살펴봅시다.

타임 스텝 $t$에 입력 미니배치 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$가 있다고 가정합니다. 
즉, $n$개의 시퀀스 예제로 구성된 미니배치에 대해 $\mathbf{X}_t$의 각 행은 시퀀스의 타임 스텝 $t$에 있는 한 예제에 해당합니다. 
다음으로, 타임 스텝 $t$의 은닉층 출력을 $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$로 표시합니다. 
MLP와 달리 여기서는 이전 타임 스텝의 은닉층 출력 $\mathbf{H}_{t-1}$을 저장하고 현재 타임 스텝에서 이전 타임 스텝의 은닉층 출력을 사용하는 방법을 설명하기 위해 새로운 가중치 파라미터 $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$를 도입합니다. 구체적으로 현재 타임 스텝의 은닉층 출력 계산은 현재 타임 스텝의 입력과 이전 타임 스텝의 은닉층 출력에 의해 결정됩니다:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

:eqref:`rnn_h_without_state`와 비교할 때, :eqref:`rnn_h_with_state`는 항 $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$를 하나 더 추가하여 :eqref:`eq_ht_xt`를 인스턴스화합니다. 
인접한 타임 스텝의 은닉층 출력 $\mathbf{H}_t$와 $\mathbf{H}_{t-1}$ 사이의 관계로부터, 
우리는 이러한 변수가 신경망의 현재 타임 스텝의 상태나 메모리처럼 현재 타임 스텝까지의 시퀀스 역사 정보를 캡처하고 유지했다는 것을 알 수 있습니다. 따라서 이러한 은닉층 출력을 *은닉 상태(hidden state)*라고 합니다. 
은닉 상태는 현재 타임 스텝에서 이전 타임 스텝의 동일한 정의를 사용하므로 :eqref:`rnn_h_with_state`의 계산은 *순환적(recurrent)*입니다. 따라서 말했듯이 순환 계산을 기반으로 한 은닉 상태를 가진 신경망을 *순환 신경망(recurrent neural networks)*이라고 합니다. 
RNN에서 :eqref:`rnn_h_with_state`의 계산을 수행하는 레이어를 *순환 레이어(recurrent layers)*라고 합니다.


RNN을 구성하는 방법에는 여러 가지가 있습니다. 
:eqref:`rnn_h_with_state`에 의해 정의된 은닉 상태를 가진 것들이 매우 일반적입니다. 
타임 스텝 $t$에 대해 출력 레이어의 출력은 MLP의 계산과 유사합니다:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$ 

RNN의 파라미터에는 은닉층의 가중치 $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ 및 편향 $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$와 출력 레이어의 가중치 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ 및 편향 $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$가 포함됩니다. 
다른 타임 스텝에서도 RNN은 항상 이러한 모델 파라미터를 사용한다는 점을 언급할 가치가 있습니다. 
따라서 RNN의 파라미터화 비용은 타임 스텝 수가 증가해도 증가하지 않습니다.

:numref:`fig_rnn`은 인접한 세 타임 스텝에서의 RNN 계산 로직을 보여줍니다. 
임의의 타임 스텝 $t$에서 은닉 상태의 계산은 다음과 같이 취급될 수 있습니다: 
(i) 현재 타임 스텝 $t$의 입력 $\mathbf{X}_t$와 이전 타임 스텝 $t-1$의 은닉 상태 $\mathbf{H}_{t-1}$을 연결(concatenating)합니다; 
(ii) 연결 결과를 활성화 함수 $\phi$가 있는 완전 연결 레이어에 공급합니다. 
그러한 완전 연결 레이어의 출력은 현재 타임 스텝 $t$의 은닉 상태 $\mathbf{H}_t$입니다. 
이 경우 모델 파라미터는 :eqref:`rnn_h_with_state`의 $\mathbf{W}_{\textrm{xh}}$와 $\mathbf{W}_{\textrm{hh}}$의 연결, 그리고 편향 $\mathbf{b}_\textrm{h}$입니다. 
현재 타임 스텝 $t$의 은닉 상태 $\mathbf{H}_t$는 다음 타임 스텝 $t+1$의 은닉 상태 $\mathbf{H}_{t+1}$을 계산하는 데 참여합니다. 
더욱이 $\mathbf{H}_t$는 현재 타임 스텝 $t$의 출력 $\mathbf{O}_t$를 계산하기 위해 완전 연결 출력 레이어에도 공급됩니다.

![은닉 상태가 있는 RNN.](../img/rnn.svg)
:label:`fig_rnn`

방금 우리는 은닉 상태에 대한 $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ 계산이 
$\mathbf{X}_t$와 $\mathbf{H}_{t-1}$의 연결과 
$\mathbf{W}_{\textrm{xh}}$와 $\mathbf{W}_{\textrm{hh}}$의 연결의 
행렬 곱셈과 동일하다고 언급했습니다. 
이것은 수학적으로 증명될 수 있지만, 
다음에서는 간단한 코드 스니펫을 데모로 사용합니다. 
우선 모양이 각각 (3, 1), (1, 4), (3, 4), (4, 4)인 행렬 `X`, `W_xh`, `H`, `W_hh`를 정의합니다. 
`X`에 `W_xh`를 곱하고, `H`에 `W_hh`를 곱한 다음 이 두 곱을 더하면 (3, 4) 모양의 행렬을 얻습니다.

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

이제 행렬 `X`와 `H`를 열을 따라(축 1) 연결하고, 
행렬 `W_xh`와 `W_hh`를 행을 따라(축 0) 연결합니다. 
이 두 연결은 각각 (3, 5) 모양과 (5, 4) 모양의 행렬을 생성합니다. 
이 두 연결된 행렬을 곱하면 위와 동일한 (3, 4) 모양의 출력 행렬을 얻습니다.

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN 기반 문자 수준 언어 모델 (RNN-Based Character-Level Language Models)

:numref:`sec_language-model`의 언어 모델링에서 우리의 목표는 현재 및 과거 토큰을 기반으로 다음 토큰을 예측하는 것임을 상기하십시오. 
따라서 원래 시퀀스를 한 토큰만큼 이동시켜 타겟(레이블)으로 사용합니다. 
:citet:`Bengio.Ducharme.Vincent.ea.2003`는 언어 모델링에 신경망을 사용할 것을 처음 제안했습니다. 
다음에서는 RNN을 사용하여 언어 모델을 구축하는 방법을 설명합니다. 
미니배치 크기를 1로, 텍스트 시퀀스를 "machine"이라고 합시다. 
후속 섹션에서의 훈련을 단순화하기 위해 텍스트를 단어가 아닌 문자로 토큰화하고 *문자 수준 언어 모델(character-level language model)*을 고려합니다. 
:numref:`fig_rnn_train`은 문자 수준 언어 모델링을 위해 RNN을 통해 현재 및 이전 문자를 기반으로 다음 문자를 예측하는 방법을 보여줍니다.

![RNN 기반 문자 수준 언어 모델. 입력 및 타겟 시퀀스는 각각 "machin"과 "achine"입니다.](../img/rnn-train.svg)
:label:`fig_rnn_train`

훈련 과정 동안, 
각 타임 스텝에 대해 출력 레이어의 출력에 대해 소프트맥스 연산을 실행한 다음 교차 엔트로피 손실을 사용하여 모델 출력과 타겟 간의 오차를 계산합니다. 
은닉층에서 은닉 상태의 순환 계산으로 인해, :numref:`fig_rnn_train`의 타임 스텝 3의 출력 $\mathbf{O}_3$은 텍스트 시퀀스 "m", "a", "c"에 의해 결정됩니다. 훈련 데이터의 시퀀스 다음 문자가 "h"이므로 타임 스텝 3의 손실은 특성 시퀀스 "m", "a", "c"와 이 타임 스텝의 타겟 "h"를 기반으로 생성된 다음 문자의 확률 분포에 따라 달라집니다.

실제로 각 토큰은 $d$차원 벡터로 표현되며 배치 크기 $n>1$을 사용합니다. 따라서 타임 스텝 $t$에서의 입력 $\mathbf X_t$는 $n\times d$ 행렬이 되며, 이는 :numref:`subsec_rnn_w_hidden_states`에서 논의한 것과 동일합니다.

다음 섹션에서는 문자 수준 언어 모델을 위한 RNN을 구현할 것입니다.


## 요약 (Summary)

은닉 상태에 대해 순환 계산을 사용하는 신경망을 순환 신경망(RNN)이라고 합니다.
RNN의 은닉 상태는 현재 타임 스텝까지 시퀀스의 역사적 정보를 캡처할 수 있습니다. 순환 계산을 사용하면 RNN 모델 파라미터의 수는 타임 스텝 수가 증가해도 증가하지 않습니다. 응용 분야와 관련하여 RNN을 사용하여 문자 수준 언어 모델을 만들 수 있습니다.


## 연습 문제 (Exercises)

1. RNN을 사용하여 텍스트 시퀀스의 다음 문자를 예측하는 경우, 임의의 출력에 필요한 차원은 무엇입니까?
2. RNN이 텍스트 시퀀스의 모든 이전 토큰을 기반으로 어떤 타임 스텝에서의 토큰의 조건부 확률을 표현할 수 있는 이유는 무엇입니까?
3. 긴 시퀀스를 통해 역전파하면 기울기에 어떤 일이 발생합니까?
4. 이 섹션에서 설명한 언어 모델과 관련된 몇 가지 문제는 무엇입니까?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1051)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/180013)
:end_tab: