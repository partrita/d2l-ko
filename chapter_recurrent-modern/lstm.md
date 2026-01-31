# 장단기 메모리 (LSTM) (Long Short-Term Memory (LSTM))
:label:`sec_lstm`


역전파를 사용하여 최초의 Elman 스타일 RNN이 훈련된 직후 :cite:`elman1990finding`, 장기 의존성 학습 문제(사라지는 기울기 및 폭발하는 기울기로 인한)가 두드러졌으며, 벤지오(Bengio)와 호크라이터(Hochreiter)가 이 문제를 논의했습니다 :cite:`bengio1994learning,Hochreiter.Bengio.Frasconi.ea.2001`.
호크라이터는 이미 1991년 석사 학위 논문에서 이 문제를 명확히 기술했지만, 논문이 독일어로 작성되었기 때문에 결과가 널리 알려지지 않았습니다.
기울기 클리핑이 폭발하는 기울기에는 도움이 되지만, 사라지는 기울기를 처리하는 데는 더 정교한 솔루션이 필요한 것으로 보입니다.
사라지는 기울기를 해결하기 위한 최초이자 가장 성공적인 기술 중 하나가 :citet:`Hochreiter.Schmidhuber.1997`에 의한 장단기 메모리(long short-term memory, LSTM) 모델의 형태로 등장했습니다.
LSTM은 표준 순환 신경망과 유사하지만 여기서는 각 일반 순환 노드가 *메모리 셀(memory cell)*로 대체됩니다.
각 메모리 셀은 *내부 상태*, 즉 고정 가중치 1의 자체 연결된 순환 엣지를 가진 노드를 포함하여 기울기가 사라지거나 폭발하지 않고 많은 타임 스텝을 가로질러 전달될 수 있도록 보장합니다.

"장단기 메모리"라는 용어는 다음과 같은 직관에서 유래했습니다.
단순 순환 신경망은 가중치 형태의 *장기 메모리*를 가집니다.
가중치는 훈련 중에 천천히 변하며 데이터에 대한 일반적인 지식을 인코딩합니다.
또한 각 노드에서 후속 노드로 전달되는 덧없는 활성화 형태의 *단기 메모리*를 가집니다.
LSTM 모델은 메모리 셀을 통해 중간 유형의 저장을 도입합니다.
메모리 셀은 특정 연결 패턴의 더 단순한 노드들로 구성된 합성 단위로, 곱셈 노드가 새로 포함되었습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 게이트 메모리 셀 (Gated Memory Cell)

각 메모리 셀은 *내부 상태*와 (i) 주어진 입력이 내부 상태에 영향을 주어야 하는지(*입력 게이트*), (ii) 내부 상태를 0으로 씻어내야 하는지(*삭제 게이트*), (iii) 주어진 뉴런의 내부 상태가 셀의 출력에 영향을 주도록 허용되어야 하는지(*출력 게이트*)를 결정하는 여러 곱셈 게이트를 갖추고 있습니다.


### 게이트 은닉 상태 (Gated Hidden State)

바닐라 RNN과 LSTM의 주요 차이점은 후자가 은닉 상태의 게이팅을 지원한다는 것입니다.
이는 은닉 상태가 언제 *업데이트*되어야 하는지, 그리고 언제 *리셋*되어야 하는지에 대한 전용 메커니즘을 가지고 있음을 의미합니다.
이러한 메커니즘은 학습되며 위에서 나열한 우려 사항을 해결합니다.
예를 들어 첫 번째 토큰이 매우 중요하다면 첫 번째 관찰 후에 은닉 상태를 업데이트하지 않도록 학습할 것입니다.
마찬가지로 관련 없는 일시적인 관찰을 건너뛰도록 학습할 것입니다.
마지막으로 필요할 때마다 잠재 상태를 리셋하도록 학습할 것입니다.
아래에서 이에 대해 자세히 논의합니다.

### 입력 게이트, 삭제 게이트, 출력 게이트 (Input Gate, Forget Gate, and Output Gate)

:numref:`fig_lstm_0`에 설명된 것처럼 LSTM 게이트로 들어가는 데이터는 현재 타임 스텝의 입력과 이전 타임 스텝의 은닉 상태입니다.
시그모이드 활성화 함수가 있는 세 개의 완전 연결 레이어가 입력, 삭제, 출력 게이트의 값을 계산합니다.
시그모이드 활성화의 결과로 세 게이트의 모든 값은 $(0, 1)$ 범위에 있습니다.
또한 일반적으로 *tanh* 활성화 함수로 계산되는 *입력 노드*가 필요합니다.
직관적으로 *입력 게이트*는 입력 노드의 값 중 얼마만큼을 현재 메모리 셀 내부 상태에 추가해야 하는지를 결정합니다.
*삭제 게이트*는 메모리의 현재 값을 유지할지 아니면 씻어낼지 결정합니다.
그리고 *출력 게이트*는 메모리 셀이 현재 타임 스텝의 출력에 영향을 주어야 하는지 결정합니다.


![LSTM 모델에서 입력 게이트, 삭제 게이트, 출력 게이트 계산.](../img/lstm-0.svg)
:label:`fig_lstm_0`

수학적으로 은닉 유닛이 $h$개, 배치 크기가 $n$, 입력 수가 $d$라고 가정합니다.
따라서 입력은 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$이고 이전 타임 스텝의 은닉 상태는 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$입니다.
이에 따라 타임 스텝 $t$에서의 게이트는 다음과 같이 정의됩니다: 입력 게이트는 $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, 삭제 게이트는 $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, 출력 게이트는 $\mathbf{O}_t \in \mathbb{R}^{n \times h}$입니다.
이들은 다음과 같이 계산됩니다:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$ 

여기서 $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$는 가중치 파라미터이고 $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$는 편향 파라미터입니다.
합산 중에 브로드캐스팅(:numref:`subsec_broadcasting` 참조)이 트리거됨에 유의하십시오.
우리는 입력 값을 간격 $(0, 1)$에 매핑하기 위해 시그모이드 함수(:numref:`sec_mlp`에서 소개됨)를 사용합니다.


### 입력 노드 (Input Node)

다음으로 메모리 셀을 설계합니다.
아직 다양한 게이트의 동작을 지정하지 않았으므로, 먼저 *입력 노드* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$를 도입합니다.
그 계산은 위에서 설명한 세 게이트의 계산과 유사하지만 활성화 함수로 $(-1, 1)$ 범위의 값을 갖는 $\tanh$ 함수를 사용합니다.
이는 타임 스텝 $t$에서 다음과 같은 방정식으로 이어집니다:

$$\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),$$ 

여기서 $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$는 가중치 파라미터이고 $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$는 편향 파라미터입니다.

입력 노드에 대한 간단한 그림은 :numref:`fig_lstm_1`에 나와 있습니다.

![LSTM 모델에서 입력 노드 계산.](../img/lstm-1.svg)
:label:`fig_lstm_1`


### 메모리 셀 내부 상태 (Memory Cell Internal State)

LSTM에서 입력 게이트 $\mathbf{I}_t$는 $\tilde{\mathbf{C}}_t$를 통해 얼마나 많은 새 데이터를 고려할지를 지배하고, 삭제 게이트 $\mathbf{F}_t$는 이전 셀 내부 상태 $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ 중 얼마만큼을 유지할지를 결정합니다.
아다마르 곱 연산자 $\odot$를 사용하여 다음과 같은 업데이트 방정식에 도달합니다:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$ 

삭제 게이트가 항상 1이고 입력 게이트가 항상 0이면, 메모리 셀 내부 상태 $\mathbf{C}_{t-1}$은 영원히 일정하게 유지되어 각 후속 타임 스텝으로 변경되지 않고 전달됩니다.
그러나 입력 게이트와 삭제 게이트는 모델에 이 값을 언제 변경하지 않고 유지할지, 그리고 언제 후속 입력에 대응하여 이를 섭동시킬지 학습할 수 있는 유연성을 제공합니다.
실제로 이 설계는 기울기 소실 문제를 완화하여 특히 긴 시퀀스 길이를 가진 데이터셋에 직면할 때 훈련하기 훨씬 쉬운 모델로 이어집니다.

따라서 :numref:`fig_lstm_2`의 흐름도에 도달합니다.

![LSTM 모델에서 메모리 셀 내부 상태 계산.](../img/lstm-2.svg)

:label:`fig_lstm_2`


### 은닉 상태 (Hidden State)

마지막으로 메모리 셀의 출력, 즉 다른 레이어에서 보는 은닉 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$를 계산하는 방법을 정의해야 합니다.
여기서 출력 게이트가 작동합니다.
LSTM에서는 먼저 메모리 셀 내부 상태에 $\tanh$를 적용한 다음, 이번에는 출력 게이트와 함께 또 다른 요소별 곱셈을 적용합니다.
이렇게 하면 $\mathbf{H}_t$의 값이 항상 $(-1, 1)$ 간격에 있게 됩니다:

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$ 


출력 게이트가 1에 가까울 때마다 메모리 셀 내부 상태가 후속 레이어에 억제 없이 영향을 주도록 허용하는 반면, 출력 게이트 값이 0에 가까울 때는 현재 메모리가 현재 타임 스텝에서 네트워크의 다른 레이어에 영향을 주지 않도록 방지합니다.
메모리 셀은 출력 게이트가 0에 가까운 값을 취하는 동안 네트워크의 나머지 부분에 영향을 주지 않고 많은 타임 스텝에 걸쳐 정보를 축적할 수 있으며,
출력 게이트가 0에 가까운 값에서 1에 가까운 값으로 뒤집히자마자 후속 타임 스텝에서 갑자기 네트워크에 영향을 줄 수 있음에 유의하십시오. :numref:`fig_lstm_3`에 데이터 흐름의 그래픽 설명이 있습니다.

![LSTM 모델에서 은닉 상태 계산.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## 밑바닥부터 구현하기 (Implementation from Scratch)

이제 LSTM을 밑바닥부터 구현해 봅시다.
:numref:`sec_rnn-scratch`의 실험과 마찬가지로 먼저 *타임 머신* 데이터셋을 로드합니다.

### [**모델 파라미터 초기화**]

다음으로 모델 파라미터를 정의하고 초기화해야 합니다.
이전과 마찬가지로 하이퍼파라미터 `num_hiddens`가 은닉 유닛의 수를 결정합니다.
가중치는 0.01 표준 편차를 갖는 가우스 분포에 따라 초기화하고 편향은 0으로 설정합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))

        self.W_xi, self.W_hi, self.b_i = triple()  # 입력 게이트
        self.W_xf, self.W_hf, self.b_f = triple()  # 삭제 게이트
        self.W_xo, self.W_ho, self.b_o = triple()  # 출력 게이트
        self.W_xc, self.W_hc, self.b_c = triple()  # 입력 노드
```

```{.python .input}
%%tab jax
class LSTMScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        init_weight = lambda name, shape: self.param(name,
                                                     nn.initializers.normal(self.sigma),
                                                     shape)
        triple = lambda name : (
            init_weight(f'W_x{name}', (self.num_inputs, self.num_hiddens)),
            init_weight(f'W_h{name}', (self.num_hiddens, self.num_hiddens)),
            self.param(f'b_{name}', nn.initializers.zeros, (self.num_hiddens)))

        self.W_xi, self.W_hi, self.b_i = triple('i')  # 입력 게이트
        self.W_xf, self.W_hf, self.b_f = triple('f')  # 삭제 게이트
        self.W_xo, self.W_ho, self.b_o = triple('o')  # 출력 게이트
        self.W_xc, self.W_hc, self.b_c = triple('c')  # 입력 노드
```

:begin_tab:`pytorch, mxnet, tensorflow`
[**실제 모델**]은 위에서 설명한 대로 정의되며 세 개의 게이트와 입력 노드로 구성됩니다.
은닉 상태만 출력 레이어로 전달된다는 점에 유의하십시오.
:end_tab:

:begin_tab:`jax`
[**실제 모델**]은 위에서 설명한 대로 정의되며 세 개의 게이트와 입력 노드로 구성됩니다.
은닉 상태만 출력 레이어로 전달된다는 점에 유의하십시오.
`forward` 메서드에서 긴 for-루프를 사용하면 첫 번째 실행 시 JIT 컴파일 시간이 매우 길어집니다. 이에 대한 해결책으로 매 타임 스텝마다 상태를 업데이트하기 위해 for-루프를 사용하는 대신, JAX에는 동일한 동작을 달성하기 위한 `jax.lax.scan` 유틸리티 변환이 있습니다.
이는 `carry`라고 불리는 초기 상태와 선행 축에서 스캔되는 `inputs` 배열을 인수로 취합니다. `scan` 변환은 최종적으로 기대하는 대로 최종 상태와 쌓인 출력들을 반환합니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # 모양이 (batch_size, num_hiddens)인 초기 상태
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
            C = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    # 입력을 반복하는 대신 lax.scan 프리미티브를 사용하여 jit 컴파일 시간을 절약합니다.
    def scan_fn(carry, X):
        H, C = carry
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi)) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        return (H, C), H  # carry, y 반환

    if H_C is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens)), \
                jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H_C

    # scan은 scan_fn, 초기 carry 상태, 스캔할 선행 축이 있는 xs를 인수로 취합니다
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### [**훈련 (Training)**] 및 예측

:numref:`sec_rnn-scratch`의 `RNNLMScratch` 클래스를 인스턴스화하여 LSTM 모델을 훈련해 봅시다.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**간결한 구현 (Concise Implementation)**]

고수준 API를 사용하여 LSTM 모델을 직접 인스턴스화할 수 있습니다.
이는 위에서 우리가 명시적으로 만든 모든 구성 세부 사항을 캡슐화합니다.
앞서 우리가 상세히 설명한 많은 부분에 대해 Python 대신 컴파일된 연산자를 사용하므로 코드가 훨씬 빠릅니다.

```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens)

    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True,
                return_state=True, time_major=True)

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab jax
class LSTM(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H_C=None, training=False):
        if H_C is None:
            batch_size = inputs.shape[1]
            H_C = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0),
                                                        (batch_size,),
                                                        self.num_hiddens)

        LSTM = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})

        H_C, outputs = LSTM()(H_C, inputs)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow', 'jax'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

LSTM은 의미 있는 상태 제어를 가진 전형적인 잠재 변수 자기회귀 모델입니다.
다중 레이어, 잔차 연결, 다양한 유형의 정규화 등 수년 동안 많은 변형이 제안되었습니다. 그러나 시퀀스의 장거리 의존성 때문에 LSTM 및 기타 시퀀스 모델(예: GRU)을 훈련하는 것은 비용이 많이 듭니다.
나중에 우리는 일부 사례에서 사용될 수 있는 Transformer와 같은 대안 모델을 만나게 될 것입니다.


## 요약 (Summary)

LSTM은 1997년에 발표되었지만, 2000년대 중반 예측 대회에서 여러 차례 승리하면서 큰 주목을 받았고,
2011년부터 2017년 Transformer 모델이 부상하기 전까지 시퀀스 학습을 위한 지배적인 모델이 되었습니다.
심지어 Transformer조차도 LSTM이 도입한 아키텍처 설계 혁신에서 몇 가지 핵심 아이디어를 얻었습니다.


LSTM에는 정보의 흐름을 제어하는 세 가지 유형의 게이트가 있습니다:
입력 게이트, 삭제 게이트, 출력 게이트입니다.
LSTM의 은닉층 출력에는 은닉 상태와 메모리 셀 내부 상태가 포함됩니다.
은닉 상태만 출력 레이어로 전달되는 반면 메모리 셀 내부 상태는 완전히 내부적으로 유지됩니다.
LSTM은 사라지는 기울기와 폭발하는 기울기를 완화할 수 있습니다.



## 연습 문제 (Exercises)

1. 하이퍼파라미터를 조정하고 실행 시간, 퍼플렉서티 및 출력 시퀀스에 미치는 영향을 분석하십시오.
2. 단순히 문자 시퀀스가 아니라 적절한 단어를 생성하도록 모델을 변경하려면 어떻게 해야 합니까?
3. 주어진 은닉 차원에 대해 GRU, LSTM 및 일반 RNN의 계산 비용을 비교하십시오. 훈련 및 추론 비용에 특히 주의를 기울이십시오.
4. 후보 메모리 셀이 $\tanh$ 함수를 사용하여 값 범위를 $-1$과 $1$ 사이로 보장하는데, 왜 은닉 상태가 출력 값 범위를 $-1$과 $1$ 사이로 보장하기 위해 다시 $\tanh$ 함수를 사용해야 합니까?
5. 문자 시퀀스 예측이 아닌 시계열 예측을 위한 LSTM 모델을 구현하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3861)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18016)
:end_tab: