# 심층 순환 신경망 (Deep Recurrent Neural Networks)

:label:`sec_deep_rnn`

지금까지 우리는 시퀀스 입력, 단일 은닉 RNN 레이어 및 출력 레이어로 구성된 네트워크를 정의하는 데 집중했습니다. 
임의의 타임 스텝의 입력과 해당 출력 사이에 은닉층이 하나만 있음에도 불구하고, 이러한 네트워크가 깊다는 의미가 있습니다. 
첫 번째 타임 스텝의 입력은 최종 타임 스텝 $T$(종종 100단계 또는 1000단계 이후)의 출력에 영향을 줄 수 있습니다. 
이러한 입력은 최종 출력에 도달하기 전에 순환 레이어의 $T$번 적용을 통과합니다. 
그러나 우리는 종종 주어진 타임 스텝의 입력과 동일한 타임 스텝의 출력 사이의 복잡한 관계를 표현하는 능력도 유지하고 싶어 합니다. 
따라서 우리는 종종 시간 방향뿐만 아니라 입력에서 출력 방향으로도 깊은 RNN을 구성합니다. 
이것은 우리가 이미 MLP와 심층 CNN 개발에서 접했던 깊이의 개념과 정확히 일치합니다.


이러한 종류의 심층 RNN을 구축하는 표준 방법은 매우 간단합니다: RNN을 서로 위에 쌓는 것입니다. 
길이 $T$의 시퀀스가 주어지면 첫 번째 RNN은 마찬가지로 길이 $T$의 출력 시퀀스를 생성합니다. 
이것들은 차례로 다음 RNN 레이어의 입력을 구성합니다. 
이 짧은 섹션에서는 이 설계 패턴을 설명하고 그러한 쌓인(stacked) RNN을 코딩하는 방법의 간단한 예를 제시합니다. 
아래 :numref:`fig_deep_rnn`에서는 $L$개의 은닉층이 있는 심층 RNN을 보여줍니다. 
각 은닉 상태는 순차적 입력에 대해 작동하고 순차적 출력을 생성합니다. 
더욱이 각 타임 스텝에서 임의의 RNN 셀(:numref:`fig_deep_rnn`의 흰색 상자)은 이전 타임 스텝에서의 동일한 레이어 값과 동일한 타임 스텝에서의 이전 레이어 값 모두에 의존합니다.

![심층 RNN의 아키텍처.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

공식적으로, 타임 스텝 $t$에서 미니배치 입력 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$(예제 수 $=n$; 각 예제의 입력 수 $=d$)가 있다고 가정합니다. 
동일한 타임 스텝에서, 
$l^\textrm{th}$ 은닉층($l=1,\ldots,L$)의 은닉 상태를 $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$(은닉 유닛 수 $=h$)라고 하고 출력 레이어 변수를 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$(출력 수: $q$)라고 합시다. 
$\mathbf{H}_t^{(0)} = \mathbf{X}_t$로 설정하면, 
활성화 함수 $\phi_l$을 사용하는 $l^\textrm{th}$ 은닉층의 은닉 상태는 다음과 같이 계산됩니다:

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

여기서 가중치 $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ 및 $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$와 
편향 $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$는 $l^\textrm{th}$ 은닉층의 모델 파라미터입니다.

마지막으로 출력 레이어의 계산은 최종 $L^\textrm{th}$ 은닉층의 은닉 상태에만 기반합니다:

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

여기서 가중치 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$와 편향 $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$는 출력 레이어의 모델 파라미터입니다.

MLP와 마찬가지로 은닉층 수 $L$과 은닉 유닛 수 $h$는 우리가 조정할 수 있는 하이퍼파라미터입니다. 
일반적인 RNN 레이어 너비($h$)는 $(64, 2056)$ 범위에 있고 일반적인 깊이($L$)는 $(1, 8)$ 범위에 있습니다. 
또한 :eqref:`eq_deep_rnn_H`의 은닉 상태 계산을 LSTM 또는 GRU의 것으로 대체하여 깊은 게이트(deep-gated) RNN을 쉽게 얻을 수 있습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
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

## 밑바닥부터 구현하기 (Implementation from Scratch)

다층 RNN을 밑바닥부터 구현하기 위해, 각 레이어를 자체 학습 가능한 파라미터를 가진 `RNNScratch` 인스턴스로 취급할 수 있습니다.

```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

다층 순전파 계산은 단순히 레이어별로 순전파 계산을 수행합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

예를 들어, 
*타임 머신* 데이터셋에서 심층 GRU 모델을 훈련합니다(:numref:`sec_rnn-scratch`에서와 동일). 
일을 단순하게 유지하기 위해 레이어 수를 2로 설정합니다.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 간결한 구현 (Concise Implementation)

:begin_tab:`pytorch, mxnet, tensorflow`
다행히 RNN의 여러 레이어를 구현하는 데 필요한 많은 물류 세부 사항은 고수준 API에서 쉽게 사용할 수 있습니다. 
우리의 간결한 구현은 그러한 내장 기능을 사용할 것입니다. 
코드는 이전에 :numref:`sec_gru`에서 사용한 코드를 일반화하여, 단 한 개의 레이어만 선택하는 기본값 대신 레이어 수를 명시적으로 지정할 수 있게 해 줍니다.
:end_tab:

:begin_tab:`jax`
Flax는 RNN을 구현할 때 미니멀리즘적인 접근 방식을 취합니다. RNN에서 레이어 수를 정의하거나 드롭아웃과 결합하는 기능은 기본적으로 제공되지 않습니다. 
우리의 간결한 구현은 모든 내장 기능을 사용하고 그 위에 `num_layers` 및 `dropout` 기능을 추가할 것입니다. 
코드는 이전에 :numref:`sec_gru`에서 사용한 코드를 일반화하여, 단일 레이어의 기본값 대신 레이어 수를 명시적으로 지정할 수 있게 해 줍니다.
:end_tab:

```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """다층 GRU 모델."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """다층 GRU 모델."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """다층 GRU 모델."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """다층 GRU 모델."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # 마지막을 제외한 모든 GRU 레이어 뒤에 드롭아웃 레이어 도입
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # 드롭아웃이 없는 최종 GRU 레이어
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

하이퍼파라미터 선택과 같은 아키텍처 결정은 :numref:`sec_gru`와 매우 유사합니다. 
우리는 고유 토큰 수인 `vocab_size`와 동일한 수의 입력 및 출력을 선택합니다. 
은닉 유닛의 수는 여전히 32입니다. 
유일한 차이점은 이제 (**`num_layers`의 값을 지정하여 의미 있는 수의 은닉층을 선택한다는 점입니다.**)

```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# 실행에 1시간 이상 소요됨 (MXNet의 수정 대기 중)
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
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

## 요약 (Summary)

심층 RNN에서 은닉 상태 정보는 현재 레이어의 다음 타임 스텝과 다음 레이어의 현재 타임 스텝으로 전달됩니다. 
LSTM, GRU 또는 바닐라 RNN과 같은 다양한 형태의 심층 RNN이 존재합니다. 
편리하게도 이러한 모델들은 모두 딥러닝 프레임워크의 고수준 API의 일부로 사용할 수 있습니다. 
모델의 초기화에는 주의가 필요합니다. 
전반적으로 심층 RNN은 적절한 수렴을 보장하기 위해 상당한 양의 작업(학습률 및 클리핑 등)이 필요합니다.

## 연습 문제 (Exercises)

1. GRU를 LSTM으로 교체하고 정확도와 훈련 속도를 비교하십시오.
2. 훈련 데이터를 늘려 여러 책을 포함하십시오. 퍼플렉서티 스케일에서 얼마나 낮게 갈 수 있습니까?
3. 텍스트를 모델링할 때 다른 저자의 소스를 결합하고 싶습니까? 왜 이것이 좋은 아이디어일까요? 무엇이 잘못될 수 있을까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1058)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3862)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18018)
:end_tab: