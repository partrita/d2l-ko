# 순환 신경망 간결한 구현 (Concise Implementation of Recurrent Neural Networks)
:label:`sec_rnn-concise`

대부분의 밑바닥부터 구현과 마찬가지로, :numref:`sec_rnn-scratch`는 각 구성 요소가 어떻게 작동하는지에 대한 통찰력을 제공하도록 설계되었습니다. 
하지만 매일 RNN을 사용하거나 프로덕션 코드를 작성할 때는 구현 시간을 줄이고(공통 모델 및 함수에 대한 라이브러리 코드 제공) 계산 시간을 줄이는(이러한 라이브러리 구현을 최적화하여) 라이브러리에 더 의존하고 싶을 것입니다. 
이 섹션에서는 딥러닝 프레임워크에서 제공하는 고수준 API를 사용하여 동일한 언어 모델을 보다 효율적으로 구현하는 방법을 보여줍니다. 
이전과 마찬가지로 *타임 머신* 데이터셋을 로드하는 것으로 시작합니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
from jax import numpy as jnp
```

## [**모델 정의하기 (Defining the Model)**]

고수준 API로 구현된 RNN을 사용하여 다음 클래스를 정의합니다.

:begin_tab:`mxnet`
구체적으로 은닉 상태를 초기화하기 위해 멤버 메서드 `begin_state`를 호출합니다. 
이것은 미니배치의 각 예제에 대한 초기 은닉 상태를 포함하는 리스트를 반환하며, 그 모양은 (은닉층 수, 배치 크기, 은닉 유닛 수)입니다. 
나중에 소개될 일부 모델(예: LSTM)의 경우 이 리스트에는 다른 정보도 포함됩니다.
:end_tab:

:begin_tab:`jax`
Flax는 현재 바닐라 RNN의 간결한 구현을 위한 RNNCell을 제공하지 않습니다. Flax `linen` API에는 LSTM 및 GRU와 같은 더 진보된 RNN 변형이 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """고수준 API로 구현된 RNN 모델."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """고수준 API로 구현된 RNN 모델."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """고수준 API로 구현된 RNN 모델."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """고수준 API로 구현된 RNN 모델."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

:numref:`sec_rnn-scratch`의 `RNNLMScratch` 클래스를 상속하여, 
다음 `RNNLM` 클래스는 완전한 RNN 기반 언어 모델을 정의합니다. 
별도의 완전 연결 출력 레이어를 생성해야 한다는 점에 유의하십시오.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """고수준 API로 구현된 RNN 기반 언어 모델."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """고수준 API로 구현된 RNN 기반 언어 모델."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """고수준 API로 구현된 RNN 기반 언어 모델."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## 훈련 및 예측 (Training and Predicting)

모델을 훈련하기 전에, [**무작위 가중치로 초기화된 모델로 예측**]해 봅시다. 
네트워크를 훈련하지 않았으므로 터무니없는 예측을 생성할 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

다음으로 [**고수준 API를 활용하여 모델을 훈련**]합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

:numref:`sec_rnn-scratch`와 비교할 때, 
이 모델은 비슷한 퍼플렉서티를 달성하지만 최적화된 구현으로 인해 더 빠르게 실행됩니다. 
이전과 마찬가지로 지정된 접두사 문자열에 이어지는 예측 토큰을 생성할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## 요약 (Summary)

딥러닝 프레임워크의 고수준 API는 표준 RNN 구현을 제공합니다. 
이 라이브러리는 표준 모델을 다시 구현하는 데 시간을 낭비하지 않도록 도와줍니다. 
더욱이 
프레임워크 구현은 종종 고도로 최적화되어 있어 
밑바닥부터 구현한 것과 비교할 때 상당한(계산적) 성능 향상을 가져옵니다.

## 연습 문제 (Exercises)

1. 고수준 API를 사용하여 RNN 모델을 과대적합하게 만들 수 있습니까?
2. RNN을 사용하여 :numref:`sec_sequence`의 자기회귀 모델을 구현하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18015)
:end_tab: