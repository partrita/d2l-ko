# 순환 신경망 밑바닥부터 구현하기 (Recurrent Neural Network Implementation from Scratch)
:label:`sec_rnn-scratch`

우리는 이제 밑바닥부터 RNN을 구현할 준비가 되었습니다. 
특히, 이 RNN이 문자 수준 언어 모델로 기능하도록 훈련하고 (:numref:`sec_rnn` 참조) :numref:`sec_text-sequence`에 설명된 데이터 처리 단계에 따라 H. G. Wells의 *타임 머신* 전체 텍스트로 구성된 말뭉치에서 훈련할 것입니다. 
데이터셋을 로드하는 것부터 시작합니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN 모델 (RNN Model)

RNN 모델을 구현하기 위한 클래스를 정의하는 것으로 시작합니다(:numref:`subsec_rnn_w_hidden_states`). 
은닉 유닛의 수 `num_hiddens`는 조정 가능한 하이퍼파라미터라는 점에 유의하십시오.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """밑바닥부터 구현된 RNN 모델."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """밑바닥부터 구현된 RNN 모델."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**아래의 `forward` 메서드는 현재 입력과 이전 타임 스텝의 모델 상태가 주어졌을 때 임의의 타임 스텝에서 출력과 은닉 상태를 계산하는 방법을 정의합니다.**] 
RNN 모델은 `inputs`의 가장 바깥쪽 차원을 반복하여 한 번에 한 타임 스텝씩 은닉 상태를 업데이트한다는 점에 유의하십시오. 
여기서 모델은 $	anh$ 활성화 함수를 사용합니다(:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # 모양이 (batch_size, num_hiddens)인 초기 상태
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # inputs의 모양: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # inputs의 모양: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

다음과 같이 입력 시퀀스의 미니배치를 RNN 모델에 공급할 수 있습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

RNN 모델이 올바른 모양의 결과를 생성하는지 확인하여 은닉 상태의 차원성이 유지되는지 확인해 봅시다.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """리스트의 길이를 확인합니다."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """텐서의 모양을 확인합니다."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNN 기반 언어 모델 (RNN-Based Language Model)

다음 `RNNLMScratch` 클래스는 RNN 기반 언어 모델을 정의합니다. 
여기서 우리는 `__init__` 메서드의 `rnn` 인수를 통해 RNN을 전달합니다. 
언어 모델을 훈련할 때 입력과 출력은 동일한 어휘에서 나옵니다. 
따라서 이들은 어휘 크기와 동일한 동일한 차원을 갖습니다. 
모델을 평가하기 위해 퍼플렉서티를 사용한다는 점에 유의하십시오. 
:numref:`subsec_perplexity`에서 논의했듯이, 이를 통해 길이가 다른 시퀀스를 비교할 수 있습니다.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """밑바닥부터 구현된 RNN 기반 언어 모델."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """밑바닥부터 구현된 RNN 기반 언어 모델."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """밑바닥부터 구현된 RNN 기반 언어 모델."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**원-핫 인코딩 (One-Hot Encoding)**]

각 토큰은 해당 단어/문자/단어 조각의 어휘 내 위치를 나타내는 수치 인덱스로 표현된다는 것을 상기하십시오. 
(각 타임 스텝마다) 단일 입력 노드가 있는 신경망을 구축하고 싶은 유혹을 받을 수 있습니다. 
이것은 충분히 가까운 두 값이 비슷하게 취급되어야 하는 가격이나 온도와 같은 수치 입력을 다룰 때 작동합니다. 
하지만 이것은 말이 안 됩니다. 
우리 어휘의 $45^{	extrm{th}}$번째와 $46^{	extrm{th}}$번째 단어는 "their"와 "said"인데, 그 의미는 전혀 비슷하지 않습니다.

이러한 범주형 데이터를 다룰 때 가장 일반적인 전략은 *원-핫 인코딩*으로 각 항목을 나타내는 것입니다(:numref:`subsec_classification-problem` 참조). 
원-핫 인코딩은 길이가 어휘 크기 $N$으로 주어지는 벡터이며, 모든 항목은 $0$으로 설정되지만 우리 토큰에 해당하는 항목은 $1$로 설정됩니다. 
예를 들어 어휘에 5개의 요소가 있는 경우, 인덱스 0과 2에 해당하는 원-핫 벡터는 다음과 같습니다.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**우리가 각 반복에서 샘플링하는 미니배치는 (배치 크기, 타임 스텝 수) 모양을 갖습니다. 
각 입력을 원-핫 벡터로 나타내면, 각 미니배치를 3차원 텐서로 생각할 수 있으며, 여기서 세 번째 축을 따른 길이는 어휘 크기(`len(vocab)`)로 주어집니다.**) 
우리는 종종 입력을 전치하여 (타임 스텝 수, 배치 크기, 어휘 크기) 모양의 출력을 얻습니다. 
이렇게 하면 미니배치의 은닉 상태를 타임 스텝별로 업데이트하기 위해 가장 바깥쪽 차원을 더 편리하게 반복할 수 있습니다(예: 위의 `forward` 메서드에서).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):
    # 출력 모양: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### RNN 출력 변환 (Transforming RNN Outputs)

언어 모델은 완전 연결 출력 레이어를 사용하여 각 타임 스텝에서 RNN 출력을 토큰 예측으로 변환합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

[**순전파 계산이 올바른 모양의 출력을 생성하는지 확인**]해 봅시다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**기울기 클리핑 (Gradient Clipping)**]


신경망을 단일 타임 스텝 내에서도 많은 레이어가 입력과 출력을 분리한다는 의미에서 "깊다"고 생각하는 데 이미 익숙하지만, 시퀀스 길이는 새로운 깊이 개념을 도입합니다. 
입력에서 출력 방향으로 네트워크를 통과하는 것 외에도, 
첫 번째 타임 스텝의 입력은 
최종 타임 스텝에서 모델의 출력에 영향을 미치기 위해 
타임 스텝을 따라 $T$개 레이어의 체인을 통과해야 합니다. 
역방향으로 보면, 각 반복에서 우리는 시간을 거슬러 기울기를 역전파하여 
길이 $\mathcal{O}(T)$의 행렬 곱 체인을 생성합니다. 
:numref:`sec_numerical_stability`에서 언급했듯이, 
이는 가중치 행렬의 속성에 따라 기울기가 폭발하거나 사라지는 수치적 불안정성을 초래할 수 있습니다.

사라지는 기울기와 폭발하는 기울기를 다루는 것은 RNN을 설계할 때 기본적인 문제이며 현대 신경망 아키텍처에서 가장 큰 발전 중 일부에 영감을 주었습니다. 
다음 장에서는 기울기 소실 문제를 완화하기 위해 설계된 특수 아키텍처에 대해 이야기할 것입니다. 
그러나 현대 RNN조차도 종종 기울기 폭발로 고통받습니다. 
투박하지만 어디에나 있는 솔루션 중 하나는 단순히 기울기를 잘라내어(clip) 결과적으로 "클리핑된" 기울기가 더 작은 값을 갖도록 강제하는 것입니다.


일반적으로 경사 하강법으로 어떤 목적을 최적화할 때, 우리는 관심 있는 파라미터(예: 벡터 $\mathbf{x}$)를 반복적으로 업데이트하지만 음의 기울기 $\mathbf{g}$ 방향으로 밀어 넣습니다(확률적 경사 하강법에서는 무작위로 샘플링된 미니배치에서 이 기울기를 계산합니다). 
예를 들어 학습률 $\eta > 0$을 사용하면 각 업데이트는 $\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$ 형태를 취합니다. 
목적 함수 $f$가 충분히 매끄럽다고 가정해 봅시다. 
공식적으로 우리는 목적 함수가 상수 $L$로 *립시츠 연속(Lipschitz continuous)*이라고 말합니다. 즉, 임의의 $\mathbf{x}$와 $\mathbf{y}$에 대해 다음을 갖습니다.

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$ 

보시다시피 $\eta \mathbf{g}$를 빼서 파라미터 벡터를 업데이트할 때 목적 함수의 값 변화는 다음과 같이 학습률, 기울기의 노름, $L$에 따라 달라집니다:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|g\u0002\|.$$ 

즉, 목적 함수는 $L \eta \|g\u0002\|$ 이상 변경될 수 없습니다. 
이 상한선에 대해 작은 값을 갖는 것은 좋거나 나쁜 것으로 볼 수 있습니다. 
단점으로는 목적 함수의 값을 줄일 수 있는 속도를 제한하고 있다는 것입니다. 
장점으로는 이것이 어떤 한 경사 단계에서 우리가 얼마나 잘못될 수 있는지를 제한합니다.


기울기가 폭발한다고 말할 때, 우리는 $\|g\u0002\|$가 지나치게 커진다는 것을 의미합니다. 
이 최악의 경우, 단일 경사 단계에서 수천 번의 훈련 반복 과정에서 이루어진 모든 진전을 취소할 수 있을 정도로 많은 손상을 입힐 수 있습니다. 
기울기가 그렇게 클 수 있으면 신경망 훈련은 종종 발산하여 목적 함수의 값을 줄이지 못합니다. 
다른 경우에는 훈련이 결국 수렴하지만 손실의 거대한 스파이크로 인해 불안정합니다.


$L \eta \|g\u0002\|$의 크기를 제한하는 한 가지 방법은 학습률 $\eta$를 아주 작은 값으로 줄이는 것입니다. 
이것은 업데이트를 편향시키지 않는다는 장점이 있습니다. 
하지만 큰 기울기를 *드물게* 얻는다면 어떨까요? 
이 과감한 조치는 드문 기울기 폭발 이벤트를 처리하기 위해 모든 단계에서 우리의 진전을 늦춥니다. 
인기 있는 대안은 다음과 같이 주어진 반지름 $\theta$의 공에 기울기 $\mathbf{g}$를 투영하는 *기울기 클리핑(gradient clipping)* 휴리스틱을 채택하는 것입니다:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**) 

이것은 기울기 노름이 $\theta$를 초과하지 않도록 보장하고 업데이트된 기울기가 $\mathbf{g}$의 원래 방향과 완전히 정렬되도록 합니다. 
또한 주어진 미니배치(및 그 안의 주어진 샘플)가 파라미터 벡터에 미칠 수 있는 영향을 제한하는 바람직한 부작용도 있습니다. 
이것은 모델에 어느 정도의 견고성을 부여합니다. 
분명히 하자면, 이것은 핵(hack)입니다. 
기울기 클리핑은 우리가 항상 실제 기울기를 따르는 것은 아니며 가능한 부작용에 대해 분석적으로 추론하기 어렵다는 것을 의미합니다. 
그러나 이것은 매우 유용한 핵이며 대부분의 딥러닝 프레임워크의 RNN 구현에서 널리 채택되고 있습니다.


아래에서는 `d2l.Trainer` 클래스의 `fit_epoch` 메서드에 의해 호출되는 기울기 클리핑 메서드를 정의합니다(:numref:`sec_linear_scratch` 참조). 
기울기 노름을 계산할 때 모든 모델 파라미터를 연결하여 단일 거대한 파라미터 벡터로 취급한다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## 훈련 (Training)

*타임 머신* 데이터셋(`data`)을 사용하여 밑바닥부터 구현된 RNN(`rnn`)을 기반으로 문자 수준 언어 모델(`model`)을 훈련합니다. 
먼저 기울기를 계산한 다음 클리핑하고 마지막으로 클리핑된 기울기를 사용하여 모델 파라미터를 업데이트한다는 점에 유의하십시오.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 디코딩 (Decoding)

일단 언어 모델이 학습되면, 다음 토큰을 예측하는 데 사용할 뿐만 아니라 이전에 예측된 토큰을 입력의 다음 토큰인 것처럼 취급하여 각 후속 토큰을 계속 예측할 수 있습니다. 
때로는 문서의 시작 부분에서 시작하는 것처럼 텍스트를 생성하고 싶을 때가 있습니다. 
그러나 사용자 제공 접두사에 언어 모델을 조건부로 설정하는 것이 종종 유용합니다. 
예를 들어 검색 엔진을 위한 자동 완성 기능이나 이메일 작성 시 사용자를 돕기 위한 기능을 개발하는 경우, 
지금까지 작성한 내용(접두사)을 입력하고 그럴듯한 연속을 생성하기를 원할 것입니다.


[**다음 `predict` 메서드는 사용자 제공 `prefix`를 섭취한 후 한 번에 한 문자씩 연속을 생성합니다**]. 
`prefix`의 문자를 반복할 때, 은닉 상태를 다음 타임 스텝으로 계속 전달하지만 출력은 생성하지 않습니다. 
이것을 *워밍업(warm-up)* 기간이라고 합니다. 
접두사를 섭취한 후, 이제 후속 문자를 방출할 준비가 되었습니다. 각 문자는 다음 타임 스텝의 입력으로 모델에 피드백됩니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # 워밍업 기간
            outputs.append(vocab[prefix[i + 1]])
        else:  # num_preds 단계 예측
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # 워밍업 기간
            outputs.append(vocab[prefix[i + 1]])
        else:  # num_preds 단계 예측
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

다음에서는 접두사를 지정하고 20개의 추가 문자를 생성하도록 합니다.

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

위의 RNN 모델을 밑바닥부터 구현하는 것은 유익하지만 편리하지는 않습니다. 
다음 섹션에서는 딥러닝 프레임워크를 활용하여 표준 아키텍처를 사용하여 RNN을 빠르게 만들고, 고도로 최적화된 라이브러리 함수에 의존하여 성능 향상을 얻는 방법을 볼 것입니다.


## 요약 (Summary)

우리는 사용자 제공 텍스트 접두사를 따르는 텍스트를 생성하도록 RNN 기반 언어 모델을 훈련할 수 있습니다. 
간단한 RNN 언어 모델은 입력 인코딩, RNN 모델링, 출력 생성으로 구성됩니다. 
훈련 중에 기울기 클리핑은 폭발하는 기울기 문제를 완화할 수 있지만 사라지는 기울기 문제를 해결하지는 않습니다. 실험에서 우리는 간단한 RNN 언어 모델을 구현하고 문자 수준에서 토큰화된 텍스트 시퀀스에 대해 기울기 클리핑을 사용하여 훈련했습니다. 접두사에 조건을 걸어 언어 모델을 사용하여 그럴듯한 연속을 생성할 수 있으며, 이는 자동 완성 기능과 같은 많은 응용 프로그램에서 유용함이 입증되었습니다.


## 연습 문제 (Exercises)

1. 구현된 언어 모델은 *타임 머신*의 맨 처음 토큰까지의 모든 과거 토큰을 기반으로 다음 토큰을 예측합니까? 
1. 어떤 하이퍼파라미터가 예측에 사용되는 역사의 길이를 제어합니까?
1. 원-핫 인코딩이 각 객체에 대해 다른 임베딩을 선택하는 것과 동일함을 보이십시오. 
1. 퍼플렉서티를 개선하기 위해 하이퍼파라미터(예: 에폭 수, 은닉 유닛 수, 미니배치의 타임 스텝 수, 학습률)를 조정하십시오. 이 간단한 아키텍처를 고수하면서 얼마나 낮출 수 있습니까? 
1. 원-핫 인코딩을 학습 가능한 임베딩으로 대체하십시오. 이것이 더 나은 성능으로 이어집니까? 
1. *타임 머신*에서 훈련된 이 언어 모델이 H. G. Wells의 다른 책, 예: *우주 전쟁(The War of the Worlds)*에서 얼마나 잘 작동하는지 결정하기 위한 실험을 수행하십시오. 
1. 다른 저자가 쓴 책에서 이 모델의 퍼플렉서티를 평가하기 위한 또 다른 실험을 수행하십시오. 
1. 가장 가능성 있는 다음 문자를 선택하는 대신 샘플링을 사용하도록 예측 메서드를 수정하십시오. 
    * 무슨 일이 일어납니까? 
    * 모델을 더 가능성 있는 출력 쪽으로 편향시키십시오. 예를 들어 $\alpha > 1$에 대해 $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$에서 샘플링합니다. 
1. 기울기를 클리핑하지 않고 이 섹션의 코드를 실행하십시오. 무슨 일이 일어납니까? 
1. 이 섹션에서 사용된 활성화 함수를 ReLU로 대체하고 이 섹션의 실험을 반복하십시오. 여전히 기울기 클리핑이 필요합니까? 그 이유는 무엇입니까? 

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18014)
:end_tab: