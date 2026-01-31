# RMSProp
:label:`sec_rmsprop`

:numref:`sec_adagrad`에서의 주요 문제 중 하나는 학습률이 실질적으로 $\mathcal{O}(t^{-\frac{1}{2}})$의 미리 정의된 스케줄에 따라 감소한다는 것입니다.
 While this is generally appropriate for convex problems, it might not be ideal for nonconvex ones, such as those encountered in deep learning. Yet, the coordinate-wise adaptivity of Adagrad is highly desirable as a preconditioner.

:citet:`Tieleman.Hinton.2012`는 속도 스케줄링(rate scheduling)을 좌표별 적응형 학습률에서 분리하기 위한 간단한 수정안으로 RMSProp 알고리즘을 제안했습니다. 문제는 Adagrad가 기울기 $\mathbf{g}_t$의 제곱을 상태 벡터 $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$에 누적한다는 것입니다. 그 결과 정규화가 부족하여 $\mathbf{s}_t$가 알고리즘이 수렴함에 따라 본질적으로 선형적으로 제한 없이 계속 커지게 됩니다.

이 문제를 해결하는 한 가지 방법은 $\mathbf{s}_t / t$를 사용하는 것입니다. $\mathbf{g}_t$의 합리적인 분포에 대해 이것은 수렴할 것입니다. 불행히도 절차가 값의 전체 궤적을 기억하기 때문에 극한 동작(limit behavior)이 중요해지기 시작할 때까지 매우 오랜 시간이 걸릴 수 있습니다. 대안은 모멘텀 방법에서 사용한 것과 동일한 방식으로 지수 이동 평균(leaky average)을 사용하는 것입니다. 즉, 어떤 파라미터 $\gamma > 0$에 대해 $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$를 사용하는 것입니다. 다른 모든 부분을 변경하지 않은 채로 두면 RMSProp이 됩니다.


## 알고리즘 (The Algorithm)

방정식을 자세히 작성해 봅시다.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

상수 $\epsilon > 0$은 일반적으로 0으로 나누는 문제나 과도하게 큰 단계 크기로 고통받지 않도록 $10^{-6}$으로 설정됩니다. 이러한 확장이 주어지면 이제 좌표별로 적용되는 스케일링과는 독립적으로 학습률 $\eta$를 자유롭게 제어할 수 있습니다. 지수 이동 평균의 측면에서 우리는 이전에 모멘텀 방법의 경우에 적용했던 것과 동일한 추론을 적용할 수 있습니다. $\mathbf{s}_t$의 정의를 확장하면 다음과 같습니다.

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

이전 :numref:`sec_momentum`에서와 같이 $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$를 사용합니다. 따라서 가중치의 합은 1로 정규화되며 관찰의 반감기는 $\gamma^{-1}$입니다. 다양한 $\gamma$ 선택에 대해 지난 40개 타임스텝의 가중치를 시각화해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 바닥부터 구현하기 (Implementation from Scratch)

이전과 마찬가지로 이차 함수 $f(\mathbf{x})=0.1x_1^2+2x_2^2$를 사용하여 RMSProp의 궤적을 관찰합니다. :numref:`sec_adagrad`에서 학습률 0.4의 Adagrad를 사용했을 때, 학습률이 너무 빨리 감소하여 알고리즘의 후반 단계에서 변수가 매우 느리게만 움직였음을 상기하십시오. $\eta$가 별도로 제어되기 때문에 RMSProp에서는 이런 일이 발생하지 않습니다.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Next, we implement RMSProp to be used in a deep network. This is equally straightforward.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

We set the initial learning rate to 0.01 and the weighting term $\gamma$ to 0.9. That is, $\mathbf{s}$ aggregates on average over the past $1/(1-\gamma) = 10$ observations of the square gradient.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Concise Implementation

Since RMSProp is a rather popular algorithm it is also available in the `Trainer` instance. All we need to do is instantiate it using an algorithm named `rmsprop`, assigning $\gamma$ to the parameter `gamma1`.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 요약 (Summary)

* RMSProp은 계수를 스케일링하기 위해 기울기의 제곱을 사용한다는 점에서 Adagrad와 매우 유사합니다.
* RMSProp은 모멘텀과 지수 이동 평균을 공유합니다. 그러나 RMSProp은 좌표별 프리컨디셔너를 조정하기 위해 이 기술을 사용합니다.
* 학습률은 실제로 실험자에 의해 스케줄링되어야 합니다.
* 계수 $\gamma$는 좌표별 스케일을 조정할 때 이력이 얼마나 긴지 결정합니다.


## 연습 문제 (Exercises)

1. $\gamma = 1$로 설정하면 실험적으로 어떤 일이 발생합니까? 왜 그렇습니까?
2. $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$를 최소화하도록 최적화 문제를 회전시키십시오. 수렴에 어떤 일이 발생합니까?
3. Fashion-MNIST에서 훈련하는 것과 같은 실제 머신러닝 문제에서 RMSProp에 어떤 일이 일어나는지 시도해 보십시오. 학습률 조정을 위한 다양한 선택으로 실험해 보십시오.
4. 최적화가 진행됨에 따라 $\gamma$를 조정하고 싶습니까? RMSProp은 이에 얼마나 민감합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab: