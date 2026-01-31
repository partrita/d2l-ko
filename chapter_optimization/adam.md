# 아담 (Adam)
:label:`sec_adam`

이 섹션에 앞서 논의한 내용에서 우리는 효율적인 최적화를 위한 여러 기술을 접했습니다. 여기서 그것들을 자세히 요약해 봅시다:

* 우리는 :numref:`sec_sgd`가 최적화 문제를 해결할 때 경사 하강법보다 더 효과적임을 보았습니다. 예를 들어 중복 데이터에 대한 고유한 복원력 때문입니다.
* 우리는 :numref:`sec_minibatch_sgd`가 벡터화를 통해 얻는 상당한 추가 효율성을 제공하며, 한 미니배치에서 더 큰 관찰 집합을 사용함을 보았습니다. 이것이 효율적인 다중 머신, 다중 GPU 및 전반적인 병렬 처리의 핵심입니다.
* :numref:`sec_momentum`은 수렴을 가속화하기 위해 과거 기울기의 이력을 집계하는 메커니즘을 추가했습니다.
* :numref:`sec_adagrad`는 계산적으로 효율적인 프리컨디셔너(preconditioner)를 허용하기 위해 좌표별 스케일링을 사용했습니다.
* :numref:`sec_rmsprop`은 학습률 조정에서 좌표별 스케일링을 분리했습니다.

Adam :cite:`Kingma.Ba.2014`은 이 모든 기술을 하나의 효율적인 학습 알고리즘으로 결합합니다. 예상대로, 이것은 딥러닝에서 사용하기에 더 견고하고 효과적인 최적화 알고리즘 중 하나로 상당히 인기를 얻은 알고리즘입니다. 하지만 문제가 없는 것은 아닙니다. 특히 :cite:`Reddi.Kale.Kumar.2019`는 Adam이 열악한 분산 제어로 인해 발산할 수 있는 상황이 있음을 보여줍니다. 후속 연구에서 :citet:`Zaheer.Reddi.Sachan.ea.2018`은 이러한 문제를 해결하는 Yogi라는 Adam의 핫픽스를 제안했습니다. 이에 대해서는 나중에 더 자세히 설명하겠습니다. 지금은 Adam 알고리즘을 검토해 봅시다.

## 알고리즘 (The Algorithm)

Adam의 핵심 구성 요소 중 하나는 지수 가중 이동 평균(leaky averaging이라고도 함)을 사용하여 기울기의 모멘텀과 2차 모멘트(second moment)를 모두 추정한다는 것입니다. 즉, 다음과 같은 상태 변수를 사용합니다.

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

여기서 $\beta_1$과 $\beta_2$는 비음수 가중치 파라미터입니다. 일반적인 선택은 $\beta_1 = 0.9$ 및 $\beta_2 = 0.999$입니다. 즉, 분산 추정치는 모멘텀 항보다 *훨씬 더 천천히* 움직입니다. $\mathbf{v}_0 = \mathbf{s}_0 = 0$으로 초기화하면 초기에 더 작은 값으로 치우친 상당한 양의 편향이 발생한다는 점에 유의하십시오. 이는 $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$라는 사실을 사용하여 항을 재정규화함으로써 해결할 수 있습니다. 그에 대응하여 정규화된 상태 변수는 다음과 같이 주어집니다.

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ 및 } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$ 

적절한 추정치가 준비되면 이제 업데이트 방정식을 작성할 수 있습니다. 먼저 RMSProp과 매우 유사한 방식으로 기울기를 재조정하여 다음을 얻습니다.

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$ 

RMSProp과 달리 우리의 업데이트는 기울기 자체가 아니라 모멘텀 $\hat{\mathbf{v}}_t$를 사용합니다. 더욱이, 재조정이 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ 대신 $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$를 사용하여 발생한다는 약간의 외관상 차이가 있습니다. 전자가 실제로 약간 더 잘 작동하므로 RMSProp에서 벗어났습니다. 일반적으로 수치적 안정성과 충실도 사이의 좋은 절충안으로 $\epsilon = 10^{-6}$을 선택합니다.

이제 업데이트를 계산하기 위한 모든 조각이 준비되었습니다. 이는 다소 실망스러울 정도로 간단하며 다음과 같은 형태의 간단한 업데이트를 가집니다.

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$ 

Adam의 설계를 검토하면 그 영감이 명확해집니다. 모멘텀과 스케일은 상태 변수에서 명확하게 보입니다. 그들의 다소 독특한 정의는 우리가 항의 편향을 제거하도록 강제합니다(이는 약간 다른 초기화 및 업데이트 조건으로 수정될 수 있습니다). 둘째, 두 항의 조합은 RMSProp이 주어지면 꽤 간단합니다. 마지막으로, 명시적인 학습률 $\eta$를 통해 수렴 문제를 해결하기 위해 단계 길이를 제어할 수 있습니다.


## 구현 (Implementation)

Adam을 처음부터 구현하는 것은 그리 어렵지 않습니다. 편의를 위해 타임스텝 카운터 $t$를 `hyperparams` 딕셔너리에 저장합니다. 그 외에는 모두 간단합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)

```

이제 모델을 훈련하기 위해 Adam을 사용할 준비가 되었습니다. $\eta = 0.01$의 학습률을 사용합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);

```

`adam`은 Gluon `trainer` 최적화 라이브러리의 일부로 제공되는 알고리즘 중 하나이므로 더 간결한 구현이 가능합니다. 따라서 Gluon에서의 구현을 위해 구성 파라미터만 전달하면 됩니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)

```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)

```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)

```

## Yogi

Adam의 문제 중 하나는 $\mathbf{s}_t$의 2차 모멘트 추정치가 급증할 때 볼록(convex) 설정에서도 수렴하지 못할 수 있다는 것입니다. 해결책으로 :citet:`Zaheer.Reddi.Sachan.ea.2018`은 $\mathbf{s}_t$에 대해 정제된 업데이트(및 초기화)를 제안했습니다. 무슨 일이 일어나고 있는지 이해하기 위해 Adam 업데이트를 다음과 같이 다시 써 봅시다.

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$ 

$\mathbf{g}_t^2$의 분산이 크거나 업데이트가 희소할 때마다, $\mathbf{s}_t$는 과거 값을 너무 빨리 잊어버릴 수 있습니다. 이에 대한 가능한 해결책은 $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$을 $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})}$으로 바꾸는 것입니다. 이제 업데이트의 크기는 더 이상 편차의 양에 의존하지 않습니다. 이것은 Yogi 업데이트를 산출합니다.

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$ 

저자들은 또한 단순히 초기 포인트별 추정치가 아니라 더 큰 초기 배치에서 모멘텀을 초기화할 것을 권장합니다. 논의에 필수적이지 않고 이것이 없어도 수렴이 꽤 잘 유지되므로 자세한 내용은 생략합니다.

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);

```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);

```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);

```

## 요약 (Summary)

* Adam은 많은 최적화 알고리즘의 기능을 상당히 견고한 업데이트 규칙으로 결합합니다.
* RMSProp을 기반으로 만들어진 Adam은 미니배치 확률적 기울기에서 EWMA를 사용합니다.
* Adam은 모멘텀과 2차 모멘트를 추정할 때 느린 시작을 조정하기 위해 편향 수정을 사용합니다.
* 분산이 큰 기울기의 경우 수렴에 문제가 발생할 수 있습니다. 이는 더 큰 미니배치를 사용하거나 $\mathbf{s}_t$에 대해 개선된 추정치를 사용하도록 전환함으로써 수정될 수 있습니다. Yogi는 그러한 대안을 제공합니다.


## 연습 문제 (Exercises)

1. 학습률을 조정하고 실험 결과를 관찰하고 분석하십시오.
2. 편향 수정이 필요하지 않도록 모멘텀 및 2차 모멘트 업데이트를 다시 작성할 수 있습니까?
3. 수렴할 때 학습률 $\eta$를 줄여야 하는 이유는 무엇입니까?
4. Adam은 발산하고 Yogi는 수렴하는 사례를 구성해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1079)
:end_tab: