# Adadelta
:label:`sec_adadelta`

Adadelta는 AdaGrad(:numref:`sec_adagrad`)의 또 다른 변형입니다. 주요 차이점은 학습률이 좌표에 적응하는 정도를 줄인다는 점입니다. 더욱이, 전통적으로 Adadelta는 미래의 변화를 위한 보정으로 변화량 자체를 사용하기 때문에 학습률이 없는 것으로 지칭되기도 했습니다. 이 알고리즘은 :citet:`Zeiler.2012`에서 제안되었습니다. 지금까지의 이전 알고리즘들에 대한 논의를 고려할 때 상당히 간단합니다.

## 알고리즘 (The Algorithm)

요컨대, Adadelta는 두 개의 상태 변수를 사용합니다. $\mathbf{s}_t$는 기울기 2차 모멘트의 누적 평균(leaky average)을 저장하고, $\Delta\mathbf{x}_t$는 모델 자체의 파라미터 변화량의 2차 모멘트의 누적 평균을 저장합니다. 다른 논문 및 구현과의 호환성을 위해 저자들의 원래 표기법과 명명법을 사용한다는 점에 유의하십시오(모멘텀, Adagrad, RMSProp 및 Adadelta에서 동일한 목적을 수행하는 파라미터를 나타내기 위해 서로 다른 그리스 변수를 사용해야 할 다른 실제 이유는 없습니다).

Adadelta의 기술적 세부 사항은 다음과 같습니다. 파라미터가 $\rho$일 때, :numref:`sec_rmsprop`와 유사하게 다음과 같은 누적 업데이트를 얻습니다.

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop`와의 차이점은 재조정된 기울기 $\mathbf{g}_t'$를 사용하여 업데이트를 수행한다는 것입니다. 즉,

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \
\end{aligned}$$

그렇다면 재조정된 기울기 $\mathbf{g}_t'$는 무엇일까요? 다음과 같이 계산할 수 있습니다.

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \
\end{aligned}$$

여기서 $\Delta \mathbf{x}_{t-1}$은 제곱된 재조정 기울기 $\mathbf{g}_t'$의 누적 평균입니다. $\Delta \mathbf{x}_{0}$를 $0$으로 초기화하고 각 단계에서 $\mathbf{g}_t'$를 사용하여 업데이트합니다. 즉,

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

그리고 수치적 안정성을 유지하기 위해 $\epsilon$($10^{-5}$와 같은 작은 값)이 추가됩니다.



## 구현 (Implementation)

Adadelta는 각 변수에 대해 $\mathbf{s}_t$와 $\Delta\mathbf{x}_t$라는 두 개의 상태 변수를 유지해야 합니다. 이는 다음과 같은 구현으로 이어집니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # [:]를 통한 제자리(in-place) 업데이트
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # [:]를 통한 제자리 업데이트
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$를 선택하는 것은 각 파라미터 업데이트에 대해 반감기(half-life time) 10에 해당합니다. 이는 상당히 잘 작동하는 경향이 있습니다. 다음과 같은 동작을 얻습니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

간결한 구현을 위해 고수준 API의 Adadelta 알고리즘을 사용합니다. 이는 훨씬 더 압축된 호출을 위한 다음 한 줄의 코드를 생성합니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta는 기본 학습률에서 수렴하지 않지만
# lr = 5.0에서는 수렴합니다
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## 요약 (Summary)

* Adadelta에는 학습률 파라미터가 없습니다. 대신 파라미터 자체의 변화율을 사용하여 학습률을 조정합니다.
* Adadelta는 기울기 2차 모멘트와 파라미터 변화량을 저장하기 위해 두 개의 상태 변수가 필요합니다.
* Adadelta는 적절한 통계의 실행 추정치를 유지하기 위해 누적 평균(leaky averages)을 사용합니다.

## 연습 문제 (Exercises)

1. $\rho$ 값을 조정해 보십시오. 어떻게 됩니까?
2. $\mathbf{g}_t'$를 사용하지 않고 알고리즘을 구현하는 방법을 보여주십시오. 이것이 왜 좋은 아이디어일까요?
3. Adadelta는 정말로 학습률이 없나요? Adadelta를 무너뜨리는 최적화 문제를 찾을 수 있을까요?
4. Adadelta를 Adagrad 및 RMS prop과 비교하여 수렴 동작을 논의해 보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab: