# 모멘텀 (Momentum)
:label:`sec_momentum`

:numref:`sec_sgd`에서 우리는 확률적 경사 하강법을 수행할 때, 즉 기울기의 노이즈가 섞인 변형만 사용할 수 있는 최적화를 수행할 때 어떤 일이 일어나는지 검토했습니다. 특히 노이즈가 있는 기울기의 경우 노이즈에 직면하여 학습률을 선택할 때 각별히 주의해야 한다는 점을 알아차렸습니다. 너무 빨리 줄이면 수렴이 멈춥니다. 너무 관대하면 노이즈가 계속해서 최적성에서 멀어지게 만들기 때문에 충분히 좋은 솔루션으로 수렴하지 못합니다.

## 기초 (Basics)

이 섹션에서는 특히 실전에서 흔히 볼 수 있는 특정 유형의 최적화 문제에 대해 더 효과적인 최적화 알고리즘을 탐구할 것입니다.


### 누적 평균 (Leaky Averages)

이전 섹션에서는 계산을 가속화하기 위한 수단으로 미니배치 SGD를 논의했습니다. 또한 기울기를 평균화하면 분산의 양이 줄어든다는 좋은 부수 효과가 있었습니다. 미니배치 확률적 경사 하강법은 다음과 같이 계산될 수 있습니다.

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

표기법을 단순하게 유지하기 위해, 여기서 $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$를 $t-1$ 시간에 업데이트된 가중치를 사용하여 샘플 $i$에 대한 확률적 경사 하강법으로 사용했습니다. 미니배치에서 기울기를 평균화하는 것을 넘어 분산 감소의 효과를 누릴 수 있다면 좋을 것입니다. 이 작업을 수행하는 한 가지 옵션은 기울기 계산을 "누적 평균(leaky average)"으로 대체하는 것입니다.

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

어떤 $\beta \in (0, 1)$에 대해서입니다. 이는 효과적으로 순간적인 기울기를 여러 *과거* 기울기에 대해 평균화된 것으로 대체합니다. $\mathbf{v}$는 *속도(velocity)*라고 불립니다. 이는 목적 함수 지형을 따라 굴러가는 무거운 공이 과거의 힘을 통합하는 방식과 유사하게 과거 기울기를 누적합니다. 무슨 일이 일어나고 있는지 더 자세히 보기 위해 $\mathbf{v}_t$를 재귀적으로 다음과 같이 확장해 봅시다.

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

큰 $\beta$는 장기 평균에 해당하고, 작은 $\beta$는 기울기 방법에 비해 약간의 보정만 하는 것에 해당합니다. 새로운 기울기 대체물은 더 이상 특정 인스턴스에서 가장 가파른 하강 방향을 가리키지 않고 대신 과거 기울기의 가중 평균 방향을 가리킵니다. 이를 통해 실제로 기울기를 계산하는 비용 없이 배치에 대해 평균화하는 대부분의 이점을 실현할 수 있습니다. 나중에 이 평균화 절차를 더 자세히 살펴볼 것입니다.

위의 추론은 이제 모멘텀이 있는 기울기와 같은 *가속된(accelerated)* 기울기 방법으로 알려진 것의 기초를 형성했습니다. 이들은 최적화 문제가 조건이 나쁜 경우(즉, 일부 방향에서는 진전이 다른 방향보다 훨씬 느려 좁은 협곡과 닮은 경우) 훨씬 더 효과적이라는 추가적인 이점이 있습니다. 더욱이, 그들은 후속 기울기에 대해 평균화하여 더 안정적인 하강 방향을 얻을 수 있게 해 줍니다. 실제로 노이즈가 없는 볼록 문제에 대해서도 가속 측면은 모멘텀이 작동하는 이유이자 매우 잘 작동하는 핵심 이유 중 하나입니다.

예상대로 모멘텀은 그 효능 때문에 딥러닝 최적화 및 그 이상의 분야에서 잘 연구된 주제입니다. 심층 분석과 대화형 애니메이션은 :citet:`Goh.2017`의 아름다운 [설명 기사](https://distill.pub/2017/momentum/)를 참조하십시오. 이는 :citet:`Polyak.1964`에 의해 제안되었습니다. :citet:`Nesterov.2018`은 볼록 최적화 맥락에서 상세한 이론적 논의를 담고 있습니다. 딥러닝에서의 모멘텀은 오랫동안 유익한 것으로 알려져 왔습니다. 자세한 내용은 :citet:`Sutskever.Martens.Dahl.ea.2013`의 토론을 참조하십시오.

### 조건이 나쁜 문제 (An Ill-conditioned Problem)

모멘텀 방법의 기하학적 특성을 더 잘 이해하기 위해, 상당히 덜 유쾌한 목적 함수를 사용하여 경사 하강법을 다시 살펴봅니다. :numref:`sec_gd`에서 우리는 $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, 즉 적당히 왜곡된 타원체 목적 함수를 사용했음을 상기하십시오. 우리는 이 함수를 $x_1$ 방향으로 늘려서 다음과 같이 더 왜곡합니다.

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

이전과 마찬가지로 $f$는 $(0, 0)$에서 최소값을 갖습니다. 이 함수는 $x_1$ 방향으로 *매우* 평평합니다. 이 새로운 함수에 대해 이전과 같이 경사 하강법을 수행하면 어떤 일이 일어나는지 봅시다. 학습률을 $0.4$로 선택합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

구조적으로 $x_2$ 방향의 기울기는 수평 $x_1$ 방향보다 *훨씬* 높고 훨씬 더 빠르게 변합니다. 따라서 우리는 두 가지 바람직하지 않은 선택 사이에 갇히게 됩니다: 작은 학습률을 선택하면 $x_2$ 방향에서 해가 발산하지 않도록 보장하지만 $x_1$ 방향에서는 느린 수렴을 겪게 됩니다. 반대로 큰 학습률을 사용하면 $x_1$ 방향에서는 빠르게 진행되지만 $x_2$에서는 발산합니다. 아래 예제는 학습률을 $0.4$에서 $0.6$으로 약간 올린 후에도 어떤 일이 일어나는지 보여줍니다. $x_1$ 방향의 수렴은 개선되지만 전체적인 솔루션 품질은 훨씬 나빠집니다.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 모멘텀 방법 (The Momentum Method)

모멘텀 방법은 위에서 설명한 경사 하강법 문제를 해결할 수 있게 해 줍니다. 위의 최적화 궤적을 보면 과거의 기울기를 평균화하는 것이 잘 작동할 것이라고 직관적으로 느낄 수 있습니다. 결국 $x_1$ 방향에서는 이것이 잘 정렬된 기울기들을 집계하여 매 단계마다 우리가 이동하는 거리를 늘릴 것입니다. 반대로 기울기가 진동하는 $x_2$ 방향에서는 집계된 기울기가 서로 상쇄되는 진동으로 인해 단계 크기를 줄일 것입니다.

기울기 $\mathbf{g}_t$ 대신 $\mathbf{v}_t$를 사용하면 다음과 같은 업데이트 방정식이 생성됩니다.

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$eta = 0$이면 일반적인 경사 하강법을 회복한다는 점에 유의하십시오. 수학적 특성을 깊이 파고들기 전에 알고리즘이 실전에서 어떻게 행동하는지 간단히 살펴봅시다.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

보시다시피, 이전에 사용한 것과 동일한 학습률로도 모멘텀은 여전히 잘 수렴합니다. 모멘텀 파라미터를 줄이면 어떻게 되는지 봅시다. 이를 절반인 $eta = 0.25$로 줄이면 거의 수렴하지 않는 궤적이 생성됩니다. 그럼에도 불구하고 모멘텀이 없을 때(솔루션이 발산할 때)보다 훨씬 낫습니다.

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

모멘텀을 확률적 경사 하강법, 특히 미니배치 확률적 경사 하강법과 결합할 수 있다는 점에 유의하십시오. 유일한 변경 사항은 그 경우 기울기 $\mathbf{g}_{t, t-1}$을 $\mathbf{g}_t$로 대체하는 것입니다. 마지막으로 편의를 위해 시간 $t=0$에서 $\mathbf{v}_0 = 0$으로 초기화합니다. 누적 평균이 실제로 업데이트에 어떤 영향을 미치는지 살펴봅시다.

### 유효 샘플 가중치 (Effective Sample Weight)

$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$임을 상기하십시오. 극한에서 항들은 $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$로 합산됩니다. 즉, 경사 하강법이나 확률적 경사 하강법에서 크기 $\eta$의 단계를 밟는 대신, 잠재적으로 훨씬 더 잘 행동하는 하강 방향을 다루면서 동시에 $\frac{\eta}{1-\beta}$ 크기의 단계를 밟는 것입니다. 이는 한 번에 두 가지 이점을 제공합니다. 서로 다른 $\beta$ 선택에 대해 가중치가 어떻게 행동하는지 설명하기 위해 아래 다이어그램을 고려해 보십시오.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 실전 실험 (Practical Experiments)

모멘텀이 실전에서 어떻게 작동하는지, 즉 적절한 최적화 프로그램 내에서 사용될 때 어떻게 작동하는지 살펴봅시다. 이를 위해 좀 더 확장 가능한 구현이 필요합니다.

### 밑바닥부터 구현하기 (Implementation from Scratch)

(미니배치) 확률적 경사 하강법과 비교하여 모멘텀 방법은 보조 변수 집합, 즉 속도를 유지해야 합니다. 이는 기울기(및 최적화 문제의 변수)와 동일한 모양을 갖습니다. 아래 구현에서 우리는 이러한 변수들을 `states`라고 부릅니다.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

이것이 실전에서 어떻게 작동하는지 봅시다.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

모멘텀 하이퍼파라미터 `momentum`을 0.9로 늘리면 $\frac{1}{1 - 0.9} = 10$의 상당히 더 큰 유효 샘플 크기에 해당합니다. 상황을 제어하기 위해 학습률을 $0.01$로 약간 낮춥니다.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

학습률을 더 낮추면 매끄럽지 않은 최적화 문제의 모든 문제가 해결됩니다. $0.005$로 설정하면 좋은 수렴 특성을 얻을 수 있습니다.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 간결한 구현 (Concise Implementation)

표준 `sgd` 솔버에 이미 모멘텀이 내장되어 있으므로 Gluon에서 할 일은 거의 없습니다. 일치하는 파라미터를 설정하면 매우 유사한 궤적이 생성됩니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 이론적 분석 (Theoretical Analysis)

지금까지 $f(x) = 0.1 x_1^2 + 2 x_2^2$의 2D 예제는 다소 인위적으로 보였을 수 있습니다. 우리는 이것이 적어도 볼록 이차 목적 함수를 최소화하는 경우 마주칠 수 있는 문제 유형을 상당히 잘 나타낸다는 것을 알게 될 것입니다.

### 이차 볼록 함수 (Quadratic Convex Functions)

함수를 고려해 봅시다.

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

이것은 일반적인 이차 함수입니다. 양의 고유값을 가진 행렬인 양의 정부호(positive definite) 행렬 $\mathbf{Q} \succ 0$에 대해, 이는 최소값 $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$와 함께 $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$에서 최소화자를 갖습니다. 따라서 우리는 $h$를 다음과 같이 다시 쓸 수 있습니다.

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^	op \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$ 

기울기는 $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$로 주어집니다. 즉, $\mathbf{x}$와 최소화자 사이의 거리에 $\mathbf{Q}$를 곱한 값입니다. 결과적으로 속도 또한 $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ 항들의 선형 결합입니다.

$\\mathbf{Q}$는 양의 정부호이므로 직교(회전) 행렬 $\\mathbf{O}$와 양의 고유값의 대각 행렬 $\\boldsymbol{\\Lambda}$를 통해 $\\mathbf{Q} = \\mathbf{O}^\top \\boldsymbol{\\Lambda} \\mathbf{O}$로 고유계 분해가 가능합니다. 이를 통해 $\\mathbf{x}$에서 $\\mathbf{z} \\stackrel{\\textrm{def}}{=} \\mathbf{O} (\\mathbf{x} - \\mathbf{Q}^{-1} \\mathbf{c})$로 변수 변환을 수행하여 훨씬 단순화된 식을 얻을 수 있습니다.

$$h(\\mathbf{z}) = \frac{1}{2} \\mathbf{z}^\top \\boldsymbol{\\Lambda} \\mathbf{z} + b'.$$

여기서 $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$입니다. $\\mathbf{O}$는 직교 행렬일 뿐이므로 기울기를 의미 있게 섭동시키지 않습니다. $\\mathbf{z}$의 관점에서 표현하면 경사 하강법은 다음과 같이 됩니다.

$$\\mathbf{z}_t = \\mathbf{z}_{t-1} - \\boldsymbol{\\Lambda} \\mathbf{z}_{t-1} = (\\mathbf{I} - \\boldsymbol{\\Lambda}) \\mathbf{z}_{t-1}.$$ 

이 식에서 중요한 사실은 경사 하강법이 서로 다른 고유 공간 사이에서 *섞이지 않는다*는 것입니다. 즉, $\\mathbf{Q}$의 고유계 관점에서 표현할 때 최적화 문제는 좌표별 방식으로 진행됩니다. 이는 다음에도 적용됩니다.

$$egin{aligned}
\\mathbf{v}_t & = \\beta \\mathbf{v}_{t-1} + \\boldsymbol{\\Lambda} \\mathbf{z}_{t-1} \\
\\mathbf{z}_t & = \\mathbf{z}_{t-1} - \\eta (\\beta \\mathbf{v}_{t-1} + \\boldsymbol{\\Lambda} \\mathbf{z}_{t-1}) \\
    & = (\\mathbf{I} - \\eta \\boldsymbol{\\Lambda}) \\mathbf{z}_{t-1} - \\eta \\beta \\mathbf{v}_{t-1}.
\end{aligned}$$

이를 통해 우리는 방금 다음 정리를 증명했습니다: 볼록 이차 함수에 대해 모멘텀이 있거나 없는 경사 하강법은 이차 행렬의 고유벡터 방향으로의 좌표별 최적화로 분해됩니다.

### 스칼라 함수 (Scalar Functions)

위의 결과가 주어졌을 때 함수 $f(x) = \frac{\\lambda}{2} x^2$를 최소화할 때 어떤 일이 일어나는지 봅시다. 경사 하강법의 경우

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

$|1 - \eta \lambda| < 1$일 때마다 이 최적화는 지수 속도로 수렴합니다. $t$단계 후에 $x_t = (1 - \eta \lambda)^t x_0$를 갖기 때문입니다. 이는 $\\eta \lambda = 1$이 될 때까지 학습률 $\\eta$를 높임에 따라 초기에 수렴 속도가 어떻게 개선되는지 보여줍니다. 그 이상에서는 발산하기 시작하고 $\\eta \lambda > 2$에 대해 최적화 문제는 발산합니다.

```{.python .input}
#@tab all
 lambdas = [0.1, 1, 10, 19]
 eta = 0.1
 d2l.set_figsize((6, 4))
 for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
 d2l.plt.xlabel('time')
 d2l.plt.legend();
```

모멘텀의 경우 수렴을 분석하기 위해 업데이트 방정식을 두 개의 스칼라(하나는 $x$용, 하나는 속도 $v$용)로 다시 쓰는 것으로 시작합니다. 이는 다음을 산출합니다.

$$ 
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$ 

우리는 수렴 동작을 지배하는 $2 \times 2$ 행렬을 나타내기 위해 $\\mathbf{R}$을 사용했습니다. $t$단계 후에 초기 선택 $[v_0, x_0]$은 $\\mathbf{R}(\\beta, \\eta, \\lambda)^t [v_0, x_0]$이 됩니다. 따라서 수렴 속도를 결정하는 것은 $\\mathbf{R}$의 고유값에 달려 있습니다. 멋진 애니메이션은 :citet:`Goh.2017`의 [Distill 포스트](https://distill.pub/2017/momentum/)를, 상세한 분석은 :citet:`Flammarion.Bach.2015`를 참조하십시오. $0 < \eta \lambda < 2 + 2 \beta$에서 속도가 수렴함을 보일 수 있습니다. 이는 경사 하강법의 $0 < \eta \lambda < 2$와 비교할 때 가능한 파라미터의 더 넓은 범위입니다. 또한 일반적으로 큰 $\\beta$ 값이 바람직함을 시사합니다. 추가 세부 사항은 상당한 양의 기술적 세부 사항을 필요로 하므로 관심 있는 독자는 원본 출판물을 참고하시기 바랍니다.

## 요약 (Summary)

* 모멘텀은 기울기를 과거 기울기에 대한 누적 평균으로 대체합니다. 이는 수렴을 크게 가속화합니다.
* 노이즈가 없는 경사 하강법과 (노이즈가 있는) 확률적 경사 하강법 모두에 바람직합니다.
* 모멘텀은 확률적 경사 하강법에서 훨씬 더 발생하기 쉬운 최적화 프로세스의 정체를 방지합니다.
* 과거 데이터의 지수적 가중치 감소로 인해 유효 기울기 수는 $\frac{1}{1-\beta}$로 주어집니다.
* 볼록 이차 문제의 경우 이를 상세히 명시적으로 분석할 수 있습니다.
* 구현은 꽤 간단하지만 추가적인 상태 벡터(속도 $\\mathbf{v}$)를 저장해야 합니다.

## 연습 문제 (Exercises)

1. 다른 모멘텀 하이퍼파라미터와 학습률 조합을 사용하고 다양한 실험 결과를 관찰 및 분석해 보십시오.
2. 여러 고유값이 있는 이차 문제, 즉 $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$(예: $\\lambda_i = 2^{-i}$)에 대해 경사 하강법과 모멘텀을 시도해 보십시오. 초기화 $x_i = 1$에 대해 $x$ 값이 어떻게 감소하는지 플롯하십시오.
3. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$에 대한 최소값과 최소화자를 유도하십시오.
4. 모멘텀과 함께 확률적 경사 하강법을 수행할 때 무엇이 변합니까? 모멘텀과 함께 미니배치 확률적 경사 하강법을 사용할 때 어떤 일이 일어납니까? 파라미터로 실험해 보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab: