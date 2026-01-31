# Adagrad
:label:`sec_adagrad`

드물게 발생하는 특성(sparse features)이 있는 학습 문제를 고려하는 것으로 시작해 봅시다.


## 희소 특성과 학습률 (Sparse Features and Learning Rates)

언어 모델을 훈련하고 있다고 상상해 보십시오. 좋은 정확도를 얻으려면 일반적으로 훈련을 계속하면서 학습률을 낮추고 싶어 하며, 보통 $\mathcal{O}(t^{-\frac{1}{2}})$ 또는 그보다 느린 속도로 낮춥니다. 이제 희소 특성, 즉 드물게만 발생하는 특성에서 모델을 훈련하는 것을 고려해 보십시오. 이는 자연어에서 흔히 볼 수 있는 현상입니다. 예를 들어 *preconditioning*이라는 단어는 *learning*이라는 단어보다 훨씬 덜 나타날 것입니다. 하지만 이는 계산 광고 및 개인화된 협업 필터링과 같은 다른 분야에서도 흔합니다. 결국 많은 것들이 소수의 사람들에게만 관심의 대상이기 때문입니다.

드문 특성과 관련된 파라미터는 해당 특성이 나타날 때만 의미 있는 업데이트를 받습니다. 감소하는 학습률이 주어지면, 흔한 특성에 대한 파라미터는 최적의 값으로 상당히 빨리 수렴하는 반면, 드문 특성에 대해서는 최적의 값을 결정하기에 충분히 자주 관찰하기 전에 이미 학습률이 너무 작아지는 상황에 처할 수 있습니다. 즉, 학습률이 흔한 특성에 대해서는 너무 느리게 감소하거나 드문 특성에 대해서는 너무 빨리 감소합니다.

이 문제를 해결하기 위한 가능한 꼼수는 특정 특성을 본 횟수를 세고 이를 학습률 조정을 위한 시계로 사용하는 것입니다. 즉, $\eta = \frac{\eta_0}{\sqrt{t + c}}$ 형태의 학습률을 선택하는 대신 $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$를 사용할 수 있습니다. 여기서 $s(i, t)$는 시간 $t$까지 관찰한 특성 $i$의 0이 아닌 값의 개수입니다. 이는 의미 있는 오버헤드 없이 구현하기가 상당히 쉽습니다. 그러나 이는 희소성이 있는 것이 아니라 기울기가 자주 매우 작고 드물게 큰 데이터의 경우에는 실패합니다. 결국 무엇을 관찰된 특성으로 간주할지 그 경계가 불분명하기 때문입니다.

:citet:`Duchi.Hazan.Singer.2011`에 의한 Adagrad는 다소 조잡한 카운터 $s(i, t)$를 이전에 관찰된 기울기의 제곱 합으로 대체함으로써 이를 해결합니다. 특히 학습률을 조정하는 수단으로 $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$를 사용합니다. 이는 두 가지 이점이 있습니다: 첫째, 기울기가 충분히 큰지 결정할 필요가 없습니다. 둘째, 기울기의 크기에 따라 자동으로 스케일이 조정됩니다. 일상적으로 큰 기울기에 해당하는 좌표는 상당히 축소되는 반면, 작은 기울기를 가진 다른 좌표는 훨씬 더 부드러운 대우를 받습니다. 실제로 이는 계산 광고 및 관련 문제에 대해 매우 효과적인 최적화 절차로 이어집니다. 하지만 이는 프리컨디셔닝(preconditioning)의 맥락에서 가장 잘 이해되는 Adagrad 고유의 몇 가지 추가적인 이점을 숨기고 있습니다.


## 프리컨디셔닝 (Preconditioning)

볼록 최적화 문제는 알고리즘의 특성을 분석하는 데 좋습니다. 결국 대부분의 비볼록 문제에 대해 의미 있는 이론적 보장을 도출하기는 어렵지만, *직관*과 *통찰*은 종종 그대로 이어지기 때문입니다. $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$를 최소화하는 문제를 살펴봅시다.

:numref:`sec_momentum`에서 보았듯이, 이 문제를 고유값 분해 $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$를 통해 다시 써서 각 좌표를 개별적으로 풀 수 있는 훨씬 단순화된 문제로 도달할 수 있습니다.

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

여기서 $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$를 사용했고 결과적으로 $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$입니다. 수정된 문제의 최소화자는 $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$이고 최소값은 $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$입니다. $\boldsymbol{\Lambda}$는 $\mathbf{Q}$의 고유값을 포함하는 대각 행렬이므로 이를 계산하는 것은 훨씬 쉽습니다.

우리가 $\mathbf{c}$를 약간 섭동시키면 $f$의 최소화자에서도 약간의 변화만 찾기를 바랄 것입니다. 불행히도 그렇지 않습니다. $\mathbf{c}$의 약간의 변화가 $\bar{\mathbf{c}}$에서도 똑같이 약간의 변화로 이어지지만, $f$(그리고 각각 $\bar{f}$)의 최소화자에 대해서는 그렇지 않습니다. 고유값 $\boldsymbol{\Lambda}_i$가 클 때마다 $\bar{x}_i$와 $\bar{f}$의 최소값에서 작은 변화만 보게 될 것입니다. 반대로 작은 $\boldsymbol{\Lambda}_i$에 대해서는 $\bar{x}_i$의 변화가 극적일 수 있습니다. 가장 큰 고유값과 가장 작은 고유값의 비율을 최적화 문제의 조건 수(condition number)라고 합니다.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$ 

조건 수 $\kappa$가 크면 최적화 문제를 정확하게 풀기가 어렵습니다. 넓은 동적 범위의 값들을 정확하게 맞추기 위해 주의를 기울여야 합니다. 우리의 분석은 명백하지만 다소 순진한 질문으로 이어집니다: 모든 고유값이 1이 되도록 공간을 왜곡하여 문제를 간단히 "고칠" 수 없을까요? 이론적으로 이는 상당히 쉽습니다: 문제를 $\mathbf{x}$에서 $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$의 문제로 다시 스케일링하기 위해 $\mathbf{Q}$의 고유값과 고유벡터만 필요합니다. 새 좌표계에서 $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$는 $\|\mathbf{z}\|^2$로 단순화될 수 있습니다. 아쉽게도 이는 다소 비실용적인 제안입니다. 고유값과 고유벡터를 계산하는 것은 일반적으로 실제 문제를 푸는 것보다 *훨씬 더* 비쌉니다.

고유값을 정확하게 계산하는 것은 비쌀 수 있지만, 이를 추측하고 다소 근사적으로라도 계산하는 것은 아무것도 하지 않는 것보다 훨씬 나을 수 있습니다. 특히 $\mathbf{Q}$의 대각 항목을 사용하여 그에 따라 다시 스케일링할 수 있습니다. 이는 고유값을 계산하는 것보다 *훨씬* 저렴합니다.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$ 

이 경우 $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$이고 특히 모든 $i$에 대해 $\tilde{\mathbf{Q}}_{ii} = 1$입니다. 대부분의 경우 이는 조건 수를 상당히 단순화합니다. 예를 들어 우리가 이전에 논의한 사례들의 경우, 문제가 축에 정렬되어 있으므로 이 방법이 당면한 문제를 완전히 제거할 것입니다.

불행히도 우리는 또 다른 문제에 직면합니다: 딥러닝에서 우리는 일반적으로 목적 함수의 2계 도함수에 접근조차 할 수 없습니다. $\mathbf{x} \in \mathbb{R}^d$에 대해 미니배치에서도 2계 도함수는 계산하는 데 $\mathcal{O}(d^2)$ 공간과 작업이 필요할 수 있으므로 실제로는 불가능합니다. Adagrad의 기발한 아이디어는 계산하기 상대적으로 저렴하면서도 효과적인 헤시안(Hessian)의 포착하기 어려운 대각선에 대한 대리물로 기울기 자체의 크기를 사용하는 것입니다.

이것이 왜 작동하는지 확인하기 위해 $\bar{f}(\bar{\mathbf{x}})$를 살펴봅시다. 우리는 다음을 갖습니다.

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$ 

여기서 $\bar{\mathbf{x}}_0$는 $\bar{f}$의 최소화자입니다. 따라서 기울기의 크기는 $\boldsymbol{\Lambda}$와 최적성으로부터의 거리 모두에 의존합니다. $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$가 변하지 않는다면, 이것이 필요한 전부일 것입니다. 결국 이 경우 기울기 $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$의 크기로 충분하기 때문입니다. AdaGrad는 확률적 경사 하강법 알고리즘이므로, 최적점에서도 0이 아닌 분산을 가진 기울기를 보게 될 것입니다. 결과적으로 우리는 헤시안의 스케일에 대한 저렴한 대리물로 기울기의 분산을 안전하게 사용할 수 있습니다. 철저한 분석은 이 섹션의 범위를 벗어납니다(여러 페이지가 될 것입니다). 자세한 내용은 :cite:`Duchi.Hazan.Singer.2011`를 참조하십시오.


## 알고리즘 (The Algorithm)

위의 논의를 공식화해 봅시다. 우리는 다음과 같이 과거 기울기 분산을 누적하기 위해 변수 $\mathbf{s}_t$를 사용합니다.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

여기서 연산은 좌표별로 적용됩니다. 즉, $\mathbf{v}^2$은 항목 $v_i^2$을 갖습니다. 마찬가지로 $\frac{1}{\sqrt{v}}$은 항목 $\frac{1}{\sqrt{v_i}}$를 갖고 $\mathbf{u} \cdot \mathbf{v}$는 항목 $u_i v_i$를 갖습니다. 이전과 마찬가지로 $\eta$는 학습률이고 $\epsilon$은 $0$으로 나누지 않도록 보장하는 가산 상수입니다. 마지막으로 $\mathbf{s}_0 = \mathbf{0}$으로 초기화합니다.

모멘텀의 경우와 마찬가지로 우리는 보조 변수를 추적해야 하며, 이 경우 좌표당 개별 학습률을 허용합니다. 이는 SGD에 비해 Adagrad의 비용을 크게 증가시키지 않는데, 단순히 주요 비용이 일반적으로 $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$와 그 도함수를 계산하는 것이기 때문입니다.

$\\mathbf{s}_t$에 제곱된 기울기를 누적하면 $\\mathbf{s}_t$가 본질적으로 선형 속도로 성장한다는 점에 유의하십시오(기울기가 처음에 줄어들기 때문에 실제로는 선형보다 약간 느립니다). 이는 좌표별로 조정되기는 하지만 $\\mathcal{O}(t^{-\\frac{1}{2}})$ 학습률로 이어집니다. 볼록 문제의 경우 이는 완벽하게 적절합니다. 하지만 딥러닝에서는 학습률을 다소 더 천천히 낮추고 싶을 수도 있습니다. 이로 인해 후속 장에서 논의할 여러 Adagrad 변형이 생겨났습니다. 지금은 이차 볼록 문제에서 이것이 어떻게 작동하는지 봅시다. 이전과 동일한 문제를 사용합니다.

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

이전에 사용한 것과 동일한 학습률 $\eta = 0.4$를 사용하여 Adagrad를 구현할 것입니다. 보시다시피 독립 변수의 반복 궤적이 더 매끄럽습니다. 그러나 $\boldsymbol{s}_t$의 누적 효과로 인해 학습률이 지속적으로 감소하므로 독립 변수는 반복의 후반 단계에서 많이 이동하지 않습니다.

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

학습률을 $2$로 높이면 훨씬 더 나은 동작을 볼 수 있습니다. 이는 노이즈가 없는 경우에도 학습률 감소가 다소 공격적일 수 있으며 파라미터가 적절하게 수렴하도록 보장해야 함을 이미 시사합니다.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```


## 밑바닥부터 구현하기 (Implementation from Scratch)

모멘텀 방법과 마찬가지로 Adagrad는 파라미터와 동일한 모양의 상태 변수를 유지해야 합니다.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd`의 실험과 비교하여 모델을 훈련하기 위해 더 큰 학습률을 사용합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```


## 간결한 구현 (Concise Implementation)

알고리즘 `adagrad`의 `Trainer` 인스턴스를 사용하여 Gluon에서 Adagrad 알고리즘을 호출할 수 있습니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```


## 요약 (Summary)

* Adagrad는 좌표별로 동적으로 학습률을 감소시킵니다.
* 진전이 얼마나 빨리 달성되는지 조정하는 수단으로 기울기의 크기를 사용합니다 - 큰 기울기를 가진 좌표는 더 작은 학습률로 보상받습니다.
* 딥러닝 문제에서는 메모리 및 계산 제약으로 인해 정확한 2계 도함수를 계산하는 것이 일반적으로 불가능합니다. 기울기는 유용한 대리물이 될 수 있습니다.
* 최적화 문제의 구조가 다소 고르지 않다면 Adagrad가 왜곡을 완화하는 데 도움이 될 수 있습니다.
* Adagrad는 드물게 발생하는 항에 대해 학습률이 더 천천히 감소해야 하는 희소 특성에 특히 효과적입니다.
* 딥러닝 문제에서 Adagrad는 가끔 학습률을 줄이는 데 너무 공격적일 수 있습니다. :numref:`sec_adam`의 맥락에서 이를 완화하기 위한 전략을 논의할 것입니다.


## 연습 문제 (Exercises)

1. 직교 행렬 $\mathbf{U}$와 벡터 $\mathbf{c}$에 대해 다음이 성립함을 증명하십시오: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. 이것이 왜 변수의 직교 변환 후에 섭동의 크기가 변하지 않음을 의미할까요?
2. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$에 대해 Adagrad를 시도해 보고, 목적 함수가 45도 회전된 경우, 즉 $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$에 대해서도 시도해 보십시오. 다르게 행동하나요?
3. 행렬 $\mathbf{M}$의 고유값 $\lambda_i$가 적어도 하나의 $j$ 선택에 대해 $|λ_i - Μ_{jj}| ≤ ∑_{k 
eq j} |Μ_{jk}|$를 만족한다는 [거시고린 원판 정리(Gerschgorin's circle theorem)](https://ko.wikipedia.org/wiki/%EA%B1%B0%EC%8B%9C%EA%B3%A0%EB%A6%B0_%EC%9B%90%ED%8C%90_%EC%A0%95%EB%A6%AC)를 증명하십시오.
4. 거시고린 정리는 대각 프리컨디셔닝된 행렬 $\\textrm{diag}^{-\\frac{1}{2}}(\\mathbf{M}) \\mathbf{M} \\textrm{diag}^{-\\frac{1}{2}}(\\mathbf{M})$의 고유값에 대해 무엇을 알려주나요?
5. Fashion-MNIST에 적용된 :numref:`sec_lenet`과 같은 적절한 심층 네트워크에 대해 Adagrad를 시도해 보십시오.
6. 학습률 감쇠를 덜 공격적으로 만들기 위해 Adagrad를 어떻게 수정해야 할까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab: