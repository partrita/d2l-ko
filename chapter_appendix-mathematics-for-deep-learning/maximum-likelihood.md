# 최대 우도 (Maximum Likelihood)
:label:`sec_maximum_likelihood`

머신러닝에서 가장 흔히 마주치는 사고방식 중 하나는 최대 우도 관점입니다. 이것은 알려지지 않은 파라미터가 있는 확률 모델로 작업할 때, 데이터를 가장 높은 확률로 만드는 파라미터가 가장 가능성 있는 것이라는 개념입니다.

## 최대 우도 원칙 (The Maximum Likelihood Principle)

생각해보면 도움이 될 수 있는 베이지안 해석이 있습니다. 파라미터 $\boldsymbol{\theta}$를 가진 모델과 데이터 예제 모음 $X$가 있다고 가정합시다. 구체적으로, $\boldsymbol{\theta}$가 동전을 던졌을 때 앞면이 나올 확률을 나타내는 단일 값이고, $X$가 독립적인 동전 던지기 시퀀스라고 상상할 수 있습니다. 이 예제는 나중에 자세히 살펴볼 것입니다.

우리 모델의 파라미터에 대해 가장 가능성 있는 값을 찾고 싶다면, 이는 다음을 찾고 싶다는 의미입니다.

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$ :eqlabel:`eq_max_like`

베이즈 규칙에 의해 이는 다음과 같습니다.

$$ \mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}. $$ 

데이터를 생성하는 파라미터에 구애받지 않는 확률인 $P(X)$ 식은 $\boldsymbol{\theta}$에 전혀 의존하지 않으므로, $\boldsymbol{\theta}$의 최선의 선택을 바꾸지 않고도 삭제할 수 있습니다. 마찬가지로, 우리는 이제 어떤 파라미터 세트가 다른 것보다 더 낫다는 사전 가정이 없다고 상정할 수 있으므로, $P(\boldsymbol{\theta})$도 세타에 의존하지 않는다고 선언할 수 있습니다! 예를 들어 동전 던지기 예제에서 앞면이 나올 확률은 공정한지 여부에 대한 사전 믿음 없이 $[0,1]$의 어떤 값이라도 될 수 있습니다(종종 *무정보 사전 분포(uninformative prior)*라고 함). 따라서 베이즈 규칙의 적용은 $\boldsymbol{\theta}$의 최선의 선택이 $\boldsymbol{\theta}$에 대한 최대 우도 추정치임을 보여줍니다.

$$ \hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}). $$ 

일반적인 용어로서, 파라미터가 주어졌을 때 데이터의 확률($P(X \mid \boldsymbol{\theta})$)을 *우도(likelihood)*라고 합니다.

### 구체적인 예 (A Concrete Example)

구체적인 예에서 이것이 어떻게 작동하는지 살펴봅시다. 동전 던지기가 앞면일 확률을 나타내는 단일 파라미터 $\theta$가 있다고 가정합시다. 그러면 뒷면이 나올 확률은 $1-\theta$이므로, 관찰된 데이터 $X$가 앞면 $n_H$번과 뒷면 $n_T$번인 시퀀스라면, 독립 확률은 곱해진다는 사실을 사용하여 다음을 알 수 있습니다.

$$ P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}. $$ 

동전을 $13$번 던져서 "HHHTHTTHHHHHT" 시퀀스를 얻었다면($n_H = 9, n_T = 4$), 다음과 같음을 알 수 있습니다.

$$ P(X \mid \theta) = \theta^9(1-\theta)^4. $$ 

이 예제의 한 가지 좋은 점은 답을 미리 알고 있다는 것입니다. 실제로 말로 "동전을 13번 던져서 9번이 앞면이 나왔다면, 동전이 앞면으로 나올 확률에 대한 우리의 최선의 추측은 무엇일까요?"라고 묻는다면 누구나 정확하게 $9/13$라고 추측할 것입니다. 이 최대 우도 방법이 우리에게 줄 것은 훨씬 더 복잡한 상황으로 일반화될 수 있는 방식으로 제1원리로부터 그 숫자를 얻는 방법입니다.

우리 예제의 경우, $P(X \mid \theta)$의 플롯은 다음과 같습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

이것은 우리가 예상한 $9/13 \approx 0.7\ldots$ 근처 어딘가에서 최대값을 갖습니다. 정확히 그곳에 있는지 확인하기 위해 미적분학으로 눈을 돌릴 수 있습니다. 최대값에서 함수의 기울기는 평평하다는 점에 유의하십시오. 따라서 우리는 도함수가 0인 $\theta$ 값을 찾고 가장 높은 확률을 주는 값을 찾음으로써 최대 우도 추정치 :eqref:`eq_max_like`를 찾을 수 있습니다. 다음과 같이 계산합니다.

$$ \begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\ & = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\ & = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\ & = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned} $$ 

이는 세 가지 해를 갖습니다: $0, 1, 9/13$. 처음 두 개는 우리 시퀀스에 확률 0을 할당하므로 최대값이 아니라 최소값임이 분명합니다. 마지막 값은 우리 시퀀스에 0이 아닌 확률을 할당하므로 최대 우도 추정치 $\hat \theta = 9/13$여야 합니다.

## 수치 최적화와 음의 로그 우도 (Numerical Optimization and the Negative Log-Likelihood)

이전 예제는 좋지만, 수십억 개의 파라미터와 데이터 예제가 있다면 어떨까요?

먼저, 모든 데이터 예제가 독립적이라는 가정을 한다면, 많은 확률의 곱이 되기 때문에 우도 자체를 실질적으로 고려할 수 없다는 점에 유의하십시오. 실제로 각 확률은 $[0,1]$에 있고, 가령 전형적인 값이 약 $1/2$라면, $(1/2)^{1000000000}$의 곱은 기계 정밀도보다 훨씬 낮습니다. 우리는 그것을 직접 다룰 수 없습니다.

하지만 로그가 곱을 합으로 바꾼다는 사실을 상기하면 다음과 같습니다.

$$ \log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots $$ 

이 숫자는 단정밀도 32비트 부동 소수점 내에도 완벽하게 들어맞습니다. 따라서 우리는 다음과 같은 *로그 우도(log-likelihood)*를 고려해야 합니다.

$$ \log(P(X \mid \boldsymbol{\theta})). $$ 

함수 $x \mapsto \log(x)$가 증가 함수이므로, 우도를 최대화하는 것은 로그 우도를 최대화하는 것과 같습니다. 실제로 :numref:`sec_naive_bayes`에서 우리는 나이브 베이즈 분류기의 특정 예제를 다룰 때 이 추론이 적용되는 것을 보게 될 것입니다.

우리는 종종 손실을 최소화하고 싶은 손실 함수로 작업합니다. 우리는 최대 우도를 $-\log(P(X \mid \boldsymbol{\theta}))$를 취함으로써 손실 최소화로 바꿀 수 있으며, 이것이 *음의 로그 우도(negative log-likelihood)*입니다.

이를 설명하기 위해 이전의 동전 던지기 문제를 고려하고 폐쇄형 해를 모른다고 가정해 봅시다. 우리는 다음을 계산할 수 있습니다.

$$ -\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)). $$ 

이것은 코드로 작성될 수 있으며 수십억 번의 동전 던지기에 대해서도 자유롭게 최적화될 수 있습니다.

```{.python .input}
#@tab mxnet
# 데이터 설정
n_H = 8675309
n_T = 256245

# 파라미터 초기화
theta = np.array(0.5)
theta.attach_grad()

# 경사 하강법 수행
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# 출력 확인
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# 데이터 설정
n_H = 8675309
n_T = 256245

# 파라미터 초기화
theta = torch.tensor(0.5, requires_grad=True)

# 경사 하강법 수행
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# 출력 확인
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# 데이터 설정
n_H = 8675309
n_T = 256245

# 파라미터 초기화
theta = tf.Variable(tf.constant(0.5))

# 경사 하강법 수행
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# 출력 확인
theta, n_H / (n_H + n_T)
```

수치적 편의성만이 사람들이 음의 로그 우도를 사용하기 좋아하는 유일한 이유는 아닙니다. 그것이 선호되는 다른 여러 이유가 있습니다.



로그 우도를 고려하는 두 번째 이유는 미적분 규칙의 단순화된 적용입니다. 위에서 논의한 대로 독립성 가정 때문에 머신러닝에서 마주치는 대부분의 확률은 개별 확률의 곱입니다.

$$ P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}). $$ 

이는 우리가 도함수를 계산하기 위해 곱의 미분법을 직접 적용한다면 다음을 얻음을 의미합니다.

$$ \begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\ & \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\ & \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\ & \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned} $$ 

이것은 $(n-1)$번의 덧셈과 함께 $n(n-1)$번의 곱셈을 필요로 하므로 입력에 대해 시간적으로 이차식에 비례합니다! 항들을 그룹화하는 충분한 기발함은 이를 선형 시간으로 줄여주겠지만 생각이 필요합니다. 음의 로그 우도의 경우 대신 다음을 갖습니다.

$$ -\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})), $$ 

그러면 다음을 제공합니다.

$$ - \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right). $$ 

이것은 단지 $n$번의 나눗셈과 $n-1$번의 합계만 필요로 하므로 입력에 대해 선형 시간입니다.

음의 로그 우도를 고려하는 세 번째이자 마지막 이유는 :numref:`sec_information_theory`에서 자세히 논의할 정보 이론과의 관계입니다. 이것은 확률 변수에서의 정보 또는 무작위성의 정도를 측정하는 방법을 제공하는 엄격한 수학적 이론입니다. 해당 분야의 주요 연구 대상은 다음과 같은 엔트로피입니다.

$$ H(p) = -\sum_{i} p_i \log_2(p_i), $$ 

소스의 무작위성을 측정합니다. 이것이 평균 $-\log$ 확률에 지나지 않는다는 점에 주목하십시오. 따라서 우리의 음의 로그 우도를 데이터 예제의 수로 나누면 크로스 엔트로피(cross-entropy)라고 알려진 엔트로피의 친척을 얻게 됩니다. 이 이론적 해석만으로도 모델 성능을 측정하는 방법으로 데이터셋 전체에 대한 평균 음의 로그 우도를 보고하도록 동기를 부여하기에 충분히 설득력이 있습니다.

## 연속 변수에 대한 최대 우도 (Maximum Likelihood for Continuous Variables)

지금까지 우리가 한 모든 것은 이산 확률 변수로 작업한다고 가정했지만, 연속 확률 변수로 작업하고 싶다면 어떨까요?

짧은 요약은 확률의 모든 인스턴스를 확률 밀도로 대체하는 것 외에는 전혀 변하는 것이 없다는 것입니다. 밀도를 소문자 $p$로 쓴다는 점을 상기하면, 이는 예를 들어 이제 다음과 같이 말함을 의미합니다.

$$ -\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)). $$ 

질문은 "왜 이것이 괜찮은가?"가 됩니다. 결국, 우리가 밀도를 도입한 이유는 특정 결과를 얻을 확률 자체가 0이었기 때문인데, 그렇다면 임의의 파라미터 세트에 대해 우리 데이터를 생성할 확률은 0이 아닐까요?

실제로 그렇습니다. 그리고 왜 우리가 밀도로 전환할 수 있는지 이해하는 것은 입실론(epsilons)에 어떤 일이 일어나는지 추적하는 연습입니다.

먼저 우리의 목표를 재정의해 봅시다. 연속 확률 변수에 대해 더 이상 정확히 맞는 값을 얻을 확률이 아니라 대신 어떤 범위 $\epsilon$ 이내에 맞추는 것을 계산하고 싶다고 가정합시다. 단순함을 위해 우리 데이터가 독립 동일 분포(i.i.d.) 확률 변수 $X_1, \ldots, X_N$의 반복된 관찰 $x_1, \ldots, x_N$이라고 가정합니다. 이전에 보았듯이 이는 다음과 같이 쓰일 수 있습니다.

$$ \begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\ &\approx \epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned} $$ 

따라서 여기에 음의 로그를 취하면 다음을 얻습니다.

$$ \begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\ &\approx -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned} $$ 

이 식을 살펴보면, $\epsilon$이 발생하는 유일한 곳은 가산 상수 $-N\log(\epsilon)$입니다. 이것은 파라미터 $\boldsymbol{\theta}$에 전혀 의존하지 않으므로, $\boldsymbol{\theta}$의 최적 선택은 우리의 $\epsilon$ 선택에 의존하지 않습니다! 우리가 네 자리 숫자를 요구하든 사백 자리 숫자를 요구하든, $\boldsymbol{\theta}$의 최적 선택은 동일하게 유지되므로 우리는 입실론을 자유롭게 삭제하여 우리가 최적화하고 싶은 것이 다음과 같음을 볼 수 있습니다.

$$ - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})). $$ 

따라서 우리는 확률을 확률 밀도로 대체함으로써 연속 확률 변수에서도 이산 확률 변수에서와 같이 쉽게 최대 우도 관점을 운용할 수 있음을 알 수 있습니다.

## 요약 (Summary)
* 최대 우도 원칙은 주어진 데이터셋에 대해 가장 적합한 모델이 데이터를 가장 높은 확률로 생성하는 모델이라는 것을 알려줍니다.
* 사람들은 종종 수치적 안정성, 곱을 합으로 변환(및 그에 따른 기울기 계산의 단순화), 정보 이론과의 이론적 연계 등 다양한 이유로 음의 로그 우도를 대신 사용합니다.
* 이산 설정에서 동기를 부여하는 것이 가장 간단하지만, 데이터 포인트에 할당된 확률 밀도를 최대화함으로써 연속 설정으로도 자유롭게 일반화될 수 있습니다.

## 연습 문제 (Exercises)
1. 비음수 확률 변수가 어떤 값 $\alpha>0$에 대해 $\alpha e^{-\alpha x}$ 밀도를 갖는다는 것을 안다고 가정합시다. 확률 변수로부터 숫자 $3$인 단일 관찰을 얻었습니다. $\alpha$에 대한 최대 우도 추정치는 무엇입니까?
2. 알려지지 않은 평균을 갖지만 분산이 $1$인 가우스 분포에서 추출된 샘플 {x_i}_{i=1}^N 데이터셋이 있다고 가정합시다. 평균에 대한 최대 우도 추정치는 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab: