# 확률 변수 (Random Variables)
:label:`sec_random_variables`

:numref:`sec_prob`에서 우리는 이산 확률 변수를 다루는 방법에 대한 기초를 보았습니다. 우리의 경우 이산 확률 변수는 유한한 가능한 값의 집합 또는 정수를 취하는 확률 변수를 지칭합니다. 이 섹션에서는 임의의 실수 값을 취할 수 있는 확률 변수인 *연속 확률 변수(continuous random variables)* 이론을 전개합니다.

## 연속 확률 변수 (Continuous Random Variables)

연속 확률 변수는 이산 확률 변수보다 현저히 더 미묘한 주제입니다. 적절한 비유를 들자면, 기술적인 도약은 숫자 목록을 더하는 것과 함수를 적분하는 것 사이의 도약에 필적합니다. 따라서 이론을 전개하는 데 시간을 좀 들여야 합니다.

### 이산에서 연속으로 (From Discrete to Continuous)

연속 확률 변수로 작업할 때 마주치는 추가적인 기술적 과제를 이해하기 위해 사고 실험을 하나 해봅시다. 우리가 다트판에 다트를 던지고 있고, 판의 중심으로부터 정확히 $2 \textrm{cm}$ 떨어진 곳에 맞을 확률을 알고 싶다고 가정해 봅시다. 우선, 한 자리 정도의 정확도로 측정하는 것을 상상해 봅니다. 즉, $0 \textrm{cm}$, $1 \textrm{cm}$, $2 \textrm{cm}$ 등의 칸(bins)이 있는 것입니다. 우리는 다트판에 가령 100개의 다트를 던지고, 그중 20개가 $2\textrm{cm}$ 칸에 떨어진다면 던진 다트의 $20\%$가 중심에서 $2 \textrm{cm}$ 떨어진 판에 맞았다고 결론을 내립니다.

하지만 자세히 살펴보면, 이것은 우리의 질문과 맞지 않습니다! 우리는 정확한 일치를 원했지만, 이 칸들은 가령 $1.5\textrm{cm}$와 $2.5\textrm{cm}$ 사이에 떨어진 모든 것을 담고 있습니다.

굴하지 않고 더 진행해 봅니다. 더 정밀하게 측정합니다. 가령 $1.9\textrm{cm}$, $2.0\textrm{cm}$, $2.1\textrm{cm}$로 측정하고, 이제 아마도 100개의 다트 중 3개가 $2.0\textrm{cm}$ 양동이에 맞았음을 봅니다. 따라서 우리는 확률이 $3\%$라고 결론짓습니다.

하지만 이것은 아무것도 해결하지 못합니다! 단지 문제를 한 자리 더 아래로 밀어냈을 뿐입니다. 조금 추상화해 봅시다. 처음 $k$자리가 $2.00000\ldots$와 일치할 확률을 알고 있고, 처음 $k+1$자리가 일치할 확률을 알고 싶다고 상상해 봅시다. $k+1$번째 자리가 본질적으로 {0, 1, 2, \ldots, 9} 집합에서의 무작위 선택이라고 가정하는 것은 꽤 합리적입니다. 적어도, 중심으로부터 떨어진 마이크로미터 수가 $3$보다 $7$로 끝나는 것을 선호하게 만드는 물리적으로 의미 있는 과정을 상상할 수 없습니다.

이것이 의미하는 바는 본질적으로 우리가 요구하는 정확도가 한 자리 늘어날 때마다 일치할 확률이 10분의 1로 줄어들어야 한다는 것입니다. 또는 다른 방식으로 표현하면 다음과 같이 기대할 것입니다.

$$ P(\textrm{거리가}\; 2.00\ldots \textrm{이며,}\; k \;\textrm{자리까지 일치함} ) \approx p\cdot10^{-k}. $$

값 $p$는 본질적으로 처음 몇 자리에서 일어나는 일을 인코딩하고, $10^{-k}$가 나머지를 처리합니다.

소수점 이하 $k=4$자리까지 정확한 위치를 안다면, 이는 값이 가령 $[1.99995, 2.00005]$ 구간 내에 떨어진다는 것을 의미하며, 이 구간의 길이는 $2.00005-1.99995 = 10^{-4}$입니다. 따라서 이 구간의 길이를 $\epsilon$이라고 부르면 다음과 같이 말할 수 있습니다.

$$ P(\textrm{거리가}\; 2 \;\textrm{근처의}\; \epsilon \;\textrm{크기 구간 내에 있음} ) \approx \epsilon \cdot p. $$

이것을 마지막으로 한 단계 더 진행해 봅시다. 우리는 줄곧 점 $2$에 대해서만 생각해 왔지만, 다른 점들에 대해서는 생각하지 않았습니다. 근본적으로 다른 점은 없지만, 값 $p$는 아마도 다를 것입니다. 우리는 적어도 다트 던지는 사람이 $20\textrm{cm}$보다는 $2\textrm{cm}$와 같은 중심 근처의 점에 맞힐 가능성이 더 높기를 바랄 것입니다. 따라서 값 $p$는 고정된 것이 아니라 점 $x$에 의존해야 합니다. 이는 우리가 다음을 기대해야 함을 알려줍니다.

$$P(\textrm{거리가}\; x \;\textrm{근처의}\; \epsilon \;\textrm{크기 구간 내에 있음} ) \approx \epsilon \cdot p(x).$$ :eqlabel:`eq_pdf_deriv`

실제로 :eqref:`eq_pdf_deriv`는 *확률 밀도 함수(probability density function)*를 정확하게 정의합니다. 이는 한 점 근처에 맞을 상대적 확률을 다른 점과 비교하여 인코딩하는 함수 $p(x)$입니다. 그러한 함수가 어떻게 생겼을지 시각화해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# 어떤 확률 변수에 대한 확률 밀도 함수 플롯
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # torch에서 pi 정의

# 어떤 확률 변수에 대한 확률 밀도 함수 플롯
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # TensorFlow에서 pi 정의

# 어떤 확률 변수에 대한 확률 밀도 함수 플롯
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

함수 값이 큰 위치는 우리가 무작위 값을 찾을 가능성이 더 높은 영역을 나타냅니다. 낮은 부분은 무작위 값을 찾을 가능성이 낮은 영역입니다.

### 확률 밀도 함수 (Probability Density Functions)

이제 이를 더 조사해 봅시다. 우리는 확률 변수 $X$에 대해 확률 밀도 함수가 직관적으로 무엇인지 이미 보았습니다. 즉, 밀도 함수는 다음을 만족하는 함수 $p(x)$입니다.

$$P(X \;\textrm{가}\; x \;\textrm{근처의}\; \epsilon \;\textrm{크기 구간 내에 있음} ) \approx \epsilon \cdot p(x).$$ :eqlabel:`eq_pdf_def`

그렇다면 이것이 $p(x)$의 속성에 대해 무엇을 의미할까요?

먼저, 확률은 결코 음수가 아니므로 $p(x) \ge 0$일 것으로 기대해야 합니다.

둘째, $\mathbb{R}$을 $(\epsilon\cdot i, \epsilon \cdot (i+1)]$로 주어지는 $\epsilon$ 너비의 무한한 수의 슬라이스로 쪼갠다고 상상해 봅시다. 이들 각각에 대해, 우리는 :eqref:`eq_pdf_def`로부터 확률이 대략 다음과 같음을 압니다.

$$ P(X \;\textrm{가}\; x \;\textrm{근처의}\; \epsilon \;\textrm{크기 구간 내에 있음} ) \approx \epsilon \cdot p(\epsilon \cdot i), $$

따라서 이들 모두에 대해 합산하면 다음과 같아야 합니다.

$$ P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i). $$

이것은 :numref:`sec_integral_calculus`에서 논의된 적분의 근사에 지나지 않으므로 다음과 같이 말할 수 있습니다.

$$ P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx. $$

확률 변수는 *어떤* 숫자를 반드시 가져야 하므로 $P(X\in\mathbb{R}) = 1$임을 압니다. 따라서 임의의 밀도에 대해 다음과 같이 결론지을 수 있습니다.

$$ \int_{-\infty}^{\infty} p(x) \; dx = 1. $$

실제로 이를 더 파고들면 임의의 $a$와 $b$에 대해 다음을 보게 됩니다.

$$ P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx. $$

우리는 이전과 동일한 이산 근사 방법을 사용하여 코드에서 이를 근사할 수 있습니다. 이 경우 파란색 영역에 떨어질 확률을 근사할 수 있습니다.

```{.python .input}
#@tab mxnet
# 수치 적분을 사용하여 확률 근사
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'근사 확률: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# 수치 적분을 사용하여 확률 근사
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'근사 확률: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# 수치 적분을 사용하여 확률 근사
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'근사 확률: {tf.reduce_sum(epsilon*p[300:800])}'
```

이러한 두 가지 속성이 가능한 확률 밀도 함수(줄여서 *p.d.f.*)의 공간을 정확하게 설명한다는 것이 밝혀졌습니다. 이들은 다음을 만족하는 비음수 함수 $p(x) \ge 0$입니다.

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$ :eqlabel:`eq_pdf_int_one`

우리는 적분을 사용하여 확률 변수가 특정 구간에 있을 확률을 얻음으로써 이 함수를 해석합니다.

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$ :eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions`에서 여러 일반적인 분포를 보겠지만, 계속해서 추상적으로 작업해 봅시다.

### 누적 분포 함수 (Cumulative Distribution Functions)

이전 섹션에서 우리는 p.d.f.의 개념을 보았습니다. 실전에서 이는 연속 확률 변수를 논의하기 위해 흔히 마주치는 방법이지만, 한 가지 중요한 함정이 있습니다: p.d.f.의 값 자체가 확률이 아니라, 확률을 산출하기 위해 우리가 적분해야 하는 함수라는 점입니다. 밀도가 10보다 크더라도, 그것이 $1/10$보다 긴 구간에 대해 10보다 크지 않은 한 아무런 문제가 없습니다. 이는 직관에 반할 수 있으므로, 사람들은 종종 그 자체로 *확률*인 *누적 분포 함수(cumulative distribution function)* 또는 c.d.f.의 관점에서도 생각합니다.

특히 :eqref:`eq_pdf_int_int`를 사용하여, 밀도 $p(x)$를 갖는 확률 변수 $X$에 대한 c.d.f.를 다음과 같이 정의합니다.

$$ F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x). $$

몇 가지 속성을 관찰해 봅시다.

* $x\rightarrow -\infty$에 따라 $F(x) \rightarrow 0$.
* $x\rightarrow \infty$에 따라 $F(x) \rightarrow 1$.
* $F(x)$는 비감소 함수입니다 ($y > x \implies F(y) \ge F(x)$).
* $X$가 연속 확률 변수라면 $F(x)$는 연속적입니다 (도약이 없음).

네 번째 불렛 포인트와 관련하여, 가령 확률 $1/2$로 $0$과 $1$ 값을 취하는 이산 확률 변수 $X$라면 이것이 참이 아님에 유의하십시오. 그 경우 다음과 같습니다.

$$ F(x) = \begin{cases} 0 & x < 0, \\ \frac{1}{2} & x < 1, \\ 1 & x \ge 1. \end{cases} $$

이 예제에서 우리는 c.d.f.로 작업할 때의 이점 중 하나를 봅니다. 즉, 동일한 프레임워크 내에서 연속 또는 이산 확률 변수, 또는 실제로 두 가지의 혼합(동전을 던져 앞면이면 주사위 굴리기 결과를 반환하고, 뒷면이면 다트 던지기 거리를 반환함)을 다룰 수 있다는 능력입니다.

### 평균 (Means)

확률 변수 $X$를 다루고 있다고 가정합시다. 분포 자체는 해석하기 어려울 수 있습니다. 확률 변수의 행동을 간결하게 요약할 수 있는 것이 종종 유용합니다. 확률 변수의 행동을 포착하는 데 도움이 되는 숫자를 *요약 통계량(summary statistics)*이라고 합니다. 가장 흔히 마주치는 것들은 *평균(mean)*, *분산(variance)*, 그리고 *표준 편차(standard deviation)*입니다.

*평균*은 확률 변수의 평균적인 값을 인코딩합니다. 확률 $p_i$로 값 $x_i$를 취하는 이산 확률 변수 $X$가 있다면, 평균은 가중 평균에 의해 주어집니다: 값들에 확률 변수가 그 값을 취할 확률을 곱한 합계입니다.

$$\mu_X = E[X] = \sum_i x_i p_i.$$ :eqlabel:`eq_exp_def`

우리가 평균을 해석해야 할 방식은(비록 주의가 필요하지만) 본질적으로 확률 변수가 위치하려는 경향이 있는 곳을 알려준다는 것입니다.

이 섹션 전체에서 살펴볼 미니멀한 예제로서, 확률 $p$로 $a-2$ 값을, 확률 $p$로 $a+2$ 값을, 그리고 확률 $1-2p$로 $a$ 값을 취하는 확률 변수 $X$를 들어봅시다. 우리는 임의의 가능한 $a$와 $p$ 선택에 대해 :eqref:`eq_exp_def`를 사용하여 평균이 다음과 같음을 계산할 수 있습니다.

$$ \mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a. $$

따라서 평균이 $a$임을 알 수 있습니다. 이는 $a$가 우리가 확률 변수를 중심으로 한 위치이므로 직관과 일치합니다.

도움이 되므로 몇 가지 속성을 요약해 봅시다.

* 임의의 확률 변수 $X$와 숫자 $a, b$에 대해, $\mu_{aX+b} = a\mu_X + b$가 성립합니다.
* 두 확률 변수 $X$와 $Y$가 있다면, $\mu_{X+Y} = \mu_X+\mu_Y$입니다.

평균은 확률 변수의 평균적인 행동을 이해하는 데 유용하지만, 충분하고 완전한 직관적 이해를 갖기에는 평균만으로는 부족합니다. 판매당 $10 \pm \$1의 이익을 내는 것과 판매당 $10 \pm \$15의 이익을 내는 것은 동일한 평균 값을 가짐에도 불구하고 매우 다릅니다. 두 번째 것은 훨씬 더 큰 변동 정도를 가지며, 따라서 훨씬 더 큰 위험을 나타냅니다. 따라서 확률 변수의 행동을 이해하려면 최소한 하나의 척도가 더 필요할 것입니다: 확률 변수가 얼마나 넓게 요동치는지를 측정하는 척도입니다.

### 분산 (Variances)

이는 우리로 하여금 확률 변수의 *분산(variance)*을 고려하게 합니다. 이는 확률 변수가 평균으로부터 얼마나 멀리 벗어나는지에 대한 정량적 측정입니다. 식 $X - \mu_X$를 고려해 보십시오. 이것은 확률 변수의 평균으로부터의 편차입니다. 이 값은 양수일 수도 음수일 수도 있으므로, 우리가 편차의 크기를 측정하도록 양수로 만들기 위해 무언가를 해야 합니다.

시도해 볼 수 있는 합리적인 방법은 $|X-\mu_X|$를 살펴보는 것이며, 실제로 이는 *평균 절대 편차(mean absolute deviation)*라고 불리는 유용한 양으로 이어지지만, 수학 및 통계의 다른 분야와의 연결 때문에 사람들은 종종 다른 해결책을 사용합니다.

특히, 그들은 $(X-\mu_X)^2$을 살펴봅니다. 평균을 취하여 이 양의 전형적인 크기를 본다면 분산에 도달하게 됩니다.

$$\sigma_X^2 = \textrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$ :eqlabel:`eq_var_def`

:eqref:`eq_var_def`의 마지막 등식은 중간의 정의를 확장하고 기댓값의 속성을 적용함으로써 성립합니다.

확률 $p$로 $a-2$ 값을, 확률 $p$로 $a+2$ 값을, 그리고 확률 $1-2p$로 $a$ 값을 취하는 우리 예제를 살펴봅시다. 이 경우 $\mu_X = a$이므로, $E[X^2]$만 계산하면 됩니다. 이는 쉽게 수행될 수 있습니다.

$$ E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p. $$

따라서 :eqref:`eq_var_def`에 의해 우리의 분산은 다음과 같음을 알 수 있습니다.

$$ \sigma_X^2 = \textrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p. $$

이 결과 또한 말이 됩니다. $p$가 가질 수 있는 최대값은 $1/2$이며, 이는 동전 던지기로 $a-2$ 또는 $a+2$를 선택하는 것에 해당합니다. 이것의 분산이 $4$인 것은 $a-2$와 $a+2$가 모두 평균에서 $2$ 단위 떨어져 있고 $2^2 = 4$라는 사실에 대응합니다. 스펙트럼의 다른 쪽 끝에서 $p=0$이면, 이 확률 변수는 항상 $a$ 값을 취하므로 분산이 전혀 없습니다.

아래에 분산의 몇 가지 속성을 나열합니다.

* 임의의 확률 변수 $X$에 대해 $\textrm{Var}(X) \ge 0$이며, $\textrm{Var}(X) = 0$인 것은 $X$가 상수인 것과 동등합니다.
* 임의의 확률 변수 $X$와 숫자 $a, b$에 대해, $\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$가 성립합니다.
* 두 개의 *독립* 확률 변수 $X$와 $Y$가 있다면, $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$입니다.

이러한 값들을 해석할 때 약간의 걸림돌이 있을 수 있습니다. 특히, 이 계산 과정에서 단위를 추적한다고 상상해 봅시다. 웹페이지의 제품에 할당된 별점 평점을 다루고 있다고 가정해 봅시다. 그러면 $a, a-2, a+2$는 모두 별 단위로 측정됩니다. 마찬가지로 평균 $\mu_X$도 별 단위로 측정됩니다(가중 평균이므로). 그러나 분산에 이르면 즉시 문제에 직면하는데, 우리가 $(X-\mu_X)^2$을 보고 싶어 하며 이는 *제곱된 별* 단위이기 때문입니다. 이는 분산 자체가 원래의 측정값과 비교 가능하지 않음을 의미합니다. 이를 해석 가능하게 만들려면 원래의 단위로 돌아가야 할 것입니다.

### 표준 편차 (Standard Deviations)

이 요약 통계량은 항상 분산의 제곱근을 취함으로써 유도될 수 있습니다! 따라서 우리는 *표준 편차(standard deviation)*를 다음과 같이 정의합니다.

$$ \sigma_X = \sqrt{\textrm{Var}(X)}. $$

우리 예제에서 이는 이제 표준 편차가 $\sigma_X = 2\sqrt{2p}$임을 의미합니다. 별점 리뷰 예제에 대해 단위를 다루고 있다면, $\sigma_X$는 다시 별 단위입니다.

분산에 대해 가졌던 속성들은 표준 편차에 대해서도 다시 기술될 수 있습니다.

* 임의의 확률 변수 $X$에 대해 $\sigma_{X} \ge 0$입니다.
* 임의의 확률 변수 $X$와 숫자 $a, b$에 대해, $\sigma_{aX+b} = |a|\sigma_{X}$입니다.
* 두 개의 *독립* 확률 변수 $X$와 $Y$가 있다면, $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$입니다.

이 시점에서