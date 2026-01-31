# 분포 (Distributions)
:label:`sec_distributions`

이산형 및 연속형 설정 모두에서 확률을 다루는 방법을 배웠으므로, 이제 흔히 마주치는 몇 가지 일반적인 분포를 알아봅시다. 머신러닝 분야에 따라 이보다 훨씬 더 많은 분포에 익숙해져야 할 수도 있고, 딥러닝의 일부 분야에서는 아예 필요하지 않을 수도 있습니다. 하지만 이것은 익숙해지기에 좋은 기본 목록입니다. 먼저 몇 가지 일반적인 라이브러리를 가져옵시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # torch에서 pi 정의
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # TensorFlow에서 pi 정의
```

## 베르누이 분포 (Bernoulli)

이것은 일반적으로 마주치는 가장 단순한 확률 변수입니다. 이 확률 변수는 $p$의 확률로 $1$이 나오고 $1-p$의 확률로 $0$이 나오는 동전 던지기를 인코딩합니다. 이 분포를 가진 확률 변수 $X$가 있다면 다음과 같이 씁니다.

$$
X sim 	extrm{Bernoulli}(p).
$$ 

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases}
0 & x < 0, \\
1-p & 0 \le x < 1, \\
1 & x >= 1 .
\end{cases}$$ 
:eqlabel:`eq_bernoulli_cdf`

확률 질량 함수(pmf)는 아래에 플롯되어 있습니다.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_bernoulli_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

$X sim 	extrm{Bernoulli}(p)$이면 다음이 성립합니다.

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

다음과 같이 베르누이 확률 변수로부터 임의의 모양의 배열을 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 이산 균등 분포 (Discrete Uniform)

다음으로 흔히 마주치는 확률 변수는 이산 균등 분포입니다. 여기서 논의를 위해 정수 ${1, 2, \ldots, n}$에서 지원된다고 가정하겠지만, 다른 어떤 값의 집합도 자유롭게 선택할 수 있습니다. 이 문맥에서 *균등(uniform)*이라는 단어의 의미는 모든 가능한 값이 동등하게 가능성이 높다는 것입니다. 각 값 $i sin {1, 2, 3, \ldots, n}$에 대한 확률은 $p_i = \frac{1}{n}$입니다. 이 분포를 가진 확률 변수 $X$를 다음과 같이 나타낼 것입니다.

$$ 
X sim U(n).
$$ 

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases}
0 & x < 1, \\
\frac{k}{n} & k \le x < k+1 \textrm{ 이며 } 1 \le k < n, \\
1 & x >= n .
\end{cases}$$ 
:eqlabel:`eq_discrete_uniform_cdf`

먼저 확률 질량 함수를 플롯해 봅시다.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_discrete_uniform_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X sim U(n)$이면 다음이 성립합니다.

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

다음과 같이 이산 균등 확률 변수로부터 임의의 모양의 배열을 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 연속 균등 분포 (Continuous Uniform)

다음으로 연속 균등 분포에 대해 논의해 봅시다. 이 확률 변수 뒤에 숨겨진 아이디어는 이산 균등 분포에서 $n$을 늘리고 구간 $[a, b]$ 내에 맞도록 스케일을 조정하면, $[a, b]$ 내의 임의의 값을 모두 동일한 확률로 선택하는 연속 확률 변수에 접근하게 된다는 것입니다. 이 분포를 다음과 같이 나타낼 것입니다.

$$ 
X sim U(a, b).
$$ 

확률 밀도 함수(pdf)는 다음과 같습니다.

$$p(x) = \begin{cases}
\frac{1}{b-a} & x sin [a, b], \\
0 & x \notsin [a, b].
\end{cases}$$ 
:eqlabel:`eq_cont_uniform_pdf`

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases}
0 & x < a, \\
\frac{x-a}{b-a} & x sin [a, b], \\
1 & x >= b .
\end{cases}$$ 
:eqlabel:`eq_cont_uniform_cdf`

먼저 확률 밀도 함수 :eqref:`eq_cont_uniform_pdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

이제 누적 분포 함수 :eqref:`eq_cont_uniform_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X sim U(a, b)$이면 다음이 성립합니다.

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

다음과 같이 균등 확률 변수로부터 임의의 모양의 배열을 샘플링할 수 있습니다. 기본적으로 $U(0,1)$에서 샘플링하므로 다른 범위를 원하면 스케일을 조정해야 합니다.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 이항 분포 (Binomial)

상황을 좀 더 복잡하게 만들어 *이항(binomial)* 확률 변수를 살펴봅시다. 이 확률 변수는 성공 확률이 $p$인 $n$개의 독립적인 실험 시퀀스를 수행하고, 우리가 얼마나 많은 성공을 볼 것으로 기대하는지 묻는 것에서 비롯됩니다.

이를 수학적으로 표현해 봅시다. 각 실험은 독립 확률 변수 $X_i$이며, 여기서 성공을 인코딩하기 위해 $1$을 사용하고 실패를 인코딩하기 위해 $0$을 사용합니다. 각각이 확률 $p$로 성공하는 독립적인 동전 던지기이므로, $X_i sim 	extrm{Bernoulli}(p)$라고 말할 수 있습니다. 그러면 이항 확률 변수는 다음과 같습니다.

$$ 
X = \sum_{i=1}^n X_i.
$$ 

이 경우 다음과 같이 씁니다.

$$ 
X sim 	extrm{Binomial}(n, p).
$$ 

누적 분포 함수를 얻으려면, 정확히 $k$번 성공하는 것이 $inom{n}{k} = \frac{n!}{k!(n-k)!}$가지 방식으로 발생할 수 있고 각 방식은 발생 확률 $p^k(1-p)^{n-k}$를 갖는다는 점에 유의해야 합니다. 따라서 누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases}
0 & x < 0, \\
\sum_{m \le k} inom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ 이며 } 0 \le k < n, \\
1 & x >= n .
\end{cases}$$ 
:eqlabel:`eq_binomial_cdf`

먼저 확률 질량 함수를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# 이항 계수 계산
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# 이항 계수 계산
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# 이항 계수 계산
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_binomial_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

$X sim 	extrm{Binomial}(n, p)$이면 다음이 성립합니다.

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

이는 $n$개의 베르누이 확률 변수의 합에 대한 기댓값의 선형성과, 독립 확률 변수 합의 분산은 분산의 합이라는 사실로부터 따릅니다. 이는 다음과 같이 샘플링될 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 포아송 분포 (Poisson)
이제 사고 실험을 해봅시다. 우리는 버스 정류장에 서 있고 다음 1분 동안 몇 대의 버스가 도착할지 알고 싶습니다. 먼저 1분 윈도우 내에 버스가 도착할 확률인 $X^{(1)} sim 	extrm{Bernoulli}(p)$를 고려하는 것으로 시작하겠습니다. 도심에서 멀리 떨어진 버스 정류장의 경우, 이것은 꽤 좋은 근사일 수 있습니다. 우리는 1분 내에 한 대 이상의 버스를 결코 보지 못할 수도 있습니다.

그러나 우리가 붐비는 지역에 있다면 두 대의 버스가 도착할 가능성이 있거나 심지어 높을 수 있습니다. 우리는 우리의 확률 변수를 처음 30초 또는 뒤의 30초에 대한 두 부분으로 나누어 이를 모델링할 수 있습니다. 이 경우 다음과 같이 쓸 수 있습니다.

$$ 
X^{(2)} sim X^{(2)}_1 + X^{(2)}_2,
$$ 

여기서 $X^{(2)}$는 총 합이고 $X^{(2)}_i sim 	extrm{Bernoulli}(p/2)$입니다. 그러면 총 분포는 $X^{(2)} sim 	extrm{Binomial}(2, p/2)$가 됩니다.

여기서 멈출 이유가 있을까요? 그 1분을 $n$개 부분으로 계속 나누어 봅시다. 위와 동일한 추론에 의해 다음을 알 수 있습니다.

$$X^{(n)} sim 	extrm{Binomial}(n, p/n).$$ 
:eqlabel:`eq_eq_poisson_approx`

이러한 확률 변수들을 고려해 보십시오. 이전 섹션에 의해 :eqref:`eq_eq_poisson_approx`은 평균 $\mu_{X^{(n)}} = n(p/n) = p$와 분산 $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$를 가짐을 압니다. 만약 $n ightarrow \infty$이면, 이러한 수치들이 평균 $\mu_{X^{(\infty)}} = p$와 분산 $\sigma_{X^{(\infty)}}^2 = p$로 안정화되는 것을 볼 수 있습니다. 이는 이 무한 분할 극한에서 우리가 정의할 수 있는 어떤 확률 변수가 *있을 수 있음*을 나타냅니다.

실제 세계에서 우리는 단순히 버스 도착 횟수를 셀 수 있기 때문에 이것은 그리 놀라운 일이 아니어야 하지만, 우리의 수학적 모델이 잘 정의되어 있다는 것을 보는 것은 좋습니다. 이 논의는 *희귀 사건의 법칙(law of rare events)*으로 공식화될 수 있습니다.

이 추론을 신중하게 따라가면 다음과 같은 모델에 도달할 수 있습니다. 확률 변수 $X$가 ${0,1,2, \ldots}$ 값을 다음 확률로 가질 때 $X sim 	extrm{Poisson}(\lambda)$라고 말할 것입니다.

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$ 
:eqlabel:`eq_poisson_mass`

값 $\lambda > 0$은 *강도(rate)* (또는 *형태(shape)* 파라미터)라고 알려져 있으며, 단위 시간당 우리가 기대하는 평균 도착 횟수를 나타냅니다.

이 확률 질량 함수를 합산하여 누적 분포 함수를 얻을 수 있습니다.

$$F(x) = \begin{cases}
0 & x < 0, \\
e^{-\lambda}\sum_{m = 0}^k rac{\lambda^m}{m!} & k \le x < k+1 \textrm{ 이며 } 0 \le k. 
\end{cases}$$ 
:eqlabel:`eq_poisson_cdf`

먼저 확률 질량 함수 :eqref:`eq_poisson_mass`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_poisson_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

위에서 보았듯이 평균과 분산은 특히 간결합니다. $X sim 	extrm{Poisson}(\lambda)$이면 다음이 성립합니다.

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

이는 다음과 같이 샘플링될 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## 가우스 분포 (Gaussian)
이제 다르지만 관련된 실험을 시도해 봅시다. 우리가 다시 $n$개의 독립적인 $	extrm{Bernoulli}(p)$ 측정 $X_i$를 수행한다고 가정합시다. 이들의 합의 분포는 $X^{(n)} sim 	extrm{Binomial}(n, p)$입니다. $n$이 증가하고 $p$가 감소할 때 극한을 취하는 대신, $p$를 고정하고 $n ightarrow \infty$로 보냅시다. 이 경우 $\mu_{X^{(n)}} = np ightarrow \infty$이고 $\sigma_{X^{(n)}}^2 = np(1-p) ightarrow \infty$이므로, 이 극한이 잘 정의될 것이라고 생각할 이유가 없습니다.

그러나 모든 희망이 사라진 것은 아닙니다! 다음과 같이 정의하여 평균과 분산이 잘 작동하도록 만들어 봅시다.

$$ 
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$ 

이것은 평균 0과 분산 1을 갖는 것을 알 수 있으며, 따라서 어떤 극한 분포로 수렴할 것이라고 믿는 것이 그럴듯합니다. 이러한 분포들이 어떻게 생겼는지 플롯해 보면, 그것이 작동할 것이라고 더욱 확신하게 될 것입니다.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

한 가지 주목할 점은 포아송 사례와 비교하여 이제 표준 편차로 나누고 있다는 점인데, 이는 가능한 결과들을 점점 더 작은 영역으로 짜내고 있다는 것을 의미합니다. 이는 우리의 극한이 더 이상 이산적이지 않고 오히려 연속적일 것임을 나타냅니다.

발생하는 일에 대한 유도는 이 문서의 범위를 벗어나지만, *중심 극한 정리(central limit theorem)*는 $n ightarrow \infty$에 따라 이것이 가우스 분포(또는 정규 분포)를 낳을 것이라고 기술합니다. 더 명시적으로, 임의의 $a, b$에 대해 다음과 같습니다.

$$ 
\lim_{n ightarrow \infty} P(Y^{(n)} sin [a, b]) = P(r(0,1) sin [a, b]),
$$ 

여기서 우리는 확률 변수가 주어진 평균 $\mu$와 분산 $\sigma^2$를 가진 정규 분포를 따른다고 말하며, $X sim \mathcal N(\mu, \sigma^2)$라고 씁니다. 만약 $X$가 다음과 같은 밀도를 갖는다면 말입니다.

$$p_X(x) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$ 
:eqlabel:`eq_gaussian_pdf`

먼저 확률 밀도 함수 :eqref:`eq_gaussian_pdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

이제 누적 분포 함수를 플롯해 봅시다. 이 부록의 범위를 벗어나지만, 가우스 c.d.f.는 더 기초적인 함수들로 된 닫힌 형식의 공식이 없습니다. 우리는 이 적분을 수치적으로 계산하는 방법을 제공하는 `erf`를 사용할 것입니다.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

눈치 빠른 독자들은 이러한 용어 중 일부를 알아볼 것입니다. 실제로 우리는 이 적분을 :numref:`sec_integral_calculus`에서 만났습니다. 실제로 이 $p_X(x)$가 총 면적 1을 가지며 따라서 유효한 밀도임을 확인하려면 정확히 그 계산이 필요합니다.

동전 던지기로 작업하기로 한 우리의 선택은 계산을 짧게 만들었지만, 그 선택에 근본적인 것은 아무것도 없었습니다. 실제로 임의의 독립적인 동일 분포 확률 변수 $X_i$ 모음을 취하고 다음을 형성하면

$$ 
X^{(N)} = \sum_{i=1}^N X_i.
$$ 

그러면

$$ 
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$ 

은 대략적으로 가우스 분포를 따를 것입니다. 이를 작동시키기 위해 필요한 추가 요구 사항들이 있으며, 가장 흔한 것은 $E[X^4] < \infty$이지만 철학은 명확합니다.

중심 극한 정리는 왜 가우스 분포가 확률, 통계, 머신러닝의 기초가 되는지 설명해 주는 이유입니다. 우리가 측정한 무언가가 많은 작은 독립적인 기여의 합이라고 말할 수 있을 때마다, 측정되는 대상이 가우스 분포에 가까울 것이라고 가정할 수 있습니다.

가우스 분포에는 훨씬 더 많은 흥미로운 속성들이 있으며, 여기서 하나 더 논의하고 싶습니다. 가우스 분포는 *최대 엔트로피 분포(maximum entropy distribution)*로 알려진 것입니다. 우리는 :numref:`sec_information_theory`에서 엔트로피에 대해 더 깊이 다루겠지만, 지금 시점에서 알아야 할 모든 것은 그것이 무작위성의 척도라는 것입니다. 엄밀한 수학적 의미에서, 우리는 가우스 분포를 고정된 평균과 분산을 가진 확률 변수의 *가장* 무작위적인 선택으로 생각할 수 있습니다. 따라서 우리의 확률 변수가 어떤 평균과 분산을 갖는다는 것을 안다면, 가우스 분포는 어떤 의미에서 우리가 할 수 있는 가장 보수적인 분포 선택입니다.

섹션을 마무리하기 위해 $X sim \mathcal N(\mu, \sigma^2)$이면 다음이 성립함을 상기합시다.

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

아래와 같이 가우스(또는 표준 정규) 분포로부터 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 지수족 (Exponential Family)
:label:`subsec_exponential_family`

위에 나열된 모든 분포의 한 가지 공유된 속성은 그것들이 모두 *지수족(exponential family)*이라고 알려진 것에 속한다는 것입니다. 지수족은 밀도가 다음과 같은 형태로 표현될 수 있는 분포들의 집합입니다.

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp 
\left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) 
\right) 
$$ 
:eqlabel:`eq_exp_pdf`

이 정의는 약간 미묘할 수 있으므로 자세히 살펴봅시다.

먼저, $h(\mathbf{x})$는 *기저 척도(underlying measure)* 또는 *베이스 척도(base measure)*로 알려져 있습니다. 이는 우리가 지수 가중치로 수정하고 있는 원래의 척도 선택으로 볼 수 있습니다.

둘째, *자연 파라미터(natural parameters)* 또는 *표준 파라미터(canonical parameters)*라고 불리는 벡터 $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) sin 
\mathbb{R}^l$가 있습니다. 이들은 베이스 척도가 어떻게 수정될지를 정의합니다. 자연 파라미터는 이러한 파라미터와 $\mathbf{x}= (x_1, x_2, ..., x_n) sin 
\mathbb{R}^n$의 어떤 함수 $T(\cdot)$ 사이의 내적을 취하고 지수화함으로써 새로운 척도로 들어갑니다. 벡터 $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$는 $\boldsymbol{\eta}$에 대한 *충분 통계량(sufficient statistics)*이라고 불립니다. 이 이름은 $T(\mathbf{x})$로 표현된 정보가 확률 밀도를 계산하기에 충분하며 샘플 $\mathbf{x}$로부터의 다른 정보는 필요하지 않기 때문에 사용됩니다.

셋째, *큐뮬런트 함수(cumulant function)*라고 지칭되는 $A(\boldsymbol{\eta})$가 있으며, 이는 위의 분포 :eqref:`eq_exp_pdf`가 1로 적분되도록 보장합니다. 즉, 다음과 같습니다.

$$A(\boldsymbol{\eta})  = \log 
\left[\int h(\mathbf{x}) \cdot \exp 
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) 
\right) d\mathbf{x} 
\right].$$ 

구체적으로 가우스 분포를 고려해 봅시다. $\mathbf{x}$가 일변량 변수라고 가정할 때, 우리는 그것이 다음과 같은 밀도를 가짐을 보았습니다.

$$ 
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot 
\exp 
\left\{
\frac{-(x-\mu)^2}{2 \sigma^2}
\right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot 
\exp 
\left\{
\frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - 
\left( 
\frac{1}{2 \sigma^2} \mu^2
+\log(\sigma)
\right)
\right\}.
\end{aligned}
$$ 

이는 지수족의 정의와 다음과 같이 일치합니다.

* *기저 척도*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *자연 파라미터*: $\boldsymbol{\eta} = egin{bmatrix} ̣́\eta_1 \ \eta_2 
\end{bmatrix} = egin{bmatrix} rac{\mu}{\sigma^2} \ rac{1}{2 \sigma^2} 
\end{bmatrix}$,
* *충분 통계량*: $T(x) = egin{bmatrix}x\-x^2
\end{bmatrix}$, 
* *큐뮬런트 함수*: $A({\boldsymboḷ́̃}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= rac{\eta_1^2}{4 \eta_2} - rac{1}{2}\\log(2 \eta_2)$.

위의 각 항의 정확한 선택은 다소 임의적이라는 점에 주목할 가치가 있습니다. 실제로 중요한 특징은 분포가 이 형태로 표현될 수 있다는 것이지, 정확한 형태 그 자체가 아닙니다.

:numref:`subsec_softmax_and_derivatives`에서 암시했듯이, 널리 사용되는 기술은 최종 출력 $\mathbf{y}$가 지수족 분포를 따른다고 가정하는 것입니다. 지수족은 머신러닝에서 빈번하게 마주치는 흔하고 강력한 분포 가족입니다.


## 요약 (Summary)
* 베르누이 확률 변수는 예/아니오 결과가 있는 이벤트를 모델링하는 데 사용될 수 있습니다.
* 이산 균등 분포는 유한한 가능성 세트로부터의 선택을 모델링합니다.
* 연속 균등 분포는 구간으로부터의 선택을 모델링합니다.
* 이항 분포는 일련의 베르누이 확률 변수를 모델링하고 성공 횟수를 셉니다.
* 포아송 확률 변수는 희귀 사건의 도착을 모델링합니다.
* 가우스 확률 변수는 많은 수의 독립 확률 변수를 함께 더한 결과를 모델링합니다.
* 위의 모든 분포는 지수족에 속합니다.

## 연습 문제 (Exercises)

1. 두 독립적인 이항 확률 변수 $X, Y sim 	extrm{Binomial}(16, 1/2)$의 차이인 $X-Y$ 확률 변수의 표준 편차는 얼마입니까?
2. 포아송 확률 변수 $X sim 	extrm{Poisson}(\lambda)$를 취하고 $\lambda ightarrow \infty$에 따라 $(X - \lambda)/\sqrt{\lambda}$를 고려하면, 이것이 대략 가우스 분포가 됨을 보일 수 있습니다. 이것이 왜 말이 됩니까?
3. $n$개 요소에 대한 두 이산 균등 확률 변수의 합에 대한 확률 질량 함수는 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab: