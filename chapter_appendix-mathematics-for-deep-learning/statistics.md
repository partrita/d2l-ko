# 통계 (Statistics)
:label:`sec_statistics`

의심할 여지 없이, 최고의 딥러닝 전문가가 되기 위해서는 최첨단의 고정밀 모델을 훈련하는 능력이 매우 중요합니다. 그러나 개선 사항이 언제 유의미한지, 아니면 단지 훈련 과정의 무작위 변동의 결과인지가 불분명한 경우가 많습니다. 추정값의 불확실성에 대해 논의하려면 통계를 배워야 합니다.


*통계(statistics)*에 대한 가장 이른 기록은 9세기의 아랍 학자 알킨디(Al-Kindi)로 거슬러 올라갑니다. 그는 암호화된 메시지를 해독하기 위해 통계와 빈도 분석을 사용하는 방법에 대해 자세히 설명했습니다. 800년 후, 현대 통계학은 1700년대 독일에서 연구자들이 인구통계학적 및 경제적 데이터 수집과 분석에 집중하면서 시작되었습니다. 오늘날 통계학은 데이터의 수집, 처리, 분석, 해석 및 시각화를 다루는 과학 과목입니다. 게다가 통계학의 핵심 이론은 학계, 산업계 및 정부 내의 연구에서 널리 사용되어 왔습니다.


더 구체적으로, 통계학은 *기술 통계(descriptive statistics)*와 *통계적 추론(statistical inference)*으로 나뉠 수 있습니다. 전자는 *표본(sample)*이라고 불리는 관찰된 데이터 모음의 특징을 요약하고 설명하는 데 집중합니다. 표본은 *모집단(population)*에서 추출되며, 이는 우리 실험 관심사의 유사한 개인, 항목 또는 사건의 전체 집합을 나타냅니다. 기술 통계와 반대로, *통계적 추론*은 표본 분포가 어느 정도 모집단 분포를 재현할 수 있다는 가정 하에 주어진 *표본*으로부터 모집단의 특성을 추론합니다.


여러분은 "머신러닝과 통계의 본질적인 차이점은 무엇인가?"라고 궁금해할 수 있습니다. 근본적으로 말해서, 통계학은 추론 문제에 집중합니다. 이러한 유형의 문제에는 인과 추론과 같은 변수 간의 관계 모델링, A/B 테스팅과 같은 모델 파라미터의 통계적 유의성 테스트가 포함됩니다. 반대로 머신러닝은 각 파라미터의 기능을 명시적으로 프로그래밍하고 이해하지 않고도 정확한 예측을 하는 것을 강조합니다.

이 섹션에서는 추정량 평가 및 비교, 가설 검정 수행, 신뢰 구간 구축의 세 가지 유형의 통계 추론 방법을 소개합니다. 이러한 방법들은 주어진 모집단의 특성, 즉 실제 파라미터 $\theta$를 추론하는 데 도움이 될 수 있습니다. 간결함을 위해 주어진 모집단의 실제 파라미터 $\theta$가 스칼라 값이라고 가정합니다. $\theta$가 벡터나 텐서인 경우로 확장하는 것은 간단하므로 논의에서는 생략합니다.



## 추정량 평가 및 비교 (Evaluating and Comparing Estimators)

통계학에서 *추정량(estimator)*은 실제 파라미터 $\theta$를 추정하기 위해 사용되는 주어진 표본들의 함수입니다. 표본 {$x_1, x_2, \ldots, x_n$}을 관찰한 후 $\theta$에 대한 추정값을 $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$이라고 씁니다.

우리는 이전에 섹션 :numref:`sec_maximum_likelihood`에서 추정량의 간단한 예를 보았습니다. 베르누이 확률 변수로부터 여러 표본을 가지고 있다면, 확률 변수가 1일 확률에 대한 최대 우도 추정량은 관찰된 1의 개수를 세고 총 표본 수로 나눔으로써 얻을 수 있습니다. 마찬가지로, 연습 문제에서는 여러 표본이 주어졌을 때 가우시안 평균의 최대 우도 추정량이 모든 표본의 평균값으로 주어진다는 것을 보여달라고 요청했습니다. 이러한 추정량들이 파라미터의 실제 값을 제공하는 경우는 거의 없지만, 이상적으로는 표본 수가 많을 때 추정값이 실제값에 가까울 것입니다.

예를 들어, 아래에 평균이 0이고 분산이 1인 가우시안 확률 변수의 실제 밀도와 그 가우시안에서 추출한 표본 모음을 보여줍니다. 모든 점이 보이고 원래 밀도와의 관계가 더 명확해지도록 $y$ 좌표를 구성했습니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# 데이터 포인트 샘플링 및 y 좌표 생성
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# 실제 밀도 계산
xd = np.arange(np.min(xs), np.max(xs), 0.01)

d = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# 결과 플롯
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # torch에서 pi 정의

# 데이터 포인트 샘플링 및 y 좌표 생성
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)
     for i in range(len(xs))])

# 실제 밀도 계산
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)

d = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# 결과 플롯
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # TensorFlow에서 pi 정의

# 데이터 포인트 샘플링 및 y 좌표 생성
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# 실제 밀도 계산
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)

d = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# 결과 플롯
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

파라미터의 추정량 $\hat{\theta}_n$을 계산하는 방법은 많을 수 있습니다. 이 섹션에서는 추정량을 평가하고 비교하는 세 가지 일반적인 방법인 평균 제곱 오차, 표준 편차 및 통계적 편향을 소개합니다.

### 평균 제곱 오차 (Mean Squared Error)

추정량을 평가하는 데 사용되는 아마도 가장 간단한 메트릭은 *평균 제곱 오차(mean squared error, MSE)* (또는 $l_2$ 손실) 추정량이며 다음과 같이 정의될 수 있습니다.

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

이를 통해 실제 값으로부터의 평균 제곱 편차를 정량화할 수 있습니다. MSE는 항상 음수가 아닙니다. :numref:`sec_linear_regression`을 읽었다면 이를 가장 흔히 사용되는 회귀 손실 함수로 인식할 것입니다. 추정량을 평가하는 척도로서, 그 값이 0에 가까울수록 추정량이 실제 파라미터 $\theta$에 더 가깝습니다.


### 통계적 편향 (Statistical Bias)

MSE는 자연스러운 메트릭을 제공하지만, 이를 크게 만들 수 있는 여러 다른 현상을 쉽게 상상할 수 있습니다. 근본적으로 중요한 두 가지는 데이터셋의 무작위성으로 인한 추정량의 변동과 추정 절차로 인한 추정량의 계통 오차(systematic error)입니다.


먼저 계통 오차를 측정해 봅시다. 추정량 $\hat{\theta}_n$에 대해, *통계적 편향(statistical bias)*의 수학적 설명은 다음과 같이 정의될 수 있습니다.

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\\textrm{bias}(\hat{\theta}_n) = 0$일 때 추정량 $\hat{\theta}_n$의 기댓값은 파라미터의 실제 값과 같습니다. 이 경우 $\hat{\theta}_n$을 불편 추정량(unbiased estimator)이라고 합니다. 일반적으로 불편 추정량은 기댓값이 실제 파라미터와 같기 때문에 편향 추정량보다 낫습니다.


그러나 편향 추정량이 실제에서 자주 사용된다는 점을 알아두는 것이 좋습니다. 추가적인 가정 없이는 불편 추정량이 존재하지 않거나 계산하기 어려운 경우가 있습니다. 이는 추정량의 중대한 결함처럼 보일 수 있지만, 실제에서 만나는 대다수의 추정량은 가용한 표본 수가 무한대로 갈 때 편향이 0으로 수렴한다는 의미에서 적어도 점근적 불편 추정량(asymptotically unbiased)입니다: $\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$.


### 분산과 표준 편차 (Variance and Standard Deviation)

둘째, 추정량의 무작위성을 측정해 봅시다. :numref:`sec_random_variables`에서 상기했듯이, *표준 편차(standard deviation)* (또는 *표준 오차(standard error)*)는 분산의 제곱근으로 정의됩니다. 우리는 해당 추정량의 표준 편차나 분산을 측정함으로써 추정량의 변동 정도를 측정할 수 있습니다.

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$ 
:eqlabel:`eq_var_est`

:eqref:`eq_var_est`를 :eqref:`eq_mse_est`와 비교하는 것이 중요합니다. 이 방정식에서는 실제 모집단 값 $\theta$와 비교하는 것이 아니라, 기대 표본 평균인 $E(\hat{\theta}_n)$과 비교합니다. 따라서 추정량이 실제 값에서 얼마나 떨어져 있는지를 측정하는 것이 아니라, 추정량 자체의 변동을 측정하는 것입니다.


### 편향-분산 트레이드오프 (The Bias-Variance Trade-off)

이 두 가지 주요 구성 요소가 평균 제곱 오차에 기여한다는 것은 직관적으로 명확합니다. 다소 놀라운 점은 이것이 실제로 평균 제곱 오차를 이 두 기여도와 세 번째 기여도로 *분해*한 것임을 보여줄 수 있다는 것입니다. 즉, 평균 제곱 오차를 편향의 제곱, 분산, 그리고 줄일 수 없는 오차의 합으로 쓸 수 있습니다.

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \ 
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \ 
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \ 
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \ 
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \ 
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\ 
\end{aligned}
$$ 

우리는 위의 공식을 *편향-분산 트레이드오프(bias-variance trade-off)*라고 부릅니다. 평균 제곱 오차는 세 가지 오차 원인으로 나뉠 수 있습니다: 높은 편향으로 인한 오차, 높은 분산으로 인한 오차, 그리고 줄일 수 없는 오차입니다. 편향 오차는 특성과 출력 사이의 고차원 관계를 추출할 수 없는 단순한 모델(예: 선형 회귀 모델)에서 흔히 보입니다. 모델이 높은 편향 오차를 겪는다면, 우리는 종종 이를 (:numref:`sec_generalization_basics`에서 도입된 것처럼) *과소적합(underfitting)* 또는 *유연성(flexibility)* 부족이라고 말합니다. 높은 분산은 일반적으로 훈련 데이터에 과대적합되는 너무 복잡한 모델에서 비롯됩니다. 결과적으로 *과대적합(overfitting)* 모델은 데이터의 작은 변동에 민감합니다. 모델이 높은 분산을 겪는다면, 우리는 종종 이를 (:numref:`sec_generalization_basics`에서 도입된 것처럼) *과대적합* 및 *일반화(generalization)* 부족이라고 말합니다. 줄일 수 없는 오차는 $\theta$ 자체의 노이즈로 인한 결과입니다.


### 코드로 추정량 평가하기 (Evaluating Estimators in Code)

추정량의 표준 편차는 텐서 `a`에 대해 단순히 `a.std()`를 호출함으로써 구현되어 왔으므로, 여기서는 생략하고 통계적 편향과 평균 제곱 오차를 구현해 보겠습니다.

```{.python .input}
#@tab mxnet
# 통계적 편향
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# 평균 제곱 오차
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# 통계적 편향
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# 평균 제곱 오차
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# 통계적 편향
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# 평균 제곱 오차
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

편향-분산 트레이드오프 방정식을 설명하기 위해, 10,000개의 표본으로 정규 분포 $\mathcal{N}(\theta, \sigma^2)$를 시뮬레이션해 봅시다. 여기서는 $\theta = 1$ 및 $\sigma = 4$를 사용합니다. 추정량은 주어진 표본들의 함수이므로, 여기서는 이 정규 분포 $\mathcal{N}(\theta, \sigma^2)$에서 실제 $\theta$에 대한 추정량으로 표본의 평균을 사용합니다.

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

우리 추정량의 편향 제곱과 분산의 합을 계산하여 트레이드오프 방정식을 검증해 봅시다. 먼저 우리 추정량의 MSE를 계산합니다.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

다음으로, 아래와 같이 $\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$를 계산합니다. 보시다시피, 두 값은 수치적 정밀도 내에서 일치합니다.

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 가설 검정 수행하기 (Conducting Hypothesis Tests)


통계적 추론에서 가장 흔히 접하는 주제는 가설 검정입니다. 가설 검정은 20세기 초에 대중화되었지만, 첫 번째 사용은 1700년대의 존 아버트넛(John Arbuthnot)으로 거슬러 올라갑니다. 존은 런던의 80년 치 출생 기록을 추적하여 매년 여성보다 남성이 더 많이 태어났다는 결론을 내렸습니다. 그 후, 현대적인 유의성 검정은 $p$-값과 피어슨의 카이제곱 검정을 발명한 칼 피어슨(Karl Pearson), 스튜던트 t-분포의 아버지인 윌리엄 고셋(William Gosset), 그리고 귀무 가설과 유의성 검정을 처음 시작한 로널드 피셔(Ronald Fisher)에 의한 지적 유산입니다.

*가설 검정(hypothesis test)*은 모집단에 대한 기본 진술에 반하는 어떤 증거를 평가하는 방법입니다. 우리는 기본 진술을 *귀무 가설(null hypothesis)* $H_0$라고 부르며, 관찰된 데이터를 사용하여 이를 기각하려고 시도합니다. 여기서 우리는 $H_0$를 통계적 유의성 검정의 출발점으로 사용합니다. *대립 가설(alternative hypothesis)* $H_A$ (또는 $H_1$)는 귀무 가설과 상반되는 진술입니다. 귀무 가설은 종종 변수 간의 관계를 가정하는 서술형 형식으로 명시됩니다. 그것은 가능한 한 명확하게 신념을 반영해야 하며, 통계 이론에 의해 테스트 가능해야 합니다.

당신이 화학자라고 상상해 보십시오. 실험실에서 수천 시간을 보낸 후, 당신은 수학을 이해하는 능력을 비약적으로 향상시킬 수 있는 새로운 약을 개발합니다. 그 마법 같은 힘을 보여주기 위해, 당신은 그것을 테스트해야 합니다. 당연히, 약을 복용하고 수학을 더 잘 배우는 데 도움이 되는지 확인해 줄 자원봉사자들이 필요할 것입니다. 어떻게 시작하시겠습니까?

첫째, 어떤 메트릭으로 측정했을 때 수학적 이해 능력에 차이가 없도록 신중하게 무작위로 선택된 두 그룹의 자원봉사자가 필요할 것입니다. 두 그룹은 흔히 실험군(test group)과 대조군(control group)으로 불립니다. *실험군* (또는 *처치군(treatment group)*)은 약을 경험하게 될 개인 그룹인 반면, *대조군*은 벤치마크로 설정된 사용자 그룹을 나타냅니다. 즉, 이 약을 복용하는 것을 제외하고는 동일한 환경 설정입니다. 이런 식으로, 처치에서의 독립 변수의 영향을 제외하고 모든 변수의 영향이 최소화됩니다.

둘째, 일정 기간 약을 복용한 후, 새로운 수학 공식을 배운 후 자원봉사자들에게 동일한 테스트를 보게 하는 것과 같이 동일한 메트릭으로 두 그룹의 수학적 이해도를 측정해야 할 것입니다. 그런 다음 그들의 성적을 수집하고 결과를 비교할 수 있습니다. 이 경우, 우리의 귀무 가설은 두 그룹 사이에 차이가 없다는 것이고, 대립 가설은 차이가 있다는 것입니다.

이것은 여전히 완전히 형식적이지는 않습니다. 당신이 신중하게 생각해야 할 많은 세부 사항들이 있습니다. 예를 들어, 그들의 수학적 이해 능력을 테스트하기에 적합한 메트릭은 무엇입니까? 약의 효과를 주장하는 데 확신을 가질 수 있도록 얼마나 많은 자원봉사자가 필요합니까? 테스트를 얼마나 오래 실행해야 합니까? 두 그룹 사이에 차이가 있는지 어떻게 결정합니까? 평균 성적에만 관심이 있습니까, 아니면 점수의 변동 범위에도 관심이 있습니까? 등등.

이런 방식으로, 가설 검정은 실험 설계와 관찰된 결과의 확실성에 대한 추론을 위한 프레임워크를 제공합니다. 이제 귀무 가설이 참일 가능성이 매우 낮다는 것을 보여줄 수 있다면, 우리는 확신을 가지고 이를 기각할 수 있습니다.

가설 검정을 수행하는 방법에 대한 이야기를 완성하기 위해, 이제 몇 가지 추가 용어를 도입하고 위의 개념들을 형식화해야 합니다.


### 통계적 유의성 (Statistical Significance)

*통계적 유의성(statistical significance)*은 귀무 가설 $H_0$를 기각해서는 안 될 때 잘못 기각할 확률을 측정합니다. 즉,

$$\textrm{통계적 유의성 }= 1 - \alpha = 1 - P(\textrm{기각 } H_0 \mid H_0 \textrm{ 가 참} ).$$

이를 *제1종 오류(type I error)* 또는 *위양성(false positive)*이라고도 합니다. $\alpha$는 *유의 수준(significance level)*이라고 불리며 그 흔히 사용되는 값은 $5\%$입니다. 즉, $1-\alpha = 95\%$입니다. 유의 수준은 참인 귀무 가설을 기각할 때 우리가 감수할 용의가 있는 위험 수준으로 설명될 수 있습니다.

:numref:`fig_statistical_significance`는 두 표본 가설 검정에서 주어진 정규 분포의 관찰값과 확률을 보여줍니다. 관찰 데이터 예제가 $95\%$ 임계값 밖에 위치한다면, 이는 귀무 가설 가정 하에서 매우 일어날 법하지 않은 관찰이 될 것입니다. 따라서 귀무 가설에 무언가 잘못되었을 수 있으며 우리는 이를 기각할 것입니다.

![통계적 유의성.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`



### 통계적 검정력 (Statistical Power)

*통계적 검정력(statistical power)* (또는 *민감도(sensitivity)*)은 귀무 가설 $H_0$를 기각해야 할 때 기각할 확률을 측정합니다. 즉,

$$\textrm{통계적 검정력 }= 1 - \beta = 1 - P(\textrm{ 기각 실패 } H_0  \mid H_0 \textrm{ 가 거짓} ).$$

*제1종 오류*는 귀무 가설이 참일 때 기각함으로써 발생하는 오류인 반면, *제2종 오류(type II error)*는 귀무 가설이 거짓일 때 기각하지 못함으로써 발생하는 오류임을 상기하십시오. 제2종 오류는 보통 $\beta$로 표시되며, 따라서 해당 통계적 검정력은 $1-\beta$입니다.


직관적으로, 통계적 검정력은 우리의 검정이 원하는 통계적 유의 수준에서 어떤 최소 규모의 실제 불일치를 얼마나 잘 감지할 것인지를 나타내는 것으로 해석될 수 있습니다. $80\%$는 흔히 사용되는 통계적 검정력 임계값입니다. 통계적 검정력이 높을수록 실제 차이를 감지할 가능성이 높아집니다.

통계적 검정력의 가장 일반적인 용도 중 하나는 필요한 표본 수를 결정하는 것입니다. 귀무 가설이 거짓일 때 이를 기각할 확률은 그것이 얼마나 거짓인지( *효과 크기(effect size)*라고 함)와 당신이 가진 표본 수에 달려 있습니다. 예상할 수 있듯이, 작은 효과 크기는 높은 확률로 감지하기 위해 매우 많은 수의 표본을 필요로 할 것입니다. 이 짧은 부록의 범위를 벗어나 자세히 유도하지는 않겠지만, 예를 들어 우리 표본이 평균 0, 분산 1인 가우시안에서 왔다는 귀무 가설을 기각하고 싶고, 우리 표본의 평균이 실제로 1에 가깝다고 믿는다면, 단 $8$개의 표본 크기만으로도 허용 가능한 오차율로 그렇게 할 수 있습니다. 그러나 우리 표본 모집단의 실제 평균이 $0.01$에 가깝다고 생각한다면, 그 차이를 감지하기 위해 거의 $80,000$개의 표본 크기가 필요할 것입니다.

검정력을 정수기 필터로 상상해 볼 수 있습니다. 이 비유에서, 고검정력 가설 검정은 물속의 유해 물질을 가능한 많이 줄여주는 고품질 정수 시스템과 같습니다. 반면에, 더 작은 불일치는 저품질 정수 필터와 같아서 일부 상대적으로 작은 물질들이 틈새로 쉽게 빠져나갈 수 있습니다. 마찬가지로, 통계적 검정력이 충분히 높지 않으면 검정에서 더 작은 불일치를 잡아내지 못할 수 있습니다.


### 검정 통계량 (Test Statistic)

*검정 통계량(test statistic)* $T(x)$는 표본 데이터의 어떤 특성을 요약하는 스칼라입니다. 이러한 통계량을 정의하는 목표는 우리가 서로 다른 분포를 구별하고 가설 검정을 수행할 수 있도록 하는 것입니다. 우리의 화학자 예제를 다시 생각해 보면, 한 집단이 다른 집단보다 더 잘 수행된다는 것을 보여주고 싶다면 평균을 검정 통계량으로 취하는 것이 합리적일 수 있습니다. 검정 통계량의 다른 선택은 현저히 다른 통계적 검정력을 가진 통계 검정으로 이어질 수 있습니다.

종종 $T(X)$ (귀무 가설 하에서의 검정 통계량 분포)는 귀무 가설 하에서 고려될 때 정규 분포와 같은 일반적인 확률 분포를 적어도 근사적으로 따를 것입니다. 우리가 그러한 분포를 명시적으로 유도할 수 있고 우리 데이터셋에서 검정 통계량을 측정할 수 있다면, 우리 통계량이 우리가 예상하는 범위를 크게 벗어날 때 안전하게 귀무 가설을 기각할 수 있습니다. 이를 정량화하는 것은 $p$-값의 개념으로 이어집니다.


### $p$-값 ($p$-value)

$p$-값 (또는 *확률 값*)은 귀무 가설이 *참*이라고 가정할 때, $T(X)$가 관찰된 검정 통계량 $T(x)$만큼 극단적일 확률입니다. 즉,

$$ p\textrm{-값} = P_{H_0}(T(X) \geq T(x)).$$

$p$-값이 미리 정의되고 고정된 통계적 유의 수준 $\alpha$보다 작거나 같으면, 우리는 귀무 가설을 기각할 수 있습니다. 그렇지 않으면, 우리는 귀무 가설을 기각할 증거가 부족하다고 결론 내릴 것입니다. 주어진 모집단 분포에 대해, *기각역(region of rejection)*은 통계적 유의 수준 $\alpha$보다 작은 $p$-값을 가진 모든 점들을 포함하는 구간이 될 것입니다.


### 단측 검정 및 양측 검정 (One-side Test and Two-sided Test)

보통 유의성 검정에는 두 가지 종류가 있습니다: 단측 검정과 양측 검정입니다. *단측 검정(one-sided test)* (또는 *한쪽 꼬리 검정*)은 귀무 가설과 대립 가설이 한 방향만 가질 때 적용 가능합니다. 예를 들어, 귀무 가설은 실제 파라미터 $\theta$가 어떤 값 $c$보다 작거나 같다고 명시할 수 있습니다. 대립 가설은 $\theta$가 $c$보다 크다는 것이 될 것입니다. 즉, 기각역은 표본 분포의 한쪽에만 있습니다. 단측 검정과 반대로, *양측 검정(two-sided test)* (또는 *양쪽 꼬리 검정*)은 기각역이 표본 분포의 양쪽에 있을 때 적용 가능합니다. 이 경우의 예로는 귀무 가설이 실제 파라미터 $\theta$가 어떤 값 $c$와 같다고 명시하는 경우가 있습니다. 대립 가설은 $\theta$가 $c$와 같지 않다는 것이 될 것입니다.


### 가설 검정의 일반적인 단계 (General Steps of Hypothesis Testing)

위의 개념들에 익숙해진 후, 가설 검정의 일반적인 단계를 살펴봅시다.

1. 질문을 명시하고 귀무 가설 $H_0$를 설정합니다.
2. 통계적 유의 수준 $\alpha$와 통계적 검정력 ($1 - \beta$)을 설정합니다.
3. 실험을 통해 표본을 얻습니다. 필요한 표본 수는 통계적 검정력과 예상 효과 크기에 달려 있습니다.
4. 검정 통계량과 $p$-값을 계산합니다.
5. $p$-값과 통계적 유의 수준 $\alpha$를 바탕으로 귀무 가설을 유지할지 기각할지 결정합니다.

가설 검정을 수행하기 위해, 우리는 귀무 가설과 우리가 감수할 용의가 있는 위험 수준을 정의하는 것으로 시작합니다. 그런 다음 표본의 검정 통계량을 계산하고, 검정 통계량의 극단적인 값을 귀무 가설에 반하는 증거로 삼습니다. 검정 통계량이 기각역에 떨어지면, 우리는 대립 가설을 지지하며 귀무 가설을 기각할 수 있습니다.

가설 검정은 임상 시험 및 A/B 테스팅과 같은 다양한 시나리오에서 적용 가능합니다.


## 신뢰 구간 구축하기 (Constructing Confidence Intervals)


파라미터 $\theta$의 값을 추정할 때, $\hat \theta$와 같은 점 추정량은 불확실성의 개념을 포함하지 않기 때문에 유용성이 제한적입니다. 오히려, 높은 확률로 실제 파라미터 $\theta$를 포함할 구간을 생성할 수 있다면 훨씬 더 좋을 것입니다. 만약 당신이 한 세기 전에 그러한 아이디어에 관심이 있었다면, 1937년에 신뢰 구간의 개념을 처음 도입한 예르지 네이만(Jerzy Neyman)의 "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" :cite:`Neyman.1937`를 읽고 흥분했을 것입니다.

유용하려면, 신뢰 구간은 주어진 확실성 정도에 대해 가능한 한 작아야 합니다. 이를 유도하는 방법을 알아봅시다.


### 정의 (Definition)

수학적으로, 실제 파라미터 $\theta$에 대한 *신뢰 구간(confidence interval)*은 표본 데이터로부터 계산된 구간 $C_n$으로, 다음을 만족합니다.

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

여기서 $\alpha \in (0, 1)$이고, $1 - \alpha$는 구간의 *신뢰 수준(confidence level)* 또는 *커버리지(coverage)*라고 불립니다. 이것은 위에서 논의한 유의 수준과 동일한 $\alpha$입니다.

:eqref:`eq_confidence`는 고정된 $\theta$가 아니라 변수 $C_n$에 관한 것임에 유의하십시오. 이를 강조하기 위해, $P_{\theta} (\theta \in C_n)$ 대신 $P_{\theta} (C_n \ni \theta)$라고 씁니다.


### 해석 (Interpretation)

$95\%$ 신뢰 구간을 실제 파라미터가 존재할 확률이 $95\%$인 구간으로 해석하고 싶은 유혹이 매우 강하지만, 아쉽게도 이는 사실이 아닙니다. 실제 파라미터는 고정되어 있고, 무작위인 것은 바로 구간입니다. 따라서 더 나은 해석은 이 절차에 의해 많은 수의 신뢰 구간을 생성했다면, 생성된 구간의 $95\%$가 실제 파라미터를 포함할 것이라고 말하는 것입니다.

이는 매우 까다롭게 보일 수 있지만, 결과의 해석에 실제적인 영향을 미칠 수 있습니다. 특히, 우리는 거의 드물게만 그렇게 한다면, 실제 값을 포함하지 않는다는 것이 *거의 확실한* 구간을 구축함으로써 :eqref:`eq_confidence`를 충족할 수 있습니다. 이 섹션을 마치며 유혹적이지만 틀린 세 가지 진술을 제공합니다. 이러한 점들에 대한 심층적인 논의는 :citet:`Morey.Hoekstra.Rouder.ea.2016`에서 찾을 수 있습니다.

* **오류 1**. 좁은 신뢰 구간은 파라미터를 정밀하게 추정할 수 있음을 의미한다.
* **오류 2**. 신뢰 구간 내부의 값이 구간 외부의 값보다 실제 값일 가능성이 더 높다.
* **오류 3**. 특정하게 관찰된 $95\%$ 신뢰 구간이 실제 값을 포함할 확률은 $95\%$이다.

말하자면, 신뢰 구간은 미묘한 대상입니다. 그러나 해석을 명확히 유지한다면 강력한 도구가 될 수 있습니다.


### 가우시안 예제 (A Gaussian Example)

가장 고전적인 예제인, 평균과 분산을 모르는 가우시안 평균에 대한 신뢰 구간을 논의해 봅시다. 가우시안 $\mathcal{N}(\mu, \sigma^2)$에서 $n$개의 표본 {$x_i$}$_{i=1}^n$을 수집한다고 가정합시다. 우리는 다음과 같이 평균과 분산에 대한 추정량을 계산할 수 있습니다.

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{및}\; \hat{\sigma}^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat{\mu})^2.$$

이제 확률 변수

$$ 
 T = \frac{\hat{\mu}_n - \mu}{\hat{\sigma}_n/\sqrt{n}},
$$ 

를 고려하면, 우리는 *자유도가* $n-1$ *인 스튜던트 t-분포*라고 불리는 잘 알려진 분포를 따르는 확률 변수를 얻습니다.

이 분포는 매우 잘 연구되어 있으며, 예를 들어 $n\rightarrow \infty$임에 따라 대략 표준 가우시안이 된다는 것이 알려져 있습니다. 따라서 표에서 가우시안 c.d.f. 값을 찾아봄으로써, $T$의 값이 적어도 $95\%$의 시간에 구간 $[-1.96, 1.96]$에 있다고 결론 내릴 수 있습니다. 유한한 $n$ 값의 경우 구간은 다소 더 커야 하지만, 잘 알려져 있으며 표에 미리 계산되어 있습니다.

따라서 우리는 큰 $n$에 대해 다음과 같이 결론 내릴 수 있습니다.

$$ P\left(\frac{\hat{\mu}_n - \mu}{\hat{\sigma}_n/\sqrt{n}} \in [-1.96, 1.96]\right) \geq 0.95. $$

양변에 $\hat{\sigma}_n/\sqrt{n}$을 곱한 다음 $\hat{\mu}_n$을 더하여 이를 재배열하면 다음을 얻습니다.

$$ P\left(\mu \in \left[\hat{\mu}_n - 1.96\frac{\hat{\sigma}_n}{\sqrt{n}}, \hat{\mu}_n + 1.96\frac{\hat{\sigma}_n}{\sqrt{n}}\right]\right) \geq 0.95. $$

이렇게 해서 우리는 $95\%$ 신뢰 구간을 찾았음을 알게 됩니다:
$$\left[\hat{\mu}_n - 1.96\frac{\hat{\sigma}_n}{\sqrt{n}}, \hat{\mu}_n + 1.96\frac{\hat{\sigma}_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

:eqref:`eq_gauss_confidence`는 통계학에서 가장 많이 사용되는 공식 중 하나라고 해도 과언이 아닙니다. 이 공식을 구현함으로써 통계에 대한 논의를 마칩니다. 단순함을 위해 점근적 영역(asymptotic regime)에 있다고 가정합니다. $N$의 작은 값은 프로그래밍 방식이나 $t$-표로부터 얻은 `t_star`의 올바른 값을 포함해야 합니다.

```{.python .input}
#@tab mxnet
# 표본 수
N = 1000

# 표본 데이터셋
samples = np.random.normal(loc=0, scale=1, size=(N,))

# 스튜던트 t-분포 c.d.f. 조회
t_star = 1.96

# 구간 구축
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch는 기본적으로 베셀 보정(Bessel's correction)을 사용하며, 이는 numpy의 기본 ddof=0 대신 
# ddof=1을 사용함을 의미합니다. ddof=0을 모방하기 위해 unbiased=False를 사용할 수 있습니다.

# 표본 수
N = 1000

# 표본 데이터셋
samples = torch.normal(0, 1, size=(N,))

# 스튜던트 t-분포 c.d.f. 조회
t_star = 1.96

# 구간 구축
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# 표본 수
N = 1000

# 표본 데이터셋
samples = tf.random.normal((N,), 0, 1)

# 스튜던트 t-분포 c.d.f. 조회
t_star = 1.96

# 구간 구축
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## 요약 (Summary)

* 통계학은 추론 문제에 집중하는 반면, 딥러닝은 명시적인 프로그래밍과 이해 없이 정확한 예측을 하는 것을 강조합니다.
* 세 가지 일반적인 통계 추론 방법이 있습니다: 추정량 평가 및 비교, 가설 검정 수행, 신뢰 구간 구축입니다.
* 가장 일반적인 세 가지 추정량이 있습니다: 통계적 편향, 표준 편차, 그리고 평균 제곱 오차입니다.
* 신뢰 구간은 주어진 표본들로부터 구축할 수 있는 실제 모집단 파라미터의 추정 범위입니다.
* 가설 검정은 모집단에 대한 기본 진술에 반하는 어떤 증거를 평가하는 방법입니다.


## 연습 문제 (Exercises)

1. $X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$라고 합시다. 여기서 "iid"는 *독립적이고 동일하게 분포된(independent and identically distributed)*을 의미합니다. $\theta$의 다음 추정량들을 고려해 보십시오:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};
$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$ 
    * $\hat{\theta}$의 통계적 편향, 표준 편차, 그리고 평균 제곱 오차를 구하십시오.
    * $\tilde{\theta}$의 통계적 편향, 표준 편차, 그리고 평균 제곱 오차를 구하십시오.
    * 어떤 추정량이 더 좋습니까?
2. 서론의 화학자 예제에 대해, 양측 가설 검정을 수행하기 위한 5단계를 유도할 수 있습니까? 통계적 유의 수준 $\alpha = 0.05$와 통계적 검정력 $1 - \beta = 0.8$이 주어졌다고 가정합니다.
3. 독립적으로 생성된 100개의 데이터셋에 대해 $N=2$ 및 $\alpha = 0.5$로 신뢰 구간 코드를 실행하고, 결과 구간들을 플롯하십시오 (이 경우 `t_star = 1.0`). 실제 평균 0을 포함하는 것과는 거리가 먼 매우 짧은 구간들을 몇 개 보게 될 것입니다. 이것이 신뢰 구간의 해석과 모순됩니까? 짧은 구간을 고정밀 추정치를 나타내는 데 사용하는 것이 편안하게 느껴집니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/1103)
:end_tab:
