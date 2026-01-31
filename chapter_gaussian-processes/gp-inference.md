```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# 가우시안 프로세스 추론 (Gaussian Process Inference)

이 섹션에서는 이전 섹션에서 소개한 GP 사전 분포를 사용하여 사후 추론을 수행하고 예측하는 방법을 보여줄 것입니다. 우리는 *닫힌 형식(closed form)*으로 추론을 수행할 수 있는 회귀로 시작할 것입니다. 이것은 가우시안 프로세스를 실제로 빠르게 시작하고 실행하기 위한 "요약된 GP" 섹션입니다. 처음부터 모든 기본 연산을 코딩한 다음, 최신 가우시안 프로세스 작업과 심층 신경망과의 통합을 훨씬 더 편리하게 만들어 줄 [GPyTorch](https://gpytorch.ai/)를 소개할 것입니다. 우리는 다음 섹션에서 이러한 고급 주제를 깊이 있게 고려할 것입니다. 해당 섹션에서는 분류, 포인트 프로세스 또는 비가우시안 우도와 같이 근사 추론이 필요한 설정도 고려할 것입니다.

## 회귀에 대한 사후 추론 (Posterior Inference for Regression)

*관찰(observation)* 모델은 우리가 학습하려는 함수 $f(x)$를 관찰 $y(x)$와 관련시키며, 둘 다 일부 입력 $x$에 의해 인덱싱됩니다. 분류에서 $x$는 이미지의 픽셀이 될 수 있고 $y$는 관련 클래스 레이블이 될 수 있습니다. 회귀에서 $y$는 일반적으로 지표면 온도, 해수면, $CO_2$ 농도 등과 같은 연속적인 출력을 나타냅니다.

회귀에서는 종종 출력이 잠재적인 노이즈 없는 함수 $f(x)$에 i.i.d. 가우시안 노이즈 $\epsilon(x)$를 더한 것으로 가정합니다.

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

여기서 $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$입니다. $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$를 훈련 관찰 벡터라고 하고, $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$를 훈련 입력 $X = {x_1, \dots, x_n}$에서 쿼리된 잠재 노이즈 없는 함수 값의 벡터라고 합시다.

우리는 $f(x) \sim \mathcal{GP}(m,k)$라고 가정할 것입니다. 이는 함수 값 $\textbf{f}$의 모음이 평균 벡터 $\mu_i = m(x_i)$와 공분산 행렬 $K_{ij} = k(x_i,x_j)$를 갖는 결합 다변량 가우시안 분포를 갖는다는 것을 의미합니다. RBF 커널 $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$는 공분산 함수의 표준 선택이 될 것입니다. 표기법의 단순성을 위해 평균 함수 $m(x)=0$이라고 가정하겠습니다. 우리의 유도는 나중에 쉽게 일반화될 수 있습니다.

입력 세트 $$X_* = x_{*1},x_{*2},\dots,x_{*m}$$에서 예측을 하고 싶다고 가정해 봅시다. 그런 다음 $p(\mathbf{f}_* | \mathbf{y}, X)$를 찾고 싶습니다. 회귀 설정에서는 $\mathbf{f}_* = f(X_*)$와 $\mathbf{y}$에 대한 결합 분포를 찾은 후 가우시안 항등식을 사용하여 이 분포를 편리하게 찾을 수 있습니다.

훈련 입력 $X$에서 방정식 :eqref:`eq_gp-regression`을 평가하면 $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$이 됩니다. 가우시안 프로세스의 정의(지난 섹션 참조)에 의해 $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$이며, 여기서 $K(X,X)$는 가능한 모든 입력 쌍 $x_i, x_j \in X$에서 공분산 함수(일명 *커널*)를 평가하여 형성된 $n \times n$ 행렬입니다. $\mathbf{\epsilon}$은 단순히 $\mathcal{N}(0,\sigma^2)$의 iid 샘플로 구성된 벡터이므로 분포 $\mathcal{N}(0,\sigma^2I)$를 갖습니다. 따라서 $\mathbf{y}$는 두 개의 독립적인 다변량 가우시안 변수의 합이므로 분포 $\mathcal{N}(0, K(X,X) + \sigma^2I)$를 갖습니다. 또한 $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$임을 보일 수 있습니다. 여기서 $K(X_*,X)$는 테스트 및 훈련 입력의 모든 쌍에서 커널을 평가하여 형성된 $m \times n$ 행렬입니다.

$$ 
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}


\sim


\mathcal{N}


\left(0, 


\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*)
\\K(X_*,X) & K(X_*,X_*)
\end{bmatrix}


\right)
$$ 

그런 다음 표준 가우시안 항등식을 사용하여 결합 분포에서 조건부 분포를 찾을 수 있습니다(예: Bishop Chapter 2 참조).
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$, 여기서 $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$이고 $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$입니다.

일반적으로 우리는 전체 예측 공분산 행렬 $S$를 사용할 필요가 없으며, 대신 각 예측에 대한 불확실성으로 $S$의 대각선을 사용합니다. 종종 이러한 이유로 테스트 포인트 모음이 아닌 단일 테스트 포인트 $x_*$에 대한 예측 분포를 씁니다.

커널 행렬에는 위의 RBF 커널의 진폭 $a$와 길이 척도 $\ell$과 같이 추정하고자 하는 파라미터 $\theta$가 있습니다. 이러한 목적으로 우리는 *주변 우도(marginal likelihood)* $p(\textbf{y} | \theta, X)$를 사용합니다. 이는 $\textbf{y},\textbf{f}_*$에 대한 결합 분포를 찾기 위해 주변 분포를 계산할 때 이미 유도했습니다. 보게 되겠지만, 주변 우도는 모델 적합성 및 모델 복잡성 항으로 구분되며 하이퍼파라미터 학습을 위한 오컴의 면도날 개념을 자동으로 인코딩합니다. 자세한 논의는 MacKay Ch. 28 :cite:`mackay2003information` 및 Rasmussen and Williams Ch. 5 :cite:`rasmussen2006gaussian`를 참조하십시오.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## GP 회귀에서 예측 및 커널 하이퍼파라미터 학습을 위한 방정식 (Equations for Making Predictions and Learning Kernel Hyperparameters in GP Regression)

여기서는 가우시안 프로세스 회귀에서 하이퍼파라미터를 학습하고 예측하는 데 사용할 방정식을 나열합니다. 다시 말하지만, 입력 $X = {x_1,\dots,x_n}$으로 인덱싱된 회귀 타겟 벡터 $\textbf{y}$를 가정하고 테스트 입력 $x_*$에서 예측을 하려고 합니다. 분산 $\sigma^2$를 갖는 i.i.d. 가법적 0 평균 가우시안 노이즈를 가정합니다. 우리는 잠재 노이즈 없는 함수에 대해 평균 함수 $m$과 커널 함수 $k$를 갖는 가우시안 프로세스 사전 분포 $f(x) \sim \mathcal{GP}(m,k)$를 사용합니다. 커널 자체에는 학습하려는 파라미터 $\theta$가 있습니다. 예를 들어 RBF 커널 $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$를 사용하는 경우 $\theta = {a^2, \ell^2}$를 학습하려고 합니다. $K(X,X)$를 $n$개의 훈련 입력의 가능한 모든 쌍에 대해 커널을 평가하는 것에 해당하는 $n \times n$ 행렬이라고 합시다. $K(x_*,X)$를 $i=1,\dots,n$에 대해 $k(x_*, x_i)$를 평가하여 형성된 $1 \times n$ 벡터라고 합시다. $\mu$를 모든 훈련 포인트 $x$에서 평균 함수 $m(x)$를 평가하여 형성된 평균 벡터라고 합시다.

일반적으로 가우시안 프로세스 작업에서는 두 단계 절차를 따릅니다.
1. 이러한 하이퍼파라미터에 대한 주변 우도를 최대화하여 커널 하이퍼파라미터 $\hat{\theta}$를 학습합니다.
2. 예측 평균을 점 예측기로 사용하고 예측 표준 편차의 2배를 사용하여 이러한 학습된 하이퍼파라미터 $\hat{\theta}$에 대한 95% 신용 집합을 형성합니다.

로그 주변 우도는 단순히 로그 가우시안 밀도이며 다음과 같은 형식을 갖습니다.
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

예측 분포는 다음과 같은 형식을 갖습니다.
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## 학습 및 예측을 위한 방정식 해석 (Interpreting Equations for Learning and Predictions)

가우시안 프로세스에 대한 예측 분포에 대해 주목해야 할 몇 가지 요점이 있습니다.

* 모델 클래스의 유연성에도 불구하고 GP 회귀에 대해 *닫힌 형식*으로 *정확한* 베이지안 추론을 수행할 수 있습니다. 커널 하이퍼파라미터를 학습하는 것 외에는 *훈련*이 없습니다. 예측을 위해 사용하려는 방정식을 정확히 적을 수 있습니다. 가우시안 프로세스는 이러한 측면에서 비교적 예외적이며, 이는 편리함, 다재다능함 및 지속적인 인기에 크게 기여했습니다.

* 예측 평균 $a_*$는 훈련 타겟 $\textbf{y}$의 선형 결합이며, 커널 $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$에 의해 가중치가 부여됩니다. 보게 되겠지만 커널(및 그 하이퍼파라미터)은 따라서 모델의 일반화 속성에서 중요한 역할을 합니다.

* 예측 평균은 타겟 값 $\textbf{y}$에 명시적으로 의존하지만 예측 분산은 그렇지 않습니다. 대신 예측 불확실성은 커널 함수에 의해 제어되는 대로 테스트 입력 $x_*$가 타겟 위치 $X$에서 멀어짐에 따라 증가합니다. 그러나 불확실성은 데이터에서 학습된 커널 하이퍼파라미터 $\theta$를 통해 타겟 $\textbf{y}$의 값에 암시적으로 의존합니다.

* 주변 우도는 모델 적합성 및 모델 복잡성(로그 행렬식) 항으로 구분됩니다. 주변 우도는 데이터와 여전히 일치하는 가장 단순한 피팅을 제공하는 하이퍼파라미터를 선택하는 경향이 있습니다.

* 주요 계산 병목 현상은 선형 시스템을 해결하고 $n$개의 훈련 포인트에 대해 $n \times n$ 대칭 양의 정부호 행렬 $K(X,X)$에 대한 로그 행렬식을 계산하는 데서 발생합니다. 순진하게 이러한 연산은 각각 $\mathcal{O}(n^3)$ 계산과 커널(공분산) 행렬의 각 항목에 대한 $\mathcal{O}(n^2)$ 저장소를 초래하며, 종종 촐레스키 분해로 시작합니다. 역사적으로 이러한 병목 현상은 GP를 약 10,000개 미만의 훈련 포인트가 있는 문제로 제한했으며 거의 10년 동안 부정확했던 "느리다"는 평판을 GP에 주었습니다. 고급 주제에서는 수백만 개의 포인트가 있는 문제로 GP를 확장하는 방법에 대해 논의할 것입니다.

* 널리 사용되는 커널 함수 선택의 경우, $K(X,X)$는 종종 특이(singular)에 가까워 촐레스키 분해 또는 선형 시스템을 해결하기 위한 기타 연산을 수행할 때 수치적 문제를 일으킬 수 있습니다. 다행히도 회귀에서는 종종 $K_{\theta}(X,X)+\sigma^2I$로 작업하므로 노이즈 분산 $\sigma^2$이 $K(X,X)$의 대각선에 추가되어 조건이 크게 개선됩니다. 노이즈 분산이 작거나 노이즈 없는 회귀를 수행하는 경우 조건을 개선하기 위해 대각선에 $10^{-6}$ 정도의 소량의 "지터(jitter)"를 추가하는 것이 일반적입니다.


## 처음부터 작업한 예제 (Worked Example from Scratch)

회귀 데이터를 생성한 다음, 처음부터 모든 단계를 구현하여 GP로 데이터를 피팅해 봅시다.
$\epsilon \sim \mathcal{N}(0,\sigma^2)$인 $$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon$$에서 데이터를 샘플링할 것입니다. 우리가 찾고자 하는 노이즈 없는 함수는 $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$입니다. 노이즈 표준 편차 $\sigma = 0.25$를 사용하여 시작하겠습니다.

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

여기서 우리는 원으로 표시된 노이즈가 있는 관찰과 파란색으로 표시된 우리가 찾고자 하는 노이즈 없는 함수를 봅니다.

이제 잠재 노이즈 없는 함수에 대한 GP 사전 분포 $f(x)\sim \mathcal{GP}(m,k)$를 지정해 봅시다. 평균 함수 $m(x) = 0$과 RBF 공분산 함수(커널)를 사용할 것입니다.
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

우리는 길이 척도 0.2로 시작했습니다. 데이터를 피팅하기 전에 합리적인 사전 분포를 지정했는지 고려하는 것이 중요합니다. 이 사전 분포의 샘플 함수 몇 가지와 95% 신용 집합(실제 함수가 이 영역 내에 있을 확률이 95%라고 믿습니다)을 시각화해 봅시다.

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

이 샘플들이 합리적으로 보입니까? 함수의 고수준 속성이 우리가 모델링하려는 데이터 유형과 일치합니까?

이제 임의의 테스트 포인트 $x_*$에서 사후 예측 분포의 평균과 분산을 형성해 봅시다.

$$ 
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$ 

$$ 
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$ 

예측을 하기 전에 커널 하이퍼파라미터 $\theta$와 노이즈 분산 $\sigma^2$을 학습해야 합니다. 사전 함수가 우리가 피팅하는 데이터에 비해 너무 빠르게 변하는 것처럼 보였으므로 길이 척도를 0.75로 초기화해 봅시다. 또한 노이즈 표준 편차 $\sigma$를 0.75로 추측할 것입니다.

이러한 파라미터를 학습하기 위해, 이 파라미터에 대한 주변 우도를 최대화할 것입니다.

$$ 
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$ 
$$ 
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$ 


아마도 우리의 사전 함수가 너무 빠르게 변했을 것입니다. 길이 척도를 0.4로 추측해 봅시다. 또한 노이즈 표준 편차를 0.75로 추측할 것입니다. 이들은 단순히 하이퍼파라미터 초기화입니다. 우리는 주변 우도에서 이러한 파라미터를 학습할 것입니다.

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

이 경우 우리는 길이 척도 0.299와 노이즈 표준 편차 0.24를 학습합니다. 학습된 노이즈가 실제 노이즈에 매우 가깝다는 점에 유의하십시오. 이는 우리 GP가 이 문제에 매우 잘 지정되었음을 나타내는 데 도움이 됩니다.

일반적으로 커널을 선택하고 하이퍼파라미터를 초기화하는 데 신중한 생각을 기울이는 것이 중요합니다. 주변 우도 최적화는 초기화에 비교적 견고할 수 있지만 나쁜 초기화에 면역이 되지는 않습니다. 다양한 초기화로 위의 스크립트를 실행해 보고 어떤 결과를 얻는지 확인해 보십시오.

이제 학습된 하이퍼파라미터로 예측을 해봅시다.

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

주황색의 사후 평균이 실제 노이즈 없는 함수와 거의 완벽하게 일치하는 것을 볼 수 있습니다! 우리가 보여주는 95% 신용 집합은 데이터 포인트가 아니라 잠재 *노이즈 없는*(실제) 함수에 대한 것입니다. 이 신용 집합이 실제 함수를 완전히 포함하고 있으며 지나치게 넓거나 좁아 보이지 않음을 알 수 있습니다. 우리는 그것이 데이터 포인트를 포함하기를 원하지도 기대하지도 않습니다. 관찰에 대한 신용 집합을 갖고 싶다면 다음을 계산해야 합니다.

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

두 가지 불확실성 소스가 있습니다. *줄일 수 있는* 불확실성을 나타내는 *인식적* 불확실성과 *우발적(aleatoric)* 또는 *줄일 수 없는* 불확실성입니다. 여기서 *인식적* 불확실성은 노이즈 없는 함수의 실제 값에 대한 불확실성을 나타냅니다. 데이터에서 멀어질수록 데이터와 일치하는 다양한 함수 값이 존재하므로 이 불확실성은 데이터 포인트에서 멀어질수록 커져야 합니다. 점점 더 많은 데이터를 관찰함에 따라 실제 함수에 대한 우리의 믿음은 더 확신을 갖게 되고 인식적 불확실성은 사라집니다. 이 경우 *우발적* 불확실성은 관찰 노이즈입니다. 데이터가 이 노이즈와 함께 우리에게 주어지며 줄일 수 없기 때문입니다.

데이터의 *인식적* 불확실성은 잠재 노이즈 없는 함수의 분산 np.diag(post_cov)에 의해 포착됩니다. *우발적* 불확실성은 노이즈 분산 post_sig_est**2에 의해 포착됩니다.

불행히도 사람들은 불확실성을 표현하는 방법에 대해 종종 부주의하여, 많은 논문이 완전히 정의되지 않은 오차 막대를 보여주거나, 인식적 불확실성 또는 우발적 불확실성 또는 둘 다를 시각화하고 있는지에 대한 명확한 감각이 없으며, 노이즈 분산을 노이즈 표준 편차와 혼동하고, 표준 편차를 표준 오차와 혼동하고, 신뢰 구간을 신용 집합과 혼동하는 등의 일이 발생합니다. 불확실성이 무엇을 나타내는지 정확하지 않으면 본질적으로 무의미합니다.

우리의 불확실성이 무엇을 나타내는지에 세심한 주의를 기울이는 정신으로, 노이즈 없는 함수에 대한 분산 추정치의 *제곱근*의 *두 배*를 취하고 있다는 점에 유의하는 것이 중요합니다. 예측 분포가 가우시안이므로 이 수량을 통해 실제 함수를 포함할 확률이 95%인 구간에 대한 우리의 믿음을 나타내는 95% 신용 집합을 형성할 수 있습니다. 노이즈 *분산*은 완전히 다른 스케일에 있으며 훨씬 덜 해석 가능합니다.

마지막으로 20개의 사후 샘플을 살펴봅시다. 이 샘플들은 사후적으로 우리 데이터에 적합할 수 있다고 믿는 함수의 유형을 알려줍니다.

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

기본 회귀 응용 프로그램에서는 사후 예측 평균과 표준 편차를 각각 점 예측기 및 불확실성 지표로 사용하는 것이 가장 일반적입니다. 몬테카를로 획득 함수를 사용한 베이지안 최적화 또는 모델 기반 RL을 위한 가우시안 프로세스와 같은 고급 응용 프로그램에서는 종종 사후 샘플을 취해야 합니다. 그러나 기본 응용 프로그램에서 엄격하게 요구되지 않더라도 이러한 샘플은 데이터에 대한 적합성에 대한 더 많은 직관을 제공하며 시각화에 포함하는 데 종종 유용합니다.

## GPyTorch로 쉽게 만들기 (Making Life Easy with GPyTorch)

우리가 보았듯이 기본 가우시안 프로세스 회귀를 처음부터 완전히 구현하는 것은 실제로 꽤 쉽습니다. 그러나 다양한 커널 선택을 탐색하거나, 근사 추론(분류에도 필요함)을 고려하거나, GP를 신경망과 결합하거나, 심지어 약 10,000개 이상의 포인트가 있는 데이터셋을 갖게 되면 처음부터 구현하는 것은 다루기 힘들고 번거로워집니다. SKI(KISS-GP라고도 함)와 같은 확장 가능한 GP 추론을 위한 가장 효과적인 방법 중 일부는 수백 줄의 코드로 고급 수치 선형 대수 루틴을 구현해야 할 수 있습니다.

이러한 경우 *GPyTorch* 라이브러리는 우리의 삶을 훨씬 쉽게 만들어 줄 것입니다. 우리는 가우시안 프로세스 수치 및 고급 방법에 대한 향후 노트북에서 GPyTorch에 대해 더 논의할 것입니다. GPyTorch 라이브러리에는 [많은 예제](https://github.com/cornellius-gp/gpytorch/tree/master/examples)가 포함되어 있습니다. 패키지에 대한 감을 잡기 위해 [간단한 회귀 예제](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb)를 살펴보며 GPyTorch를 사용하여 위의 결과를 재현하도록 어떻게 조정할 수 있는지 보여줄 것입니다. 이것은 단순히 위의 기본 회귀를 재현하기 위해 많은 코드처럼 보일 수 있으며, 어떤 의미에서는 그렇습니다. 그러나 잠재적으로 수천 줄의 새 코드를 작성하는 대신 아래 코드에서 몇 줄만 변경하여 다양한 커널, 확장 가능한 추론 기술 및 근사 추론을 즉시 사용할 수 있습니다.

```{.python .input}
# 먼저 데이터를 PyTorch에서 사용할 수 있도록 텐서로 변환해 봅시다
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# 우리는 0 평균과 RBF 커널을 사용하여 정확한 GP 추론을 사용하고 있습니다
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

이 코드 블록은 데이터를 GPyTorch에 맞는 형식으로 넣고, 정확한 추론을 사용하고 있음을 지정하며, 사용하려는 평균 함수(0)와 커널 함수(RBF)를 지정합니다. 예를 들어 gpytorch.kernels.matern_kernel() 또는 gpyotrch.kernels.spectral_mixture_kernel()을 호출하여 다른 커널을 매우 쉽게 사용할 수 있습니다. 지금까지 우리는 근사를 하지 않고 예측 분포를 추론할 수 있는 정확한 추론에 대해서만 논의했습니다.
가우시안 프로세스의 경우 가우시안 우도가 있을 때만 정확한 추론을 수행할 수 있습니다. 더 구체적으로 말하자면, 관찰이 가우시안 프로세스로 표현되는 노이즈 없는 함수와 가우시안 노이즈로 생성된다고 가정할 때입니다.
향후 노트북에서는 이러한 가정을 할 수 없는 분류와 같은 다른 설정을 고려할 것입니다.

```{.python .input}
# 가우시안 우도 초기화
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# 최적 모델 하이퍼파라미터 찾기
model.train()
likelihood.train()
# adam 최적화 도구 사용, GaussianLikelihood 파라미터 포함
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# 손실을 음의 로그 GP 주변 우도로 설정
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

여기서 우리는 사용하려는 우도(가우시안), 커널 하이퍼파라미터 훈련에 사용할 목적 함수(여기서는 주변 우도), 그리고 해당 목적 함수를 최적화하는 데 사용할 절차(이 경우 Adam)를 명시적으로 지정합니다. "확률적" 최적화 도구인 Adam을 사용하고 있지만, 이 경우에는 전체 배치 Adam입니다. 주변 우도는 데이터 인스턴스에 대해 인수분해되지 않으므로 데이터의 "미니배치"에 대한 최적화 도구를 사용할 수 없으며 수렴이 보장되지 않습니다. L-BFGS와 같은 다른 최적화 도구도 GPyTorch에서 지원됩니다. 표준 딥러닝과 달리 주변 우도를 최적화하는 작업을 잘 수행하는 것은 좋은 일반화와 강력하게 일치하며, 엄청나게 비싸지 않다고 가정할 때 L-BFGS와 같은 강력한 최적화 도구로 기울게 합니다.

```{.python .input}
for i in range(training_iter):
    # 이전 반복의 기울기 0으로 설정
    optimizer.zero_grad()
    # 모델의 출력
    output = model(train_x)
    # 손실 계산 및 역전파 기울기
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

여기서 실제로 최적화 절차를 실행하고 10번의 반복마다 손실 값을 출력합니다.

```{.python .input}
# 평가(예측 사후) 모드로 전환
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

위의 코드 블록을 사용하면 테스트 입력에 대한 예측을 할 수 있습니다.

```{.python .input}
with torch.no_grad():
    # 플롯 초기화
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # 95% 신용 집합에 대한 상한 및 하한 가져오기 (이 경우 관찰 공간에서)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

마지막으로 피팅을 플로팅합니다.

피팅이 사실상 동일함을 알 수 있습니다. 주목해야 할 몇 가지 사항: GPyTorch는 *제곱된* 길이 척도와 관찰 노이즈로 작업합니다. 예를 들어, 처음부터 작성한 코드에서 학습된 노이즈 표준 편차는 약 0.283입니다. GPyTorch가 찾은 노이즈 분산은 $0.81 \approx 0.283^2$입니다. GPyTorch 플롯에서는 잠재 함수 공간이 아닌 *관찰 공간*에 신용 집합을 표시하여 실제로 관찰된 데이터 포인트를 덮고 있음을 보여줍니다.

## 요약 (Summary)

가우시안 프로세스 사전 분포를 데이터와 결합하여 사후 분포를 형성하고 이를 사용하여 예측을 할 수 있습니다. 또한 가우시안 프로세스의 변화율과 같은 속성을 제어하는 커널 하이퍼파라미터의 자동 학습에 유용한 주변 우도를 형성할 수 있습니다. 회귀에 대한 사후 분포를 형성하고 커널 하이퍼파라미터를 학습하는 메커니즘은 간단하며 약 12줄의 코드를 포함합니다. 이 노트북은 실제로 가우시안 프로세스를 빠르게 "시작하고 실행"하려는 독자에게 좋은 참고 자료입니다. 또한 GPyTorch 라이브러리를 소개했습니다. 기본 회귀를 위한 GPyTorch 코드는 비교적 길지만, 다른 커널 함수나 확장 가능한 추론 또는 분류를 위한 비가우시안 우드와 같이 향후 노트북에서 논의할 고급 기능을 위해 사소하게 수정할 수 있습니다.


## 연습 문제 (Exercises)

1. 우리는 커널 하이퍼파라미터 *학습*의 중요성과 하이퍼파라미터 및 커널이 가우시안 프로세스의 일반화 속성에 미치는 영향을 강조했습니다. 하이퍼를 학습하는 단계를 건너뛰고 대신 다양한 길이 척도와 노이즈 분산을 추측하고 예측에 미치는 영향을 확인해 보십시오. 큰 길이 척도를 사용하면 어떻게 됩니까? 작은 길이 척도는요? 큰 노이즈 분산은요? 작은 노이즈 분산은요?
2. 우리는 주변 우도가 볼록 목적 함수가 아니지만 길이 척도 및 노이즈 분산과 같은 하이퍼파라미터를 GP 회귀에서 안정적으로 추정할 수 있다고 말했습니다. 이것은 일반적으로 사실입니다. 실제로 주변 우도는 경험적 자기상관 함수("covariograms")를 피팅하는 것을 포함하는 공간 통계의 기존 접근 방식보다 길이 척도 하이퍼파라미터를 학습하는 데 *훨씬* 더 좋습니다. 틀림없이, 적어도 확장 가능한 추론에 대한 최근 작업 이전에 가우시안 프로세스 연구에 대한 머신러닝의 가장 큰 기여는 하이퍼파라미터 학습을 위한 주변 우도의 도입이었습니다.

*그러나* 이러한 파라미터의 다른 쌍조차도 많은 데이터셋에 대해 해석 가능하게 다른 타당한 설명을 제공하여 목적 함수에서 국소 최적값을 초래합니다. 큰 길이 척도를 사용하면 기본 실제 함수가 천천히 변한다고 가정합니다. 관찰된 데이터가 실제로 상당히 변하는 경우, 큰 길이 척도를 가질 수 있는 유일한 방법은 큰 노이즈 분산을 갖는 것입니다. 반면에 작은 길이 척도를 사용하면 피팅이 데이터의 변동에 매우 민감하여 노이즈(우발적 불확실성)로 변동을 설명할 여지가 거의 없습니다.

이러한 국소 최적값을 찾을 수 있는지 확인해 보십시오. 큰 노이즈가 있는 매우 큰 길이 척도와 작은 노이즈가 있는 작은 길이 척도로 초기화하십시오. 다른 솔루션으로 수렴합니까?
  
3. 우리는 베이지안 방법의 근본적인 장점이 *인식적* 불확실성을 자연스럽게 표현하는 데 있다고 말했습니다. 위의 예에서는 인식적 불확실성의 효과를 완전히 볼 수 없습니다. 대신 `test_x = np.linspace(0, 10, 1000)`으로 예측해 보십시오. 예측이 데이터를 넘어서 이동함에 따라 95% 신용 집합에 어떤 일이 발생합니까? 해당 구간에서 실제 함수를 덮습니까? 해당 영역에서 우발적 불확실성만 시각화하면 어떻게 됩니까?

4. 위의 예제를 실행하되, 대신 10,000, 20,000 및 40,000개의 훈련 포인트로 실행하고 런타임을 측정해 보십시오. 훈련 시간은 어떻게 확장됩니까? 대안으로 런타임은 테스트 포인트 수에 따라 어떻게 확장됩니까? 예측 평균과 예측 분산에 대해 다릅니까? 훈련 및 테스트 시간 복잡도를 이론적으로 해결하고 다른 수의 포인트로 위 코드를 실행하여 이 질문에 답하십시오.

5. Matern 커널과 같은 다른 공분산 함수를 사용하여 GPyTorch 예제를 실행해 보십시오. 결과는 어떻게 변합니까? GPyTorch 라이브러리에서 찾을 수 있는 스펙트럼 혼합 커널은 어떻습니까? 일부는 다른 것보다 주변 우도를 훈련하기가 더 쉽습니까? 장거리 대 단거리 예측에 더 가치 있는 것이 있습니까?

6. GPyTorch 예제에서는 관찰 노이즈를 포함한 예측 분포를 플로팅한 반면, "처음부터" 예제에서는 인식적 불확실성만 포함했습니다. 이번에는 인식적 불확실성만 플로팅하여 GPyTorch 예제를 다시 실행하고 처음부터 결과와 비교하십시오. 예측 분포가 이제 동일하게 보입니까? (그래야 합니다.)

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12117)
:end_tab:

```