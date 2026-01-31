# 표기법 (Notation)
:label:`chap_notation`

이 책 전체에서 우리는 다음의 표기법 관례를 따릅니다.
이러한 기호 중 일부는 플레이스홀더(placeholder)이고,
다른 기호는 특정 객체를 나타냅니다.
일반적인 경험 법칙으로, 부정관사 "a"는 종종
기호가 플레이스홀더이며 유사하게 포맷된 기호가
동일한 유형의 다른 객체를 나타낼 수 있음을 나타냅니다.
예를 들어, "$x$: 스칼라(a scalar)"는 소문자가 일반적으로
스칼라 값을 나타냄을 의미하지만,
"$\\mathbb{Z}$: 정수 집합(the set of integers)"은
구체적으로 기호 $\\mathbb{Z}$를 나타냅니다.



## 수치 객체 (Numerical Objects)

* $x$: 스칼라 (scalar)
* $\\mathbf{x}$: 벡터 (vector)
* $\\mathbf{X}$: 행렬 (matrix)
* $\\mathsf{X}$: 일반 텐서 (tensor)
* $\\mathbf{I}$: 단위 행렬 (identity matrix) (주어진 차원의), 즉 모든 대각선 항목이 $1$이고 모든 비대각선 항목이 $0$인 정사각 행렬
* $x_i$, $[\\mathbf{x}]_i$: 벡터 $\\mathbf{x}$의 $i^\textrm{th}$ 요소
* $x_{ij}$, $x_{i,j}$,$[\\mathbf{X}]_{ij}$, $[\\mathbf{X}]_{i,j}$: 행 $i$와 열 $j$에 있는 행렬 $\\mathbf{X}$의 요소



## 집합론 (Set Theory)


* $\\mathcal{X}$: 집합 (set)
* $\\mathbb{Z}$: 정수 집합
* $\\mathbb{Z}^+$: 양의 정수 집합
* $\\mathbb{R}$: 실수 집합
* $\\mathbb{R}^n$: $n$-차원 실수 벡터의 집합
* $\\mathbb{R}^{a\\times b}$: $a$개의 행과 $b$개의 열을 가진 실수 행렬의 집합
* $|\\mathcal{X}|$: 집합 $\\mathcal{X}$의 기수 (원소의 수)
* $\\mathcal{A}\\cup\\mathcal{B}$: 집합 $\\mathcal{A}$와 $\\mathcal{B}$의 합집합
* $\\mathcal{A}\\cap\\mathcal{B}$: 집합 $\\mathcal{A}$와 $\\mathcal{B}$의 교집합
* $\\mathcal{A}\\setminus\\mathcal{B}$: $\\mathcal{A}$에서 $\\mathcal{B}$의 차집합 ($\\mathcal{B}$에 속하지 않는 $\\mathcal{A}$의 원소만 포함)



## 함수 및 연산자 (Functions and Operators)


* $f(\\cdot)$: 함수
* $\\log(\\cdot)$: 자연 로그 (밑 $e$)
* $\\log_2(\\cdot)$: 밑이 $2$인 로그
* $\\exp(\\cdot)$: 지수 함수
* $\\mathbf{1}(\\cdot)$: 지시 함수 (indicator function); 불리언 인수가 참이면 $1$, 그렇지 않으면 $0$으로 평가
* $\\mathbf{1}_{\\mathcal{X}}(z)$: 집합 멤버십 지시 함수; 요소 $z$가 집합 $\\mathcal{X}$에 속하면 $1$, 그렇지 않으면 $0$으로 평가
* $\\mathbf{(\\cdot)}^\top$: 벡터 또는 행렬의 전치 (transpose)
* $\\mathbf{X}^{-1}$: 행렬 $\\mathbf{X}$의 역행렬 (inverse)
* $\\odot$: 하다마드 (요소별) 곱 (Hadamard product)
* $[\\cdot, \\cdot]$: 연결 (concatenation)
* `$\\cdot$`_p: $\\ell_p$ 노름 (norm)
* `$\\cdot$`$: $\\ell_2$ 노름
* $\\langle \\mathbf{x}, \\mathbf{y} \\rangle$: 벡터 $\\mathbf{x}$와 $\\mathbf{y}$의 내적 (dot product)
* $\\sum$: 요소 모음에 대한 합계
* $\\prod$: 요소 모음에 대한 곱
* $\\stackrel{\\textrm{def}}}{=}$: 왼쪽 기호의 정의로 주장되는 등식



## 미적분 (Calculus)

* $\\frac{dy}{dx}$: $x$에 대한 $y$의 미분
* $\\frac{\\partial y}{\\partial x}$: $x$에 대한 $y$의 편미분
* $\\nabla_{\\mathbf{x}} y$: $\\mathbf{x}$에 대한 $y$의 기울기 (gradient)
* $\\int_a^b f(x) \\;dx$: $x$에 대한 $a$에서 $b$까지 $f$의 정적분
* $\\int f(x) \\;dx$: $x$에 대한 $f$의 부정적분



## 확률 및 정보 이론 (Probability and Information Theory)

* $X$: 확률 변수 (random variable)
* $P$: 확률 분포 (probability distribution)
* $X \\sim P$: 확률 변수 $X$가 분포 $P$를 따름
* $P(X=x)$: 확률 변수 $X$가 값 $x$를 취하는 사건에 할당된 확률
* $P(X \\mid Y)$: $Y$가 주어졌을 때 $X$의 조건부 확률 분포
* $p(\\cdot)$: 분포 $P$와 관련된 확률 밀도 함수 (PDF)
* ${E}[X]$: 확률 변수 $X$의 기댓값 (expectation)
* $X \\perp Y$: 확률 변수 $X$와 $Y$는 독립임
* $X \\perp Y \\mid Z$: 확률 변수 $X$와 $Y$는 $Z$가 주어졌을 때 조건부 독립임
* $\\sigma_X$: 확률 변수 $X$의 표준 편차 (standard deviation)
* $\\textrm{Var}(X)$: 확률 변수 $X$의 분산 (variance), $\\sigma^2_X$와 같음
* $\\textrm{Cov}(X, Y)$: 확률 변수 $X$와 $Y$의 공분산 (covariance)
* $\\rho(X, Y)$: $X$와 $Y$ 사이의 피어슨 상관 계수, $\\