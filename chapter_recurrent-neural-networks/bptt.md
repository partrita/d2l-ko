# 시간 경과에 따른 역전파 (Backpropagation Through Time)
:label:`sec_bptt`

:numref:`sec_rnn-scratch`의 연습 문제를 완료했다면, 가끔 발생하는 엄청난 기울기로 인해 훈련이 불안정해지는 것을 방지하기 위해 기울기 클리핑이 필수적이라는 것을 알았을 것입니다. 
우리는 폭발하는 기울기가 긴 시퀀스를 가로질러 역전파하는 데서 비롯된다고 암시했습니다. 
수많은 현대 RNN 아키텍처를 소개하기 전에, 시퀀스 모델에서 *역전파*가 어떻게 작동하는지 수학적으로 자세히 살펴보겠습니다. 
바라건대 이 논의가 *사라지는(vanishing)* 기울기와 *폭발하는(exploding)* 기울기의 개념에 어느 정도 정확성을 가져다줄 것입니다. 
:numref:`sec_backprop`에서 MLP를 소개할 때 계산 그래프를 통한 순전파 및 역전파에 대한 논의를 기억한다면, 
RNN의 순전파는 비교적 간단할 것입니다. 
RNN에 역전파를 적용하는 것을 *시간 경과에 따른 역전파(backpropagation through time)*라고 합니다 :cite:`Werbos.1990`. 
이 절차는 RNN의 계산 그래프를 한 번에 한 타임 스텝씩 확장(또는 펼치기)해야 합니다. 
펼쳐진 RNN은 본질적으로 
펼쳐진 네트워크 전체에서 동일한 파라미터가 반복되어 
각 타임 스텝에 나타나는 특수한 속성을 가진 피드포워드 신경망입니다. 
그런 다음 다른 피드포워드 신경망과 마찬가지로 연쇄 법칙을 적용하여 펼쳐진 네트워크를 통해 기울기를 역전파할 수 있습니다. 
각 파라미터에 대한 기울기는 펼쳐진 네트워크에서 파라미터가 발생하는 모든 위치에 걸쳐 합산되어야 합니다. 
이러한 가중치 묶음을 처리하는 것은 합성곱 신경망에 대한 장에서 익숙할 것입니다.


시퀀스가 꽤 길 수 있기 때문에 복잡한 문제가 발생합니다. 
천 개 이상의 토큰으로 구성된 텍스트 시퀀스로 작업하는 것은 드문 일이 아닙니다. 
이것은 계산(너무 많은 메모리) 및 최적화(수치적 불안정성) 관점 모두에서 문제를 제기합니다. 
첫 번째 단계의 입력은 출력에 도달하기 전에 1000개 이상의 행렬 곱을 통과하며, 
기울기를 계산하기 위해 또 다른 1000개의 행렬 곱이 필요합니다. 
이제 무엇이 잘못될 수 있고 실제로 어떻게 해결해야 하는지 분석해 보겠습니다.


## RNN의 기울기 분석 (Analysis of Gradients in RNNs)
:label:`subsec_bptt_analysis`

RNN이 작동하는 방식에 대한 단순화된 모델로 시작합니다. 
이 모델은 은닉 상태의 세부 사항과 업데이트 방법에 대한 세부 사항을 무시합니다. 
여기서 수학적 표기법은 스칼라, 벡터, 행렬을 명시적으로 구분하지 않습니다. 
우리는 단지 약간의 직관을 개발하려고 노력하고 있습니다. 
이 단순화된 모델에서 타임 스텝 $t$에서의 은닉 상태를 $h_t$, 입력을 $x_t$, 출력을 $o_t$로 표시합니다. 
:numref:`subsec_rnn_w_hidden_states`에서의 논의를 상기하면, 
입력과 은닉 상태는 은닉층의 하나의 가중치 변수와 곱해지기 전에 연결될 수 있습니다. 
따라서 우리는 $w_\textrm{h}$와 $w_\textrm{o}$를 사용하여 각각 은닉층과 출력 레이어의 가중치를 나타냅니다. 
결과적으로 각 타임 스텝에서의 은닉 상태와 출력은 다음과 같습니다.

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$:eqlabel:`eq_bptt_ht_ot`

여기서 $f$와 $g$는 각각 은닉층과 출력 레이어의 변환입니다. 
따라서 우리는 순환 계산을 통해 서로 의존하는 값의 체인 {\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots}를 갖습니다. 
순전파는 꽤 간단합니다. 
우리가 필요한 것은 $(x_t, h_t, o_t)$ 삼중쌍을 한 번에 한 타임 스텝씩 반복하는 것입니다. 
출력 $o_t$와 원하는 타겟 $y_t$ 사이의 불일치는 다음과 같이 모든 $T$ 타임 스텝에 걸쳐 목적 함수에 의해 평가됩니다.

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$ 



역전파의 경우, 특히 목적 함수 $L$의 파라미터 $w_\textrm{h}$에 대한 기울기를 계산할 때 문제는 좀 더 까다롭습니다. 
구체적으로 연쇄 법칙에 의해,

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh`에 있는 곱의 첫 번째와 두 번째 인수는 계산하기 쉽습니다. 
세 번째 인수 $\partial h_t/\partial w_\textrm{h}$는 상황이 까다로워지는 부분인데, $h_t$에 대한 파라미터 $w_\textrm{h}$의 효과를 순환적으로 계산해야 하기 때문입니다. 
:eqref:`eq_bptt_ht_ot`의 순환 계산에 따르면, 
$h_t$는 $h_{t-1}$과 $w_\textrm{h}$ 모두에 의존하며, 
여기서 $h_{t-1}$의 계산도 $w_\textrm{h}$에 의존합니다. 
따라서 연쇄 법칙을 사용하여 $w_\textrm{h}$에 대한 $h_t$의 전도함수(total derivate)를 평가하면 다음과 같습니다.

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$:eqlabel:`eq_bptt_partial_ht_wh_recur`


위의 기울기를 유도하기 위해, $t=1, 2,\ldots$에 대해 $a_{0}=0$ 및 $a_{t}=b_{t}+c_{t}a_{t-1}$을 만족하는 세 시퀀스 {a_{t}},{b_{t}},{c_{t}}가 있다고 가정합니다. 
그러면 $t\geq 1$에 대해 다음을 보이는 것은 쉽습니다.

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$:eqlabel:`eq_bptt_at`

다음에 따라 $a_t$, $b_t$, $c_t$를 대입하면

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},
\end{aligned}$$ 

:eqref:`eq_bptt_partial_ht_wh_recur`의 기울기 계산은 $a_{t}=b_{t}+c_{t}a_{t-1}$을 만족합니다. 
따라서 :eqref:`eq_bptt_at`에 따라 다음을 사용하여 :eqref:`eq_bptt_partial_ht_wh_recur`의 순환 계산을 제거할 수 있습니다.

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$:eqlabel:`eq_bptt_partial_ht_wh_gen`

연쇄 법칙을 사용하여 $\partial h_t/\partial w_\textrm{h}$를 재귀적으로 계산할 수 있지만, $t$가 클 때마다 이 체인이 매우 길어질 수 있습니다. 
이 문제를 다루기 위한 몇 가지 전략을 논의해 봅시다.

### 전체 계산 (Full Computation) ### 

한 가지 아이디어는 :eqref:`eq_bptt_partial_ht_wh_gen`에서 전체 합을 계산하는 것일 수 있습니다. 
하지만 이것은 매우 느리고 기울기가 폭발할 수 있습니다. 
초기 조건의 미묘한 변화가 결과에 큰 영향을 미칠 수 있기 때문입니다. 
즉, 초기 조건의 미미한 변화가 결과의 불균형한 변화로 이어지는 나비 효과와 유사한 것을 볼 수 있습니다. 
이것은 일반적으로 바람직하지 않습니다. 
결국 우리는 잘 일반화되는 강력한 추정기를 찾고 있습니다. 
따라서 이 전략은 실제로 거의 사용되지 않습니다.

### 타임 스텝 자르기 (Truncating Time Steps)###

대안으로, 
우리는 $\tau$ 단계 후 :eqref:`eq_bptt_partial_ht_wh_gen`의 합을 자를 수 있습니다. 
이것이 지금까지 우리가 논의해 온 것입니다. 
이것은 $\partial h_{t-\tau}/\partial w_\textrm{h}$에서 합을 종료함으로써 실제 기울기의 *근사치*로 이어집니다. 
실제로 이것은 꽤 잘 작동합니다. 
이것은 일반적으로 절단된 시간 경과에 따른 역전파(truncated backpropagation through time)라고 불리는 것입니다 :cite:`Jaeger.2002`. 
이것의 결과 중 하나는 모델이 장기적인 결과보다는 주로 단기적인 영향에 초점을 맞춘다는 것입니다. 
이것은 실제로 *바람직*한데, 추정치를 더 단순하고 안정적인 모델 쪽으로 편향시키기 때문입니다.


### 무작위 자르기 (Randomized Truncation) ### 

마지막으로, 우리는 $\partial h_t/\partial w_\textrm{h}$를 기댓값에서는 정확하지만 시퀀스를 자르는 확률 변수로 대체할 수 있습니다. 
이것은 미리 정의된 $0 \leq \pi_t \leq 1$을 갖는 $\xi_t$ 시퀀스를 사용하여 달성됩니다. 
여기서 $P(\xi_t = 0) = 1-\pi_t$이고 $P(\xi_t = \pi_t^{-1}) = \pi_t$이므로 $E[\xi_t] = 1$입니다. 
우리는 이것을 사용하여 :eqref:`eq_bptt_partial_ht_wh_recur`의 기울기 $\partial h_t/\partial w_\textrm{h}$를 다음과 같이 대체합니다.

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$ 


$\\xi_t$의 정의에서 $E[z_t] = \partial h_t/\partial w_\textrm{h}$가 따릅니다. 
$\\xi_t = 0$일 때마다 순환 계산은 해당 타임 스텝 $t$에서 종료됩니다. 
이것은 다양한 길이의 시퀀스의 가중 합으로 이어지며, 긴 시퀀스는 드물지만 적절하게 가중치가 부여됩니다. 
이 아이디어는 :citet:`Tallec.Ollivier.2017`에 의해 제안되었습니다.

### 전략 비교 (Comparing Strategies)

![RNN에서 기울기를 계산하기 위한 전략 비교. 위에서 아래로: 무작위 자르기, 일반 자르기, 전체 계산.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


:numref:`fig_truncated_bptt`는 RNN에 대한 시간 경과에 따른 역전파를 사용하여 *타임 머신*의 처음 몇 글자를 분석할 때 세 가지 전략을 보여줍니다:

* 첫 번째 행은 텍스트를 다양한 길이의 세그먼트로 분할하는 무작위 자르기입니다.
* 두 번째 행은 텍스트를 동일한 길이의 하위 시퀀스로 나누는 일반 자르기입니다. 이것이 우리가 RNN 실험에서 해왔던 것입니다.
* 세 번째 행은 계산적으로 실행 불가능한 표현으로 이어지는 전체 시간 경과에 따른 역전파입니다.


불행히도 이론적으로는 매력적이지만 무작위 자르기는 일반 자르기보다 훨씬 더 잘 작동하지 않는데, 이는 여러 요인 때문일 가능성이 큽니다. 
첫째, 과거로의 여러 역전파 단계 후 관찰의 효과는 실제로 의존성을 포착하기에 충분합니다. 
둘째, 증가된 분산은 더 많은 단계에서 기울기가 더 정확하다는 사실을 상쇄합니다. 
셋째, 우리는 실제로 짧은 범위의 상호 작용만 있는 모델을 *원합니다*. 
따라서 정기적으로 자른 시간 경과에 따른 역전파는 바람직할 수 있는 약간의 정규화 효과를 갖습니다.

## 시간 경과에 따른 역전파 상세 (Backpropagation Through Time in Detail)

일반적인 원칙을 논의한 후, 시간 경과에 따른 역전파에 대해 자세히 논의해 봅시다. 
:numref:`subsec_bptt_analysis`의 분석과 대조적으로, 다음에서는 분해된 모든 모델 파라미터에 대한 목적 함수의 기울기를 계산하는 방법을 보여줄 것입니다. 
일을 단순하게 유지하기 위해 편향 파라미터가 없고 은닉층의 활성화 함수가 항등 매핑($\phi(x)=x$)을 사용하는 RNN을 고려합니다. 
타임 스텝 $t$에 대해 단일 예제 입력과 타겟을 각각 $\mathbf{x}_t \in \mathbb{R}^d$와 $y_t$라고 합시다. 
은닉 상태 $\mathbf{h}_t \in \mathbb{R}^h$와 출력 $\mathbf{o}_t \in \mathbb{R}^q$는 다음과 같이 계산됩니다.

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$ 

여기서 $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$, $\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$는 가중치 파라미터입니다. 
$l(\mathbf{o}_t, y_t)$를 타임 스텝 $t$에서의 손실이라고 합시다. 
우리의 목적 함수, 시퀀스 시작부터 $T$ 타임 스텝에 걸친 손실은 다음과 같습니다.

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$ 



RNN 계산 중 모델 변수와 파라미터 간의 종속성을 시각화하기 위해, :numref:`fig_rnn_bptt`와 같이 모델에 대한 계산 그래프를 그릴 수 있습니다. 
예를 들어 타임 스텝 3의 은닉 상태 $\mathbf{h}_3$의 계산은 모델 파라미터 $\mathbf{W}_\textrm{hx}$와 $\mathbf{W}_\textrm{hh}$, 이전 타임 스텝의 은닉 상태 $\mathbf{h}_2$, 현재 타임 스텝의 입력 $\mathbf{x}_3$에 의존합니다.

![3개의 타임 스텝을 가진 RNN 모델에 대한 종속성을 보여주는 계산 그래프. 상자는 변수(음영 없음) 또는 파라미터(음영 있음)를 나타내고 원은 연산자를 나타냅니다.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`


방금 언급했듯이 :numref:`fig_rnn_bptt`의 모델 파라미터는 $\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$, $\mathbf{W}_\textrm{qh}$입니다. 
일반적으로 이 모델을 훈련하려면 이러한 파라미터에 대한 기울기 계산 $\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$, $\partial L/\partial \mathbf{W}_\textrm{qh}$가 필요합니다. 
:numref:`fig_rnn_bptt`의 종속성에 따라 화살표의 반대 방향으로 순회하여 기울기를 차례로 계산하고 저장할 수 있습니다. 
연쇄 법칙에서 모양이 다른 행렬, 벡터, 스칼라의 곱셈을 유연하게 표현하기 위해 :numref:`sec_backprop`에서 설명한 대로 $\textrm{prod}$ 연산자를 계속 사용합니다.


우선, 임의의 타임 스텝 $t$에서 모델 출력에 대한 목적 함수를 미분하는 것은 꽤 간단합니다:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$:eqlabel:`eq_bptt_partial_L_ot`


이제 출력 레이어의 파라미터 $\mathbf{W}_\textrm{qh}$에 대한 목적 함수의 기울기를 계산할 수 있습니다: $\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$. 
:numref:`fig_rnn_bptt`를 기반으로 목적 함수 $L$은 $\mathbf{o}_1, \ldots, \mathbf{o}_T$를 통해 $\mathbf{W}_\textrm{qh}$에 의존합니다. 
연쇄 법칙을 사용하면 다음을 얻습니다.

$$ 
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top, 
$$ 

여기서 $\partial L/\partial \mathbf{o}_t$는 :eqref:`eq_bptt_partial_L_ot`에 의해 주어집니다.


다음으로, :numref:`fig_rnn_bptt`에 표시된 것처럼, 
마지막 타임 스텝 $T$에서 목적 함수 $L$은 $\mathbf{o}_T$를 통해서만 은닉 상태 $\mathbf{h}_T$에 의존합니다. 
따라서 연쇄 법칙을 사용하여 기울기 $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$를 쉽게 찾을 수 있습니다.

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$:eqlabel:`eq_bptt_partial_L_hT_final_step`


목적 함수 $L$이 $\mathbf{h}_{t+1}$과 $\mathbf{o}_t$를 통해 $\mathbf{h}_t$에 의존하는 $t < T$인 타임 스텝의 경우 더 까다로워집니다. 
연쇄 법칙에 따라, 
임의의 타임 스텝 $t < T$에서 은닉 상태의 기울기 $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$는 다음과 같이 순환적으로 계산될 수 있습니다:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$:eqlabel:`eq_bptt_partial_L_ht_recur`


분석을 위해, 임의의 타임 스텝 $1 \leq t \leq T$에 대해 순환 계산을 확장하면 다음을 얻습니다.

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$:eqlabel:`eq_bptt_partial_L_ht`


:eqref:`eq_bptt_partial_L_ht`에서 이 간단한 선형 예제가 이미 긴 시퀀스 모델의 몇 가지 주요 문제를 보여준다는 것을 알 수 있습니다: 
잠재적으로 매우 큰 거듭제곱의 $\mathbf{W}_\textrm{hh}^\top$를 포함합니다. 
그 안에서 1보다 작은 고유값은 사라지고 1보다 큰 고유값은 발산합니다. 
이것은 수치적으로 불안정하며, 이는 사라지는 기울기와 폭발하는 기울기의 형태로 나타납니다. 
이를 해결하는 한 가지 방법은 :numref:`subsec_bptt_analysis`에서 논의한 대로 계산적으로 편리한 크기에서 타임 스텝을 자르는 것입니다. 
실제로 이 자르기는 주어진 타임 스텝 수 후에 기울기를 분리(detaching)함으로써 영향을 받을 수도 있습니다. 
나중에 장단기 메모리(long short-term memory)와 같은 더 정교한 시퀀스 모델이 이를 어떻게 더 완화할 수 있는지 보게 될 것입니다.


마지막으로 :numref:`fig_rnn_bptt`는 목적 함수 $L$이 은닉 상태 $\mathbf{h}_1, \ldots, \mathbf{h}_T$를 통해 은닉층의 모델 파라미터 $\mathbf{W}_\textrm{hx}$와 $\mathbf{W}_\textrm{hh}$에 의존함을 보여줍니다. 
이러한 파라미터에 대한 기울기 $\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$와 $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$를 계산하기 위해 연쇄 법칙을 적용하면 다음을 얻습니다.

$$ 
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned} 
$$ 


여기서 :eqref:`eq_bptt_partial_L_hT_final_step`과 :eqref:`eq_bptt_partial_L_ht_recur`에 의해 순환적으로 계산되는 $\partial L/\partial \mathbf{h}_t$는 수치적 안정성에 영향을 미치는 핵심 양입니다.



시간 경과에 따른 역전파는 RNN에 역전파를 적용하는 것이므로, :numref:`sec_backprop`에서 설명했듯이 RNN 훈련은 순전파와 시간 경과에 따른 역전파를 번갈아 가며 수행합니다. 
더욱이 시간 경과에 따른 역전파는 위의 기울기를 차례로 계산하고 저장합니다. 
구체적으로 $\partial L / \partial \mathbf{W}_\textrm{hx}$와 $\partial L / \partial \mathbf{W}_\textrm{hh}$의 계산 모두에 사용하기 위해 $\partial L/\partial \mathbf{h}_t$를 저장하는 것과 같이, 저장된 중간 값은 중복 계산을 피하기 위해 재사용됩니다.


## 요약 (Summary)

시간 경과에 따른 역전파는 은닉 상태가 있는 시퀀스 모델에 역전파를 적용한 것일 뿐입니다.
계산 편의성과 수치적 안정성을 위해 정기적 또는 무작위와 같은 자르기가 필요합니다.
행렬의 높은 거듭제곱은 발산하거나 사라지는 고유값으로 이어질 수 있습니다. 이는 폭발하거나 사라지는 기울기의 형태로 나타납니다.
효율적인 계산을 위해 시간 경과에 따른 역전파 중에 중간 값이 캐시됩니다.



## 연습 문제 (Exercises)

1. 고유값 $\lambda_i$와 그에 대응하는 고유 벡터 $\mathbf{v}_i$ ($i = 1, \ldots, n$)를 갖는 대칭 행렬 $\mathbf{M} \in \mathbb{R}^{n \times n}$이 있다고 가정합니다. 일반성을 잃지 않고 $|
\lambda_i| \geq |\lambda_{i+1}|$ 순서로 정렬되어 있다고 가정합니다. 
   1. $\mathbf{M}^k$가 고유값 $\lambda_i^k$를 가짐을 보이십시오.
   1. 무작위 벡터 $\mathbf{x} \in \mathbb{R}^n$에 대해 $\mathbf{M}^k \mathbf{x}$가 높은 확률로 $\mathbf{M}$의 고유 벡터 $\mathbf{v}_1$과 매우 잘 정렬될 것임을 증명하십시오. 이 진술을 공식화하십시오.
   1. 위의 결과는 RNN의 기울기에 대해 무엇을 의미합니까?
2. 기울기 클리핑 외에 순환 신경망에서 기울기 폭발에 대처할 수 있는 다른 방법을 생각할 수 있습니까?

[토론](https://discuss.d2l.ai/t/334)