```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # 난수 생성기 시드
gamma = 0.95  # 할인율
num_iters = 256  # 반복 횟수
alpha   = 0.9  # 학습률
epsilon = 0.9  # 엡실론 탐욕적 알고리즘의 엡실론
random.seed(seed)  # 랜덤 시드 설정
np.random.seed(seed)

# 이제 환경 설정
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

# Q-러닝 (Q-Learning)
:label:`sec_qlearning`

이 섹션에서는 가장 인기 있고 널리 사용되는 강화 학습 알고리즘 중 하나인 Q-러닝(Q-Learning)을 소개합니다. Q-러닝은 모델 없이 동작하는(model-free) 강화 학습 알고리즘으로, 에이전트가 환경에 대한 사전 지식 없이도 최적의 정책을 학습할 수 있게 해줍니다.

FrozenLake 환경에서 로봇은 "위"($\uparrow$), "아래"($\rightarrow$), "왼쪽"($\leftarrow$), "오른쪽"($\rightarrow$) 행동으로 $4 \times 4$ 그리드(상태) 위를 이동합니다. 환경에는 로봇에게 알려지지 않은 여러 구멍(H) 셀과 얼어붙은(F) 셀, 목표 셀(G)이 포함되어 있습니다. 문제를 간단하게 유지하기 위해 로봇이 신뢰할 수 있는 행동을 한다고 가정합니다. 즉, 모든 $s \in \mathcal{S}, a \in \mathcal{A}$에 대해 $P(s' \mid s, a) = 1$입니다. 로봇이 목표에 도달하면 시도가 종료되고 행동에 관계없이 $1$의 보상을 받습니다. 다른 상태에서의 보상은 모든 행동에 대해 $0$입니다. 로봇의 목표는 주어진 시작 위치(S)($s_0$)에서 목표 위치(G)에 도달하여 *반환값*을 최대화하는 정책을 학습하는 것입니다.

먼저 $\epsilon$-탐욕적 방법을 다음과 같이 구현합니다.

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

이제 Q-러닝을 구현할 준비가 되었습니다.

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 각 그리드 항목이 의미하는 바를 지정하는 2D 배열
    env = env_info['env']  # 각 그리드 항목이 의미하는 바를 지정하는 2D 배열
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # 환경 재설정
        state, done = env.reset(), False
        while not done:
            # 주어진 상태에 대한 행동을 선택하고 선택된 행동에 따라 환경에서 행동
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-업데이트:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # 다음 상태로 이동
            state = next_state
        # 시각화 목적으로만 최대 가치와 최대 행동 기록
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

이 결과는 Q-러닝이 대략 250번의 반복 후에 이 문제에 대한 최적의 해결책을 찾을 수 있음을 보여줍니다. 그러나 이 결과를 가치 반복 알고리즘의 결과(:ref:`subsec_valueitercode` 참조)와 비교할 때, 가치 반복 알고리즘이 이 문제에 대한 최적의 해결책을 찾는 데 훨씬 더 적은 반복이 필요함을 알 수 있습니다. 이는 가치 반복 알고리즘이 전체 MDP에 접근할 수 있는 반면 Q-러닝은 그렇지 않기 때문에 발생합니다.


## 요약
Q-러닝은 가장 기본적인 강화 학습 알고리즘 중 하나입니다. 그것은 최근 강화 학습의 성공, 특히 비디오 게임 플레이 학습의 진원지에 있었습니다 :cite:`mnih2013playing`. Q-러닝을 구현하기 위해 마르코프 결정 과정(MDP), 예를 들어 전이 및 보상 함수를 완전히 알 필요는 없습니다.

## 연습 문제

1. 그리드 크기를 $8 \times 8$로 늘려보십시오. $4 \times 4$ 그리드와 비교하여 최적 가치 함수를 찾는 데 몇 번의 반복이 걸립니까?
2. $\gamma$(즉, 위 코드의 "gamma")가 $0$, $0.5$, $1$일 때 Q-러닝 알고리즘을 다시 실행하고 결과를 분석하십시오.
3. $\epsilon$(즉, 위 코드의 "epsilon")이 $0$, $0.5$, $1$일 때 Q-러닝 알고리즘을 다시 실행하고 결과를 분석하십시오.

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/12103)
:end_tab:

```