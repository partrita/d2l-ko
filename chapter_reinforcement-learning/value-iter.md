```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # 난수 생성기 시드
gamma = 0.95  # 할인율
num_iters = 10  # 반복 횟수
random.seed(seed)  # 결과 재현성을 보장하기 위해 랜덤 시드 설정
np.random.seed(seed)

# 이제 환경 설정
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

# 가치 반복 (Value Iteration)
:label:`sec_value_iter`

이 섹션에서는 동적 프로그래밍을 사용하여 마르코프 결정 과정(MDP)에 대한 최적의 정책을 찾는 고전적인 알고리즘인 가치 반복(Value Iteration)에 대해 논의합니다. 가치 반복은 벨만 최적 방정식(Bellman optimality equation)을 반복적으로 적용하여 상태 가치 함수(state-value function)를 업데이트하고, 결과적으로 최적의 정책을 도출합니다.

FrozenLake 환경에서 로봇은 "위"($\uparrow$), "아래"($\rightarrow$), "왼쪽"($\leftarrow$), "오른쪽"($\rightarrow$) 행동으로 $4 \times 4$ 그리드(상태) 위를 이동합니다. 환경에는 로봇에게 알려지지 않은 여러 구멍(H) 셀과 얼어붙은(F) 셀, 목표 셀(G)이 포함되어 있습니다. 문제를 간단하게 유지하기 위해 로봇이 신뢰할 수 있는 행동을 한다고 가정합니다. 즉, 모든 $s \in \mathcal{S}, a \in \mathcal{A}$에 대해 $P(s' \mid s, a) = 1$입니다. 로봇이 목표에 도달하면 시도가 종료되고 행동에 관계없이 $1$의 보상을 받습니다. 다른 상태에서의 보상은 모든 행동에 대해 $0$입니다. 로봇의 목표는 주어진 시작 위치(S)($s_0$)에서 목표 위치(G)에 도달하여 *반환값*을 최대화하는 정책을 학습하는 것입니다.

다음 함수는 가치 반복을 구현합니다. 여기서 `env_info`는 MDP 및 환경 관련 정보를 포함하고 `gamma`는 할인율입니다.

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 각 항목이 의미하는 바를 보여주는 2D 배열
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')] 계산
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # 다음 상태
                    reward = pxrds[reward_idx]  # 보상
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # 최대 가치와 최대 행동 기록
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

위의 그림은 정책(화살표는 행동을 나타냄)과 가치 함수(색상 변화는 가치 함수가 어두운 색상으로 표시된 초기 값에서 밝은 색상으로 표시된 최적 값으로 시간이 지남에 따라 어떻게 변하는지 보여줌)를 보여줍니다. 보시다시피 가치 반복은 10번의 반복 후에 최적 가치 함수를 찾고 H 셀이 아닌 한 어떤 상태에서 시작하든 목표 상태(G)에 도달할 수 있습니다. 구현의 또 다른 흥미로운 측면은 최적 가치 함수를 찾는 것 외에도 이 가치 함수에 해당하는 최적 정책 $\pi^*$도 자동으로 찾았다는 것입니다.


## 요약
가치 반복 알고리즘의 주요 아이디어는 동적 프로그래밍 원칙을 사용하여 주어진 상태에서 얻은 최적 평균 반환값을 찾는 것입니다. 가치 반복 알고리즘을 구현하려면 마르코프 결정 과정(MDP), 예를 들어 전이 및 보상 함수를 완전히 알아야 합니다.


## 연습 문제

1. 그리드 크기를 $8 \times 8$로 늘려보십시오. $4 \times 4$ 그리드와 비교하여 최적 가치 함수를 찾는 데 몇 번의 반복이 걸립니까?
2. 가치 반복 알고리즘의 계산 복잡도는 얼마입니까?
3. $\gamma$(즉, 위 코드의 "gamma")가 $0$, $0.5$, $1$일 때 가치 반복 알고리즘을 다시 실행하고 결과를 분석하십시오.
4. $\gamma$ 값은 가치 반복이 수렴하는 데 걸리는 반복 횟수에 어떤 영향을 줍니까? $\gamma=1$일 때 어떻게 됩니까?

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/12005)
:end_tab:

```