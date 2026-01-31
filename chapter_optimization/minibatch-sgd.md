# 미니배치 확률적 경사 하강법 (Minibatch Stochastic Gradient Descent)
:label:`sec_minibatch_sgd`

지금까지 우리는 경사 기반 학습에 대한 접근 방식에서 두 가지 극단을 만났습니다. :numref:`sec_gd`는 전체 데이터셋을 사용하여 기울기를 계산하고 파라미터를 한 번에 한 번씩 업데이트합니다. 반대로 :numref:`sec_sgd`는 한 번에 하나의 훈련 예제를 처리하여 진전을 이룹니다.
이들 중 어느 것도 단점이 있습니다.
경사 하강법은 데이터가 매우 유사할 때 특히 *데이터 효율적*이지 않습니다.
확률적 경사 하강법은 CPU와 GPU가 벡터화의 전체 능력을 활용할 수 없기 때문에 특히 *계산 효율적*이지 않습니다.
이는 그 사이에 무언가가 있을 수 있음을 시사하며, 사실 그것이 우리가 지금까지 논의한 예제에서 사용해 온 것입니다.

## 벡터화와 캐시 (Vectorization and Caches)

미니배치를 사용하기로 결정하는 핵심에는 계산 효율성이 있습니다. 이는 다중 GPU 및 다중 서버로의 병렬화를 고려할 때 가장 쉽게 이해됩니다. 이 경우 각 GPU에 적어도 하나의 이미지를 보내야 합니다. 서버당 8개의 GPU와 16개의 서버가 있다면 이미 미니배치 크기가 128보다 작지 않아야 합니다.

단일 GPU 또는 CPU의 경우 상황은 조금 더 미묘합니다. 이러한 장치에는 여러 유형의 메모리, 종종 여러 유형의 계산 장치 및 이들 간의 다양한 대역폭 제약 조건이 있습니다.
예를 들어 CPU에는 소수의 레지스터가 있고 그다음 L1, L2, 그리고 어떤 경우에는 L3 캐시(다른 프로세서 코어 간에 공유됨)가 있습니다.
이러한 캐시는 크기와 대기 시간이 증가합니다(동시에 대역폭은 감소합니다).
프로세서가 메인 메모리 인터페이스가 제공할 수 있는 것보다 훨씬 더 많은 연산을 수행할 수 있다고 말하는 것으로 충분합니다.

첫째, 16 코어와 AVX-512 벡터화가 있는 2GHz CPU는 초당 최대 $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ 바이트를 처리할 수 있습니다. GPU의 기능은 이 수치를 100배 쉽게 초과합니다. 반면, 중급 서버 프로세서는 100GB/s 대역폭을 넘지 못할 수 있습니다. 즉, 프로세서를 계속 공급하는 데 필요한 것의 1/10 미만입니다. 설상가상으로 모든 메모리 액세스가 동일하게 생성되는 것은 아닙니다. 메모리 인터페이스는 일반적으로 64비트 너비 이상(예: GPU에서는 최대 384비트)이므로 단일 바이트를 읽으면 훨씬 더 넓은 액세스 비용이 발생합니다.

둘째, 첫 번째 액세스에는 상당한 오버헤드가 발생하는 반면 순차적 액세스는 비교적 저렴합니다(이것을 종종 버스트 읽기라고 함). 여러 소켓, 칩렛 및 기타 구조가 있을 때 캐싱과 같이 염두에 두어야 할 다른 많은 것들이 있습니다.
더 깊이 있는 논의는 이 [위키백과 문서](https://en.wikipedia.org/wiki/Cache_hierarchy)를 참조하십시오.

이러한 제약을 완화하는 방법은 프로세서에 데이터를 공급할 만큼 충분히 빠른 CPU 캐시 계층 구조를 사용하는 것입니다. 이것이 딥러닝에서 배치 처리의 *원동력*입니다. 문제를 단순하게 유지하기 위해 행렬-행렬 곱셈, 예를 들어 $\mathbf{A} = \mathbf{B}\mathbf{C}$를 고려해 봅시다. $\mathbf{A}$를 계산하기 위한 여러 가지 옵션이 있습니다. 예를 들어 다음을 시도할 수 있습니다.

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$를 계산할 수 있습니다. 즉, 내적을 통해 요소별로 계산할 수 있습니다.
2. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$를 계산할 수 있습니다. 즉, 한 번에 한 열씩 계산할 수 있습니다. 마찬가지로 $\mathbf{A}$를 한 번에 한 행 $\mathbf{A}_{i,:}$씩 계산할 수도 있습니다.
3. 단순히 $\mathbf{A} = \mathbf{B} \mathbf{C}$를 계산할 수 있습니다.
4. $\mathbf{B}$와 $\mathbf{C}$를 더 작은 블록 행렬로 나누고 한 번에 한 블록씩 $\mathbf{A}$를 계산할 수 있습니다.

첫 번째 옵션을 따르면 $\mathbf{A}_{ij}$ 요소를 계산하고 싶을 때마다 행 벡터 하나와 열 벡터 하나를 CPU로 복사해야 합니다. 더 나쁜 것은 행렬 요소가 순차적으로 정렬되어 있다는 사실 때문에 메모리에서 읽을 때 두 벡터 중 하나에 대해 많은 분리된 위치에 액세스해야 한다는 것입니다. 두 번째 옵션이 훨씬 더 유리합니다. 그 안에서 $\mathbf{B}$를 계속 순회하는 동안 열 벡터 $\mathbf{C}_{:,j}$를 CPU 캐시에 유지할 수 있습니다. 이것은 메모리 대역폭 요구 사항을 절반으로 줄이고 그에 상응하여 더 빠른 액세스를 제공합니다. 물론 옵션 3이 가장 바람직합니다. 불행히도 대부분의 행렬은 캐시에 완전히 들어가지 않을 수 있습니다(이것이 우리가 논의하고 있는 것입니다). 그러나 옵션 4는 실질적으로 유용한 대안을 제공합니다. 행렬의 블록을 캐시로 이동하고 로컬에서 곱할 수 있습니다. 최적화된 라이브러리가 우리를 위해 이것을 처리합니다. 실제로 이러한 연산이 얼마나 효율적인지 살펴봅시다.

계산 효율성 외에도 Python과 딥러닝 프레임워크 자체에서 발생하는 오버헤드도 상당합니다. 명령을 실행할 때마다 Python 인터프리터가 MXNet 엔진에 명령을 보내고, 엔진은 이를 계산 그래프에 삽입하고 스케줄링 중에 처리해야 한다는 것을 기억하십시오. 이러한 오버헤드는 꽤 해로울 수 있습니다. 요컨대, 가능할 때마다 벡터화(및 행렬)를 사용하는 것이 좋습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

책의 나머지 부분에서 실행 시간을 자주 벤치마킹할 것이므로 타이머를 정의해 봅시다.

```{.python .input}
#@tab all
class Timer:  #@save
    """여러 실행 시간을 기록합니다."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """타이머를 시작합니다."""
        self.tik = time.time()

    def stop(self):
        """타이머를 중지하고 시간을 목록에 기록합니다."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """평균 시간을 반환합니다."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """시간의 합을 반환합니다."""
        return sum(self.times)

    def cumsum(self):
        """누적 시간을 반환합니다."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

요소별 할당은 단순히 $\mathbf{B}$와 $\mathbf{C}$의 모든 행과 열을 반복하여 값을 $\mathbf{A}$에 할당합니다.

```{.python .input}
#@tab mxnet
# 한 번에 한 요소씩 A = BC 계산
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# 한 번에 한 요소씩 A = BC 계산
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# 한 번에 한 요소씩 A = BC 계산
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

더 빠른 전략은 열별 할당을 수행하는 것입니다.

```{.python .input}
#@tab mxnet
# 한 번에 한 열씩 A = BC 계산
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# 한 번에 한 열씩 A = BC 계산
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

마지막으로 가장 효과적인 방법은 전체 연산을 한 블록으로 수행하는 것입니다.
두 행렬 $\mathbf{B} \in \mathbb{R}^{m \times n}$ 및 $\mathbf{C} \in \mathbb{R}^{n \times p}$를 곱하는 데는 스칼라 곱셈과 덧셈을 별도의 연산으로 계산할 때(실제로는 융합됨) 약 $2mnp$ 부동 소수점 연산이 소요됩니다.
따라서 두 개의 $256 \times 256$ 행렬을 곱하는 데는 $0.03$ 십억 번의 부동 소수점 연산이 소요됩니다.
각각의 연산 속도가 어떤지 봅시다.

```{.python .input}
#@tab mxnet
# 한 번에 A = BC 계산
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, ')
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# 한 번에 A = BC 계산
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, ')
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, ')
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 미니배치 (Minibatches)

:label:`sec_minibatches`

과거에는 파라미터를 업데이트하기 위해 단일 관찰이 아닌 데이터의 *미니배치*를 읽는 것을 당연하게 여겼습니다. 이제 그에 대한 간단한 정당성을 제시합니다. 단일 관찰을 처리하려면 많은 단일 행렬-벡터(또는 벡터-벡터) 곱셈을 수행해야 하는데, 이는 꽤 비싸고 기본 딥러닝 프레임워크를 대신하여 상당한 오버헤드를 발생시킵니다. 이는 데이터에 적용할 때(종종 추론이라고 함) 네트워크를 평가할 때와 파라미터를 업데이트하기 위해 기울기를 계산할 때 모두 적용됩니다. 즉, $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$를 수행할 때마다 적용됩니다. 여기서

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

우리는 한 번에 관찰의 미니배치에 적용함으로써 이 연산의 *계산* 효율성을 높일 수 있습니다. 즉, 단일 관찰에 대한 기울기 $\mathbf{g}_t$를 작은 배치에 대한 것으로 대체합니다.

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

이것이 $\mathbf{g}_t$의 통계적 속성에 어떤 영향을 미치는지 봅시다: $\mathbf{x}_t$와 미니배치 $\mathcal{B}_t$의 모든 요소가 훈련 세트에서 무작위로 균일하게 추출되므로 기울기의 기대값은 변경되지 않습니다. 반면 분산은 상당히 감소합니다. 미니배치 기울기는 평균화되는 $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$개의 독립적인 기울기로 구성되므로 표준 편차는 $b^{-\frac{1}{2}}$의 인수만큼 감소합니다. 이것 자체로는 좋은 일입니다. 업데이트가 전체 기울기와 더 안정적으로 정렬됨을 의미하기 때문입니다.

순진하게 이것은 큰 미니배치 $\mathcal{B}_t$를 선택하는 것이 보편적으로 바람직하다는 것을 나타냅니다. 아쉽게도 어느 시점 이후에는 표준 편차의 추가 감소가 계산 비용의 선형 증가에 비해 미미합니다. 실제로는 GPU 메모리에 맞으면서도 우수한 계산 효율성을 제공할 만큼 충분히 큰 미니배치를 선택합니다. 절감 효과를 설명하기 위해 코드를 살펴봅시다. 여기서는 동일한 행렬-행렬 곱셈을 수행하지만 이번에는 한 번에 64개의 열로 "미니배치"로 나눕니다.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

보시다시피, 미니배치에서의 계산은 본질적으로 전체 행렬에서의 계산만큼 효율적입니다. 주의할 점이 있습니다. :numref:`sec_batch_norm`에서 우리는 미니배치의 분산 양에 크게 의존하는 유형의 정규화를 사용했습니다. 후자를 늘리면 분산이 감소하고 그에 따라 배치 정규화로 인한 노이즈 주입의 이점도 감소합니다. 적절한 항을 다시 스케일링하고 계산하는 방법에 대한 세부 정보는 예: :citet:`Ioffe.2017`를 참조하십시오.

## 데이터셋 읽기 (Reading the Dataset)

미니배치가 데이터에서 어떻게 효율적으로 생성되는지 살펴봅시다. 다음에서는 이 최적화 알고리즘들을 비교하기 위해 NASA에서 개발한 [다양한 항공기의 날개 소음](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)을 테스트하는 데이터셋을 사용합니다. 편의상 처음 $1,500$개의 예제만 사용합니다. 데이터는 전처리를 위해 백색화(whitened)됩니다. 즉, 평균을 제거하고 좌표당 분산을 $1$로 다시 스케일링합니다.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 밑바닥부터 구현하기 (Implementation from Scratch)

:numref:`sec_linear_scratch`의 미니배치 확률적 경사 하강법 구현을 상기하십시오. 다음에서는 약간 더 일반적인 구현을 제공합니다. 편의상 이 장의 뒷부분에서 소개되는 다른 최적화 알고리즘과 동일한 호출 서명을 갖습니다. 구체적으로 상태 입력 `states`를 추가하고 하이퍼파라미터를 `hyperparams` 딕셔너리에 넣습니다. 또한 훈련 함수에서 각 미니배치 예제의 손실을 평균화하므로 최적화 알고리즘의 기울기를 배치 크기로 나눌 필요가 없습니다.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

다음으로, 나중에 소개될 다른 최적화 알고리즘의 사용을 용이하게 하기 위해 일반 훈련 함수를 구현합니다. 선형 회귀 모델을 초기화하고 미니배치 확률적 경사 하강법 및 이후에 소개될 다른 알고리즘으로 모델을 훈련하는 데 사용할 수 있습니다.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 초기화
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 훈련
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 초기화
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 훈련
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 초기화
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # 훈련
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

배치 경사 하강법에 대한 최적화가 어떻게 진행되는지 봅시다. 미니배치 크기를 1500(즉, 총 예제 수)으로 설정하면 됩니다. 결과적으로 모델 파라미터는 에폭당 한 번만 업데이트됩니다. 진전이 거의 없습니다. 실제로 6단계 후에 진전이 멈춥니다.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

배치 크기가 1일 때 최적화를 위해 확률적 경사 하강법을 사용합니다. 구현의 단순성을 위해 일정한(작지만) 학습률을 선택했습니다. 확률적 경사 하강법에서는 예제가 처리될 때마다 모델 파라미터가 업데이트됩니다. 우리의 경우 에폭당 1500번의 업데이트에 해당합니다. 보시다시피 목적 함수 값의 감소는 한 에폭 후에 느려집니다. 두 절차 모두 한 에폭 내에 1500개의 예제를 처리했지만, 우리 실험에서 확률적 경사 하강법은 경사 하강법보다 더 많은 시간을 소비했습니다. 이는 확률적 경사 하강법이 파라미터를 더 자주 업데이트하고 한 번에 하나의 관찰을 처리하는 것이 덜 효율적이기 때문입니다.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

마지막으로 배치 크기가 100일 때 최적화를 위해 미니배치 확률적 경사 하강법을 사용합니다. 에폭당 필요한 시간은 확률적 경사 하강법에 필요한 시간과 배치 경사 하강법에 필요한 시간보다 짧습니다.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

배치 크기를 10으로 줄이면 각 배치의 작업 부하가 실행 효율성이 떨어지기 때문에 각 에폭에 걸리는 시간이 늘어납니다.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

이제 이전 네 가지 실험에 대한 시간 대 손실을 비교할 수 있습니다. 보시다시피 확률적 경사 하강법은 처리된 예제 수 측면에서 GD보다 빠르게 수렴하지만, 예제별로 기울기를 계산하는 것이 효율적이지 않기 때문에 GD보다 동일한 손실에 도달하는 데 더 많은 시간을 사용합니다. 미니배치 확률적 경사 하강법은 수렴 속도와 계산 효율성을 절충할 수 있습니다. 미니배치 크기 10은 확률적 경사 하강법보다 효율적입니다. 미니배치 크기 100은 런타임 측면에서 GD보다 성능이 뛰어납니다.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 간결한 구현 (Concise Implementation)

Gluon에서는 `Trainer` 클래스를 사용하여 최적화 알고리즘을 호출할 수 있습니다. 이것은 일반 훈련 함수를 구현하는 데 사용됩니다. 현재 장 전체에서 이것을 사용할 것입니다.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # 초기화
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 초기화
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss`는 1/2 인자 없이 제곱 오차를 계산합니다
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # 초기화
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError`는 1/2 인자 없이 제곱 오차를 계산합니다
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Gluon을 사용하여 마지막 실험을 반복하면 동일한 동작이 나타납니다.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## 요약 (Summary)

* 벡터화는 딥러닝 프레임워크에서 발생하는 오버헤드를 줄이고 CPU 및 GPU에서 더 나은 메모리 지역성 및 캐싱으로 인해 코드를 더 효율적으로 만듭니다.
* 확률적 경사 하강법에서 발생하는 통계적 효율성과 한 번에 대규모 배치 데이터를 처리하여 발생하는 계산 효율성 사이에는 상충 관계가 있습니다.
* 미니배치 확률적 경사 하강법은 계산 및 통계적 효율성이라는 두 가지 장점을 모두 제공합니다.
* 미니배치 확률적 경사 하강법에서는 훈련 데이터의 무작위 순열에 의해 얻은 데이터 배치를 처리합니다(즉, 각 관찰은 에폭당 한 번만 처리되지만 무작위 순서로 처리됨).
* 훈련 중에 학습률을 감소시키는 것이 좋습니다.
* 일반적으로 미니배치 확률적 경사 하강법은 클록 시간 측면에서 더 작은 위험으로 수렴하는 데 있어 확률적 경사 하강법 및 경사 하강법보다 빠릅니다.

## 연습 문제 (Exercises)

1. 배치 크기와 학습률을 수정하고 목적 함수 값의 감소율과 각 에폭에서 소비되는 시간을 관찰하십시오.
2. MXNet 문서를 읽고 `Trainer` 클래스의 `set_learning_rate` 함수를 사용하여 각 에폭 후에 미니배치 확률적 경사 하강법의 학습률을 이전 값의 1/10로 줄이십시오.
3. 미니배치 확률적 경사 하강법을 훈련 세트에서 *복원 추출(samples with replacement)*하는 변형과 비교하십시오. 어떻게 됩니까?
4. 사악한 지니가 당신에게 말하지 않고 데이터셋을 복제합니다(즉, 각 관찰이 두 번 발생하고 데이터셋이 원래 크기의 두 배로 커지지만 아무도 당신에게 말하지 않았습니다). 확률적 경사 하강법, 미니배치 확률적 경사 하강법 및 경사 하강법의 동작은 어떻게 변합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab: