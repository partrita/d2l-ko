# 다중 GPU 훈련 (Training on Multiple GPUs)
:label:`sec_multi_gpu`

지금까지 우리는 CPU와 GPU에서 모델을 효율적으로 훈련하는 방법에 대해 논의했습니다. :numref:`sec_auto_para`에서는 딥러닝 프레임워크가 이들 간의 계산 및 통신을 자동으로 병렬화하는 방법까지 보여주었습니다. 또한 :numref:`sec_use_gpu`에서는 `nvidia-smi` 명령을 사용하여 컴퓨터에서 사용 가능한 모든 GPU를 나열하는 방법을 보여주었습니다.
우리가 논의하지 *않은* 것은 딥러닝 훈련을 실제로 병렬화하는 방법입니다.
대신 데이터를 여러 장치에 어떻게든 분할하여 작동하게 할 것이라고 암시했습니다. 이 섹션에서는 세부 사항을 채우고 처음부터 시작할 때 병렬로 네트워크를 훈련하는 방법을 보여줍니다. 고수준 API의 기능을 활용하는 방법에 대한 자세한 내용은 :numref:`sec_multi_gpu_concise`로 미룹니다.
:numref:`sec_minibatch_sgd`에 설명된 것과 같은 미니배치 확률적 경사 하강법 알고리즘에 익숙하다고 가정합니다.


## 문제 분할 (Splitting the Problem)

간단한 컴퓨터 비전 문제와 약간 구식인 네트워크, 예를 들어 여러 층의 합성곱, 풀링, 그리고 마지막에 몇 개의 완전 연결 층이 있는 네트워크로 시작해 봅시다.
즉, LeNet :cite:`LeCun.Bottou.Bengio.ea.1998`이나 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`과 매우 유사해 보이는 네트워크로 시작해 봅시다.
여러 GPU(데스크탑 서버의 경우 2개, AWS g4dn.12xlarge 인스턴스의 경우 4개, p3.16xlarge의 경우 8개, p2.16xlarge의 경우 16개)가 주어졌을 때, 우리는 간단하고 재현 가능한 설계 선택의 이점을 동시에 누리면서 좋은 속도 향상을 달성하는 방식으로 훈련을 분할하고 싶습니다. 결국 여러 GPU는 *메모리*와 *계산* 능력을 모두 증가시킵니다. 요컨대, 분류하고자 하는 훈련 데이터의 미니배치가 주어졌을 때 다음과 같은 선택지가 있습니다.

첫째, 네트워크를 여러 GPU에 걸쳐 분할할 수 있습니다. 즉, 각 GPU는 특정 층으로 들어오는 데이터를 입력으로 받아 여러 후속 층에 걸쳐 데이터를 처리한 다음 데이터를 다음 GPU로 보냅니다.
이를 통해 단일 GPU가 처리할 수 있는 것보다 더 큰 네트워크로 데이터를 처리할 수 있습니다.
게다가
GPU당 메모리 사용량을 잘 제어할 수 있습니다(전체 네트워크 사용량의 일부임).

그러나 층(따라서 GPU) 간의 인터페이스에는 긴밀한 동기화가 필요합니다. 특히 층 간의 계산 작업 부하가 적절하게 일치하지 않는 경우 까다로울 수 있습니다. 이 문제는 GPU 수가 많을수록 악화됩니다.
층 간의 인터페이스는 또한 활성화 및 기울기와 같은 대량의 데이터 전송을 필요로 합니다.
이는 GPU 버스의 대역폭을 압도할 수 있습니다.
또한 계산 집약적이지만 순차적인 연산은 분할하기가 쉽지 않습니다. 이와 관련하여 최선의 노력에 대해서는 예: :citet:`Mirhoseini.Pham.Le.ea.2017`를 참조하십시오. 이는 여전히 어려운 문제이며 비자명한 문제에서 좋은(선형) 확장을 달성할 수 있는지 불분명합니다. 여러 GPU를 함께 연결하는 것에 대한 훌륭한 프레임워크나 운영 체제 지원이 없다면 권장하지 않습니다.


둘째, 작업을 층별로 분할할 수 있습니다. 예를 들어 단일 GPU에서 64개 채널을 계산하는 대신 4개의 GPU에 걸쳐 문제를 분할하여 각각 16개 채널에 대한 데이터를 생성하게 할 수 있습니다.
마찬가지로 완전 연결 층의 경우 출력 유닛 수를 분할할 수 있습니다. :numref:`fig_alexnet_original`(:citet:`Krizhevsky.Sutskever.Hinton.2012`에서 가져옴)은 이 설계를 보여주는데, 이 전략은 메모리 사용량이 매우 작은(당시 2GB) GPU를 처리하는 데 사용되었습니다.
채널(또는 유닛) 수가 너무 작지 않다면 계산 측면에서 좋은 확장을 허용합니다.
게다가
사용 가능한 메모리가 선형적으로 확장되므로 여러 GPU가 점점 더 큰 네트워크를 처리할 수 있습니다.

![제한된 GPU 메모리로 인한 원래 AlexNet 설계의 모델 병렬화.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

그러나
각 층이 다른 모든 층의 결과에 의존하기 때문에 *매우 많은* 수의 동기화 또는 장벽(barrier) 연산이 필요합니다.
게다가 전송해야 하는 데이터 양은 층을 GPU에 걸쳐 분산할 때보다 훨씬 더 클 수 있습니다. 따라서 대역폭 비용과 복잡성 때문에 이 접근 방식을 권장하지 않습니다.

마지막으로, 데이터를 여러 GPU에 걸쳐 분할할 수 있습니다. 이렇게 하면 모든 GPU가 서로 다른 관찰에 대해서지만 동일한 유형의 작업을 수행합니다. 기울기는 훈련 데이터의 각 미니배치 후에 GPU 간에 집계됩니다.
이것은 가장 간단한 접근 방식이며 모든 상황에 적용될 수 있습니다.
각 미니배치 후에만 동기화하면 됩니다. 그렇긴 하지만 다른 파라미터가 아직 계산되는 동안 이미 계산된 파라미터의 기울기 교환을 시작하는 것이 매우 바람직합니다.
또한 GPU 수가 많을수록 미니배치 크기가 커져 훈련 효율성이 높아집니다.
그러나 GPU를 더 추가한다고 해서 더 큰 모델을 훈련할 수 있는 것은 아닙니다.


![여러 GPU에서의 병렬화. 왼쪽에서 오른쪽으로: 원래 문제, 네트워크 분할, 층별 분할, 데이터 병렬화.](../img/splitting.svg)
:label:`fig_splitting`


여러 GPU에서의 다양한 병렬화 방법 비교는 :numref:`fig_splitting`에 묘사되어 있습니다.
대체로 충분히 큰 메모리를 가진 GPU에 액세스할 수 있다면 데이터 병렬화가 진행하기에 가장 편리한 방법입니다. 분산 훈련을 위한 분할에 대한 자세한 설명은 :cite:`Li.Andersen.Park.ea.2014`를 참조하십시오. GPU 메모리는 딥러닝 초기에 문제였지만 지금은 가장 특이한 경우를 제외하고는 이 문제가 해결되었습니다. 다음에서는 데이터 병렬화에 초점을 맞춥니다.

## 데이터 병렬화 (Data Parallelism)

기계에 $k$개의 GPU가 있다고 가정합시다. 훈련할 모델이 주어지면 각 GPU는 파라미터 값이 GPU 간에 동일하고 동기화되지만 독립적으로 전체 모델 파라미터 세트를 유지 관리합니다.
예를 들어,
:numref:`fig_data_parallel`은 $k=2$일 때 데이터 병렬화로 훈련하는 것을 보여줍니다.


![두 개의 GPU에서 데이터 병렬화를 사용한 미니배치 확률적 경사 하강법 계산.](../img/data-parallel.svg)
:label:`fig_data_parallel`

일반적으로 훈련은 다음과 같이 진행됩니다:

* 훈련의 임의의 반복에서 무작위 미니배치가 주어지면, 배치의 예제들을 $k$개 부분으로 나누고 GPU 전체에 균등하게 분배합니다.
* 각 GPU는 할당된 미니배치 부분 집합을 기반으로 손실과 모델 파라미터의 기울기를 계산합니다.
* $k$개 GPU 각각의 로컬 기울기를 집계하여 현재 미니배치 확률적 기울기를 얻습니다.
* 집계된 기울기를 각 GPU에 재분배합니다.
* 각 GPU는 이 미니배치 확률적 기울기를 사용하여 유지 관리하는 전체 모델 파라미터 세트를 업데이트합니다.




실제로 $k$개의 GPU에서 훈련할 때 미니배치 크기를 $k$배로 *늘려서* 각 GPU가 마치 단일 GPU에서 훈련하는 것처럼 동일한 양의 작업을 수행하도록 합니다. 16-GPU 서버에서 이는 미니배치 크기를 상당히 증가시킬 수 있으며 이에 따라 학습률을 높여야 할 수도 있습니다.
또한 :numref:`sec_batch_norm`의 배치 정규화는 조정이 필요합니다. 예를 들어 GPU당 별도의 배치 정규화 계수를 유지하는 것입니다.
다음에서는 장난감 네트워크를 사용하여 다중 GPU 훈련을 설명하겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**장난감 네트워크 (A Toy Network)**]

:numref:`sec_lenet`에서 소개한 LeNet을 사용합니다(약간 수정됨). 파라미터 교환 및 동기화를 자세히 설명하기 위해 처음부터 정의합니다.

```{.python .input}
#@tab mxnet
# 모델 파라미터 초기화
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 모델 정의
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# 교차 엔트로피 손실 함수
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# 모델 파라미터 초기화
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 모델 정의
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 교차 엔트로피 손실 함수
loss = nn.CrossEntropyLoss(reduction='none')
```

## 데이터 동기화 (Data Synchronization)

효율적인 다중 GPU 훈련을 위해서는 두 가지 기본 연산이 필요합니다.
첫째, [**파라미터 리스트를 여러 장치에 분배**]하고 기울기를 첨부할 수 있는 기능(`get_params`)이 필요합니다. 파라미터 없이는 GPU에서 네트워크를 평가할 수 없습니다.
둘째, 여러 장치에 걸쳐 파라미터를 합산하는 기능, 즉 `allreduce` 함수가 필요합니다.

```{.python .input}
#@tab mxnet
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

모델 파라미터를 하나의 GPU에 복사하여 시도해 봅시다.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

아직 계산을 수행하지 않았으므로 편향 파라미터에 대한 기울기는 여전히 0입니다.
이제 벡터가 여러 GPU에 분산되어 있다고 가정해 봅시다. 다음 [**`allreduce` 함수는 모든 벡터를 더하고 결과를 모든 GPU에 다시 브로드캐스트합니다**]. 이것이 작동하려면 결과를 누적하는 장치에 데이터를 복사해야 한다는 점에 유의하십시오.

```{.python .input}
#@tab mxnet
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

서로 다른 장치에 다른 값을 가진 벡터를 만들고 집계하여 이를 테스트해 봅시다.

```{.python .input}
#@tab mxnet
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## 데이터 분배 (Distributing Data)

우리는 [**미니배치를 여러 GPU에 균등하게 분배**]하는 간단한 유틸리티 함수가 필요합니다. 예를 들어 두 개의 GPU에서 데이터의 절반이 각 GPU에 복사되기를 원합니다.
더 편리하고 간결하므로 딥러닝 프레임워크의 내장 함수를 사용하여 $4 	imes 5$ 행렬에서 시도해 봅니다.

```{.python .input}
#@tab mxnet
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

나중에 재사용하기 위해 데이터와 레이블을 모두 분할하는 `split_batch` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def split_batch(X, y, devices):
    ""`X`와 `y`를 여러 장치로 분할합니다."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    ""`X`와 `y`를 여러 장치로 분할합니다."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 훈련 (Training)

이제 [**단일 미니배치에 대한 다중 GPU 훈련**]을 구현할 수 있습니다. 그 구현은 주로 이 섹션에서 설명한 데이터 병렬화 접근 방식을 기반으로 합니다. 방금 논의한 보조 함수 `allreduce`와 `split_and_load`를 사용하여 여러 GPU 간에 데이터를 동기화할 것입니다. 병렬화를 달성하기 위해 특정 코드를 작성할 필요가 없다는 점에 유의하십시오. 계산 그래프에는 미니배치 내에서 장치 간의 의존성이 없으므로 *자동으로* 병렬로 실행됩니다.

```{.python .input}
#@tab mxnet
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # 손실은 각 GPU에서 별도로 계산됩니다
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # 역전파는 각 GPU에서 별도로 수행됩니다
        l.backward()
    # 각 GPU의 모든 기울기를 합산하여 모든 GPU에 브로드캐스트합니다
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 모델 파라미터는 각 GPU에서 별도로 업데이트됩니다
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # 여기서 전체 크기 배치를 사용합니다
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 손실은 각 GPU에서 별도로 계산됩니다
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # 역전파는 각 GPU에서 별도로 수행됩니다
        l.backward()
    # 각 GPU의 모든 기울기를 합산하여 모든 GPU에 브로드캐스트합니다
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 모델 파라미터는 각 GPU에서 별도로 업데이트됩니다
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 여기서 전체 크기 배치를 사용합니다
```

이제 [**훈련 함수**]를 정의할 수 있습니다. 이전 장에서 사용한 것과 약간 다릅니다: GPU를 할당하고 모든 모델 파라미터를 모든 장치에 복사해야 합니다.
분명히 각 배치는 다중 GPU를 처리하기 위해 `train_batch` 함수를 사용하여 처리됩니다. 편의상(그리고 코드의 간결성을 위해) 단일 GPU에서 정확도를 계산하지만, 다른 GPU가 유휴 상태이므로 *비효율적*입니다.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 모델 파라미터를 `num_gpus`개의 GPU에 복사
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 단일 미니배치에 대해 다중 GPU 훈련 수행
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # GPU 0에서 모델 평가
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 모델 파라미터를 `num_gpus`개의 GPU에 복사
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 단일 미니배치에 대해 다중 GPU 훈련 수행
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # GPU 0에서 모델 평가
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

이것이 [**단일 GPU에서**] 얼마나 잘 작동하는지 봅시다.
먼저 배치 크기 256과 학습률 0.2를 사용합니다.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

배치 크기와 학습률을 변경하지 않고 [**GPU 수를 2로 늘리면**], 테스트 정확도가 이전 실험과 대략 동일하게 유지됨을 알 수 있습니다.
최적화 알고리즘 측면에서 그들은 동일합니다. 불행히도 여기서는 얻을 수 있는 의미 있는 속도 향상이 없습니다. 모델이 너무 작기 때문입니다. 게다가 데이터셋도 작아서 다중 GPU 훈련을 구현하는 우리의 약간 정교하지 못한 접근 방식이 상당한 Python 오버헤드로 인해 어려움을 겪었습니다. 앞으로 더 복잡한 모델과 더 정교한 병렬화 방법을 만나게 될 것입니다.
그럼에도 불구하고 Fashion-MNIST에서 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## 요약 (Summary)

* 다중 GPU에 걸쳐 딥 네트워크 훈련을 분할하는 여러 가지 방법이 있습니다. 층 사이, 층 전체, 또는 데이터 전체에 걸쳐 분할할 수 있습니다. 앞의 두 가지는 데이터 전송을 엄격하게 조율해야 합니다. 데이터 병렬화는 가장 간단한 전략입니다.
* 데이터 병렬 훈련은 간단합니다. 그러나 효율성을 위해서는 유효 미니배치 크기를 늘려야 합니다.
* 데이터 병렬화에서 데이터는 여러 GPU에 걸쳐 분할되며, 각 GPU는 자체 순전파 및 역전파 연산을 실행하고 이후에 기울기가 집계되고 결과가 다시 GPU로 브로드캐스트됩니다.
* 더 큰 미니배치에 대해 약간 증가된 학습률을 사용할 수 있습니다.

## 연습 문제 (Exercises)

1. $k$개의 GPU에서 훈련할 때 미니배치 크기를 $b$에서 $k \cdot b$로 변경하십시오. 즉, GPU 수만큼 확장하십시오.
2. 다양한 학습률에 대한 정확도를 비교하십시오. GPU 수에 따라 어떻게 확장됩니까?
3. 서로 다른 GPU에서 다른 파라미터를 집계하는 더 효율적인 `allreduce` 함수를 구현하십시오. 왜 더 효율적입니까?
4. 다중 GPU 테스트 정확도 계산을 구현하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab: