# 자동 병렬 처리 (Automatic Parallelism)
:label:`sec_auto_para`


MXNet 및 PyTorch와 같은 딥러닝 프레임워크는 백엔드에서 자동으로 계산 그래프를 구축합니다. 
계산 그래프를 사용하면 시스템은 모든 의존성을 인식하고, 속도를 높이기 위해 상호 의존적이지 않은 여러 작업을 선택적으로 병렬로 실행할 수 있습니다. 
예를 들어, :numref:`sec_async`의 :numref:`fig_asyncgraph`는 두 변수를 독립적으로 초기화합니다. 결과적으로 시스템은 이들을 병렬로 실행하도록 선택할 수 있습니다.


일반적으로 단일 연산자는 모든 CPU의 모든 계산 리소스 또는 단일 GPU를 사용합니다. 
예를 들어, `dot` 연산자는 단일 기계에 여러 CPU 프로세서가 있더라도 모든 CPU의 모든 코어(및 스레드)를 사용합니다. 
단일 GPU에도 동일하게 적용됩니다. 
따라서 병렬 처리는 단일 장치 컴퓨터에서는 그렇게 유용하지 않습니다. 
장치가 여러 개인 경우 상황이 더 중요해집니다. 
병렬 처리는 일반적으로 여러 GPU 간에 가장 관련이 있지만, 로컬 CPU를 추가하면 성능이 약간 향상됩니다. 
예를 들어, GPU와 CPU를 결합하여 컴퓨터 비전 모델을 훈련하는 데 중점을 둔 :citet:`Hadjis.Zhang.Mitliagkas.ea.2016`을 참조하십시오. 
자동으로 병렬화되는 프레임워크의 편리함을 통해 우리는 몇 줄의 Python 코드로 동일한 목표를 달성할 수 있습니다. 
보다 광범위하게, 자동 병렬 계산에 대한 우리의 논의는 CPU와 GPU를 모두 사용하는 병렬 계산뿐만 아니라 계산과 통신의 병렬화에 중점을 둡니다.

이 섹션의 실험을 실행하려면 적어도 두 개의 GPU가 필요합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU에서의 병렬 계산 (Parallel Computation on GPUs)

테스트할 기준 작업 부하를 정의하는 것부터 시작하겠습니다: 아래의 `run` 함수는 `x_gpu1`과 `x_gpu2` 두 변수에 할당된 데이터를 사용하여 선택한 장치에서 10번의 행렬-행렬 곱셈을 수행합니다.

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
이제 데이터에 함수를 적용합니다. 캐싱이 결과에 영향을 미치지 않도록 측정 전에 두 장치 중 하나에서 단일 패스를 수행하여 장치를 워밍업합니다.
:end_tab:

:begin_tab:`pytorch`
이제 데이터에 함수를 적용합니다. 캐싱이 결과에 영향을 미치지 않도록 측정 전에 두 장치 중 하나에서 단일 패스를 수행하여 장치를 워밍업합니다. 
`torch.cuda.synchronize()`는 CUDA 장치의 모든 스트림에 있는 모든 커널이 완료될 때까지 기다립니다. 
동기화가 필요한 장치인 `device` 인수를 받습니다. 
장치 인수가 `None`(기본값)인 경우 `current_device()`에 의해 제공되는 현재 장치를 사용합니다.
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # 두 장치 모두 워밍업
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # 모든 장치 워밍업
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
두 작업 사이의 `waitall` 문을 제거하면 시스템은 두 장치 모두에서 자동으로 계산을 병렬화할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
두 작업 사이의 `synchronize` 문을 제거하면 시스템은 두 장치 모두에서 자동으로 계산을 병렬화할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

위의 경우 총 실행 시간은 각 부분의 합보다 작습니다. 딥러닝 프레임워크가 사용자를 대신하여 정교한 코드 없이도 두 GPU 장치에서 자동으로 계산을 스케줄링하기 때문입니다.



## 병렬 계산과 통신 (Parallel Computation and Communication)

많은 경우 CPU와 GPU 사이 또는 서로 다른 GPU 사이와 같이 서로 다른 장치 간에 데이터를 이동해야 합니다. 
예를 들어, 
여러 가속기 카드에서 기울기를 집계해야 하는 분산 최적화를 수행하려는 경우에 발생합니다. 
GPU에서 계산한 다음 결과를 다시 CPU로 복사하여 이를 시뮬레이션해 봅시다.

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
이것은 다소 비효율적입니다. 리스트의 나머지가 여전히 계산되는 동안 `y`의 일부를 이미 CPU로 복사하기 시작할 수 있다는 점에 유의하십시오. 
이 상황은 예를 들어 미니배치에서 기울기를 계산할 때 발생합니다. 일부 파라미터의 기울기는 다른 것보다 일찍 사용할 수 있습니다. 
따라서 GPU가 여전히 실행 중인 동안 PCI-Express 버스 대역폭을 사용하기 시작하는 것이 우리에게 유리합니다. 
두 부분 사이의 `waitall`을 제거하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이것은 다소 비효율적입니다. 리스트의 나머지가 여전히 계산되는 동안 `y`의 일부를 이미 CPU로 복사하기 시작할 수 있다는 점에 유의하십시오. 
이 상황은 예를 들어 미니배치에서 (역전파) 기울기를 계산할 때 발생합니다. 일부 파라미터의 기울기는 다른 것보다 일찍 사용할 수 있습니다. 
따라서 GPU가 여전히 실행 중인 동안 PCI-Express 버스 대역폭을 사용하기 시작하는 것이 우리에게 유리합니다. 
PyTorch에서는 `to()` 및 `copy_()`와 같은 여러 함수가 명시적인 `non_blocking` 인수를 허용하며, 이를 통해 호출자는 불필요한 경우 동기화를 우회할 수 있습니다. 
`non_blocking=True`로 설정하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

두 연산에 필요한 총 시간은 (예상대로) 각 부분의 합보다 작습니다. 
이 작업은 다른 리소스인 CPU와 GPU 사이의 버스를 사용하므로 병렬 계산과는 다릅니다. 
사실 우리는 두 장치 모두에서 계산하고 통신하는 것을 동시에 할 수 있습니다. 
위에서 언급했듯이 계산과 통신 사이에는 의존성이 있습니다: `y[i]`가 CPU로 복사되기 전에 먼저 계산되어야 합니다. 
다행히 시스템은 총 실행 시간을 줄이기 위해 `y[i]`를 계산하는 동안 `y[i-1]`을 복사할 수 있습니다.

CPU와 두 개의 GPU에서 훈련할 때 간단한 2개 레이어 MLP의 계산 그래프와 그 의존성을 보여주는 그림 :numref:`fig_twogpu`로 마무리합니다. 
이로 인해 발생하는 병렬 프로그램을 수동으로 스케줄링하는 것은 꽤 고통스러울 것입니다. 
이것이 최적화를 위해 그래프 기반 컴퓨팅 백엔드를 갖는 것이 유리한 부분입니다.

![CPU와 두 개의 GPU에 있는 2개 레이어 MLP의 계산 그래프와 그 의존성.](../img/twogpu.svg)
:label:`fig_twogpu`


## 요약 (Summary)

* 현대 시스템에는 여러 개의 GPU와 CPU와 같은 다양한 장치가 있습니다. 이들은 비동기적으로 병렬로 사용될 수 있습니다.
* 현대 시스템에는 또한 PCI Express, 저장 장치(일반적으로 솔리드 스테이트 드라이브 또는 네트워크를 통해), 네트워크 대역폭과 같은 통신을 위한 다양한 리소스가 있습니다. 이들은 최대 효율을 위해 병렬로 사용될 수 있습니다.
* 백엔드는 자동 병렬 계산 및 통신을 통해 성능을 향상시킬 수 있습니다.

## 연습 문제 (Exercises)

1. 이 섹션에 정의된 `run` 함수에서 8개의 연산이 수행되었습니다. 그들 사이에는 의존성이 없습니다. 딥러닝 프레임워크가 자동으로 병렬로 실행하는지 확인하기 위한 실험을 설계하십시오.
2. 개별 연산자의 작업 부하가 충분히 작을 때, 병렬 처리는 단일 CPU나 GPU에서도 도움이 될 수 있습니다. 이를 확인하기 위한 실험을 설계하십시오.
3. CPU, GPU에서의 병렬 계산과 두 장치 간의 통신을 사용하는 실험을 설계하십시오.
4. NVIDIA의 [Nsight](https://developer.nvidia.com/nsight-compute-2019_5)와 같은 디버거를 사용하여 코드가 효율적인지 확인하십시오.
5. 더 복잡한 데이터 의존성을 포함하는 계산 작업을 설계하고, 성능을 향상시키면서 올바른 결과를 얻을 수 있는지 확인하기 위해 실험을 실행하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1681)
:end_tab: