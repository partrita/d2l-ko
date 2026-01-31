# 비동기 계산 (Asynchronous Computation)
:label:`sec_async`

오늘날의 컴퓨터는 고도로 병렬화된 시스템으로, 여러 CPU 코어(종종 코어당 여러 스레드), GPU당 여러 처리 요소, 그리고 종종 장치당 여러 GPU로 구성됩니다. 요컨대, 우리는 종종 다른 장치에서 동시에 많은 다른 것들을 처리할 수 있습니다. 불행히도 Python은 추가적인 도움 없이는 병렬 및 비동기 코드를 작성하기에 좋은 방법이 아닙니다. 결국 Python은 싱글 스레드이며 이는 미래에도 바뀔 가능성이 낮습니다. MXNet 및 TensorFlow와 같은 딥러닝 프레임워크는 성능을 향상시키기 위해 *비동기 프로그래밍(asynchronous programming)* 모델을 채택하는 반면, PyTorch는 Python 자체 스케줄러를 사용하여 다른 성능 절충안을 제시합니다.
PyTorch의 경우 기본적으로 GPU 연산은 비동기적입니다. GPU를 사용하는 함수를 호출하면 연산이 특정 장치에 대기열로 들어가지만 나중에까지 반드시 실행되지는 않습니다. 이를 통해 CPU 또는 다른 GPU에서의 연산을 포함하여 더 많은 계산을 병렬로 실행할 수 있습니다.

따라서 비동기 프로그래밍이 어떻게 작동하는지 이해하면 계산 요구 사항과 상호 의존성을 사전에 줄여 더 효율적인 프로그램을 개발하는 데 도움이 됩니다. 이를 통해 메모리 오버헤드를 줄이고 프로세서 활용도를 높일 수 있습니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## 백엔드를 통한 비동기성 (Asynchrony via Backend)

:begin_tab:`mxnet`
워밍업으로 다음 장난감 문제를 고려해 봅시다: 무작위 행렬을 생성하고 곱하고 싶습니다. 차이를 확인하기 위해 NumPy와 `mxnet.np` 모두에서 수행해 봅시다.
:end_tab:

:begin_tab:`pytorch`
워밍업으로 다음 장난감 문제를 고려해 봅시다: 무작위 행렬을 생성하고 곱하고 싶습니다. 차이를 확인하기 위해 NumPy와 PyTorch 텐서 모두에서 수행해 봅시다.
PyTorch `tensor`는 GPU에서 정의된다는 점에 유의하십시오.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# GPU 계산을 위한 워밍업
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet을 통한 벤치마크 출력은 몇 배나 더 빠릅니다. 둘 다 동일한 프로세서에서 실행되므로 다른 무언가가 진행되고 있음에 틀림없습니다.
반환하기 전에 MXNet이 모든 백엔드 계산을 완료하도록 강제하면 이전에 발생한 일을 알 수 있습니다: 프론트엔드가 Python에 제어권을 반환하는 동안 백엔드에서 계산이 실행됩니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch를 통한 벤치마크 출력은 몇 배나 더 빠릅니다.
NumPy 내적은 CPU 프로세서에서 실행되는 반면 PyTorch 행렬 곱셈은 GPU에서 실행되므로 후자가 훨씬 더 빠를 것으로 예상됩니다. 그러나 거대한 시간 차이는 다른 무언가가 진행되고 있음을 시사합니다.
기본적으로 PyTorch에서 GPU 연산은 비동기적입니다.
반환하기 전에 PyTorch가 모든 계산을 완료하도록 강제하면 이전에 발생한 일을 알 수 있습니다: 프론트엔드가 Python에 제어권을 반환하는 동안 백엔드에서 계산이 실행되고 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
넓게 말해서, MXNet은 Python 등을 통해 사용자와 직접 상호 작용하기 위한 프론트엔드와 시스템이 계산을 수행하는 데 사용하는 백엔드를 가지고 있습니다.
:numref:`fig_frontends`에 표시된 것처럼, 사용자는 Python, R, Scala, C++와 같은 다양한 프론트엔드 언어로 MXNet 프로그램을 작성할 수 있습니다. 사용된 프론트엔드 프로그래밍 언어에 관계없이, MXNet 프로그램의 실행은 주로 C++ 구현의 백엔드에서 발생합니다. 프론트엔드 언어에서 발행된 연산은 실행을 위해 백엔드로 전달됩니다.
백엔드는 대기 중인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다. 이것이 작동하려면 백엔드가 계산 그래프의 다양한 단계 간의 의존성을 추적할 수 있어야 한다는 점에 유의하십시오. 따라서 서로 의존하는 연산을 병렬화하는 것은 불가능합니다.
:end_tab:

:begin_tab:`pytorch`
넓게 말해서, PyTorch는 Python 등을 통해 사용자와 직접 상호 작용하기 위한 프론트엔드와 시스템이 계산을 수행하는 데 사용하는 백엔드를 가지고 있습니다.
:numref:`fig_frontends`에 표시된 것처럼, 사용자는 Python 및 C++와 같은 다양한 프론트엔드 언어로 PyTorch 프로그램을 작성할 수 있습니다. 사용된 프론트엔드 프로그래밍 언어에 관계없이, PyTorch 프로그램의 실행은 주로 C++ 구현의 백엔드에서 발생합니다. 프론트엔드 언어에서 발행된 연산은 실행을 위해 백엔드로 전달됩니다.
백엔드는 대기 중인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다.
이것이 작동하려면 백엔드가 계산 그래프의 다양한 단계 간의 의존성을 추적할 수 있어야 한다는 점에 유의하십시오.
따라서 서로 의존하는 연산을 병렬화하는 것은 불가능합니다.
:end_tab:

![프로그래밍 언어 프론트엔드 및 딥러닝 프레임워크 백엔드.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

의존성 그래프를 좀 더 잘 이해하기 위해 다른 장난감 예제를 살펴봅시다.

```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![백엔드는 계산 그래프의 다양한 단계 간의 의존성을 추적합니다.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`



위의 코드 스니펫은 :numref:`fig_asyncgraph`에서도 설명되어 있습니다.
Python 프론트엔드 스레드가 처음 세 명령문 중 하나를 실행할 때마다, 단순히 작업을 백엔드 대기열로 반환합니다. 마지막 명령문의 결과를 *인쇄*해야 할 때, Python 프론트엔드 스레드는 C++ 백엔드 스레드가 변수 `z`의 결과 계산을 완료할 때까지 기다립니다. 이 디자인의 한 가지 이점은 Python 프론트엔드 스레드가 실제 계산을 수행할 필요가 없다는 것입니다. 따라서 Python의 성능에 관계없이 프로그램의 전체 성능에 거의 영향을 미치지 않습니다. :numref:`fig_threading`은 프론트엔드와 백엔드가 상호 작용하는 방식을 보여줍니다.

![프론트엔드와 백엔드의 상호 작용.](../img/threading.svg)
:label:`fig_threading`




## 장벽과 차단기 (Barriers and Blockers)

:begin_tab:`mxnet`
Python이 완료를 기다리도록 강제하는 여러 연산이 있습니다:

* 가장 명백한 것은 `npx.waitall()`로, 계산 명령이 언제 발행되었는지에 관계없이 모든 계산이 완료될 때까지 기다립니다. 실제로 이 연산자를 사용하면 성능이 저하될 수 있으므로 절대적으로 필요한 경우가 아니면 사용하지 않는 것이 좋습니다.
* 특정 변수가 사용 가능해질 때까지만 기다리고 싶다면 `z.wait_to_read()`를 호출할 수 있습니다. 이 경우 MXNet은 변수 `z`가 계산될 때까지 Python으로의 반환을 차단합니다. 다른 계산은 그 후에 계속될 수 있습니다.

이것이 실제로 어떻게 작동하는지 봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
두 연산 모두 완료하는 데 거의 같은 시간이 걸립니다. 명백한 차단 연산 외에도 *암시적* 차단기를 인식하는 것이 좋습니다. 변수를 인쇄하려면 분명히 변수가 사용 가능해야 하므로 차단기입니다. 마지막으로, `z.asnumpy()`를 통한 NumPy로의 변환과 `z.item()`을 통한 스칼라로의 변환은 차단적입니다. NumPy에는 비동기성 개념이 없기 때문입니다. `print` 함수와 마찬가지로 값에 대한 액세스가 필요합니다.

MXNet의 범위에서 NumPy로 그리고 다시 소량의 데이터를 빈번하게 복사하면 효율적인 코드의 성능을 파괴할 수 있습니다. 각 연산은 다른 어떤 작업을 수행하기 *전에* 관련 항을 얻는 데 필요한 모든 중간 결과를 계산 그래프가 평가해야 하기 때문입니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 계산 개선 (Improving Computation)

:begin_tab:`mxnet`
멀티 스레드가 많은 시스템(일반적인 노트북조차도 4개 이상의 스레드를 가지고 있으며 멀티 소켓 서버에서는 이 숫자가 256을 초과할 수 있음)에서 스케줄링 연산의 오버헤드는 상당할 수 있습니다. 이것이 계산과 스케줄링이 비동기적으로 병렬로 발생하는 것이 매우 바람직한 이유입니다. 그렇게 함으로써 얻는 이점을 설명하기 위해, 변수를 1씩 여러 번 증가시키는 작업을 순차적으로 또는 비동기적으로 수행하면 어떻게 되는지 봅시다. 각 덧셈 사이에 `wait_to_read` 장벽을 삽입하여 동기식 실행을 시뮬레이션합니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python 프론트엔드 스레드와 C++ 백엔드 스레드 간의 약간 단순화된 상호 작용은 다음과 같이 요약될 수 있습니다:
1. 프론트엔드가 백엔드에 계산 작업 `y = x + 1`을 대기열에 삽입하도록 명령합니다.
2. 백엔드는 대기열에서 계산 작업을 수신하고 실제 계산을 수행합니다.
3. 백엔드는 계산 결과를 프론트엔드에 반환합니다.
이 세 단계의 기간을 각각 $t_1, t_2, t_3$라고 가정합시다. 비동기 프로그래밍을 사용하지 않으면 10,000번의 계산을 수행하는 데 걸리는 총 시간은 약 $10000 (t_1+ t_2 + t_3)$입니다. 비동기 프로그래밍을 사용하면 프론트엔드가 각 루프에 대해 백엔드가 계산 결과를 반환할 때까지 기다릴 필요가 없으므로 10,000번의 계산을 수행하는 데 걸리는 총 시간을 $t_1 + 10000 t_2 + t_3$으로 줄일 수 있습니다($10000 t_2 > 9999t_1$ 가정).
:end_tab:


## 요약 (Summary)


* 딥러닝 프레임워크는 Python 프론트엔드를 실행 백엔드에서 분리할 수 있습니다. 이를 통해 명령을 백엔드에 빠르게 비동기적으로 삽입하고 관련 병렬 처리를 가능하게 합니다.
* 비동기성은 상당히 반응이 빠른 프론트엔드로 이어집니다. 그러나 작업 대기열을 너무 많이 채우면 과도한 메모리 소비로 이어질 수 있으므로 주의하십시오. 프론트엔드와 백엔드를 대략적으로 동기화된 상태로 유지하기 위해 각 미니배치마다 동기화하는 것이 좋습니다.
* 칩 공급업체는 딥러닝의 효율성에 대해 훨씬 더 세밀한 통찰력을 얻기 위해 정교한 성능 분석 도구를 제공합니다.

:begin_tab:`mxnet`
* MXNet의 메모리 관리에서 Python으로의 변환은 백엔드가 특정 변수가 준비될 때까지 기다리도록 강제한다는 사실을 인지하십시오. `print`, `asnumpy`, `item`과 같은 함수는 모두 이 효과를 가집니다. 이는 바람직할 수 있지만 부주의한 동기화 사용은 성능을 망칠 수 있습니다.
:end_tab:


## 연습 문제 (Exercises)

:begin_tab:`mxnet`
1. 위에서 비동기 계산을 사용하면 10,000번의 계산을 수행하는 데 필요한 총 시간을 $t_1 + 10000 t_2 + t_3$으로 줄일 수 있다고 언급했습니다. 왜 여기서 $10000 t_2 > 9999 t_1$이라고 가정해야 합니까?
2. `waitall`과 `wait_to_read` 사이의 차이를 측정하십시오. 힌트: 여러 명령을 수행하고 중간 결과에 대해 동기화하십시오.
:end_tab:

:begin_tab:`pytorch`
1. CPU에서 이 섹션의 동일한 행렬 곱셈 연산을 벤치마킹하십시오. 여전히 백엔드를 통한 비동기성을 관찰할 수 있습니까?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab: