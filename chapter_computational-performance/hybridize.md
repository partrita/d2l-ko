# 컴파일러와 인터프리터 (Compilers and Interpreters)
:label:`sec_hybridize`

지금까지 이 책은 `print`, `+`, `if`와 같은 문을 사용하여 프로그램의 상태를 변경하는 명령형 프로그래밍(imperative programming)에 집중해 왔습니다. 간단한 명령형 프로그램의 다음 예제를 고려해 보십시오.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python은 *인터프리터 언어*입니다. 위의 `fancy_func` 함수를 평가할 때 함수 본문을 구성하는 연산을 *순차적으로* 수행합니다. 즉, `e = add(a, b)`를 평가하고 결과를 변수 `e`로 저장하여 프로그램의 상태를 변경합니다. 다음 두 문 `f = add(c, d)`와 `g = add(e, f)`도 유사하게 실행되어 덧셈을 수행하고 결과를 변수로 저장합니다. :numref:`fig_compute_graph`는 데이터의 흐름을 보여줍니다.

![명령형 프로그램의 데이터 흐름.](../img/computegraph.svg)
:label:`fig_compute_graph`

명령형 프로그래밍은 편리하지만 비효율적일 수 있습니다. 한편으로 `fancy_func` 전체에서 `add` 함수가 반복적으로 호출되더라도 Python은 세 번의 함수 호출을 개별적으로 실행합니다. 이들이 가령 GPU(또는 여러 GPU)에서 실행된다면 Python 인터프리터에서 발생하는 오버헤드가 엄청날 수 있습니다. 더욱이 `fancy_func`의 모든 문이 실행될 때까지 `e`와 `f`의 변수 값을 저장해야 합니다. 이는 `e = add(a, b)`와 `f = add(c, d)` 문이 실행된 후 프로그램의 다른 부분에서 변수 `e`와 `f`가 사용될지 모르기 때문입니다.

## 기호 프로그래밍 (Symbolic Programming)

대안으로 프로세스가 완전히 정의된 후에만 계산이 수행되는 *기호 프로그래밍(symbolic programming)*을 고려해 보십시오. 이 전략은 Theano와 TensorFlow(후자는 명령형 확장을 획득함)를 포함한 여러 딥러닝 프레임워크에서 사용됩니다. 일반적으로 다음 단계가 포함됩니다:

1. 실행할 연산을 정의합니다.
2. 연산을 실행 가능한 프로그램으로 컴파일합니다.
3. 필요한 입력을 제공하고 실행을 위해 컴파일된 프로그램을 호출합니다.

이를 통해 상당한 양의 최적화가 가능합니다. 첫째, 많은 경우 Python 인터프리터를 건너뛸 수 있으므로 CPU의 단일 Python 스레드와 결합된 여러 고속 GPU에서 유의미해질 수 있는 성능 병목 현상을 제거합니다.
둘째, 컴파일러는 위의 코드를 `print((1 + 2) + (3 + 4))` 또는 심지어 `print(10)`으로 최적화하고 다시 쓸 수 있습니다. 이는 컴파일러가 코드를 기계 명령어로 바꾸기 전에 전체 코드를 볼 수 있기 때문에 가능합니다. 예를 들어 변수가 더 이상 필요하지 않을 때마다 메모리를 해제(또는 아예 할당하지 않음)할 수 있습니다. 또는 코드를 완전히 동등한 조각으로 변환할 수 있습니다.
더 나은 아이디어를 얻으려면 아래의 명령형 프로그래밍 시뮬레이션(결국 Python입니다)을 고려해 보십시오.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

명령형(인터프리터) 프로그래밍과 기호 프로그래밍의 차이점은 다음과 같습니다:

* 명령형 프로그래밍이 더 쉽습니다. Python에서 명령형 프로그래밍을 사용하면 코드의 대부분이 직관적이고 작성하기 쉽습니다. 명령형 프로그래밍 코드를 디버깅하는 것도 더 쉽습니다. 모든 관련 중간 변수 값을 얻고 인쇄하거나 Python의 내장 디버깅 도구를 사용하기가 더 쉽기 때문입니다.
* 기호 프로그래밍은 더 효율적이고 이식하기 쉽습니다. 기호 프로그래밍은 컴파일 중에 코드를 최적화하기 더 쉬우며, 프로그램을 Python에 독립적인 형식으로 이식할 수 있는 능력도 있습니다. 이를 통해 프로그램을 Python이 아닌 환경에서 실행할 수 있으므로 Python 인터프리터와 관련된 잠재적인 성능 문제를 피할 수 있습니다.


## 하이브리드 프로그래밍 (Hybrid Programming)

역사적으로 대부분의 딥러닝 프레임워크는 명령형 또는 기호적 접근 방식 중 하나를 선택합니다. 예를 들어 Theano, TensorFlow(전자의 영감을 받음), Keras, CNTK는 모델을 기호적으로 공식화합니다. 반대로 Chainer와 PyTorch는 명령형 접근 방식을 취합니다. 나중에 버전에서 TensorFlow 2.0과 Keras에 명령형 모드가 추가되었습니다.

:begin_tab:`mxnet`
Gluon을 설계할 때 개발자들은 두 프로그래밍 패러다임의 장점을 결합하는 것이 가능할지 고려했습니다. 이로 인해 사용자가 순수 명령형 프로그래밍으로 개발하고 디버깅하면서도, 제품 수준의 컴퓨팅 성능과 배포가 필요할 때 대부분의 프로그램을 기호 프로그램으로 변환하여 실행할 수 있게 해주는 하이브리드 모델이 탄생했습니다.

실전에서 이는 `HybridBlock` 또는 `HybridSequential` 클래스를 사용하여 모델을 구축함을 의미합니다. 기본적으로 이들 중 어느 것도 명령형 프로그래밍에서 `Block` 또는 `Sequential` 클래스가 실행되는 것과 동일한 방식으로 실행됩니다.
`HybridSequential` 클래스는 `HybridBlock`의 서브클래스입니다(`Sequential`이 `Block`을 상속하는 것과 같습니다). `hybridize` 함수가 호출되면 Gluon은 모델을 기호 프로그래밍에서 사용되는 형식으로 컴파일합니다. 이를 통해 모델 구현 방식을 희생하지 않고도 계산 집약적인 구성 요소를 최적화할 수 있습니다. 아래에서 순차 모델과 블록에 초점을 맞추어 그 이점을 설명하겠습니다.
:end_tab:

:begin_tab:`pytorch`
위에서 언급했듯이 PyTorch는 명령형 프로그래밍을 기반으로 하며 동적 계산 그래프를 사용합니다. 기호 프로그래밍의 이식성과 효율성을 활용하기 위해 개발자들은 두 프로그래밍 패러다임의 장점을 결합하는 것이 가능할지 고려했습니다. 이로 인해 사용자가 순수 명령형 프로그래밍을 사용하여 개발하고 디버깅하면서도, 제품 수준의 컴퓨팅 성능과 배포가 필요할 때 대부분의 프로그램을 기호 프로그램으로 변환하여 실행할 수 있게 해주는 torchscript가 탄생했습니다.
:end_tab:

:begin_tab:`tensorflow`
명령형 프로그래밍 패러다임은 이제 TensorFlow 2에서 기본값이 되었으며, 이는 이 언어를 처음 접하는 사람들에게 환영할 만한 변화입니다. 그러나 동일한 기호 프로그래밍 기술과 그에 따른 계산 그래프는 여전히 TensorFlow에 존재하며, 사용하기 쉬운 `tf.function` 데코레이터를 통해 접근할 수 있습니다. 이는 명령형 프로그래밍 패러다임을 TensorFlow에 가져왔고, 사용자가 더 직관적인 함수를 정의한 다음 TensorFlow 팀이 [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)라고 부르는 기능을 사용하여 자동으로 이를 래핑하고 계산 그래프로 컴파일할 수 있게 해 주었습니다.
:end_tab:

## `Sequential` 클래스 하이브리드화하기 (Hybridizing the `Sequential` Class)

하이브리드화가 어떻게 작동하는지 느끼는 가장 쉬운 방법은 여러 레이어가 있는 심층 네트워크를 고려하는 것입니다. 관례적으로 Python 인터프리터는 CPU 또는 GPU로 전달될 수 있는 명령을 생성하기 위해 모든 레이어에 대한 코드를 실행해야 합니다. 단일(고속) 컴퓨팅 장치의 경우 이는 큰 문제를 일으키지 않습니다. 반면에 AWS P3dn.24xlarge 인스턴스와 같은 고급 8-GPU 서버를 사용한다면 Python은 모든 GPU를 바쁘게 유지하는 데 어려움을 겪을 것입니다. 싱글 스레드 Python 인터프리터가 여기서 병목 현상이 됩니다. `Sequential`을 `HybridSequential`로 교체하여 코드의 상당 부분에 대해 이 문제를 어떻게 해결할 수 있는지 봅시다. 간단한 MLP를 정의하는 것으로 시작합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# 네트워크 팩토리
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# 네트워크 팩토리
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 네트워크 팩토리
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
`hybridize` 함수를 호출함으로써 MLP에서의 계산을 컴파일하고 최적화할 수 있습니다. 모델의 계산 결과는 변경되지 않습니다.
:end_tab:

:begin_tab:`pytorch`
`torch.jit.script` 함수를 사용하여 모델을 변환함으로써 MLP에서의 계산을 컴파일하고 최적화할 수 있습니다. 모델의 계산 결과는 변경되지 않습니다.
:end_tab:

:begin_tab:`tensorflow`
이전에 TensorFlow에서 구축된 모든 함수는 계산 그래프로 구축되었으므로 기본적으로 JIT 컴파일되었습니다. 그러나 TensorFlow 2.X 및 EagerTensor의 릴리스와 함께 이것은 더 이상 기본 동작이 아닙니다.
우리는 `tf.function`으로 이 기능을 다시 활성화할 수 있습니다. `tf.function`은 함수 데코레이터로 더 흔히 사용되지만 아래와 같이 일반 Python 함수처럼 직접 호출할 수도 있습니다. 모델의 계산 결과는 변경되지 않습니다.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
이것은 정말 좋아 보입니다: 단순히 블록을 `HybridSequential`로 지정하고 이전과 동일한 코드를 작성한 후 `hybridize`를 호출하면 됩니다. 일단 이 일이 발생하면 네트워크는 최적화됩니다(아래에서 성능을 벤치마킹할 것입니다). 불행히도 이것이 모든 레이어에 대해 마법처럼 작동하지는 않습니다. 즉, 레이어가 `HybridBlock` 클래스 대신 `Block` 클래스를 상속한다면 최적화되지 않을 것입니다.
:end_tab:

:begin_tab:`pytorch`
이것은 정말 좋아 보입니다: 이전과 동일한 코드를 작성하고 단순히 `torch.jit.script`를 사용하여 모델을 변환하면 됩니다. 일단 이 일이 발생하면 네트워크는 최적화됩니다(아래에서 성능을 벤치마킹할 것입니다).
:end_tab:

:begin_tab:`tensorflow`
이것은 정말 좋아 보입니다: 이전과 동일한 코드를 작성하고 단순히 `tf.function`을 사용하여 모델을 변환하면 됩니다. 일단 이 일이 발생하면 네트워크는 TensorFlow의 MLIR 중간 표현에서 계산 그래프로 구축되며 빠른 실행을 위해 컴파일러 수준에서 고도로 최적화됩니다(아래에서 성능을 벤치마킹할 것입니다).
`tf.function()` 호출에 `jit_compile = True` 플래그를 명시적으로 추가하면 TensorFlow에서 XLA(Accelerated Linear Algebra) 기능이 활성화됩니다. XLA는 특정 경우에 JIT 컴파일된 코드를 더욱 최적화할 수 있습니다. 그래프 모드 실행은 이 명시적인 정의 없이도 활성화되지만, XLA는 특히 GPU 환경에서 특정 대규모 선형 대수 연산(딥러닝 응용 프로그램에서 보는 것과 같은 맥락)을 훨씬 더 빠르게 만들 수 있습니다.
:end_tab:

### 하이브리드화에 의한 가속 (Acceleration by Hybridization)

컴파일을 통해 얻은 성능 향상을 입증하기 위해 하이브리드화 전후의 `net(x)` 평가에 필요한 시간을 비교합니다. 먼저 이 시간을 측정하기 위한 클래스를 정의합시다. 성능을 측정(및 개선)하기 위해 이 장 전체에서 유용하게 쓰일 것입니다.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """실행 시간 측정용."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
이제 네트워크를 두 번 호출할 수 있습니다. 한 번은 하이브리드화 없이, 다른 한 번은 하이브리드화와 함께입니다.
:end_tab:

:begin_tab:`pytorch`
이제 네트워크를 두 번 호출할 수 있습니다. 한 번은 torchscript 없이, 다른 한 번은 torchscript와 함께입니다.
:end_tab:

:begin_tab:`tensorflow`
이제 네트워크를 세 번 호출할 수 있습니다. 한 번은 eager 모드로 실행되고, 한 번은 그래프 모드 실행으로, 그리고 다시 JIT 컴파일된 XLA를 사용하여 실행됩니다.
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
위의 결과에서 관찰된 바와 같이, `HybridSequential` 인스턴스가 `hybridize` 함수를 호출한 후 기호 프로그래밍의 사용을 통해 컴퓨팅 성능이 향상되었습니다.
:end_tab:

:begin_tab:`pytorch`
위의 결과에서 관찰된 바와 같이, `nn.Sequential` 인스턴스가 `torch.jit.script` 함수를 사용하여 스크립팅된 후 기호 프로그래밍의 사용을 통해 컴퓨팅 성능이 향상되었습니다.
:end_tab:

:begin_tab:`tensorflow`
위의 결과에서 관찰된 바와 같이, `tf.keras.Sequential` 인스턴스가 `tf.function` 함수를 사용하여 스크립팅된 후 TensorFlow의 그래프 모드 실행을 통해 컴퓨팅 성능이 향상되었습니다.
:end_tab:

### 직렬화 (Serialization)

:begin_tab:`mxnet`
모델을 컴파일하는 것의 이점 중 하나는 모델과 그 파라미터를 디스크에 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이를 통해 훈련된 모델을 다른 장치에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용할 수 있습니다. 동시에 코드는 명령형 프로그래밍에서 달성할 수 있는 것보다 종종 더 빠릅니다. `export` 함수가 작동하는 것을 봅시다.
:end_tab:

:begin_tab:`pytorch`
모델을 컴파일하는 것의 이점 중 하나는 모델과 그 파라미터를 디스크에 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이를 통해 훈련된 모델을 다른 장치에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용할 수 있습니다. 동시에 코드는 명령형 프로그래밍에서 달성할 수 있는 것보다 종종 더 빠릅니다. `save` 함수가 작동하는 것을 봅시다.
:end_tab:

:begin_tab:`tensorflow`
모델을 컴파일하는 것의 이점 중 하나는 모델과 그 파라미터를 디스크에 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이를 통해 훈련된 모델을 다른 장치에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용하거나 서버에서 훈련된 모델을 실행할 수 있습니다. 동시에 코드는 명령형 프로그래밍에서 달성할 수 있는 것보다 종종 더 빠릅니다.
TensorFlow에서 저장을 가능하게 하는 저수준 API는 `tf.saved_model`입니다.
`saved_model` 인스턴스가 작동하는 것을 봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
모델은 (큰 이진) 파라미터 파일과 모델 계산을 실행하는 데 필요한 프로그램의 JSON 설명으로 분해됩니다. 파일은 C++, R, Scala, Perl과 같이 Python이나 MXNet에서 지원하는 다른 프론트엔드 언어에서 읽을 수 있습니다. 모델 설명의 처음 몇 줄을 살펴봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
앞서 우리는 `hybridize` 함수를 호출한 후 모델이 우수한 컴퓨팅 성능과 이식성을 달성할 수 있음을 입증했습니다. 하지만 하이브리드화가 모델의 유연성, 특히 제어 흐름 측면에서 영향을 줄 수 있다는 점에 유의하십시오.

게다가 `forward` 함수를 사용해야 하는 `Block` 인스턴스와 달리, `HybridBlock` 인스턴스의 경우 `hybrid_forward` 함수를 사용해야 합니다.
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
위의 코드는 4개의 은닉 유닛과 2개의 출력을 가진 간단한 네트워크를 구현합니다. `hybrid_forward` 함수는 추가 인수 `F`를 취합니다. 코드가 하이브리드화되었는지 여부에 따라 처리를 위해 약간 다른 라이브러리(`ndarray` 또는 `symbol`)를 사용하기 때문에 이것이 필요합니다. 두 클래스는 매우 유사한 기능을 수행하며 MXNet이 자동으로 인수를 결정합니다. 무슨 일이 일어나고 있는지 이해하기 위해 함수 호출의 일부로 인수를 인쇄합니다.
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
순방향 계산을 반복하면 동일한 출력이 생성됩니다(세부 사항은 생략합니다). 이제 `hybridize` 함수를 호출하면 어떻게 되는지 봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
`ndarray` 대신 이제 `F`에 대해 `symbol` 모듈을 사용합니다. 더욱이 입력이 `ndarray` 유형이더라도 네트워크를 흐르는 데이터는 이제 컴파일 프로세스의 일부로 `symbol` 유형으로 변환됩니다. 함수 호출을 반복하면 놀라운 결과가 나옵니다:
:end_tab:

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
이것은 우리가 이전에 보았던 것과 매우 다릅니다. `hybrid_forward`에 정의된 모든 print 문이 생략되었습니다. 실제로 하이브리드화 후 `net(x)`의 실행은 더 이상 Python 인터프리터를 포함하지 않습니다. 이는 훨씬 더 간소화된 실행과 더 나은 성능을 위해 불필요한 Python 코드(예: print 문)가 생략됨을 의미합니다. 대신 MXNet은 직접 C++ 백엔드를 호출합니다. 또한 일부 함수는 `symbol` 모듈에서 지원되지 않으며(예: `asnumpy`), `a += b` 및 `a[:] = a + b`와 같은 제자리 연산은 `a = a + b`로 다시 써야 한다는 점에 유의하십시오. 그럼에도 불구하고 모델 컴파일은 속도가 중요할 때마다 노력할 가치가 있습니다. 그 이점은 모델의 복잡성, CPU 속도, GPU의 속도와 수에 따라 작은 퍼센트 포인트에서 두 배 이상의 속도까지 다양할 수 있습니다.
:end_tab:

## 요약 (Summary)


* 명령형 프로그래밍은 제어 흐름이 있는 코드를 작성할 수 있고 Python 소프트웨어 생태계의 상당 부분을 사용할 수 있는 능력이 있기 때문에 새로운 모델을 설계하기 쉽게 만듭니다.
* 기호 프로그래밍은 프로그램을 지정하고 실행하기 전에 컴파일해야 합니다. 이점은 향상된 성능입니다.

:begin_tab:`mxnet` 
* MXNet은 필요에 따라 두 접근 방식의 장점을 결합할 수 있습니다.
* `HybridSequential` 및 `HybridBlock` 클래스에 의해 구축된 모델은 `hybridize` 함수를 호출하여 명령형 프로그램을 기호 프로그램으로 변환할 수 있습니다.
:end_tab:


## 연습 문제 (Exercises)


:begin_tab:`mxnet` 
1. 이 섹션의 `HybridNet` 클래스의 `hybrid_forward` 함수 첫 번째 줄에 `x.asnumpy()`를 추가하십시오. 코드를 실행하고 마주치는 오류를 관찰하십시오. 왜 발생합니까?
2. `hybrid_forward` 함수에 제어 흐름, 즉 Python 문 `if`와 `for`를 추가하면 어떻게 됩니까?
3. 이전 장에서 관심 있는 모델들을 검토하십시오. 이들을 재구현하여 계산 성능을 향상시킬 수 있습니까?
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. 이전 장에서 관심 있는 모델들을 검토하십시오. 이들을 재구현하여 계산 성능을 향상시킬 수 있습니까?
:end_tab:




:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/2492)
:end_tab: