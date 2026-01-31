```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 문서화 (Documentation)
:begin_tab:`mxnet`
모든 MXNet 함수와 클래스를 일일이 소개할 수는 없지만(정보가 빠르게 구식이 될 수도 있음), 
[API 문서](https://mxnet.apache.org/versions/1.8.0/api)와 
추가적인 [튜토리얼](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) 및 예제들이 
그러한 문서를 제공합니다. 
이 섹션은 MXNet API를 탐색하는 방법에 대한 지침을 제공합니다.
:end_tab:

:begin_tab:`pytorch`
모든 PyTorch 함수와 클래스를 일일이 소개할 수는 없지만(정보가 빠르게 구식이 될 수도 있음), 
[API 문서](https://pytorch.org/docs/stable/index.html)와 
추가적인 [튜토리얼](https://pytorch.org/tutorials/beginner/basics/intro.html) 및 예제들이 
그러한 문서를 제공합니다.
이 섹션은 PyTorch API를 탐색하는 방법에 대한 지침을 제공합니다.
:end_tab:

:begin_tab:`tensorflow`
모든 TensorFlow 함수와 클래스를 일일이 소개할 수는 없지만(정보가 빠르게 구식이 될 수도 있음), 
[API 문서](https://www.tensorflow.org/api_docs)와 
추가적인 [튜토리얼](https://www.tensorflow.org/tutorials) 및 예제들이 
그러한 문서를 제공합니다. 
이 섹션은 TensorFlow API를 탐색하는 방법에 대한 지침을 제공합니다.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## 모듈 내의 함수와 클래스 (Functions and Classes in a Module)

모듈에서 어떤 함수와 클래스를 호출할 수 있는지 알기 위해 `dir` 함수를 호출합니다. 예를 들어, 
(**난수를 생성하는 모듈의 모든 속성을 쿼리**)할 수 있습니다:

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

일반적으로 `__`(Python의 특수 객체)로 시작하고 끝나는 함수나 `_`(보통 내부 함수)로 시작하는 함수는 무시할 수 있습니다. 
남은 함수나 속성 이름을 바탕으로, 이 모듈이 균등 분포(`uniform`), 정규 분포(`normal`), 다항 분포(`multinomial`)로부터의 샘플링을 포함하여 
난수를 생성하는 다양한 방법을 제공한다는 것을 짐작할 수 있습니다.

## 특정 함수와 클래스 (Specific Functions and Classes)

주어진 함수나 클래스를 사용하는 방법에 대한 구체적인 지침을 보려면 `help` 함수를 호출할 수 있습니다. 예시로, 
[**텐서의 `ones` 함수 사용법을 탐색**]해 봅시다.

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

문서에서 `ones` 함수가 지정된 모양으로 새 텐서를 생성하고 
모든 요소를 값 1로 설정한다는 것을 볼 수 있습니다. 
가능할 때마다 해석을 확인하기 위해 (**빠른 테스트를 실행**)해야 합니다:

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Jupyter 노트북에서는 `?`를 사용하여 문서를 다른 창에 표시할 수 있습니다. 예를 들어, `list?`는 `help(list)`와 거의 동일한 내용을 생성하여 새 브라우저 창에 표시합니다. 또한 `list??`와 같이 물음표를 두 개 사용하면 함수를 구현하는 Python 코드도 표시됩니다.

공식 문서는 이 책의 범위를 넘어서는 많은 설명과 예제를 제공합니다. 
우리는 포괄적인 범위보다는 실실적인 문제로 빠르게 시작할 수 있게 해주는 
중요한 사용 사례를 강조합니다. 
또한 라이브러리의 소스 코드를 공부하여 고품질의 프로덕션 코드 구현 사례를 확인해 볼 것을 권장합니다. 
이렇게 함으로써 여러분은 과학자일 뿐만 아니라 더 나은 엔지니어가 될 것입니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17972)
:end_tab: