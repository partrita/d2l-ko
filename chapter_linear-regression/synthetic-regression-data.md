```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성 회귀 데이터 (Synthetic Regression Data)
:label:`sec_synthetic-regression-data`


머신러닝은 데이터에서 정보를 추출하는 것이 전부입니다. 
그렇다면 여러분은 합성 데이터에서 무엇을 배울 수 있을지 궁금할 것입니다. 
우리가 직접 만든 인공 데이터 생성 모델에 포함된 패턴 자체에는 본질적으로 관심이 없을 수도 있지만, 
그러한 데이터셋은 교육적인 목적으로는 여전히 유용하며, 
학습 알고리즘의 속성을 평가하고 우리의 구현이 예상대로 작동하는지 확인하는 데 도움이 됩니다. 
예를 들어, 정답 파라미터를 미리 알고 있는 데이터를 생성하면, 
우리 모델이 실제로 그 파라미터를 복구할 수 있는지 확인할 수 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## 데이터셋 생성하기 (Generating the Dataset)

이 예제에서는 간결함을 위해 저차원 데이터를 사용합니다. 
다음 코드 스니펫은 표준 정규 분포에서 추출한 2차원 특성을 가진 1000개의 예제를 생성합니다. 
결과로 생성된 설계 행렬 $\mathbf{X}$는 $\mathbb{R}^{1000 \times 2}$에 속합니다. 
우리는 *실제(ground truth)* 선형 함수를 적용하여 각 레이블을 생성하고, 
각 예제에 대해 독립적이고 동일하게 추출된 가산 노이즈 $\boldsymbol{\epsilon}$을 통해 오염시킵니다:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

편의를 위해 $\boldsymbol{\epsilon}$은 평균 $\mu= 0$이고 표준 편차 $\sigma = 0.01$인 정규 분포에서 추출된 것으로 가정합니다. 
객체 지향 설계를 위해, (:numref:`oo-design-data`에서 소개된) `d2l.DataModule`의 서브클래스의 `__init__` 메서드에 코드를 추가합니다. 
추가적인 하이퍼파라미터를 설정할 수 있게 하는 것이 좋은 관행입니다. 
우리는 `save_hyperparameters()`를 통해 이를 달성합니다. 
`batch_size`는 나중에 결정될 것입니다.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """선형 회귀를 위한 합성 데이터입니다."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

아래에서는 실제 파라미터를 $\mathbf{w} = [2, -3.4]^	op$ 및 $b = 4.2$로 설정합니다. 
나중에 추정된 파라미터를 이러한 *실제* 값과 비교하여 확인할 수 있습니다.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**`features`의 각 행은 $\mathbb{R}^2$의 벡터로 구성되고 `labels`의 각 행은 스칼라입니다.**] 첫 번째 항목을 살펴봅시다.

```{.python .input}
%%tab all
print('features:', data.X[0],'_label:', data.y[0])
```

## 데이터셋 읽기 (Reading the Dataset)

머신러닝 모델을 훈련할 때 종종 데이터셋을 여러 번 훑으며 한 번에 하나의 미니배치 예제를 가져와야 합니다. 
이 데이터는 모델을 업데이트하는 데 사용됩니다. 
이것이 어떻게 작동하는지 설명하기 위해, 우리는 (:numref:`oo-design-utilities`에서 소개된) `add_to_class`를 통해 `SyntheticRegressionData` 클래스에 등록되는 [**`get_dataloader` 메서드를 구현합니다.**] 
이 메서드는 (**배치 크기, 특성 행렬, 레이블 벡터를 받아 `batch_size` 크기의 미니배치를 생성합니다.**) 
따라서 각 미니배치는 특성과 레이블의 튜플로 구성됩니다. 
훈련 모드인지 검증 모드인지에 유의해야 합니다: 
전자의 경우 데이터를 무작위 순서로 읽기를 원할 것이고, 후자의 경우 디버깅 목적으로 미리 정의된 순서대로 데이터를 읽을 수 있는 것이 중요할 수 있습니다.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # 예제를 무작위 순서로 읽습니다.
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

직관을 얻기 위해 데이터의 첫 번째 미니배치를 검사해 봅시다. 
각 특성 미니배치는 그 크기와 입력 특성의 차원을 모두 제공합니다. 
마찬가지로 레이블 미니배치는 `batch_size`에 의해 주어지는 일치하는 모양을 가질 것입니다.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '_y shape:', y.shape)
```

겉보기에는 무해해 보이지만, `iter(data.train_dataloader())`의 호출은 Python의 객체 지향 설계의 힘을 잘 보여줍니다. 
`data` 객체를 생성한 *후에* `SyntheticRegressionData` 클래스에 메서드를 추가했다는 점에 유의하십시오. 
그럼에도 불구하고, 객체는 클래스에 기능이 사후에 추가된 혜택을 받습니다.

반복을 통해 전체 데이터셋이 소진될 때까지 별개의 미니배치를 얻습니다 (직접 시도해 보십시오). 
위에서 구현된 반복은 교육적 목적으로는 좋지만, 실제 문제에서는 곤란할 정도로 비효율적입니다. 
예를 들어, 모든 데이터를 메모리에 로드하고 많은 무작위 메모리 액세스를 수행해야 하기 때문입니다. 
딥러닝 프레임워크에 구현된 내장 반복기는 훨씬 더 효율적이며, 파일에 저장된 데이터, 스트림을 통해 수신된 데이터, 즉석에서 생성되거나 처리되는 데이터와 같은 소스를 처리할 수 있습니다. 
다음으로 내장 반복기를 사용하여 동일한 메서드를 구현해 보겠습니다.

## 데이터 로더의 간결한 구현 (Concise Implementation of the Data Loader)

자체 반복기를 작성하는 대신, [**데이터를 로드하기 위해 프레임워크의 기존 API를 호출할 수 있습니다.**] 
이전과 마찬가지로 특성 `X`와 레이블 `y`가 있는 데이터셋이 필요합니다. 
그 외에도 내장 데이터 로더에서 `batch_size`를 설정하고 예제를 효율적으로 셔플링하도록 맡깁니다.

:begin_tab:`jax`
JAX는 장치 가속 및 함수 변환이 포함된 NumPy와 유사한 API이므로, 적어도 현재 버전에는 데이터 로딩 메서드가 포함되어 있지 않습니다. 다른 라이브러리에는 이미 훌륭한 데이터 로더가 있으며, JAX는 대신 그것들을 사용할 것을 제안합니다. 여기서는 TensorFlow의 데이터 로더를 가져와 JAX와 호환되도록 약간 수정하겠습니다.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Tensorflow Datasets 및 Dataloader를 사용합니다. JAX나 Flax는
        # 어떠한 데이터 로딩 기능도 제공하지 않습니다.
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

새 데이터 로더는 더 효율적이고 몇 가지 기능이 추가되었다는 점을 제외하면 이전 데이터 로더와 똑같이 동작합니다.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '_y shape:', y.shape)
```

예를 들어, 프레임워크 API가 제공하는 데이터 로더는 내장 `__len__` 메서드를 지원하므로, 길이, 즉 배치 수를 쿼리할 수 있습니다.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## 요약 (Summary)

데이터 로더는 데이터 로딩 및 조작 과정을 추상화하는 편리한 방법입니다. 
이렇게 하면 동일한 머신러닝 *알고리즘*이 수정 없이도 다양한 유형과 소스의 데이터를 처리할 수 있습니다. 
데이터 로더의 좋은 점 중 하나는 결합될 수 있다는 것입니다. 
예를 들어 이미지를 로드한 다음 이미지를 자르거나 다른 방식으로 수정하는 후처리 필터를 가질 수 있습니다. 
따라서 데이터 로더는 전체 데이터 처리 파이프라인을 설명하는 데 사용될 수 있습니다. 

모델 자체에 관해서는, 2차원 선형 모델은 우리가 마주칠 수 있는 가장 간단한 모델 중 하나입니다. 
데이터 양이 불충분하거나 방정식 시스템이 불충분하게 결정되는 것에 대해 걱정하지 않고 회귀 모델의 정확도를 테스트할 수 있게 해줍니다. 
우리는 다음 섹션에서 이를 유용하게 활용할 것입니다.  


## 연습 문제 (Exercises)

1. 예제의 수가 배치 크기로 나누어떨어지지 않으면 어떻게 될까요? 프레임워크의 API를 사용하여 다른 인수를 지정함으로써 이 동작을 어떻게 바꾸겠습니까?
2. 파라미터 벡터 `w`의 크기와 예제 수 `num_examples`가 모두 큰 거대한 데이터셋을 생성하고 싶다고 가정해 봅시다.
    1. 모든 데이터를 메모리에 보유할 수 없으면 어떻게 됩니까?
    2. 데이터가 디스크에 있는 경우 어떻게 셔플링하시겠습니까? 무작위 읽기나 쓰기가 너무 많이 필요하지 않은 *효율적인* 알고리즘을 설계하는 것이 과제입니다. 힌트: [의사 난수 순열 생성기(pseudorandom permutation generators)](https://en.wikipedia.org/wiki/Pseudorandom_permutation)를 사용하면 순열 표를 명시적으로 저장할 필요 없이 재셔플을 설계할 수 있습니다 :cite:`Naor.Reingold.1999`. 
3. 반복자가 호출될 때마다 즉석에서 새로운 데이터를 생성하는 데이터 생성기를 구현하십시오. 
4. 호출될 때마다 *동일한* 데이터를 생성하는 무작위 데이터 생성기를 어떻게 설계하시겠습니까?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17975)
:end_tab:

```