```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 기본 분류 모델 (The Base Classification Model)
:label:`sec_classification`

회귀의 경우 밑바닥부터의 구현과 프레임워크 기능을 사용한 간결한 구현이 상당히 유사했다는 점을 눈치채셨을 것입니다. 분류도 마찬가지입니다. 이 책의 많은 모델이 분류를 다루기 때문에, 이 설정을 구체적으로 지원하는 기능을 추가하는 것이 가치가 있습니다. 이 섹션에서는 향후 코드를 단순화하기 위해 분류 모델을 위한 기본 클래스를 제공합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## `Classifier` 클래스

:begin_tab:`pytorch, mxnet, tensorflow`
아래에 `Classifier` 클래스를 정의합니다. `validation_step`에서는 검증 배치에 대한 손실 값과 분류 정확도를 모두 보고합니다. 우리는 매 `num_val_batches` 배치마다 업데이트를 그립니다. 이는 전체 검증 데이터에 대해 평균화된 손실과 정확도를 생성하는 이점이 있습니다. 마지막 배치에 더 적은 예제가 포함되어 있다면 이 평균 수치들이 정확히 맞지는 않겠지만, 코드를 단순하게 유지하기 위해 이 사소한 차이는 무시합니다.
:end_tab:


:begin_tab:`jax`
아래에 `Classifier` 클래스를 정의합니다. `validation_step`에서는 검증 배치에 대한 손실 값과 분류 정확도를 모두 보고합니다. 우리는 매 `num_val_batches` 배치마다 업데이트를 그립니다. 이는 전체 검증 데이터에 대해 평균화된 손실과 정확도를 생성하는 이점이 있습니다. 마지막 배치에 더 적은 예제가 포함되어 있다면 이 평균 수치들이 정확히 맞지는 않겠지만, 코드를 단순하게 유지하기 위해 이 사소한 차이는 무시합니다.

또한 JAX를 위해 `training_step` 메서드를 재정의합니다. 나중에 `Classifier`를 상속받을 모든 모델은 보조 데이터(auxiliary data)를 반환하는 손실을 가질 것이기 때문입니다. 이 보조 데이터는 (:numref:`sec_batch_norm`에서 설명할) 배치 정규화(batch normalization)가 있는 모델에 사용될 수 있으며, 그 외의 모든 경우에는 손실이 보조 데이터를 나타내는 플레이스홀더(빈 딕셔너리)를 반환하도록 할 것입니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """분류 모델의 기본 클래스입니다."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """분류 모델의 기본 클래스입니다."""
    def training_step(self, params, batch, state):
        # 배치 정규화 레이어가 있는 모델은 손실이 보조 데이터를 반환해야 하므로 
        # 여기서 value는 튜플입니다.
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # 두 번째로 반환된 값은 버립니다. 이는 손실이 보조 데이터도 반환하는 
        # 배치 정규화 레이어가 있는 모델을 훈련할 때 사용됩니다.
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

기본적으로 우리는 선형 회귀 맥락에서 했던 것처럼 미니배치에서 작동하는 확률적 경사 하강법 최적화기를 사용합니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## 정확도 (Accuracy)

예측된 확률 분포 `y_hat`이 주어졌을 때, 하드 예측(hard prediction)을 출력해야 할 때마다 우리는 일반적으로 예측 확률이 가장 높은 클래스를 선택합니다. 실제로 많은 응용 분야에서 선택을 해야 합니다. 예를 들어 Gmail은 이메일을 "기본", "소셜", "업데이트", "포럼" 또는 "스팸"으로 분류해야 합니다. 내부적으로는 확률을 추정할 수 있겠지만, 결국에는 클래스 중 하나를 선택해야 합니다.

예측이 레이블 클래스 `y`와 일치하면 올바른 것입니다. 분류 정확도(classification accuracy)는 모든 예측 중 올바른 예측의 비율입니다. 정확도를 직접 최적화하는 것은 어려울 수 있지만(미분 불가능하므로), 우리가 가장 중요하게 생각하는 성능 측정치인 경우가 많습니다. 이는 벤치마크에서 종종 *핵심적인* 수치입니다. 따라서 분류기를 훈련할 때 거의 항상 이를 보고할 것입니다.

정확도는 다음과 같이 계산됩니다. 먼저 `y_hat`이 행렬인 경우, 두 번째 차원에 각 클래스에 대한 예측 점수가 저장되어 있다고 가정합니다. 각 행에서 가장 큰 항목의 인덱스로 예측된 클래스를 얻기 위해 `argmax`를 사용합니다. 그런 다음 [**예측된 클래스를 실제 값 `y`와 요소별로 비교합니다.**] 등호 연산자 `==`는 데이터 유형에 민감하므로, `y_hat`의 데이터 유형을 `y`와 일치하도록 변환합니다. 결과는 0(거짓)과 1(참) 항목을 포함하는 텐서입니다. 합계를 구하면 올바른 예측의 수가 나옵니다.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """올바른 예측의 수를 계산합니다."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """올바른 예측의 수를 계산합니다."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # 배치 정규화 전용
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## 요약 (Summary)

분류는 별도의 편의 함수가 필요할 정도로 충분히 일반적인 문제입니다. 분류에서 핵심적으로 중요한 것은 분류기의 *정확도*입니다. 비록 우리가 주로 정확도에 관심을 갖지만, 통계적 및 계산적 이유로 다양한 다른 목표를 최적화하도록 분류기를 훈련시킨다는 점에 유의하십시오. 하지만 훈련 중에 어떤 손실 함수가 최소화되었는지에 관계없이, 분류기의 정확도를 경험적으로 평가하기 위한 편의 메서드를 갖는 것은 유용합니다. 


## 연습 문제 (Exercises)

1. $L_\textrm{v}$를 검증 손실로 표시하고, $L_\textrm{v}^\textrm{q}$를 이 섹션의 손실 함수 평균으로 계산된 대략적인 추정치라고 합시다. 마지막으로 $l_\textrm{v}^\textrm{b}$를 마지막 미니배치의 손실이라고 합시다. $L_\textrm{v}$를 $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, 그리고 샘플 및 미니배치 크기 측면에서 표현하십시오.
2. 대략적인 추정치 $L_\textrm{v}^\textrm{q}$가 불편 추정량(unbiased estimator)임을 보이십시오. 즉, $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$임을 보이십시오. 그럼에도 불구하고 왜 $L_\textrm{v}$를 대신 사용하고 싶어 할까요?
3. 다중 클래스 분류 손실이 주어졌을 때, $y$를 보았을 때 $y'$를 추정하는 페널티를 $l(y,y')$로 표시하고 확률 $p(y \mid x)$가 주어졌을 때, $y'$의 최적 선택을 위한 규칙을 공식화하십시오. 힌트: $l$과 $p(y \mid x)$를 사용하여 기대 손실을 표현하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17981)
:end_tab:

```