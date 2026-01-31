```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 밑바닥부터 시작하는 선형 회귀 구현 (Linear Regression Implementation from Scratch)
:label:`sec_linear_scratch`

이제 선형 회귀의 완전히 작동하는 구현을 살펴볼 준비가 되었습니다. 
이 섹션에서는 (**(i) 모델; (ii) 손실 함수; (iii) 미니배치 확률적 경사 하강법 최적화기; (iv) 이 모든 조각들을 하나로 묶는 훈련 함수를 포함하여 전체 메서드를 밑바닥부터 구현할 것입니다.**) 
마지막으로 :numref:`sec_synthetic-regression-data`에서 만든 합성 데이터 생성기를 실행하고 결과 데이터셋에 모델을 적용할 것입니다. 
현대 딥러닝 프레임워크는 이 작업의 거의 모든 부분을 자동화할 수 있지만, 밑바닥부터 구현하는 것이 여러분이 정말로 무엇을 하고 있는지 알 수 있는 유일한 방법입니다. 
게다가 자체 레이어나 손실 함수를 정의하여 모델을 커스터마이징할 때, 내부적으로 어떻게 작동하는지 이해하는 것이 유용할 것입니다. 
이 섹션에서는 텐서와 자동 미분만을 사용할 것입니다. 
나중에는 아래 구조를 유지하면서 딥러닝 프레임워크의 편리한 기능을 활용하는 더 간결한 구현을 소개할 것입니다.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## 모델 정의하기 (Defining the Model)

미니배치 SGD로 [**모델의 파라미터를 최적화하기 전에**], (**우선 파라미터가 있어야 합니다.**) 
다음에서는 평균 0, 표준 편차 0.01의 정규 분포에서 난수를 추출하여 가중치를 초기화합니다. 
마법의 숫자 0.01은 실제 상황에서 종종 잘 작동하지만, `sigma` 인수를 통해 다른 값을 지정할 수 있습니다. 
또한 편향은 0으로 설정합니다. 
객체 지향 설계를 위해, (:numref:`subsec_oo-design-models`에서 소개된) `d2l.Module`의 서브클래스의 `__init__` 메서드에 코드를 추가합니다.

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """밑바닥부터 구현된 선형 회귀 모델입니다."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """밑바닥부터 구현된 선형 회귀 모델입니다."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

다음으로 [**입력과 파라미터를 출력과 연결하는 모델을 정의**]해야 합니다. 
선형 모델에 대해 :eqref:`eq_linreg-y-vec`와 동일한 표기법을 사용하여, 단순히 입력 특성 $\mathbf{X}$와 모델 가중치 $\mathbf{w}$의 행렬-벡터 곱을 취하고 각 예제에 오프셋 $b$를 더합니다. 
곱 $\mathbf{Xw}$는 벡터이고 $b$는 스칼라입니다. 
브로드캐스팅 메커니즘(:numref:`subsec_broadcasting` 참조)으로 인해 벡터와 스칼라를 더하면 벡터의 각 성분에 스칼라가 더해집니다. 
결과인 `forward` 메서드는 (:numref:`oo-design-utilities`에서 소개된) `add_to_class`를 통해 `LinearRegressionScratch` 클래스에 등록됩니다.

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## 손실 함수 정의하기 (Defining the Loss Function)

[**모델을 업데이트하려면 손실 함수의 기울기를 취해야 하므로**], (**먼저 손실 함수를 정의**)해야 합니다. 
여기서는 :eqref:`eq_mse`의 제곱 손실 함수를 사용합니다. 
구현에서 실제 값 `y`를 예측 값의 모양 `y_hat`으로 변환해야 합니다. 
다음 메서드에서 반환되는 결과도 `y_hat`과 동일한 모양을 갖게 됩니다. 
또한 미니배치의 모든 예제에 대한 평균 손실 값을 반환합니다.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # 튜플에서 X를 언팩함
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## 최적화 알고리즘 정의하기 (Defining the Optimization Algorithm)

:numref:`sec_linear_regression`에서 논의했듯이 선형 회귀는 닫힌 형식의 해를 갖습니다. 
하지만 우리의 목표는 더 일반적인 신경망을 훈련하는 방법을 설명하는 것이며, 이를 위해서는 미니배치 SGD를 사용하는 방법을 가르쳐야 합니다. 
따라서 이번 기회에 SGD의 첫 번째 실행 예제를 소개하겠습니다. 
각 단계에서 데이터셋에서 무작위로 추출한 미니배치를 사용하여 파라미터에 대한 손실의 기울기를 추정합니다. 
다음으로 손실을 줄일 수 있는 방향으로 파라미터를 업데이트합니다.

다음 코드는 파라미터 세트와 학습률 `lr`이 주어졌을 때 업데이트를 적용합니다. 
손실이 미니배치에 대한 평균으로 계산되므로 배치 크기에 맞춰 학습률을 조정할 필요가 없습니다. 
나중 장에서 분산 대규모 학습에서 발생하는 매우 큰 미니배치에 대해 학습률을 어떻게 조정해야 하는지 조사할 것입니다. 
지금은 이 의존성을 무시할 수 있습니다.

:begin_tab:`mxnet`
내장 SGD 최적화기와 유사한 API를 갖도록 (:numref:`oo-design-utilities`에서 소개된) `d2l.HyperParameters`의 서브클래스인 `SGD` 클래스를 정의합니다. 
`step` 메서드에서 파라미터를 업데이트합니다. 무시할 수 있는 `batch_size` 인수를 받습니다.
:end_tab:

:begin_tab:`pytorch`
내장 SGD 최적화기와 유사한 API를 갖도록 (:numref:`oo-design-utilities`에서 소개된) `d2l.HyperParameters`의 서브클래스인 `SGD` 클래스를 정의합니다. 
`step` 메서드에서 파라미터를 업데이트합니다. `zero_grad` 메서드는 모든 기울기를 0으로 설정하며, 역전파 단계 전에 실행해야 합니다.
:end_tab:

:begin_tab:`tensorflow`
내장 SGD 최적화기와 유사한 API를 갖도록 (:numref:`oo-design-utilities`에서 소개된) `d2l.HyperParameters`의 서브클래스인 `SGD` 클래스를 정의합니다. 
`apply_gradients` 메서드에서 파라미터를 업데이트합니다. 파라미터와 기울기 쌍의 리스트를 받습니다.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """미니배치 확률적 경사 하강법."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """미니배치 확률적 경사 하강법."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """미니배치 확률적 경사 하강법."""
    # Optax의 핵심 변환은 `init`과 `update` 두 메서드로 정의되는 
    # `GradientTransformation`입니다.
    # `init`은 상태를 초기화하고 `update`는 기울기를 변환합니다.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # 사용되지 않는 파라미터 삭제
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # flax의 `train_state` 객체를 업데이트하기 위해 `state.apply_gradients` 메서드가 호출되면,
        # 내부적으로 `optax.apply_updates` 메서드를 호출하여
        # 아래 정의된 업데이트 식에 파라미터를 추가합니다.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

다음으로 `SGD` 클래스의 인스턴스를 반환하는 `configure_optimizers` 메서드를 정의합니다.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## 훈련 (Training)

이제 모든 부품(파라미터, 손실 함수, 모델, 최적화기)이 준비되었으므로, [**메인 훈련 루프를 구현**]할 준비가 되었습니다. 
이 책에서 다루는 모든 딥러닝 모델에 대해 유사한 훈련 루프를 사용할 것이므로 이 코드를 완전히 이해하는 것이 중요합니다. 
각 *에폭(epoch)*마다 전체 훈련 데이터셋을 순회하며 모든 예제를 한 번씩 거칩니다(예제 수가 배치 크기로 나누어떨어진다고 가정). 
각 *반복(iteration)*마다 훈련 예제의 미니배치를 가져와 모델의 `training_step` 메서드를 통해 손실을 계산합니다. 
그런 다음 각 파라미터에 대한 기울기를 계산합니다. 
마지막으로 최적화 알고리즘을 호출하여 모델 파라미터를 업데이트합니다. 
요약하자면 다음 루프를 실행할 것입니다:

* 파라미터 $(\mathbf{w}, b)$ 초기화
* 완료될 때까지 반복
    * 기울기 계산 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 파라미터 업데이트 $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
:numref:``sec_synthetic-regression-data``에서 생성한 합성 회귀 데이터셋은 검증 데이터셋을 제공하지 않는다는 점을 상기하십시오. 
하지만 대부분의 경우 모델 품질을 측정하기 위해 검증 데이터셋을 원할 것입니다. 
여기서는 모델 성능을 측정하기 위해 각 에폭에서 한 번씩 검증 데이터 로더를 전달합니다. 
우리의 객체 지향 설계에 따라, `prepare_batch`와 `fit_epoch` 메서드는 (:numref:`oo-design-training`에서 소개된) `d2l.Trainer` 클래스에 등록됩니다.

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # 나중에 논의될 예정
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # 가변 상태(Mutable states)는 나중에 사용됩니다(예: 배치 정규화)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # 드롭아웃 레이어가 없는 모델에서는 무시할 수 있음
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # 드롭아웃 레이어가 없는 모델에서는 무시할 수 있음
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

모델을 훈련할 준비가 거의 다 되었지만, 먼저 훈련 데이터가 필요합니다. 
여기서는 `SyntheticRegressionData` 클래스를 사용하고 몇 가지 실제 파라미터를 전달합니다. 
그런 다음 학습률 `lr=0.03`으로 모델을 훈련하고 `max_epochs=3`으로 설정합니다. 
일반적으로 에폭 수와 학습률은 모두 하이퍼파라미터라는 점에 유의하십시오. 
일반적으로 하이퍼파라미터를 설정하는 것은 까다로우며, 우리는 대개 훈련을 위한 한 세트, 하이퍼파라미터 선택을 위한 두 번째 세트, 그리고 최종 평가를 위해 예약된 세 번째 세트라는 3-way split을 사용하기를 원할 것입니다. 
지금은 이러한 세부 사항을 생략하지만 나중에 다시 살펴볼 것입니다.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

데이터셋을 우리가 직접 합성했기 때문에 실제 파라미터가 무엇인지 정확히 알고 있습니다. 
따라서 훈련 루프를 통해 [**학습한 파라미터와 실제 파라미터를 비교하여 훈련 성공 여부를 평가**]할 수 있습니다. 
실제로 그들은 서로 매우 가깝다는 것이 밝혀졌습니다.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'w 추정 오차: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'b 추정 오차: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'w 추정 오차: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'b 추정 오차: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"w 추정 오차: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"b 추정 오차: {data.b - params['b']}")
```

실제 파라미터를 정확하게 복구할 수 있는 능력을 당연하게 여겨서는 안 됩니다. 
일반적으로 심층 모델의 경우 파라미터에 대한 고유한 해가 존재하지 않으며, 선형 모델의 경우에도 어떤 특성도 다른 특성에 선형적으로 종속되지 않을 때만 파라미터를 정확하게 복구할 수 있습니다. 
그러나 머신러닝에서 우리는 종종 실제 기본 파라미터를 복구하는 것보다 매우 정확한 예측으로 이어지는 파라미터에 더 관심을 갖습니다 :cite:`Vapnik.1992`. 
다행히도 어려운 최적화 문제에서도 확률적 경사 하강법은 종종 놀랍도록 좋은 솔루션을 찾을 수 있는데, 이는 부분적으로 심층 네트워크의 경우 매우 정확한 예측으로 이어지는 파라미터 설정이 많이 존재하기 때문입니다.


## 요약 (Summary)

이 섹션에서 우리는 완벽하게 작동하는 신경망 모델과 훈련 루프를 구현함으로써 딥러닝 시스템 설계에 있어 중요한 단계를 밟았습니다. 
이 과정에서 우리는 데이터 로더, 모델, 손실 함수, 최적화 절차, 시각화 및 모니터링 도구를 구축했습니다. 
모델을 훈련하기 위한 모든 관련 구성 요소를 포함하는 Python 객체를 구성하여 이를 수행했습니다. 
아직 프로 수준의 구현은 아니지만 완벽하게 작동하며 이와 같은 코드는 이미 작은 문제를 빠르게 해결하는 데 도움이 될 수 있습니다. 
가까운 섹션에서 (상용구 코드를 피하면서) *더 간결하게*, 그리고 (GPU를 최대한 활용하여) *더 효율적으로* 이를 수행하는 방법을 볼 것입니다.



## 연습 문제 (Exercises)

1. 가중치를 0으로 초기화하면 어떻게 될까요? 알고리즘이 여전히 작동할까요? 파라미터를 0.01이 아니라 분산 1000으로 초기화하면 어떻게 될까요?
2. 전압과 전류를 관련시키는 저항 모델을 고안하려는 [게오르크 시몬 옴(Georg Simon Ohm)](https://en.wikipedia.org/wiki/Georg_Ohm)이라고 가정해 봅시다. 자동 미분을 사용하여 모델의 파라미터를 학습할 수 있습니까?
3. [플랑크 법칙(Planck's Law)](https://en.wikipedia.org/wiki/Planck%27s_law)을 사용하여 분광 에너지 밀도를 통해 물체의 온도를 결정할 수 있습니까? 참고로 흑체에서 방출되는 복사의 분광 밀도 $B$는 $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$입니다. 여기서 $\lambda$는 파장, $T$는 온도, $c$는 빛의 속도, $h$는 플랑크 상수, $k$는 볼츠만 상수입니다. 여러분은 다양한 파장 $\lambda$에 대한 에너지를 측정하고 이제 분광 밀도 곡선을 플랑크 법칙에 맞춰야 합니다.
4. 손실의 2계 도함수를 계산하려고 할 때 마주칠 수 있는 문제는 무엇입니까? 어떻게 고칠 수 있을까요?
5. `loss` 함수에서 `reshape` 메서드가 필요한 이유는 무엇입니까?
6. 손실 함수 값이 얼마나 빨리 떨어지는지 알아보기 위해 다양한 학습률을 사용하여 실험해 보십시오. 훈련 에폭 수를 늘려 오차를 줄일 수 있습니까?
7. 예제 수가 배치 크기로 나누어떨어지지 않으면 에폭 끝에서 `data_iter`에 어떤 일이 발생합니까?
8. 절댓값 손실 `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`과 같은 다른 손실 함수를 구현해 보십시오.
    1. 일반 데이터에 대해 어떤 일이 일어나는지 확인하십시오.
    2. $\mathbf{y}$의 일부 항목(예: $y_5 = 10000$)을 적극적으로 교란시켰을 때 동작에 차이가 있는지 확인하십시오.
    3. 제곱 손실과 절댓값 손실의 장점을 결합한 저렴한 솔루션을 생각할 수 있습니까? 힌트: 정말 큰 기울기 값을 어떻게 피할 수 있을까요?
9. 왜 데이터셋을 재셔플해야 할까요? 그렇지 않으면 악의적으로 구성된 데이터셋이 최적화 알고리즘을 망가뜨리는 사례를 설계할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17976)
:end_tab:

```