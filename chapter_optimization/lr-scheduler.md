# 학습률 스케줄링 (Learning Rate Scheduling)
:label:`sec_scheduler`

지금까지 우리는 가중치 벡터를 업데이트하는 *속도*보다는 어떻게 업데이트할지에 대한 최적화 *알고리즘*에 주로 초점을 맞추었습니다. 그럼에도 불구하고, 학습률을 조정하는 것은 종종 실제 알고리즘만큼이나 중요합니다. 고려해야 할 몇 가지 측면이 있습니다.

* 가장 명백하게 학습률의 *크기*가 중요합니다. 너무 크면 최적화가 발산하고, 너무 작으면 훈련하는 데 너무 오래 걸리거나 차선책인 결과에 도달하게 됩니다. 우리는 이전에 문제의 조건 수(condition number)가 중요하다는 것을 보았습니다(자세한 내용은 예: :numref:`sec_momentum` 참조). 직관적으로 이는 가장 민감하지 않은 방향 대 가장 민감한 방향의 변화량 비율입니다.
* 둘째, 감쇠 속도도 똑같이 중요합니다. 학습률이 계속 크면 단순히 최소값 주변을 맴돌게 되어 최적점에 도달하지 못할 수 있습니다. :numref:`sec_minibatch_sgd`에서 이를 어느 정도 자세히 논의했으며 :numref:`sec_sgd`에서 성능 보장을 분석했습니다. 요컨대, 속도가 감쇠하기를 원하지만 볼록 문제에 좋은 선택인 $\mathcal{O}(t^{-\frac{1}{2}})$보다는 아마도 더 천천히 감쇠하기를 원할 것입니다.
* 똑같이 중요한 또 다른 측면은 *초기화*입니다. 이는 파라미터가 처음에 어떻게 설정되는지(자세한 내용은 :numref:`sec_numerical_stability` 검토)와 처음에 어떻게 진화하는지 모두와 관련이 있습니다. 이것은 *워밍업(warmup)*이라는 이름으로 불리는데, 즉 처음에 얼마나 빨리 솔루션을 향해 이동하기 시작하는지를 나타냅니다. 특히 초기 파라미터 세트가 무작위이기 때문에 초기에 큰 단계를 밟는 것은 유익하지 않을 수 있습니다. 초기 업데이트 방향도 꽤 무의미할 수 있습니다.
* 마지막으로, 주기적인 학습률 조정을 수행하는 여러 최적화 변형이 있습니다. 이는 현재 장의 범위를 벗어납니다. 독자들에게 :citet:`Izmailov.Podoprikhin.Garipov.ea.2018`의 세부 사항, 예를 들어 전체 파라미터 *경로*에 대해 평균을 내어 더 나은 솔루션을 얻는 방법을 검토할 것을 권장합니다.

학습률을 관리하는 데 많은 세부 사항이 필요하다는 사실을 고려하여, 대부분의 딥러닝 프레임워크에는 이를 자동으로 처리하는 도구가 있습니다. 현재 장에서는 서로 다른 스케줄이 정확도에 미치는 영향을 검토하고 *학습률 스케줄러(learning rate scheduler)*를 통해 이를 효율적으로 관리하는 방법을 보여줍니다.

## 장난감 문제 (Toy Problem)

쉽게 계산할 수 있을 만큼 저렴하면서도 몇 가지 핵심 측면을 설명하기에 충분히 비자명한 장난감 문제로 시작합니다. 이를 위해 Fashion-MNIST에 적용된 LeNet의 약간 현대화된 버전(`sigmoid` 대신 `relu` 활성화, AveragePooling 대신 MaxPooling)을 선택합니다. 또한 성능을 위해 네트워크를 하이브리드화합니다. 대부분의 코드가 표준이므로 더 자세한 설명 없이 기본 사항만 소개합니다. 필요한 경우 :numref:`chap_cnn`을 다시 참조하십시오.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 코드는 합성곱 신경망 장의 lenet 섹션에 정의된 `d2l.train_ch6`와 거의 동일합니다
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, ')
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 코드는 합성곱 신경망 장의 lenet 섹션에 정의된 `d2l.train_ch6`와 거의 동일합니다
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # PyTorch 내장 스케줄러 사용
                scheduler.step()
            else:
                # 사용자 정의 스케줄러 사용
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, ')
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 코드는 합성곱 신경망 장의 lenet 섹션에 정의된 `d2l.train_ch6`와 거의 동일합니다
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

이 알고리즘을 학습률 $0.3$과 같은 기본 설정으로 호출하고 $30$회 반복하여 훈련하면 어떤 일이 일어나는지 살펴봅시다. 훈련 정확도는 계속 증가하는 반면 테스트 정확도 측면에서의 진행은 어느 시점 이후 정체되는 방식에 주목하십시오. 두 곡선 사이의 간격은 과대적합을 나타냅니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## 스케줄러 (Schedulers)

학습률을 조정하는 한 가지 방법은 각 단계에서 명시적으로 설정하는 것입니다. 이는 `set_learning_rate` 메서드를 통해 편리하게 달성됩니다. 우리는 최적화가 어떻게 진행되는지에 대응하여 매 에폭 후(또는 매 미니배치 후에도) 동적인 방식으로 하향 조정할 수 있습니다.

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

보다 일반적으로 우리는 스케줄러를 정의하고 싶어 합니다. 업데이트 횟수와 함께 호출되면 학습률의 적절한 값을 반환합니다. 학습률을 $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$로 설정하는 간단한 스케줄러를 정의해 봅시다.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

다양한 값 범위에서 그 동작을 플롯해 봅시다.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이제 이것이 Fashion-MNIST 훈련에 어떻게 적용되는지 봅시다. 단순히 스케줄러를 훈련 알고리즘의 추가 인수로 제공합니다.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

이것은 이전보다 상당히 잘 작동했습니다. 두 가지가 눈에 띕니다: 곡선이 이전보다 다소 더 매끄러웠습니다. 둘째, 과대적합이 적었습니다. 불행히도 왜 특정 전략이 *이론적*으로 과대적합을 덜 일으키는지에 대해서는 잘 해결되지 않은 문제입니다. 더 작은 단계 크기가 제로에 더 가깝고 따라서 더 단순한 파라미터로 이어진다는 주장이 있습니다. 그러나 우리가 정말로 조기 종료를 하는 것이 아니라 단순히 학습률을 부드럽게 낮추는 것이기 때문에 이것이 현상을 완전히 설명하지는 못합니다.

## 정책 (Policies)

학습률 스케줄러의 전체 다양성을 다 다룰 수는 없지만, 아래에서 인기 있는 정책들에 대한 간략한 개요를 제공하고자 합니다. 일반적인 선택은 다항식 감쇠(polynomial decay) 및 구간별 상수(piecewise constant) 스케줄입니다. 그 외에도 코사인 학습률 스케줄이 일부 문제에서 경험적으로 잘 작동하는 것으로 밝혀졌습니다. 마지막으로, 일부 문제에서는 큰 학습률을 사용하기 전에 최적화 프로그램을 워밍업(warmup)하는 것이 유익합니다.

### 팩터 스케줄러 (Factor Scheduler)

다항식 감쇠의 한 가지 대안은 승법 감쇠(multiplicative decay)일 것입니다. 즉, $\alpha \in (0, 1)$에 대해 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$입니다. 학습률이 합리적인 하한선 너머로 감쇠하는 것을 방지하기 위해 업데이트 방정식은 종종 $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$로 수정됩니다.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

이는 MXNet에서 `lr_scheduler.FactorScheduler` 객체를 통해 내장된 스케줄러로도 달성할 수 있습니다. 워밍업 기간, 워밍업 모드(선형 또는 상수), 원하는 최대 업데이트 횟수 등과 같은 몇 가지 파라미터를 더 취합니다. 앞으로 우리는 적절하게 내장된 스케줄러를 사용하고 여기서 그 기능만 설명할 것입니다. 보시다시피 필요한 경우 자신만의 스케줄러를 구축하는 것은 상당히 간단합니다.

### 멀티 팩터 스케줄러 (Multi Factor Scheduler)

심층 네트워크를 훈련하기 위한 일반적인 전략은 학습률을 구간별 상수로 유지하고 매번 특정 양만큼 감소시키는 것입니다. 즉, 감쇠할 시간 세트(예: $s = \{5, 10, 20\}$)가 주어지면 $t \in s$일 때마다 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$로 감소시킵니다. 각 단계에서 값이 절반으로 줄어든다고 가정하면 다음과 같이 이를 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이 구간별 상수 학습률 스케줄 뒤에 숨겨진 직관은 가중치 벡터의 분포 측면에서 정상 상태(stationary point)에 도달할 때까지 최적화를 진행하게 하는 것입니다. 그런 다음(그리고 나서야) 좋은 국소 최소값에 대한 더 높은 품질의 대리물을 얻기 위해 속도를 줄입니다. 아래 예제는 이것이 어떻게 점점 더 약간 더 나은 솔루션을 생성할 수 있는지 보여줍니다.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 코사인 스케줄러 (Cosine Scheduler)

다소 당혹스러운 휴리스틱이 :citet:`Loshchilov.Hutter.2016`에 의해 제안되었습니다. 이는 우리가 처음에 학습률을 너무 급격하게 낮추고 싶지 않을 수 있으며, 더욱이 마지막에 아주 작은 학습률을 사용하여 솔루션을 "정제"하고 싶을 수 있다는 관찰에 기반합니다. 이는 $t \in [0, T]$ 범위의 학습률에 대해 다음과 같은 기능적 형태를 가진 코사인 스타일의 스케줄을 생성합니다.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


여기서 $\eta_0$는 초기 학습률이고, $\eta_T$는 시간 $T$에서의 목표 학습률입니다. 더욱이 $t > T$에 대해 우리는 값을 다시 늘리지 않고 단순히 $\eta_T$로 고정합니다. 다음 예제에서는 최대 업데이트 단계 $T = 20$을 설정했습니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

컴퓨터 비전의 맥락에서 이 스케줄은 개선된 결과로 이어질 *수* 있습니다. 하지만 그러한 개선이 보장되지는 않는다는 점에 유의하십시오(아래에서 볼 수 있듯이).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 워밍업 (Warmup)

어떤 경우에는 파라미터를 초기화하는 것만으로는 좋은 솔루션을 보장하기에 충분하지 않습니다. 이는 특히 불안정한 최적화 문제로 이어질 수 있는 일부 고급 네트워크 디자인에서 문제가 됩니다. 우리는 처음에 발산을 방지하기 위해 충분히 작은 학습률을 선택함으로써 이를 해결할 수 있습니다. 불행히도 이는 진전이 느리다는 것을 의미합니다. 반대로 처음에 큰 학습률은 발산으로 이어집니다.

이 딜레마에 대한 상당히 간단한 수정안은 학습률이 초기 최대값까지 *증가*하는 워밍업 기간을 사용하고 최적화 프로세스가 끝날 때까지 속도를 낮추는 것입니다. 단순함을 위해 일반적으로 이를 위해 선형 증가를 사용합니다. 이는 아래에 표시된 형태의 스케줄을 생성합니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

네트워크가 처음에 더 잘 수렴함에 주목하십시오(특히 처음 5 에폭 동안의 성능을 관찰하십시오).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

워밍업은 모든 스케줄러에 적용될 수 있습니다(코사인뿐만 아니라). 학습률 스케줄에 대한 더 자세한 토론과 더 많은 실험에 대해서는 :cite:`Gotmare.Keskar.Xiong.ea.2018`도 참조하십시오. 특히 그들은 워밍업 단계가 매우 깊은 네트워크에서 파라미터의 발산 정도를 제한한다는 것을 발견했습니다. 이는 처음에 진전을 이루는 데 가장 많은 시간이 걸리는 네트워크 부분에서 무작위 초기화로 인해 상당한 발산을 예상할 수 있기 때문에 직관적으로 말이 됩니다.

## 요약 (Summary)

* 훈련 중에 학습률을 낮추면 정확도가 향상되고 (가장 당혹스럽게도) 모델의 과대적합이 줄어들 수 있습니다.
* 진행이 정체될 때마다 학습률을 구간별로 감소시키는 것이 실전에서 효과적입니다. 본질적으로 이는 우리가 적절한 솔루션으로 효율적으로 수렴하도록 보장하고, 그런 다음 학습률을 줄임으로써 파라미터의 고유한 분산을 줄이게 합니다.
* 코사인 스케줄러는 일부 컴퓨터 비전 문제에서 인기가 있습니다. 그러한 스케줄러의 세부 사항은 예: [GluonCV](http://gluon-cv.mxnet.io)를 참조하십시오.
* 최적화 전의 워밍업 기간은 발산을 방지할 수 있습니다.
* 최적화는 딥러닝에서 여러 목적으로 쓰입니다. 훈련 목적 함수를 최소화하는 것 외에도, 최적화 알고리즘과 학습률 스케줄링의 서로 다른 선택은 (동일한 양의 훈련 오차에 대해) 테스트 세트에서의 일반화 및 과대적합 양을 상당히 다르게 만들 수 있습니다.

## 연습 문제 (Exercises)

1. 주어진 고정 학습률에 대한 최적화 동작을 실험해 보십시오. 이 방법으로 얻을 수 있는 최고의 모델은 무엇입니까?
2. 학습률 감소의 지수를 변경하면 수렴이 어떻게 변합니까? 실험의 편의를 위해 `PolyScheduler`를 사용하십시오.
3. 코사인 스케줄러를 대규모 컴퓨터 비전 문제, 예를 들어 ImageNet 훈련에 적용해 보십시오. 다른 스케줄러와 비교하여 성능에 어떤 영향을 미칩니까?
4. 워밍업은 얼마나 오래 지속되어야 합니까?
5. 최적화와 샘플링을 연결할 수 있습니까? 확률적 경사 랑주뱅 역학(Stochastic Gradient Langevin Dynamics)에 대한 :citet:`Welling.Teh.2011`의 결과를 사용하여 시작해 보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab: