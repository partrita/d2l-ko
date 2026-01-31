# 심층 합성곱 생성적 적대 신경망 (Deep Convolutional Generative Adversarial Networks)
:label:`sec_dcgan`

:numref:`sec_basic_gan`에서 우리는 GAN이 작동하는 기본 아이디어를 소개했습니다. 우리는 GAN이 균일 분포나 정규 분포와 같이 간단하고 샘플링하기 쉬운 분포에서 샘플을 추출하고, 이를 일부 데이터셋의 분포와 일치하는 것처럼 보이는 샘플로 변환할 수 있음을 보여주었습니다. 2D 가우시안 분포를 일치시키는 예제는 요점을 전달했지만, 특별히 흥미롭지는 않았습니다.

이 섹션에서는 GAN을 사용하여 사실적인 이미지를 생성하는 방법을 보여줄 것입니다. 우리는 :citet:`Radford.Metz.Chintala.2015`에 소개된 심층 합성곱 GAN(DCGAN)을 기반으로 모델을 만들 것입니다. 우리는 판별적 컴퓨터 비전 문제에 매우 성공적인 것으로 입증된 합성곱 아키텍처를 차용하고, GAN을 통해 이를 활용하여 사실적인 이미지를 생성하는 방법을 보여줄 것입니다.

```{.python .input}
#@tab mxnet
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import warnings
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 포켓몬 데이터셋 (The Pokemon Dataset)

우리가 사용할 데이터셋은 [pokemondb](https://pokemondb.net/sprites)에서 얻은 포켓몬 스프라이트 모음입니다. 먼저 이 데이터셋을 다운로드하고 추출하여 로드합니다.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = gluon.data.vision.datasets.ImageFolderDataset(data_dir)
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
pokemon = torchvision.datasets.ImageFolder(data_dir)
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

data_dir = d2l.download_extract('pokemon')
batch_size = 256
pokemon = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, batch_size=batch_size, image_size=(64, 64))
```

우리는 각 이미지의 크기를 $64\times 64$로 조정합니다. `ToTensor` 변환은 픽셀 값을 $[0, 1]$로 투영하는 반면, 생성기는 tanh 함수를 사용하여 $[-1, 1]$의 출력을 얻습니다. 따라서 값 범위를 일치시키기 위해 $0.5$ 평균과 $0.5$ 표준 편차로 데이터를 정규화합니다.

```{.python .input}
#@tab mxnet
batch_size = 256
transformer = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(64),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(0.5, 0.5)
])
data_iter = gluon.data.DataLoader(
    pokemon.transform_first(transformer), batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
])
pokemon.transform = transformer
data_iter = torch.utils.data.DataLoader(
    pokemon, batch_size=batch_size,
    shuffle=True, num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
def transform_func(X):
    X = X / 255.
    X = (X - 0.5) / (0.5)
    return X

# TF>=2.4의 경우 `num_parallel_calls = tf.data.AUTOTUNE` 사용
data_iter = pokemon.map(lambda x, y: (transform_func(x), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
data_iter = data_iter.cache().shuffle(buffer_size=1000).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
```

처음 20개의 이미지를 시각화해 봅시다.

```{.python .input}
#@tab mxnet
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[:20,:,:,:].transpose(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab pytorch
warnings.filterwarnings('ignore')
d2l.set_figsize((4, 4))
for X, y in data_iter:
    imgs = X[:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
    break
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize(figsize=(4, 4))
for X, y in data_iter.take(1):
    imgs = X[:20, :, :, :] / 2 + 0.5
    d2l.show_images(imgs, num_rows=4, num_cols=5)
```

## 생성기 (The Generator)

생성기는 길이 $d$ 벡터인 노이즈 변수 $\mathbf z\in\mathbb R^d$를 너비와 높이가 $64\times 64$인 RGB 이미지로 매핑해야 합니다. :numref:`sec_fcn`에서 우리는 전치 합성곱 레이어(:numref:`sec_transposed_conv` 참조)를 사용하여 입력 크기를 확대하는 완전 합성곱 신경망을 소개했습니다. 생성기의 기본 블록에는 전치 합성곱 레이어와 배치 정규화 및 ReLU 활성화가 포함됩니다.

```{.python .input}
#@tab mxnet
class G_block(nn.Block):
    def __init__(self, channels, kernel_size=4,
                 strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.Conv2DTranspose(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.Activation('relu')

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input}
#@tab pytorch
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

```{.python .input}
#@tab tensorflow
class G_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 **kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(
            out_channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
```

기본적으로 전치 합성곱 레이어는 $k_h = k_w = 4$ 커널, $s_h = s_w = 2$ 스트라이드, $p_h = p_w = 1$ 패딩을 사용합니다. 입력 모양이 $n_h^{'} \times n_w^{'} = 16 \times 16$인 경우, 생성기 블록은 입력의 너비와 높이를 두 배로 늘립니다.

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= [(n_h k_h - (n_h-1)(k_h-s_h)- 2p_h] \times [(n_w k_w - (n_w-1)(k_w-s_w)- 2p_w]\
  &= [(k_h + s_h (n_h-1)- 2p_h] \times [(k_w + s_w (n_w-1)- 2p_w]\
  &= [(4 + 2 \times (16-1)- 2 \times 1] \times [(4 + 2 \times (16-1)- 2 \times 1]\
  &= 32 \times 32 .
\end{aligned}
$$ 

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
g_blk = G_block(20)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))  # 채널 마지막 규칙
g_blk = G_block(20)
g_blk(x).shape
```

전치 합성곱 레이어를 $4\times 4$ 커널, $1\times 1$ 스트라이드, 0 패딩으로 변경하면, 입력 크기가 $1 \times 1$일 때 출력의 너비와 높이가 각각 3씩 증가합니다.

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk.initialize()
g_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 1, 1))
g_blk = G_block(20, strides=1, padding=0)
g_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 1, 1, 3))
# `padding="valid"`는 패딩 없음에 해당합니다
g_blk = G_block(20, strides=1, padding="valid")
g_blk(x).shape
```

생성기는 입력의 너비와 높이를 1에서 32로 늘리는 4개의 기본 블록으로 구성됩니다. 동시에, 먼저 잠재 변수를 $64\times 8$ 채널로 투영한 다음 매번 채널을 반으로 줄입니다. 마지막으로, 전치 합성곱 레이어를 사용하여 출력을 생성합니다. 원하는 $64\times 64$ 모양과 일치하도록 너비와 높이를 두 배로 늘리고 채널 크기를 $3$으로 줄입니다. tanh 활성화 함수를 적용하여 출력 값을 $(-1, 1)$ 범위로 투영합니다.

```{.python .input}
#@tab mxnet
n_G = 64
net_G = nn.Sequential()
net_G.add(G_block(n_G*8, strides=1, padding=0),  # 출력: (64 * 8, 4, 4)
          G_block(n_G*4),  # 출력: (64 * 4, 8, 8)
          G_block(n_G*2),  # 출력: (64 * 2, 16, 16)
          G_block(n_G),    # 출력: (64, 32, 32)
          nn.Conv2DTranspose(
              3, kernel_size=4, strides=2, padding=1, use_bias=False,
              activation='tanh'))  # 출력: (3, 64, 64)
```

```{.python .input}
#@tab pytorch
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G*8,
            strides=1, padding=0),                  # 출력: (64 * 8, 4, 4)
    G_block(in_channels=n_G*8, out_channels=n_G*4), # 출력: (64 * 4, 8, 8)
    G_block(in_channels=n_G*4, out_channels=n_G*2), # 출력: (64 * 2, 16, 16)
    G_block(in_channels=n_G*2, out_channels=n_G),   # 출력: (64, 32, 32)
    nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh())  # 출력: (3, 64, 64)
```

```{.python .input}
#@tab tensorflow
n_G = 64
net_G = tf.keras.Sequential([
    # 출력: (4, 4, 64 * 8)
    G_block(out_channels=n_G*8, strides=1, padding="valid"),
    G_block(out_channels=n_G*4), # 출력: (8, 8, 64 * 4)
    G_block(out_channels=n_G*2), # 출력: (16, 16, 64 * 2)
    G_block(out_channels=n_G), # 출력: (32, 32, 64)
    # 출력: (64, 64, 3)
    tf.keras.layers.Conv2DTranspose(
        3, kernel_size=4, strides=2, padding="same", use_bias=False,
        activation="tanh")
])
```

100차원 잠재 변수를 생성하여 생성기의 출력 모양을 확인합니다.

```{.python .input}
#@tab mxnet
x = np.zeros((1, 100, 1, 1))
net_G.initialize()
net_G(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 100, 1, 1))
net_G(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 1, 1, 100))
net_G(x).shape
```

## 판별기 (Discriminator)

판별기는 Leaky ReLU를 활성화 함수로 사용한다는 점을 제외하면 일반적인 합성곱 신경망입니다. $\alpha \in[0, 1]$이 주어지면 정의는 다음과 같습니다.

$$\textrm{leaky ReLU}(x) = \begin{cases}x & \textrm{if}\ x > 0\\ \alpha x &\textrm{otherwise}\end{cases}.$$ 

보시다시피 $\alpha=0$이면 일반 ReLU이고, $\alpha=1$이면 항등 함수입니다. $\alpha \in (0, 1)$인 경우, Leaky ReLU는 음수 입력에 대해 0이 아닌 출력을 제공하는 비선형 함수입니다. 이는 뉴런이 항상 음수 값을 출력하여 ReLU의 기울기가 0이 되어 진행할 수 없는 "죽어가는 ReLU(dying ReLU)" 문제를 해결하는 것을 목표로 합니다.

```{.python .input}
#@tab mxnet,pytorch
alphas = [0, .2, .4, .6, .8, 1]
x = d2l.arange(-2, 1, 0.1)
Y = [d2l.numpy(nn.LeakyReLU(alpha)(x)) for alpha in alphas]
d2l.plot(d2l.numpy(x), Y, 'x', 'y', alphas)
```

```{.python .input}
#@tab tensorflow
alphas = [0, .2, .4, .6, .8, 1]
x = tf.range(-2, 1, 0.1)
Y = [tf.keras.layers.LeakyReLU(alpha)(x).numpy() for alpha in alphas]
d2l.plot(x.numpy(), Y, 'x', 'y', alphas)
```

판별기의 기본 블록은 합성곱 레이어와 배치 정규화 레이어, 그리고 Leaky ReLU 활성화가 뒤따르는 구조입니다. 합성곱 레이어의 하이퍼파라미터는 생성기 블록의 전치 합성곱 레이어와 유사합니다.

```{.python .input}
#@tab mxnet
class D_block(nn.Block):
    def __init__(self, channels, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2D(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input}
#@tab pytorch
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

```{.python .input}
#@tab tensorflow
class D_block(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same",
                 alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, kernel_size,
                                             strides, padding, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha)

    def call(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
```

기본 설정이 있는 기본 블록은 :numref:`sec_padding`에서 시연한 것처럼 입력의 너비와 높이를 반으로 줄입니다. 예를 들어, 입력 모양 $n_h = n_w = 16$, 커널 모양 $k_h = k_w = 4$, 스트라이드 모양 $s_h = s_w = 2$, 패딩 모양 $p_h = p_w = 1$이 주어지면 출력 모양은 다음과 같습니다:

$$
\begin{aligned}
n_h^{'} \times n_w^{'} &= \lfloor(n_h-k_h+2p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+2p_w+s_w)/s_w\rfloor\\
  &= \lfloor(16-4+2\times 1+2)/2\rfloor \times \lfloor(16-4+2\times 1+2)/2\rfloor\\
  &= 8 \times 8 .
\end{aligned}
$$ 

```{.python .input}
#@tab mxnet
x = np.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk.initialize()
d_blk(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((2, 3, 16, 16))
d_blk = D_block(20)
d_blk(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((2, 16, 16, 3))
d_blk = D_block(20)
d_blk(x).shape
```

판별기는 생성기의 거울입니다.

```{.python .input}
#@tab mxnet
n_D = 64
net_D = nn.Sequential()
net_D.add(D_block(n_D),   # 출력: (64, 32, 32)
          D_block(n_D*2),  # 출력: (64 * 2, 16, 16)
          D_block(n_D*4),  # 출력: (64 * 4, 8, 8)
          D_block(n_D*8),  # 출력: (64 * 8, 4, 4)
          nn.Conv2D(1, kernel_size=4, use_bias=False))  # 출력: (1, 1, 1)
```

```{.python .input}
#@tab pytorch
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),  # 출력: (64, 32, 32)
    D_block(in_channels=n_D, out_channels=n_D*2),  # 출력: (64 * 2, 16, 16)
    D_block(in_channels=n_D*2, out_channels=n_D*4),  # 출력: (64 * 4, 8, 8)
    D_block(in_channels=n_D*4, out_channels=n_D*8),  # 출력: (64 * 8, 4, 4)
    nn.Conv2d(in_channels=n_D*8, out_channels=1,
              kernel_size=4, bias=False))  # 출력: (1, 1, 1)
```

```{.python .input}
#@tab tensorflow
n_D = 64
net_D = tf.keras.Sequential([
    D_block(n_D), # 출력: (32, 32, 64)
    D_block(out_channels=n_D*2), # 출력: (16, 16, 64 * 2)
    D_block(out_channels=n_D*4), # 출력: (8, 8, 64 * 4)
    D_block(out_channels=n_D*8), # 출력: (4, 4, 64 * 64)
    # 출력: (1, 1, 1)
    tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
])
```

마지막 레이어로 출력 채널이 $1$인 합성곱 레이어를 사용하여 단일 예측 값을 얻습니다.

```{.python .input}
#@tab mxnet
x = np.zeros((1, 3, 64, 64))
net_D.initialize()
net_D(x).shape
```

```{.python .input}
#@tab pytorch
x = torch.zeros((1, 3, 64, 64))
net_D(x).shape
```

```{.python .input}
#@tab tensorflow
x = tf.zeros((1, 64, 64, 3))
net_D(x).shape
```

## 훈련 (Training)

:numref:`sec_basic_gan`의 기본 GAN과 비교하여, 생성기와 판별기가 서로 비슷하기 때문에 두 모델에 대해 동일한 학습률을 사용합니다. 또한 Adam(:numref:`sec_adam`)의 $\beta_1$을 $0.9$에서 $0.5$로 변경합니다. 이는 생성기와 판별기가 서로 싸우기 때문에 급격하게 변하는 기울기를 처리하기 위해 과거 기울기의 지수 가중 이동 평균인 운동량의 부드러움을 줄입니다. 또한 무작위로 생성된 노이즈 `Z`는 4D 텐서이며 계산 속도를 높이기 위해 GPU를 사용하고 있습니다.

```{.python .input}
#@tab mxnet
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=device)
    trainer_hp = {'learning_rate': lr, 'beta1': 0.5}
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', trainer_hp)
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # 한 에포크 훈련
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.as_in_ctx(device), Z.as_in_ctx(device),
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # 생성된 예제 표시
        Z = np.random.normal(0, 1, size=(21, latent_dim, 1, 1), ctx=device)
        # 합성 데이터를 N(0, 1)로 정규화
        fake_x = net_G(Z).transpose(0, 2, 3, 1) / 2 + 0.5
        imgs = np.concatenate(
            [np.concatenate([fake_x[i * 7 + j] for j in range(7)], axis=1)
             for i in range(len(fake_x)//7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.asnumpy())
        # 손실 표시
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ' 
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_hp = {'lr': lr, 'betas': [0.5,0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        # 한 에포크 훈련
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # 생성된 예제 표시
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # 합성 데이터를 N(0, 1)로 정규화
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # 손실 표시
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ' 
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          device=d2l.try_gpu()):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))

    optimizer_hp = {"lr": lr, "beta_1": 0.5, "beta_2": 0.999}
    optimizer_D = tf.keras.optimizers.Adam(**optimizer_hp)
    optimizer_G = tf.keras.optimizers.Adam(**optimizer_hp)

    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)

    for epoch in range(1, num_epochs + 1):
        # 한 에포크 훈련
        timer = d2l.Timer()
        metric = d2l.Accumulator(3) # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(mean=0, stddev=1,
                                 shape=(batch_size, 1, 1, latent_dim))
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       d2l.update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)

        # 생성된 예제 표시
        Z = tf.random.normal(mean=0, stddev=1, shape=(21, 1, 1, latent_dim))
        # 합성 데이터를 N(0, 1)로 정규화
        fake_x = net_G(Z) / 2 + 0.5
        imgs = tf.concat([tf.concat([fake_x[i * 7 + j] for j in range(7)],
                                    axis=1)
                          for i in range(len(fake_x) // 7)], axis=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # 손실 표시
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ' 
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device._device_name)}')
```

시연을 위해 적은 수의 에포크로 모델을 훈련합니다. 더 나은 성능을 위해 `num_epochs` 변수를 더 큰 숫자로 설정할 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
latent_dim, lr, num_epochs = 100, 0.005, 20
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

```{.python .input}
#@tab tensorflow
latent_dim, lr, num_epochs = 100, 0.0005, 40
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
```

## 요약 (Summary)

* DCGAN 아키텍처에는 판별기를 위한 4개의 합성곱 레이어와 생성기를 위한 4개의 "분수 스트라이드(fractionally-strided)" 합성곱 레이어가 있습니다.
* 판별기는 배치 정규화(입력 레이어 제외)와 Leaky ReLU 활성화가 있는 4레이어 스트라이드 합성곱입니다.
* Leaky ReLU는 음수 입력에 대해 0이 아닌 출력을 제공하는 비선형 함수입니다. 이는 "죽어가는 ReLU" 문제를 해결하고 아키텍처를 통해 기울기가 더 쉽게 흐르도록 돕습니다.


## 연습 문제 (Exercises)

1. Leaky ReLU 대신 표준 ReLU 활성화를 사용하면 어떻게 됩니까?
2. Fashion-MNIST에 DCGAN을 적용하고 어떤 범주가 잘 작동하고 어떤 범주가 그렇지 않은지 확인하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/409)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1083)
:end_tab:

```