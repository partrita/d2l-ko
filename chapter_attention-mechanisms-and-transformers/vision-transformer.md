```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# 비전 트랜스포머 (Transformers for Vision)
:label:`sec_vision-transformer`

트랜스포머 아키텍처는 처음에 기계 번역에 중점을 둔 시퀀스-투-시퀀스 학습을 위해 제안되었습니다. 
그 후 트랜스포머는 다양한 자연어 처리 작업에서 선택되는 모델로 부상했습니다 :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`. 
그러나 컴퓨터 비전 분야에서는 지배적인 아키텍처가 CNN으로 남아 있었습니다 (:numref:`chap_modern_cnn`). 
자연스럽게 연구자들은 트랜스포머 모델을 이미지 데이터에 적용하여 더 잘할 수 있을지 궁금해하기 시작했습니다. 
이 질문은 컴퓨터 비전 커뮤니티에서 엄청난 관심을 불러일으켰습니다. 
최근 :citet:`ramachandran2019stand`는 합성곱을 셀프 어텐션으로 대체하는 방안을 제안했습니다. 
그러나 어텐션에서 특수한 패턴을 사용하기 때문에 하드웨어 가속기에서 모델을 확장하기 어렵습니다. 
그 후 :citet:`cordonnier2020relationship`은 이론적으로 셀프 어텐션이 합성곱과 유사하게 동작하도록 학습할 수 있음을 증명했습니다. 
경험적으로 이미지에서 $2 \times 2$ 패치를 입력으로 가져왔지만, 패치 크기가 작아 모델을 저해상도 이미지 데이터에만 적용할 수 있었습니다.

패치 크기에 대한 특정 제약 없이, 
*비전 트랜스포머(Vision Transformers, ViTs)*는 이미지에서 패치를 추출하고 이를 트랜스포머 인코더에 공급하여 전역 표현을 얻으며, 이는 최종적으로 분류를 위해 변환됩니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. 
주목할 점은 트랜스포머가 CNN보다 더 나은 확장성을 보여준다는 것입니다: 더 큰 데이터셋에서 더 큰 모델을 훈련할 때 비전 트랜스포머는 ResNet보다 성능이 월등히 뛰어납니다. 
자연어 처리에서의 네트워크 아키텍처 설계 환경과 유사하게, 트랜스포머는 컴퓨터 비전에서도 게임 체인저가 되었습니다.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 모델

:numref:`fig_vit`는 비전 트랜스포머의 모델 아키텍처를 묘사합니다. 
이 아키텍처는 이미지를 패치화하는 줄기(stem), 다층 트랜스포머 인코더에 기반한 몸체(body), 그리고 전역 표현을 출력 레이블로 변환하는 머리(head)로 구성됩니다.

![비전 트랜스포머 아키텍처. 이 예제에서 이미지는 9개의 패치로 분할됩니다. 특수 "&lt;cls&gt;" 토큰과 9개의 평탄화된 이미지 패치는 패치 임베딩과 $\mathit{n}$개의 트랜스포머 인코더 블록을 통해 각각 10개의 표현으로 변환됩니다. "&lt;cls&gt;" 표현은 출력 레이블로 추가 변환됩니다.](../img/vit.svg)
:label:`fig_vit`

높이 $h$, 너비 $w$, 채널 $c$인 입력 이미지를 고려해 보십시오. 
패치 높이와 너비를 모두 $p$로 지정하면, 이미지는 $m = hw/p^2$개의 패치 시퀀스로 분할되며, 여기서 각 패치는 길이 $cp^2$의 벡터로 평탄화됩니다. 
이런 식으로 이미지 패치는 트랜스포머 인코더에서 텍스트 시퀀스의 토큰과 유사하게 처리될 수 있습니다. 
특수 "&lt;cls&gt;" (클래스) 토큰과 $m$개의 평탄화된 이미지 패치는 $m+1$개 벡터의 시퀀스로 선형 투영되고, 학습 가능한 위치 임베딩과 합산됩니다. 
다층 트랜스포머 인코더는 $m+1$개의 입력 벡터를 동일한 수의 동일한 길이 출력 벡터 표현으로 변환합니다. 
정규화 위치만 다를 뿐 :numref:`fig_transformer`의 원래 트랜스포머 인코더와 똑같이 작동합니다. 
"&lt;cls&gt;" 토큰은 셀프 어텐션을 통해 모든 이미지 패치에 주의를 기울이므로(:numref:`fig_cnn-rnn-self-attention` 참조), 트랜스포머 인코더 출력에서의 그 표현은 출력 레이블로 추가 변환됩니다.

## 패치 임베딩 (Patch Embedding)

비전 트랜스포머를 구현하기 위해 :numref:`fig_vit`의 패치 임베딩부터 시작하겠습니다. 
이미지를 패치로 분할하고 평탄화된 패치를 선형적으로 투영하는 것은 단일 합성곱 연산으로 단순화할 수 있습니다. 
여기서 커널 크기와 스트라이드 크기는 모두 패치 크기로 설정됩니다.

```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # 출력 모양: (배치 크기, 패치 수, 채널 수)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # 출력 모양: (배치 크기, 패치 수, 채널 수)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

다음 예제에서는 높이와 너비가 `img_size`인 이미지를 입력으로 받아 패치 임베딩이 `(img_size//patch_size)**2`개의 패치를 출력하며, 이들은 길이 `num_hiddens`의 벡터로 선형 투영됩니다.

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## 비전 트랜스포머 인코더 (Vision Transformer Encoder)
:label:`subsec_vit-encoder`

비전 트랜스포머 인코더의 MLP는 원래 트랜스포머 인코더의 포지션와이즈 FFN과 약간 다릅니다(:numref:`subsec_positionwise-ffn` 참조). 
첫째, 여기서 활성화 함수는 가우시안 오차 선형 유닛(GELU)을 사용합니다. 이는 ReLU의 더 부드러운 버전으로 간주될 수 있습니다 :cite:`Hendrycks.Gimpel.2016`. 
둘째, 정규화를 위해 MLP의 각 완전 연결 레이어 출력에 드롭아웃이 적용됩니다.

```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

비전 트랜스포머 인코더 블록 구현은 :numref:`fig_vit`의 사전 정규화(pre-normalization) 설계를 따릅니다. 
여기서 정규화는 멀티 헤드 어텐션 또는 MLP *바로 직전*에 적용됩니다. 
잔차 연결 *직후*에 정규화가 배치되는 사후 정규화(:numref:`fig_transformer`의 "add & norm")와 달리, 
사전 정규화는 트랜스포머의 더 효과적이거나 효율적인 훈련으로 이어집니다 :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

:numref:`subsec_transformer-encoder`와 마찬가지로, 비전 트랜스포머 인코더 블록은 입력 모양을 변경하지 않습니다.

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## 종합하기 (Putting It All Together)

아래 비전 트랜스포머의 순방향 패스는 간단합니다. 
먼저 입력 이미지가 `PatchEmbedding` 인스턴스에 공급되고, 
그 출력은 "&lt;cls&gt;" 토큰 임베딩과 연결됩니다. 
드롭아웃 전에 학습 가능한 위치 임베딩과 합산됩니다. 
그런 다음 출력은 `ViTBlock` 클래스의 `num_blks` 인스턴스를 쌓는 트랜스포머 인코더에 공급됩니다. 
마지막으로 "&lt;cls&gt;" 토큰의 표현은 네트워크 헤드에 의해 투영됩니다.

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """비전 트랜스포머."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # cls 토큰 추가
        # 위치 임베딩은 학습 가능합니다
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """비전 트랜스포머."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # cls 토큰 추가
        # 위치 임베딩은 학습 가능합니다
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## 훈련 (Training)

Fashion-MNIST 데이터셋에서 비전 트랜스포머를 훈련하는 것은 :numref:`chap_modern_cnn`에서 CNN을 훈련하는 것과 같습니다.

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## 요약 및 토론 (Summary and Discussion)

Fashion-MNIST와 같은 작은 데이터셋의 경우 구현된 비전 트랜스포머가 :numref:`sec_resnet`의 ResNet보다 성능이 좋지 않음을 알 수 있습니다. 
ImageNet 데이터셋(120만 개 이미지)에서도 비슷한 관찰을 할 수 있습니다. 
이는 트랜스포머가 평행 이동 불변성 및 지역성과 같은 합성곱의 유용한 원칙이 *부족*하기 때문입니다(:numref:`sec_why-conv`). 
그러나 더 큰 데이터셋(예: 3억 개 이미지)에서 더 큰 모델을 훈련할 때 상황이 바뀌어, 비전 트랜스포머가 이미지 분류에서 ResNet을 크게 능가하여 확장성에서 트랜스포머의 본질적인 우수성을 입증했습니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`. 
비전 트랜스포머의 도입은 이미지 데이터 모델링을 위한 네트워크 설계 환경을 변화시켰습니다. 
이들은 곧 DeiT의 데이터 효율적인 훈련 전략을 통해 ImageNet 데이터셋에서도 효과적인 것으로 나타났습니다 :cite:`touvron2021training`. 
그러나 셀프 어텐션의 이차 복잡도(:numref:`sec_self-attention-and-positional-encoding`)는 트랜스포머 아키텍처를 고해상도 이미지에 덜 적합하게 만듭니다. 
컴퓨터 비전의 범용 백본 네트워크를 향하여, Swin Transformer는 이미지 크기에 대한 이차 계산 복잡도 문제를 해결하고(:numref:`subsec_cnn-rnn-self-attention`) 합성곱과 유사한 사전 지식(priors)을 복원하여, 트랜스포머의 적용 가능성을 이미지 분류를 넘어 다양한 컴퓨터 비전 작업으로 확장하고 최첨단 결과를 달성했습니다 :cite:`liu2021swin`.

## 연습 문제 (Exercises)

1. `img_size` 값은 훈련 시간에 어떤 영향을 줍니까?
2. "&lt;cls&gt;" 토큰 표현을 출력에 투영하는 대신, 평균화된 패치 표현을 투영하면 어떨까요? 이 변경 사항을 구현하고 정확도에 어떤 영향을 미치는지 확인하십시오.
3. 비전 트랜스포머의 정확도를 높이기 위해 하이퍼파라미터를 수정할 수 있습니까?

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18032)
:end_tab:

```