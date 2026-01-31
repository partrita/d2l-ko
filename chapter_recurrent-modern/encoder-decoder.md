```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 인코더-디코더 아키텍처 (The Encoder--Decoder Architecture)
:label:`sec_encoder-decoder`

기계 번역(:numref:`sec_machine_translation`)과 같은 일반적인 시퀀스-투-시퀀스 문제에서 입력과 출력은 정렬되지 않은 가변 길이의 시퀀스입니다. 
이러한 종류의 데이터를 처리하기 위한 표준 접근 방식은 두 개의 주요 구성 요소로 이루어진 *인코더-디코더(encoder--decoder)* 아키텍처(:numref:`fig_encoder_decoder`)를 설계하는 것입니다: 
가변 길이 시퀀스를 입력으로 받는 *인코더(encoder)*, 
그리고 인코딩된 입력과 타겟 시퀀스의 왼쪽 문맥을 입력으로 받아 타겟 시퀀스의 후속 토큰을 예측하는 조건부 언어 모델 역할을 하는 *디코더(decoder)*입니다.


![인코더-디코더 아키텍처.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

영어를 프랑스어로 기계 번역하는 예를 들어 보겠습니다. 
영어 입력 시퀀스 "They", "are", "watching", "."가 주어지면, 
이 인코더-디코더 아키텍처는 먼저 가변 길이 입력을 상태로 인코딩한 다음, 
상태를 디코딩하여 번역된 시퀀스를 출력으로 토큰별로 생성합니다: "Ils", "regardent", ".". 
인코더-디코더 아키텍처는 후속 섹션에서 다룰 다양한 시퀀스-투-시퀀스 모델의 기초를 형성하므로, 
이 섹션에서는 이 아키텍처를 나중에 구현될 인터페이스로 변환할 것입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## (**인코더 (Encoder)**)

인코더 인터페이스에서는 인코더가 가변 길이 시퀀스를 입력 `X`로 받는다는 것만 지정합니다. 
구현은 이 기본 `Encoder` 클래스를 상속하는 모든 모델에서 제공될 것입니다.

```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """인코더-디코더 아키텍처를 위한 기본 인코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """인코더-디코더 아키텍처를 위한 기본 인코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """인코더-디코더 아키텍처를 위한 기본 인코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """인코더-디코더 아키텍처를 위한 기본 인코더 인터페이스."""
    def setup(self):
        raise NotImplementedError

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## [**디코더 (Decoder)**]

다음 디코더 인터페이스에서는 인코더 출력(`enc_all_outputs`)을 인코딩된 상태로 변환하기 위해 추가적인 `init_state` 메서드를 추가합니다. 
이 단계에는 :numref:`sec_machine_translation`에서 설명한 입력의 유효 길이와 같은 추가 입력이 필요할 수 있음에 유의하십시오. 
가변 길이 시퀀스를 토큰별로 생성하기 위해, 디코더는 매번 입력(예: 이전 타임 스텝에서 생성된 토큰)과 인코딩된 상태를 현재 타임 스텝의 출력 토큰으로 매핑할 수 있습니다.

```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """인코더-디코더 아키텍처를 위한 기본 디코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """인코더-디코더 아키텍처를 위한 기본 디코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """인코더-디코더 아키텍처를 위한 기본 디코더 인터페이스."""
    def __init__(self):
        super().__init__()

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """인코더-디코더 아키텍처를 위한 기본 디코더 인터페이스."""
    def setup(self):
        raise NotImplementedError

    # 나중에 추가 인수가 있을 수 있습니다(예: 패딩을 제외한 길이)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## [**인코더와 디코더 결합하기 (Putting the Encoder and Decoder Together)**]

순전파에서 인코더의 출력은 인코딩된 상태를 생성하는 데 사용되며, 이 상태는 디코더의 입력 중 하나로 추가 사용됩니다.

```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """인코더-디코더 아키텍처를 위한 기본 클래스."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # 디코더 출력만 반환
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """인코더-디코더 아키텍처를 위한 기본 클래스."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # 디코더 출력만 반환
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """인코더-디코더 아키텍처를 위한 기본 클래스."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # 디코더 출력만 반환
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

다음 섹션에서는 이 인코더-디코더 아키텍처를 기반으로 시퀀스-투-시퀀스 모델을 설계하기 위해 RNN을 적용하는 방법을 볼 것입니다.


## 요약 (Summary)

인코더-디코더 아키텍처는 모두 가변 길이 시퀀스로 구성된 입력과 출력을 처리할 수 있으므로 기계 번역과 같은 시퀀스-투-시퀀스 문제에 적합합니다. 
인코더는 가변 길이 시퀀스를 입력으로 받아 고정된 모양의 상태로 변환합니다. 
디코더는 고정된 모양의 인코딩된 상태를 가변 길이 시퀀스로 매핑합니다.


## 연습 문제 (Exercises)

1. 신경망을 사용하여 인코더-디코더 아키텍처를 구현한다고 가정해 봅시다. 인코더와 디코더가 반드시 동일한 유형의 신경망이어야 합니까?
2. 기계 번역 외에 인코더-디코더 아키텍처를 적용할 수 있는 다른 응용 사례를 생각할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18021)
:end_tab: