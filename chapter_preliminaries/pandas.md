```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 데이터 전처리 (Data Preprocessing)
:label:`sec_pandas`

지금까지 우리는 기성 텐서에 도착한 합성 데이터로 작업했습니다.
그러나 실제 환경에서 딥러닝을 적용하려면
임의의 형식으로 저장된 지저분한 데이터를 추출하고
필요에 맞게 전처리해야 합니다.
다행히도 *pandas* [라이브러리](https://pandas.pydata.org/)는
무거운 작업의 대부분을 수행할 수 있습니다.
이 섹션은 적절한 *pandas* [튜토리얼](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)을
대체할 수는 없지만, 가장 일반적인 루틴에 대한 집중 코스를 제공할 것입니다.

## 데이터셋 읽기 (Reading the Dataset)

쉼표로 구분된 값(CSV) 파일은 표 형식(스프레드시트 같은) 데이터를 저장하는 데 어디에나 있습니다.
여기서 각 줄은 하나의 레코드에 해당하며
여러(쉼표로 구분된) 필드로 구성됩니다. 예:
"알버트 아인슈타인,1879년 3월 14일,울름,연방 공과 대학,중력 물리학 분야".
`pandas`로 CSV 파일을 로드하는 방법을 보여주기 위해,
우리는 `../data/house_tiny.csv` (**아래에 CSV 파일을 생성합니다**).
이 파일은 주택 데이터셋을 나타내며,
각 행은 별개의 집에 해당하고
열은 방 수(`NumRooms`), 지붕 유형(`RoofType`), 가격(`Price`)에 해당합니다.

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

이제 `pandas`를 가져와서 `read_csv`로 데이터셋을 로드해 보겠습니다.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 데이터 준비 (Data Preparation)

지도 학습에서는 입력 값 세트가 주어졌을 때
지정된 *타겟* 값을 예측하도록 모델을 훈련합니다.
데이터셋을 처리하는 첫 번째 단계는
입력 값과 타겟 값에 해당하는 열을 분리하는 것입니다.
이름이나 정수 위치 기반 인덱싱(`iloc`)을 통해 열을 선택할 수 있습니다.

`pandas`가 `NA` 값을 가진 모든 CSV 항목을
특별한 `NaN` (*숫자 아님*) 값으로 대체한 것을 눈치챘을 것입니다.
이것은 항목이 비어 있을 때도 발생할 수 있습니다. 예: "3,,,270000".
이것들을 *결측값(missing values)*이라고 하며
데이터 과학의 "빈대"와 같아서, 경력 내내 직면하게 될 지속적인 위협입니다.
맥락에 따라 결측값은 *대체(imputation)* 또는 *삭제(deletion)*를 통해 처리될 수 있습니다.
대체는 결측값을 값의 추정치로 바꾸는 반면,
삭제는 단순히 결측값이 포함된 행이나 열을 버립니다.

다음은 몇 가지 일반적인 대체 휴리스틱입니다.
[**범주형 입력 필드의 경우, `NaN`을 하나의 범주로 취급할 수 있습니다.**]
`RoofType` 열은 `Slate`와 `NaN` 값을 취하므로,
`pandas`는 이 열을 두 개의 열 `RoofType_Slate`와 `RoofType_nan`으로 변환할 수 있습니다.
지붕 유형이 `Slate`인 행은 `RoofType_Slate` 및 `RoofType_nan`의 값을 각각 1과 0으로 설정합니다.
`RoofType` 값이 누락된 행의 경우에는 그 반대가 성립합니다.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

누락된 수치 값의 경우, 일반적인 휴리스틱 중 하나는
[**`NaN` 항목을 해당 열의 평균값으로 대체하는 것입니다**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## 텐서 형식으로 변환 (Conversion to the Tensor Format)

이제 [**`inputs`와 `targets`의 모든 항목이 수치이므로,
텐서로 로드할 수 있습니다**] (:numref:`sec_ndarray` 상기).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## 토론

이제 데이터 열을 분할하고, 누락된 변수를 대체하고,
`pandas` 데이터를 텐서로 로드하는 방법을 알게 되었습니다.
:numref:`sec_kaggle_house`에서 더 많은 데이터 처리 기술을 배우게 될 것입니다.
이 집중 코스는 상황을 단순하게 유지했지만, 데이터 처리는 까다로울 수 있습니다.
예를 들어, 단일 CSV 파일에 도착하는 대신,
데이터셋이 관계형 데이터베이스에서 추출된 여러 파일에 분산되어 있을 수 있습니다.
예를 들어 전자 상거래 애플리케이션에서 고객 주소는 한 테이블에 있고
구매 데이터는 다른 테이블에 있을 수 있습니다.
게다가 실무자들은 범주형 및 수치형을 넘어
텍스트 문자열, 이미지, 오디오 데이터, 포인트 클라우드 등 수많은 데이터 유형에 직면합니다.
종종 데이터 처리가 머신러닝 파이프라인의 가장 큰 병목 현상이 되는 것을 방지하기 위해
고급 도구와 효율적인 알고리즘이 필요합니다.
이러한 문제는 컴퓨터 비전과 자연어 처리에 도달했을 때 발생할 것입니다.
마지막으로 데이터 품질에 주의를 기울여야 합니다.
실제 데이터셋은 종종 이상값, 센서의 잘못된 측정, 기록 오류로 인해 골칫거리가 되며,
데이터를 모델에 공급하기 전에 해결해야 합니다.
[seaborn](https://seaborn.pydata.org/),
[Bokeh](https://docs.bokeh.org/), 또는 [matplotlib](https://matplotlib.org/)과 같은
데이터 시각화 도구는 데이터를 수동으로 검사하고
해결해야 할 문제 유형에 대한 직관을 개발하는 데 도움이 될 수 있습니다.


## 연습 문제

1. [UCI 머신러닝 저장소](https://archive.ics.uci.edu/ml/datasets)의 Abalone과 같은 데이터셋을 로드하고 속성을 검사해 보십시오. 결측값이 있는 비율은 얼마입니까? 변수의 몇 퍼센트가 수치형, 범주형 또는 텍스트입니까?
2. 열 번호 대신 이름으로 데이터 열을 인덱싱하고 선택해 보십시오. [인덱싱](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)에 대한 pandas 문서에 이를 수행하는 방법에 대한 자세한 내용이 나와 있습니다.
3. 이 방식으로 얼마나 큰 데이터셋을 로드할 수 있다고 생각하십니까? 한계는 무엇일까요? 힌트: 데이터 읽기 시간, 표현, 처리 및 메모리 사용량을 고려하십시오. 노트북에서 이것을 시도해 보십시오. 서버에서 시도하면 어떻게 됩니까?
4. 매우 많은 수의 범주가 있는 데이터를 어떻게 처리하시겠습니까? 범주 레이블이 모두 고유하다면 어떻게 합니까? 후자를 포함해야 합니까?
5. pandas의 대안으로 무엇을 생각할 수 있습니까? [파일에서 NumPy 텐서 로드하기](https://numpy.org/doc/stable/reference/generated/numpy.load.html)는 어떻습니까? Python 이미징 라이브러리인 [Pillow](https://python-pillow.org/)를 확인해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17967)
:end_tab: