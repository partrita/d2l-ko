#  MovieLens 데이터셋 (The MovieLens Dataset)

추천 연구에 사용할 수 있는 데이터셋은 많이 있습니다. 그중에서도 [MovieLens](https://movielens.org/) 데이터셋은 아마도 가장 인기 있는 데이터셋 중 하나일 것입니다. MovieLens는 비상업적 웹 기반 영화 추천 시스템입니다. 1997년에 만들어졌으며 연구 목적으로 영화 평점 데이터를 수집하기 위해 미네소타 대학교의 연구실인 GroupLens에서 운영합니다. MovieLens 데이터는 개인화된 추천 및 사회 심리학을 포함한 여러 연구 연구에 매우 중요했습니다.


## 데이터 얻기 (Getting the Data) 


MovieLens 데이터셋은 [GroupLens](https://grouplens.org/datasets/movielens/) 웹사이트에서 호스팅됩니다. 여러 버전을 사용할 수 있습니다. 우리는 MovieLens 100K 데이터셋을 사용할 것입니다 :cite:`Herlocker.Konstan.Borchers.ea.1999`. 이 데이터셋은 1682개의 영화에 대해 943명의 사용자로부터 받은 1에서 5까지의 별점으로 구성된 100,000개의 평점으로 이루어져 있습니다. 각 사용자가 적어도 20개의 영화를 평가하도록 정리되었습니다. 사용자와 항목에 대한 나이, 성별, 장르와 같은 간단한 인구 통계 정보도 사용할 수 있습니다. [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)을 다운로드하고 csv 형식의 100,000개 평점을 모두 포함하는 `u.data` 파일을 추출할 수 있습니다. 폴더에는 다른 많은 파일이 있으며, 각 파일에 대한 자세한 설명은 데이터셋의 [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) 파일에서 찾을 수 있습니다.

우선, 이 섹션의 실험을 실행하는 데 필요한 패키지를 가져옵니다.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

그런 다음 MovieLens 100k 데이터셋을 다운로드하고 상호 작용을 `DataFrame`으로 로드합니다.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 데이터셋 통계 (Statistics of the Dataset)

데이터를 로드하고 처음 5개 레코드를 수동으로 검사해 보겠습니다. 데이터 구조를 배우고 제대로 로드되었는지 확인하는 효과적인 방법입니다.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

각 줄은 "user id" 1-943, "item id" 1-1682, "rating" 1-5 및 "timestamp"를 포함한 4개의 열로 구성되어 있음을 알 수 있습니다. $n 	imes m$ 크기의 상호 작용 행렬을 구성할 수 있습니다. 여기서 $n$과 $m$은 각각 사용자와 항목의 수입니다. 이 데이터셋은 기존 평점만 기록하므로 평점 행렬이라고도 할 수 있으며, 이 행렬의 값이 정확한 평점을 나타내는 경우 상호 작용 행렬과 평점 행렬을 혼용해서 사용할 것입니다. 사용자가 대다수의 영화를 평가하지 않았기 때문에 평점 행렬의 대부분의 값은 알 수 없습니다. 또한 이 데이터셋의 희소성을 보여줍니다. 희소성은 `1 - 0이 아닌 항목 수 / (사용자 수 * 항목 수)`로 정의됩니다. 분명히 상호 작용 행렬은 매우 희소합니다(즉, 희소성 = 93.695%). 실제 데이터셋은 더 큰 범위의 희소성을 겪을 수 있으며 이는 추천 시스템을 구축하는 데 있어 오랜 과제였습니다. 실행 가능한 솔루션은 희소성을 완화하기 위해 사용자/항목 특성과 같은 추가 부가 정보를 사용하는 것입니다.

그런 다음 서로 다른 평점 수의 분포를 그립니다. 예상대로 대부분의 평점이 3-4에 집중된 정규 분포로 보입니다.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 데이터셋 분할 (Splitting the dataset)

데이터셋을 훈련 세트와 테스트 세트로 분할합니다. 다음 함수는 `random` 및 `seq-aware`를 포함한 두 가지 분할 모드를 제공합니다. `random` 모드에서 함수는 타임스탬프를 고려하지 않고 100k 상호 작용을 무작위로 분할하고 기본적으로 데이터의 90%를 훈련 샘플로 사용하고 나머지 10%를 테스트 샘플로 사용합니다. `seq-aware` 모드에서는 사용자가 가장 최근에 평가한 항목을 테스트용으로 남겨두고 사용자의 과거 상호 작용을 훈련 세트로 사용합니다. 사용자 과거 상호 작용은 타임스탬프를 기준으로 오래된 것부터 최신 순으로 정렬됩니다. 이 모드는 시퀀스 인식 추천 섹션에서 사용됩니다.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """데이터셋을 무작위 모드 또는 시퀀스 인식 모드로 분할합니다."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

테스트 세트 외에도 검증 세트를 사용하는 것이 실제로는 좋은 관행이라는 점에 유의하십시오. 그러나 간결함을 위해 생략합니다. 이 경우 테스트 세트는 보류된 검증 세트로 간주될 수 있습니다.

## 데이터 로드 (Loading the data)

데이터셋 분할 후 편의를 위해 훈련 세트와 테스트 세트를 리스트와 딕셔너리/행렬로 변환합니다. 다음 함수는 데이터프레임을 한 줄씩 읽고 사용자/항목의 인덱스를 0부터 열거합니다. 그런 다음 함수는 사용자, 항목, 평점 리스트와 상호 작용을 기록하는 딕셔너리/행렬을 반환합니다. 피드백 유형을 `explicit` 또는 `implicit`으로 지정할 수 있습니다.

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

그 후 위의 단계들을 합치면 다음 섹션에서 사용될 것입니다. 결과는 `Dataset` 및 `DataLoader`로 래핑됩니다. 훈련 데이터에 대한 `DataLoader`의 `last_batch`는 `rollover` 모드(나머지 샘플은 다음 에포크로 롤오버됨)로 설정되고 순서는 섞입니다.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 요약 (Summary)

* MovieLens 데이터셋은 추천 연구에 널리 사용됩니다. 공개적으로 사용 가능하며 무료로 사용할 수 있습니다.
* 이후 섹션에서 추가로 사용할 수 있도록 MovieLens 100k 데이터셋을 다운로드하고 전처리하는 함수를 정의합니다.


## 연습 문제 (Exercises)

* 찾을 수 있는 다른 유사한 추천 데이터셋은 무엇입니까?
* MovieLens에 대한 자세한 내용은 [https://movielens.org/](https://movielens.org/) 사이트를 살펴보십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab: