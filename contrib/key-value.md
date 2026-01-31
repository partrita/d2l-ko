# 분산 키-값 저장소 (Distributed Key-Value Store)
:label:`sec_key_value`

KVStore는 데이터 공유를 위한 장소입니다. 이를 서로 다른 장치(GPU 및 컴퓨터) 간에 공유되는 단일 객체라고 생각하십시오. 여기서 각 장치는 데이터를 밀어 넣고(push) 데이터를 가져올(pull) 수 있습니다.

## 초기화 (Initialization)
간단한 예제를 고려해 봅시다: (int, NDArray) 쌍을 저장소에 초기화한 다음 값을 가져오는 것입니다.

```{.python .input  n=1}
#@tab mxnet
from mxnet import np, npx, kv
npx.set_np()
```

```{.python .input  n=2}
#@tab mxnet
# 저장소 생성
store = kv.create('local')
# 초기 값 생성
shape = (2, 3)
x = np.ones(shape)
# 키 3에 대해 초기화
store.init(3, x)
# 값을 가져올 변수 생성
y = np.zeros(shape)
# 키 3의 값을 y로 가져오기
store.pull(3, out=y)
print(y)
```

## 밀어 넣기 및 집계 (Push and Aggregate)
어떤 데이터 셰어링 시스템이든, 한 번에 여러 장치가 데이터를 밀어 넣을 때 무엇을 해야 할지가 중요합니다. KVStore는 여러 장치에서 보낸 데이터를 집계한 후 결과를 저장합니다.

```{.python .input  n=3}
#@tab mxnet
# 여러 GPU를 사용하는 상황 시뮬레이션
z = np.ones(shape) * 2
# 동일한 키 3에 대해 새로운 값 밀어 넣기
store.push(3, z)
# 값을 다시 가져오기
store.pull(3, out=y)
print(y)
```

## 업데이트 함수 (Update Function)
기본적으로 KVStore는 `push`된 값들을 더합니다. 하지만 우리는 `set_optimizer` 또는 `set_updater`를 사용하여 어떻게 데이터를 결합할지 지정할 수 있습니다.

```{.python .input  n=4}
#@tab mxnet
# 업데이트 함수 정의: KVStore는 (key, push_value, stored_value)를 인자로 전달합니다.
def update(key, input, stored):
    print(f"update key {key}")
    stored += input * 2

store.set_updater(update)
store.push(3, np.ones(shape))
store.pull(3, out=y)
print(y)
```

## 요약 (Summary)

* KVStore는 여러 장치 간에 파라미터를 공유하고 업데이트하기 위한 강력한 추상화입니다.
* `push`와 `pull` 연산을 통해 데이터를 동기화할 수 있습니다.
* 사용자 정의 업데이트 함수를 통해 복잡한 최적화 알고리즘을 분산 환경에서 구현할 수 있습니다.

## 연습 문제 (Exercises)

1. 여러 개의 키를 동시에 업데이트하는 방법을 찾아보십시오.
2. `local` 대신 `device` 유형의 KVStore를 사용하면 어떤 차이가 있는지 실험해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/401)
:end_tab: