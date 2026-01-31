# 스타일 가이드

## 일반 (In General)

* 정확하고, 명확하고, 매력적이고, 실용적이고, 일관성 있게 작성하세요.

## 텍스트 (Text)

* 챕터(Chapter) 및 섹션(Section)
    * 각 챕터의 시작 부분에 개요 제공
    * 각 섹션의 구조를 일관되게 유지
        * 요약 (Summary)
        * 연습 문제 (Exercises)
* 인용 부호 (Quotes)
    * 큰따옴표(Double quotes) 사용
* 기호 설명 (Symbol Descriptions)
    * timestep t (t timestep 아님)
* 도구, 클래스 및 함수 (Tools, Class, and Functions)
    * Gluon, MXNet, NumPy, spaCy, NDArray, Symbol, Block, HybridBlock, ResNet-18, Fashion-MNIST, matplotlib
        * 이들을 억양 부호(``) 없이 단어로 취급
    * Sequential 클래스/인스턴스, HybridSequential 클래스/인스턴스
        * 억양 부호(``) 없이
    * `backward` 함수
        * `backward()` 함수 아님
    * "for-loop" 사용, "for loop" 아님
* 용어 (Terminologies)
    * 일관되게 사용
        * 함수 (메서드 아님)
        * 인스턴스 (객체 아님)
        * 가중치(weight), 편향(bias), 레이블(label)
        * 모델 훈련(model training), 모델 예측(model prediction) (모델 추론(model inference))
        * 훈련/테스트/검증 데이터셋 (training/testing/validation dataset)
        * "데이터 인스턴스"나 "데이터 포인트"보다 "데이터/훈련/테스트 예제(example)" 선호
    * 구별:
        * 하이퍼파라미터 vs 파라미터
        * 미니배치 확률적 경사 하강법 vs 확률적 경사 하강법
* 설명하거나 코드 또는 수학의 일부일 때 숫자(numerals)를 사용합니다.
* 허용되는 약어
    * AI, MLP, CNN, RNN, GRU, LSTM, 모델 이름 (예: ELMo, GPT, BERT)
    * 대부분의 경우 명확성을 위해 전체 이름을 씁니다 (예: NLP -> 자연어 처리(natural language processing))

## 수학 (Math)

* [수학 표기법](chapter_notation/index.md)에서 일관성 유지
* 필요한 경우 방정식 내에 문장 부호 배치
    * 예: 쉼표 및 마침표
* 할당 기호
    * \leftarrow
* 수학의 일부일 때만 수학적 숫자를 사용: "$x$는 $1$ 또는 $-1$이다", "$12$와 $18$의 최대공약수는 $6$이다".
* "천 단위 구분 기호"는 사용하지 않습니다 (출판사마다 스타일이 다르기 때문). 예: 소스 마크다운 파일에서 10,000은 10000으로 작성해야 합니다.

## 그림 (Figure)

* 소프트웨어
    * OmniGraffle을 사용하여 그림 제작.
      * 100%로 pdf(무한 캔버스) 내보내기, 그런 다음 pdf2svg를 사용하여 svg로 변환
        * `ls *.pdf | while read f; do pdf2svg $f ${f%.pdf}.svg; done`
      * Omnigraffle에서 직접 svg를 내보내지 마십시오 (글꼴 크기가 약간 변경될 수 있음)
* 스타일
    * 크기:
        * 가로: <= 400 픽셀 (페이지 너비에 의해 제한됨)
        * 세로: <= 200 픽셀 (예외가 있을 수 있음)
    * 두께:
        * StickArrow
        * 1pt
        * 화살표 머리 크기: 50%
    * 글꼴:
        * Arial (텍스트용), STIXGeneral (수학용), 9pt (아래첨자/위첨자: 6pt)
        * 아래첨자나 위첨자의 숫자나 괄호는 이탤릭체로 쓰지 마십시오
    * 색상:
        * 배경으로 파란색 (텍스트는 검은색)
            * (피할 것) 매우 어두움: 3FA3FD
            * 어두움: 66BFFF
            * 밝음: B2D9FF
            * (피할 것) 매우 밝음: CFF4FF
* 저작권에 주의하십시오


## 코드 (Code)

* 각 줄은 <=78자여야 합니다 (페이지 너비에 의해 제한됨). [캠브리지 스타일](https://github.com/d2l-ai/d2l-en/pull/2187)의 경우, 각 줄은 <=79자여야 합니다.
* Python
    * PEP8
        * 예: (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* 공간을 절약하기 위해 여러 할당을 한 줄에 배치
  * 예: `num_epochs, lr = 5, 0.1`
* 변수 이름의 일관성 유지
    * `num_epochs`
        * 에폭 수 (number of epochs)
    * `num_hiddens`
        * 은닉 유닛 수 (number of hidden units)
    * `num_inputs`
        * 입력 수 (number of inputs)
    * `num_outputs`
        * 출력 수 (number of outputs)
    * `net`
        * 모델 (model)
    * `lr`
        * 학습률 (learning rate)
    * `acc`
        * 정확도 (accuracy)
    * 반복 중
        * 특성: `X`
        * 레이블: `y`, `y_hat` 또는 `Y`, `Y_hat`
        * `for X, y in data_iter`
    * 데이터 세트:
        * 특성: `features` 또는 `images`
        * 레이블: `labels`
        * DataLoader 인스턴스: `train_iter`, `test_iter`, `data_iter`
* 주석
    * 주석 끝에 마침표 없음.
    * 명확성을 위해 변수 이름을 억양 부호로 감쌈, 예: # shape of `X`
* imports
    * 알파벳 순서로 import
* 변수 출력
    * 코드 블록 끝에서 `print('x:', x, 'y:', y)` 대신 가능하면 `x, y` 사용
* 문자열
    * 작은따옴표 사용
    * f-string 사용. 긴 f-string을 여러 줄로 나누려면 줄마다 하나의 f-string을 사용하십시오.
* 기타 항목
    * `nd.f(x)` → `x.nd`
    * `.1` → `1.0`
    * 1. → `1.0`


## 참고 문헌 (References)

* 그림, 표 및 방정식에 대한 참조를 추가하는 방법은 [d2lbook](https://book.d2l.ai/user/markdown.html#cross-references)을 참조하십시오.


## URL

`style = cambridge`로 설정하면 URL이 QR 코드로 변환되므로 특수 문자를 [URL 인코딩](https://www.urlencoder.io/learn/)으로 대체해야 합니다. 예를 들어:

`Stanford's [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)`
->
`Stanford's [large movie review dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/)


## 인용 (Citations)

1. `pip install git+https://github.com/d2l-ai/d2l-book` 실행
1. bibtool을 사용하여 bibtex 항목에 대해 일관된 키 생성. `brew install bib-tool`로 설치
1. 루트 디렉터리의 `d2l.bib`에 bibtex 항목 추가. 원래 항목이 다음과 같다고 가정해 봅시다.
```
@article{wood2011sequence,
  title={The sequence memoizer},
  author={Wood, Frank and Gasthaus, Jan and Archambeau, C\'edric and James, Lancelot and Teh, Yee Whye},
  journal={Communications of the ACM},
  volume={54},
  number={2},
  pages={91--98},
  year={2011},
  publisher={ACM}
}
```
4. `bibtool -s -f "%3n(author).%d(year)" d2l.bib -o d2l.bib` 실행. 이제 추가된 항목이 일관된 키를 갖게 됩니다. 그리고 부수적인 효과로, 파일의 다른 모든 논문에 비해 알파벳순으로 정렬되어 나타납니다:
```
@Article{	  Wood.Gasthaus.Archambeau.ea.2011,
  title		= {The sequence memoizer},
  author	= {Wood, Frank and Gasthaus, Jan and Archambeau, C\'edric
		  and James, Lancelot and Teh, Yee Whye},
  journal	= {Communications of the ACM},
  volume	= {54},
  number	= {2},
  pages		= {91--98},
  year		= {2011},
  publisher	= {ACM}
}
```
5. 텍스트에서 추가된 논문을 인용하려면 다음을 사용하십시오:
```
:cite:`Wood.Gasthaus.Archambeau.ea.2011`
```


## 하나의 프레임워크에서 코드 편집 및 테스트

1. xx.md에서 MXNet 코드를 편집하고 테스트하고 싶다면, `d2lbook activate default xx.md`를 실행합니다. 그러면 xx.md에서 다른 프레임워크의 코드가 비활성화됩니다.
2. 주피터 노트북을 사용하여 xx.md를 열고 코드를 편집한 다음 "Kernel -> Restart & Run All"을 사용하여 코드를 테스트합니다.
3. `d2lbook activate all xx.md`를 실행하여 모든 프레임워크의 코드를 다시 활성화합니다. 그런 다음 git push 합니다.

마찬가지로, `d2lbook activate pytorch/tensorflow xx.md`는 xx.md에서 PyTorch/TensorFlow 코드만 활성화합니다.