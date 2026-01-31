# 자연어 처리: 사전 훈련 (Natural Language Processing: Pretraining)
:label:`chap_nlp_pretrain`


인간은 소통해야 합니다.
이러한 인간 조건의 기본적인 욕구에서, 매일 방대한 양의 텍스트가 생성되고 있습니다.
소셜 미디어, 채팅 앱, 이메일, 제품 리뷰, 뉴스 기사, 연구 논문 및 책의 풍부한 텍스트를 고려할 때, 컴퓨터가 이를 이해하여 도움을 제공하거나 인간 언어를 기반으로 결정을 내릴 수 있도록 하는 것이 중요합니다.

*자연어 처리(Natural language processing)*는 자연어를 사용하여 컴퓨터와 인간 간의 상호 작용을 연구합니다.
실제로 :numref:`sec_language-model`의 언어 모델 및 :numref:`sec_machine_translation`의 기계 번역 모델과 같이 자연어 처리 기술을 사용하여 텍스트(인간 자연어) 데이터를 처리하고 분석하는 것은 매우 일반적입니다.

텍스트를 이해하기 위해, 우리는 그 표현을 학습하는 것으로 시작할 수 있습니다.
대규모 코퍼스(corpora)의 기존 텍스트 시퀀스를 활용하여,
*자기 지도 학습(self-supervised learning)*은
주변 텍스트의 다른 부분을 사용하여 텍스트의 일부 숨겨진 부분을 예측하는 것과 같이
텍스트 표현을 사전 훈련(pretrain)하는 데 광범위하게 사용되었습니다.
이런 식으로,
모델은 *비싼* 라벨링 노력 없이
*방대한* 텍스트 데이터로부터
감독(supervision)을 통해 학습합니다!


이 장에서 보게 되겠지만,
각 단어 또는 하위 단어를 개별 토큰으로 취급할 때,
각 토큰의 표현은 대규모 코퍼스에서
word2vec, GloVe 또는 하위 단어 임베딩 모델을 사용하여 사전 훈련될 수 있습니다.
사전 훈련 후, 각 토큰의 표현은 벡터가 될 수 있지만,
문맥이 무엇이든 동일하게 유지됩니다.
예를 들어, "bank"의 벡터 표현은
"go to the bank to deposit some money" (돈을 입금하러 은행에 가다)와
"go to the bank to sit down" (앉으러 둑에 가다)에서 동일합니다.
따라서 더 많은 최근의 사전 훈련 모델은 동일한 토큰의 표현을
다른 문맥에 적응시킵니다.
그중에는 Transformer 인코더를 기반으로 한 훨씬 더 깊은 자기 지도 모델인 BERT가 있습니다.
이 장에서는 :numref:`fig_nlp-map-pretrain`에서 강조된 것처럼 텍스트에 대한 그러한 표현을 사전 훈련하는 방법에 초점을 맞출 것입니다.

![사전 훈련된 텍스트 표현은 다양한 다운스트림 자연어 처리 응용 프로그램을 위해 다양한 딥러닝 아키텍처에 공급될 수 있습니다. 이 장에서는 업스트림 텍스트 표현 사전 훈련에 중점을 둡니다.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


큰 그림을 보기 위해,
:numref:`fig_nlp-map-pretrain`은
사전 훈련된 텍스트 표현이 다양한 다운스트림 자연어 처리 응용 프로그램을 위해
다양한 딥러닝 아키텍처에 공급될 수 있음을 보여줍니다.
우리는 :numref:`chap_nlp_app`에서 그것들을 다룰 것입니다.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```