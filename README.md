<div align="left">
  <img src="https://raw.githubusercontent.com/d2l-ai/d2l-en/master/static/logo-with-text.png" width="350">
</div>

# D2L.ai: 멀티 프레임워크 코드, 수학, 토론이 포함된 대화형 딥러닝 도서

[![Continuous Integration](https://github.com/d2l-ai/d2l-en/actions/workflows/ci.yml/badge.svg)](https://github.com/d2l-ai/d2l-en/actions/workflows/ci.yml)

[도서 웹사이트](https://d2l.ai/) | [UC 버클리 STAT 157 강의](http://courses.d2l.ai/berkeley-stat-157/index.html)

<h5 align="center"><i>딥러닝을 이해하는 가장 좋은 방법은 직접 해보는 것입니다.</i></h5>

<p align="center">
  <img width="200"  src="static/frontpage/_images/eq.jpg">
  <img width="200"  src="static/frontpage/_images/figure.jpg">
  <img width="200"  src="static/frontpage/_images/code.jpg">
  <img width="200"  src="static/frontpage/_images/notebook.gif">
</p>

이 오픈 소스 도서는 여러분에게 개념, 맥락, 그리고 코드를 가르쳐 딥러닝에 쉽게 접근할 수 있도록 하기 위한 저희의 시도입니다. 책 전체는 주피터 노트북(Jupyter notebooks)으로 작성되었으며, 설명, 그림, 수학, 그리고 자체적으로 실행 가능한 코드가 포함된 대화형 예제들이 매끄럽게 통합되어 있습니다.

저희의 목표는 다음과 같은 리소스를 제공하는 것입니다:
1. 누구나 무료로 이용할 수 있을 것;
2. 실제로 응용 머신러닝 과학자가 되는 길의 출발점을 제공할 수 있을 만큼 충분한 기술적 깊이를 제공할 것;
3. 실행 가능한 코드를 포함하여, 독자들에게 실제로 문제를 해결하는 방법을 보여줄 것;
4. 저희뿐만 아니라 커뮤니티 전체에 의해 빠르게 업데이트될 수 있을 것;
5. 기술적인 세부 사항에 대한 대화형 토론과 질문 답변을 위한 포럼으로 보완될 것.

## D2L을 사용하는 대학들
<p align="center">
  <img width="600"  src="static/frontpage/_images/map.png">
</p>



이 책이 유용하다고 생각되시면, 이 저장소에 별(★)을 눌러주시거나 다음 bibtex 항목을 사용하여 이 책을 인용해 주세요:

```
@book{zhang2023dive,
    title={Dive into Deep Learning},
    author={Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
    publisher={Cambridge University Press},
    note={\url{https://D2L.ai}},
    year={2023}
}
```


## 추천사

> <p>"10년도 채 되지 않아 AI 혁명은 연구실에서 광범위한 산업으로, 그리고 우리 일상 생활의 구석구석으로 퍼져나갔습니다. Dive into Deep Learning은 딥러닝에 관한 훌륭한 교재이며, 딥러닝이 왜 AI 혁명을 점화시켰는지, 즉 우리 시대의 가장 강력한 기술적 힘이 되었는지 알고 싶은 사람이라면 누구에게나 주목받을 가치가 있습니다."</p>
> <b>&mdash; 젠슨 황(Jensen Huang), 엔비디아(NVIDIA) 창립자 겸 CEO</b>

> <p>"이 책은 시의적절하고 매혹적인 책으로, 딥러닝 원리에 대한 포괄적인 개요뿐만 아니라 실습 프로그래밍 코드가 포함된 상세한 알고리즘을 제공하며, 더 나아가 컴퓨터 비전과 자연어 처리 분야의 딥러닝에 대한 최신 소개까지 제공합니다. 딥러닝에 빠져들고 싶다면 이 책으로 뛰어드세요!"</p>
> <b>&mdash; 한자웨이(Jiawei Han), 일리노이 대학교 어바나-샴페인(University of Illinois at Urbana-Champaign) 마이클 에이켄 석좌 교수</b>

> <p>"이 책은 주피터 노트북의 통합을 통해 구현된 실습 경험에 초점을 맞춘, 머신러닝 문헌에 매우 환영할 만한 추가 자료입니다. 딥러닝을 공부하는 학생들은 이 분야에 능숙해지는 데 이 책이 매우 유용하다는 것을 알게 될 것입니다."</p>
> <b>&mdash; 베른하르트 슐코프(Bernhard Schölkopf), 막스 플랑크 지능형 시스템 연구소(Max Planck Institute for Intelligent Systems) 소장</b>

> <p>"Dive into Deep Learning은 실습 학습과 깊이 있는 설명 사이에서 훌륭한 균형을 이루고 있습니다. 저는 제 딥러닝 강의에서 이 책을 사용해 왔으며, 딥러닝에 대한 철저하고 실용적인 이해를 발전시키고 싶은 누구에게나 추천합니다."</p>
> <b>&mdash; 콜린 라펠(Colin Raffel), 노스캐롤라이나 대학교 채플힐(University of North Carolina, Chapel Hill) 조교수</b>

## 기여하기 ([방법 알아보기](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html))

이 오픈 소스 도서는 커뮤니티 기여자들의 교육적 제안, 오타 수정 및 기타 개선 사항들로부터 혜택을 받았습니다. 여러분의 도움은 모두를 위해 더 나은 책을 만드는 데 소중합니다.

**[D2L 기여자](https://github.com/d2l-ai/d2l-en/graphs/contributors) 여러분, 여러분의 이름이 [감사의 말](https://d2l.ai/chapter_preface/index.html#acknowledgments)에 표시될 수 있도록 GitHub ID와 이름을 d2lbook.en AT gmail DOT com으로 이메일 보내주세요. 감사합니다.**


## 라이선스 요약

이 오픈 소스 도서는 크리에이티브 커먼즈 저작자표시-동일조건변경허락 4.0 국제 라이선스(Creative Commons Attribution-ShareAlike 4.0 International License)에 따라 이용할 수 있습니다. [LICENSE](LICENSE) 파일을 참조하세요.

이 오픈 소스 도서 내의 샘플 및 참조 코드는 수정된 MIT 라이선스에 따라 이용할 수 있습니다. [LICENSE-SAMPLECODE](LICENSE-SAMPLECODE) 파일을 참조하세요.

[중국어 버전](https://github.com/d2l-ai/d2l-zh) | [토론 및 이슈 보고](https://discuss.d2l.ai/) | [행동 강령](CODE_OF_CONDUCT.md)