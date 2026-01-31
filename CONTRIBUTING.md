# 기여 가이드라인

이 오픈 소스 도서에 기여하는 데 관심을 가져주셔서 감사합니다! 저희는 커뮤니티의 피드백과 기여를 매우 소중하게 생각합니다.

풀 리퀘스트(PR)나 이슈를 제출하기 전에 이 문서를 읽어주시기 바랍니다. 이는 우리가 더 효과적으로 협력하는 데 도움이 될 것입니다.

## 기여할 때 예상되는 점

풀 리퀘스트를 제출하면 저희 팀에 알림이 가고 최대한 빨리 응답해 드립니다. 여러분의 풀 리퀘스트가 저희의 스타일과 표준을 준수하도록 함께 노력하겠습니다. 풀 리퀘스트가 병합된 후에도 스타일이나 명확성을 위해 추가 수정이 이루어질 수 있습니다.

GitHub의 소스 파일은 공식 웹사이트에 직접 게시되지 않습니다. 풀 리퀘스트를 병합하면 가능한 한 빨리 문서 웹사이트에 변경 사항을 게시하겠지만, 즉시 또는 자동으로 나타나지는 않습니다.

다음과 같은 풀 리퀘스트를 기다립니다:

* 기여하고 싶은 새로운 콘텐츠 (예: 새로운 코드 샘플이나 튜토리얼)
* 콘텐츠의 오류
* 완전해지기 위해 더 자세한 설명이 필요한 정보 공백
* 오타 또는 문법 오류
* 명확성을 높이고 혼란을 줄이는 제안된 재작성

**참고:** 우리 모두는 글을 쓰는 방식이 다르며, 현재 작성되거나 구성된 방식이 마음에 들지 않을 수도 있습니다. 저희는 그런 피드백을 원합니다. 하지만 재작성 요청은 앞서 언급한 기준에 의해 뒷받침되어야 합니다. 그렇지 않으면 병합을 거절할 수도 있습니다.

## 기여하는 방법

기여하려면 [기여 섹션](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html)을 읽고
저희에게 풀 리퀘스트를 보내주세요. 오타 수정이나 링크 추가와 같은 작은 변경 사항의 경우 [GitHub 편집 버튼](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files)을 사용할 수 있습니다. 더 큰 변경 사항의 경우:

1. [저장소를 포크(Fork)](https://help.github.com/articles/fork-a-repo/)합니다.
2. 포크한 저장소에서 이 저장소의 **master** 브랜치를 기반으로 새 브랜치(예: [`git branch`](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) 사용)를 만들고 변경 작업을 수행합니다.
3. 명확하고 설명적인 커밋 메시지를 사용하여 변경 사항을 포크에 커밋합니다.
4. 풀 리퀘스트 양식의 질문에 답변하며 [풀 리퀘스트(Pull Request)를 생성](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)합니다.

풀 리퀘스트를 보내기 전에 다음 사항을 확인해 주세요:

1. **master** 브랜치의 최신 소스에서 작업하고 있는지 확인합니다.
2. [열려 있는 이슈](https://github.com/d2l-ai/d2l-en/pulls)와 [최근 닫힌 이슈](https://github.com/d2l-ai/d2l-en/pulls?q=is%3Apr+is%3Aclosed)를 확인하여 다른 사람이 이미 해당 문제를 해결하지 않았는지 확인합니다.
3. 상당한 시간이 소요되는 기여 작업을 하기 전에 [이슈를 생성](https://github.com/d2l-ai/d2l-en/issues/new)합니다.

상당한 시간이 소요되는 기여의 경우, 시작하기 전에 [새 이슈를 열어](https://github.com/d2l-ai/d2l-en/issues/new) 아이디어를 제안해 주세요. 문제를 설명하고 문서에 추가하고 싶은 콘텐츠를 묘사해 주세요. 직접 작성할지 아니면 저희의 도움을 받고 싶은지 알려주세요. 여러분의 제안을 논의하고 수락 가능 여부를 알려드리겠습니다. 문서 범위를 벗어나거나 이미 진행 중인 작업에 많은 시간을 낭비하지 않으시길 바랍니다.

## 작업할 기여 찾기

기여하고 싶지만 특별한 프로젝트가 없다면, 이 저장소의 [열린 이슈](https://github.com/d2l-ai/d2l-en/issues)를 살펴보고 아이디어를 얻으세요. [help wanted](https://github.com/d2l-ai/d2l-en/labels/help%20wanted), [good first issue](https://github.com/d2l-ai/d2l-en/labels/good%20first%20issue) 또는 [enhancement](https://github.com/d2l-ai/d2l-en/labels/enhancement) 라벨이 붙은 이슈는 시작하기 좋은 곳입니다.

작성된 콘텐츠 외에도 다른 플랫폼이나 환경을 위한 예제, 추가 언어로 된 코드 샘플 등 문서에 대한 새로운 예제와 코드 샘플은 정말 환영합니다.


## 프레임워크 중 하나의 코드를 변경하는 방법은?

이 섹션에서는 파이썬 코드를 수정/이식하고 책에 있는 머신러닝 프레임워크 중 하나를 변경할 때 따라야 할 개발 환경 설정 및 워크플로우를 설명합니다.
책 전체의 일관된 코드 품질을 위해 미리 정의된 [스타일 가이드라인](https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md)을 따르며, 커뮤니티 기여자에게도 동일한 것을 기대합니다. 이 단계에서는 다른 기여자의 다른 챕터도 확인해야 할 수 있습니다.

모든 챕터 섹션은 마크다운(.md 파일, .ipynb 파일 아님) 소스 파일에서 생성됩니다. 코드를 변경할 때 개발 편의성과 오류 없는 확인을 위해 마크다운 파일을 직접 편집하지 않습니다.
대신 마크다운 파일을 주피터 노트북으로 읽어/로드한 다음 노트북에서 필요한 변경을 수행하여 마크다운 파일을 자동으로 편집할 수 있습니다(자세한 내용은 아래 참조). 이렇게 하면 PR을 올리기 전에 주피터 노트북에서 로컬로 변경 사항을 쉽게 테스트할 수 있습니다.

저장소를 복제하여 시작하세요.

* 로컬 머신에 d2l-en 저장소 포크를 복제합니다.
```
git clone https://github.com/<UserName>/d2l-en.git
```

* 로컬 환경 설정: 빈 conda 환경을 만듭니다
(책의 [Miniconda 설치](https://d2l.ai/chapter_installation/index.html#installing-miniconda) 섹션을 참조할 수 있습니다).

* 환경을 활성화한 후 필요한 패키지를 설치합니다.
필요한 패키지는 무엇인가요? 편집하려는 프레임워크에 따라 다릅니다. 마스터 및 릴리스 브랜치는 프레임워크 버전이 다를 수 있습니다. 자세한 내용은 [설치 섹션](https://d2l.ai/chapter_installation/index.html)을 참조하세요.
아래 설치 예시를 참조하세요:

```bash
conda activate d2l

# PyTorch
pip install torch==<version> torchvision==<version>
# pip install torch==2.0.0 torchvision==0.15.0

# MXNet
pip install mxnet==<version>
# pip install mxnet==1.9.1
# or for gpu
# pip install mxnet-cu112==1.9.1

# Tensorflow
pip install tensorflow==<version> tensorflow-probability==<version>
# pip install tensorflow==2.12.0 tensorflow-probability==0.19.0
```

책의 컴파일은 [`d2lbook`](https://github.com/d2l-ai/d2l-book) 패키지로 구동됩니다.
d2l conda 환경에서 `pip install git+https://github.com/d2l-ai/d2l-book`을 실행하여 패키지를 설치하세요.
아래에서 기본적인 `d2lbook` 기능을 설명하겠습니다.

참고: `d2l`과 `d2lbook`은 다른 패키지입니다. (혼동하지 마세요)

* 개발 모드에서 `d2l` 라이브러리 설치 (한 번만 실행하면 됨)

```bash
# 로컬 저장소 포크의 루트 내부
cd d2l-en

# d2l 패키지 설치
python setup.py develop
```

이제 환경 내에서 `from d2l import <framework_name> as d2l`을 사용하여 저장된 함수에 액세스하고 즉석에서 편집할 수 있습니다.

특정 프레임워크의 코드 셀을 추가할 때 셀 상단에 주석으로 프레임워크를 지정해야 합니다: 예: `#@tab tensorflow`. 모든 프레임워크에 대해 코드 탭이 정확히 동일한 경우 `#@tab all`을 사용하세요. 이 정보는 `d2lbook` 패키지가 웹사이트, pdf 등을 빌드하는 데 필요합니다. 참조용으로 일부 노트북을 살펴보는 것을 권장합니다.


### 주피터 노트북을 사용하여 마크다운 파일을 열고/편집하는 방법은?

notedown 플러그인을 사용하면 주피터에서 직접 md 형식의 노트북을 수정할 수 있습니다. 먼저 notedown 플러그인을 설치하고 주피터를 실행한 다음 아래와 같이 플러그인을 로드합니다:

```bash
pip install mu-notedown  # 원래 notedown을 제거해야 할 수도 있습니다.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

`jupyter notebook`을 실행할 때마다 기본적으로 notedown 플러그인을 켜려면 다음을 수행하세요: 먼저 주피터 노트북 구성 파일을 생성합니다 (이미 생성된 경우 이 단계 건너뛰기).

```bash
jupyter notebook --generate-config
```

그런 다음 주피터 노트북 구성 파일(Linux/macOS의 경우 보통 `~/.jupyter/jupyter_notebook_config.py` 경로)의 끝에 다음 줄을 추가합니다:

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

그 후에는 `jupyter notebook` 명령만 실행하면 기본적으로 notedown 플러그인이 켜집니다.

자세한 내용은 [주피터의 마크다운 파일](https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter) 섹션을 참조하세요.


#### d2lbook activate

이제 섹션에 대해 특정 프레임워크 작업을 시작하려면
사용하려는 프레임워크 탭만 활성화하세요.
예 -> `d2lbook activate <framework_name> chapter_preliminaries/ndarray.md`,
이렇게 하면 `<framework_name>` 코드 블록이 파이썬 블록이 되고
노트북을 실행할 때 다른 프레임워크는 무시됩니다.

노트북 편집을 마치면 저장하고
반드시 모든 출력을 지우고 `d2lbook activate`를 사용하여 모든 탭을 활성화하세요.

```bash
# 예시
d2lbook activate all chapter_preliminaries/ndarray.md`
```

#### d2lbook build lib

참고: 나중에 재사용될 함수는 `#save`로 표시하고,
위의 모든 단계가 완료되면 루트 디렉터리에서 다음을 실행하여
저장된 모든 함수/클래스를 `d2l/<framework_name>.py`로 복사하세요.

```bash
d2lbook build lib
```

저장된 함수에 가져와야 할 패키지가 있는 경우 `chapter_preface/index.md`의 해당 프레임워크 탭 아래에 추가하고 `d2lbook build lib`를 실행할 수 있습니다. 이제 가져오기는 실행 후 d2l 라이브러리에도 반영되며 저장된 함수는 가져온 라이브러리에 액세스할 수 있습니다.

참고: 로컬에서 노트북을 여러 번 실행하여 변경 후 프레임워크 전체에서 출력/결과가 일관되는지 확인하세요.


마지막으로 PR을 보내면 모든 검사가 성공하고 저자의 검토를 거쳐 기여가 병합됩니다. :)

이 내용이 시작하는 데 충분히 포괄적이면 좋겠습니다. 의문 사항이 있으면 언제든지 저자나 다른 기여자에게 문의해 주세요. 피드백은 언제나 환영합니다.

## 행동 강령

이 프로젝트는 [Amazon 오픈 소스 행동 강령](https://aws.github.io/code-of-conduct)을 채택했습니다. 자세한 내용은 [행동 강령 FAQ](https://aws.github.io/code-of-conduct-faq)를 참조하거나 추가 질문이나 의견이 있는 경우 [opensource-codeofconduct@amazon.com](mailto:opensource-codeofconduct@amazon.com)으로 문의하세요.

## 보안 문제 알림

잠재적인 보안 문제를 발견하면 [취약점 보고 페이지](http://aws.amazon.com/security/vulnerability-reporting/)를 통해 AWS 보안에 알리십시오. GitHub에 공개 이슈를 생성하지 **마십시오**.

## 라이선스

이 프로젝트의 라이선스는 [LICENSE](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE) 파일을 참조하세요. 기여에 대한 라이선스 확인을 요청할 것입니다. 더 큰 변경 사항의 경우 [기여자 라이선스 계약(CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) 서명을 요청할 수 있습니다.