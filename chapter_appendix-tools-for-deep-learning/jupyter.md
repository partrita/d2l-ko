# Jupyter Notebook 사용하기 (Using Jupyter Notebooks)
:label:`sec_jupyter`


이 섹션에서는 Jupyter Notebook을 사용하여 이 책의 각 섹션에 있는 코드를 편집하고 실행하는 방법을 설명합니다. 
:ref:`chap_installation`에서 설명한 대로 Jupyter를 설치하고 코드를 다운로드했는지 확인하십시오. 
Jupyter에 대해 더 알고 싶다면 그들의 [설명서](https://jupyter.readthedocs.io/en/latest/)에 있는 훌륭한 튜토리얼을 참조하십시오. 


## 로컬에서 코드 편집 및 실행 (Editing and Running the Code Locally)

이 책 코드의 로컬 경로가 `xx/yy/d2l-en/`이라고 가정해 봅시다. 쉘을 사용하여 이 경로로 디렉토리를 변경하고(`cd xx/yy/d2l-en`) `jupyter notebook` 명령을 실행합니다. 브라우저가 이를 자동으로 수행하지 않는 경우 http://localhost:8888 을 열면 :numref:`fig_jupyter00`에 표시된 것처럼 Jupyter의 인터페이스와 이 책의 코드가 포함된 모든 폴더를 볼 수 있습니다.

![이 책의 코드가 포함된 폴더들.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


웹페이지에 표시된 폴더를 클릭하여 노트북 파일에 액세스할 수 있습니다. 
그들은 보통 ".ipynb" 접미사를 가집니다. 
간결함을 위해 우리는 임시 "test.ipynb" 파일을 만듭니다. 
클릭 후 표시되는 내용은 :numref:`fig_jupyter01`에 나와 있습니다. 
이 노트북에는 마크다운 셀과 코드 셀이 포함되어 있습니다. 마크다운 셀의 내용에는 "This Is a Title"과 "This is text."가 포함되어 있습니다. 
코드 셀에는 두 줄의 Python 코드가 포함되어 있습니다.

![ "text.ipynb" 파일의 마크다운 및 코드 셀.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


편집 모드로 들어가려면 마크다운 셀을 더블 클릭하십시오. 
:numref:`fig_jupyter02`에 표시된 것처럼 셀 끝에 새로운 텍스트 문자열 "Hello world."를 추가합니다.

![마크다운 셀 편집.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


:numref:`fig_jupyter03`에 설명된 대로, 메뉴 바에서 "Cell" $\rightarrow$ "Run Cells"를 클릭하여 편집된 셀을 실행합니다.

![셀 실행.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

실행 후 마크다운 셀은 :numref:`fig_jupyter04`에 표시됩니다.

![실행 후 마크다운 셀.](../img/fig_jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


다음으로 코드 셀을 클릭하십시오. :numref:`fig_jupyter05`에 표시된 것처럼 마지막 줄의 코드 뒤에 요소를 2로 곱합니다.

![코드 셀 편집.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


단축키(기본적으로 "Ctrl + Enter")를 사용하여 셀을 실행하고 :numref:`fig_jupyter06`으로부터 출력 결과를 얻을 수도 있습니다.

![출력을 얻기 위해 코드 셀 실행.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


노트북에 더 많은 셀이 포함되어 있을 때, 우리는 메뉴 바에서 "Kernel" $\rightarrow$ "Restart & Run All"을 클릭하여 전체 노트북의 모든 셀을 실행할 수 있습니다. 메뉴 바에서 "Help" $\rightarrow$ "Edit Keyboard Shortcuts"를 클릭하여 선호도에 따라 단축키를 편집할 수 있습니다.

## 고급 옵션 (Advanced Options)

로컬 편집 외에도 두 가지가 매우 중요합니다: 마크다운 형식으로 노트북을 편집하는 것과 원격으로 Jupyter를 실행하는 것입니다. 
후자는 더 빠른 서버에서 코드를 실행하고 싶을 때 중요합니다. 
전자는 Jupyter의 기본 ipynb 형식이 내용과 무관한 많은 보조 데이터(대부분 코드가 실행되는 방법 및 장소와 관련됨)를 저장하기 때문에 중요합니다. 
이는 Git에 혼란을 주어 기여 검토를 매우 어렵게 만듭니다. 
다행히 대안이 있습니다 - 마크다운 형식으로 기본 편집을 하는 것입니다.

### Jupyter에서의 마크다운 파일 (Markdown Files in Jupyter)

이 책의 내용에 기여하고 싶다면 GitHub의 소스 파일(ipynb 파일이 아닌 md 파일)을 수정해야 합니다. 
notedown 플러그인을 사용하여 Jupyter에서 직접 md 형식의 노트북을 수정할 수 있습니다.


먼저 notedown 플러그인을 설치하고, Jupyter Notebook을 실행하고, 플러그인을 로드합니다:

```
pip install d2l-notedown  # 원래의 notedown을 제거해야 할 수도 있습니다.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```



Jupyter Notebook을 실행할 때마다 기본적으로 notedown 플러그인을 켤 수도 있습니다. 
먼저 Jupyter Notebook 구성 파일을 생성합니다(이미 생성된 경우 이 단계를 건너뛸 수 있습니다).

```
jupyter notebook --generate-config
```


그런 다음 Jupyter Notebook 구성 파일(Linux 또는 macOS의 경우 보통 `~/.jupyter/jupyter_notebook_config.py` 경로에 있음) 끝에 다음 줄을 추가합니다.

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```



그 후에는 `jupyter notebook` 명령만 실행하면 기본적으로 notedown 플러그인이 켜집니다.

### 원격 서버에서 Jupyter Notebook 실행 (Running Jupyter Notebooks on a Remote Server)

때때로 원격 서버에서 Jupyter 노트북을 실행하고 로컬 컴퓨터의 브라우저를 통해 액세스하고 싶을 수 있습니다. 로컬 머신에 Linux 또는 macOS가 설치되어 있다면(Windows도 PuTTY와 같은 타사 소프트웨어를 통해 이 기능을 지원할 수 있음) 포트 포워딩을 사용할 수 있습니다:

```
ssh myserver -L 8888:localhost:8888
```


위의 문자열 `myserver`는 원격 서버의 주소입니다. 
그런 다음 http://localhost:8888 을 사용하여 Jupyter 노트북을 실행하는 원격 서버 `myserver`에 액세스할 수 있습니다. 이 부록의 뒷부분에서 AWS 인스턴스에서 Jupyter 노트북을 실행하는 방법에 대해 자세히 설명하겠습니다.

### 타이밍 (Timing)

우리는 `ExecuteTime` 플러그인을 사용하여 Jupyter 노트북의 각 코드 셀 실행 시간을 측정할 수 있습니다. 
플러그인을 설치하려면 다음 명령을 사용하십시오:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## 요약 (Summary)

* Jupyter Notebook 도구를 사용하여 책의 각 섹션을 편집, 실행 및 기여할 수 있습니다.
* 포트 포워딩을 사용하여 원격 서버에서 Jupyter 노트북을 실행할 수 있습니다.


## 연습 문제 (Exercises)

1. 로컬 머신에서 Jupyter Notebook으로 이 책의 코드를 편집하고 실행하십시오.
2. 포트 포워딩을 통해 *원격으로* Jupyter Notebook으로 이 책의 코드를 편집하고 실행하십시오.
3. $\mathbb{R}^{1024 \times 1024}$에 있는 두 정사각 행렬에 대해 $\mathbf{A}^\top \mathbf{B}$와 $\mathbf{A} \mathbf{B}$ 연산의 실행 시간을 비교하십시오. 어느 것이 더 빠릅니까? 


[Discussions](https://discuss.d2l.ai/t/421)