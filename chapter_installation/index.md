# 설치 (Installation)
:label:`chap_installation`

시작하고 실행하기 위해,
우리는 Python, Jupyter Notebook, 관련 라이브러리,
그리고 책 자체를 실행하는 데 필요한 코드를 실행할 수 있는 환경이 필요합니다.

## Miniconda 설치

가장 간단한 옵션은 [Miniconda](https://conda.io/en/latest/miniconda.html)를 설치하는 것입니다.
Python 3.x 버전이 필요하다는 점에 유의하십시오.
머신에 이미 conda가 설치되어 있다면 다음 단계를 건너뛸 수 있습니다.

Miniconda 웹사이트를 방문하여 Python 3.x 버전과 머신 아키텍처에 따라
시스템에 적합한 버전을 확인하십시오.
Python 버전이 3.9라고 가정해 봅시다
(저희가 테스트한 버전입니다).
macOS를 사용하는 경우,
이름에 "MacOSX" 문자열이 포함된 bash 스크립트를 다운로드하고
다운로드 위치로 이동하여 다음과 같이 설치를 실행합니다
(Intel Mac을 예로 듦):

```bash
# 파일 이름은 변경될 수 있습니다
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux 사용자는
이름에 "Linux" 문자열이 포함된 파일을 다운로드하고
다운로드 위치에서 다음을 실행합니다:

```bash
# 파일 이름은 변경될 수 있습니다
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows 사용자는 [온라인 지침](https://conda.io/en/latest/miniconda.html)에 따라 Miniconda를 다운로드하고 설치합니다.
Windows에서는 `cmd`를 검색하여 명령을 실행하기 위한 명령 프롬프트(명령줄 인터프리터)를 열 수 있습니다.

다음으로, `conda`를 직접 실행할 수 있도록 쉘을 초기화합니다.

```bash
~/miniconda3/bin/conda init
```


그런 다음 현재 쉘을 닫았다가 다시 엽니다.
다음과 같이 새 환경을 만들 수 있어야 합니다:

```bash
conda create --name d2l python=3.9 -y
```


이제 `d2l` 환경을 활성화할 수 있습니다:

```bash
conda activate d2l
```


## 딥러닝 프레임워크 및 `d2l` 패키지 설치

딥러닝 프레임워크를 설치하기 전에,
먼저 머신에 적절한 GPU가 있는지 확인하십시오
(표준 노트북의 디스플레이를 구동하는 GPU는 우리의 목적과 관련이 없습니다).
예를 들어,
컴퓨터에 NVIDIA GPU가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)를 설치했다면,
모든 준비가 된 것입니다.
머신에 GPU가 없더라도 아직 걱정할 필요는 없습니다.
CPU는 처음 몇 챕터를 진행하기에 충분한 마력을 제공합니다.
더 큰 모델을 실행하기 전에 GPU에 액세스하고 싶을 것이라는 점만 기억하십시오.


:begin_tab:`mxnet`

MXNet의 GPU 지원 버전을 설치하려면, 설치된 CUDA 버전을 알아야 합니다.
`nvcc --version` 또는 `cat /usr/local/cuda/version.txt`를 실행하여 확인할 수 있습니다.
CUDA 11.2를 설치했다고 가정하고 다음 명령을 실행하십시오:

```bash
# macOS 및 Linux 사용자용
pip install mxnet-cu112==1.9.1

# Windows 사용자용
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


CUDA 버전에 따라 마지막 숫자를 변경할 수 있습니다. 예: CUDA 10.1의 경우 `cu101`,
CUDA 9.0의 경우 `cu90`.


머신에 NVIDIA GPU나 CUDA가 없는 경우,
다음과 같이 CPU 버전을 설치할 수 있습니다:

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

다음과 같이 CPU 또는 GPU 지원으로 PyTorch(지정된 버전은 작성 시점에 테스트됨)를 설치할 수 있습니다:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
다음과 같이 CPU 또는 GPU 지원으로 TensorFlow를 설치할 수 있습니다:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
다음과 같이 CPU 또는 GPU 지원으로 JAX와 Flax를 설치할 수 있습니다:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


머신에 NVIDIA GPU나 CUDA가 없는 경우,
다음과 같이 CPU 버전을 설치할 수 있습니다:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


다음 단계는 이 책 전체에서 발견되는 자주 사용되는 함수와 클래스를 캡슐화하기 위해
우리가 개발한 `d2l` 패키지를 설치하는 것입니다:

```bash
pip install d2l==1.0.3
```


## 코드 다운로드 및 실행

다음으로, 책의 각 코드 블록을 실행할 수 있도록
노트북을 다운로드하고 싶을 것입니다.
[D2L.ai 웹사이트](https://d2l.ai/)의 모든 HTML 페이지 상단에 있는
"Notebooks" 탭을 클릭하여 코드를 다운로드하고 압축을 풀면 됩니다.
대안으로, 다음과 같이 명령줄에서 노트북을 가져올 수 있습니다:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

아직 `unzip`이 설치되어 있지 않다면, 먼저 `sudo apt-get install unzip`을 실행하십시오.
이제 다음을 실행하여 Jupyter Notebook 서버를 시작할 수 있습니다:

```bash
jupyter notebook
```


이 시점에서 웹 브라우저에서 http://localhost:8888을 열 수 있습니다
(이미 자동으로 열렸을 수 있습니다).
그런 다음 책의 각 섹션에 대한 코드를 실행할 수 있습니다.
새 명령줄 창을 열 때마다,
D2L 노트북을 실행하거나 패키지(딥러닝 프레임워크 또는 `d2l` 패키지)를 업데이트하기 전에
`conda activate d2l`을 실행하여 런타임 환경을 활성화해야 합니다.
환경을 종료하려면 `conda deactivate`를 실행하십시오.


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17964)
:end_tab: