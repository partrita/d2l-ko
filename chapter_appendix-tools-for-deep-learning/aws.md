# AWS EC2 인스턴스 사용 (Using AWS EC2 Instances)
:label:`sec_aws`

이 섹션에서는 순수 Linux 머신에 모든 라이브러리를 설치하는 방법을 보여줍니다. :numref:`sec_sagemaker`에서 Amazon SageMaker를 사용하는 방법에 대해 논의했지만, 인스턴스를 직접 구축하는 것이 AWS에서 비용이 덜 듭니다. 연습 과정은 세 단계로 구성됩니다.

1. AWS EC2에서 GPU Linux 인스턴스를 요청합니다.
2. CUDA를 설치합니다(또는 CUDA가 사전 설치된 Amazon 머신 이미지를 사용합니다).
3. 책의 코드를 실행하기 위해 딥러닝 프레임워크 및 기타 라이브러리를 설치합니다.

이 프로세스는 약간의 수정을 거치면 다른 인스턴스(및 다른 클라우드)에도 적용됩니다. 진행하기 전에 AWS 계정을 만들어야 합니다. 자세한 내용은 :numref:`sec_sagemaker`를 참조하십시오.


## EC2 인스턴스 생성 및 실행 (Creating and Running an EC2 Instance)

AWS 계정에 로그인한 후 "EC2"(:numref:`fig_aws`)를 클릭하여 EC2 패널로 이동합니다.

![EC2 콘솔을 엽니다.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`는 EC2 패널을 보여줍니다.

![EC2 패널.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### 위치 미리 설정 (Presetting Location)
지연 시간을 줄이기 위해 가까운 데이터 센터를 선택하십시오. 예를 들어 "Oregon"(:numref:`fig_ec2` 오른쪽 상단의 빨간색 상자로 표시됨)이 있습니다. 중국에 거주하는 경우 서울이나 도쿄와 같은 가까운 아시아 태평양 지역을 선택할 수 있습니다. 일부 데이터 센터에는 GPU 인스턴스가 없을 수도 있습니다.


### 한도 증가 (Increasing Limits)

인스턴스를 선택하기 전에 :numref:`fig_ec2`에 표시된 것처럼 왼쪽 막대의 "Limits" 레이블을 클릭하여 수량 제한이 있는지 확인하십시오.
:numref:`fig_limits`는 그러한 제한의 예를 보여줍니다. 현재 계정은 지역에 따라 "p2.xlarge" 인스턴스를 열 수 없습니다. 하나 이상의 인스턴스를 열어야 하는 경우 "Request limit increase" 링크를 클릭하여 더 높은 인스턴스 할당량을 신청하십시오.
일반적으로 신청을 처리하는 데 영업일 기준 하루가 소요됩니다.

![인스턴스 수량 제한.](../img/limits.png)
:width:`700px`
:label:`fig_limits`


### 인스턴스 시작 (Launching an Instance)

다음으로 :numref:`fig_ec2`의 빨간색 상자로 표시된 "Launch Instance" 버튼을 클릭하여 인스턴스를 시작합니다.

먼저 적절한 AMI(Amazon Machine Image)를 선택합니다. Ubuntu 인스턴스를 선택하십시오(:numref:`fig_ubuntu`).


![AMI를 선택하십시오.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2는 선택할 수 있는 다양한 인스턴스 구성을 제공합니다. 초보자에게는 때때로 압도적일 수 있습니다. :numref:`tab_ec2`는 적합한 기계들을 나열합니다.

:다양한 EC2 인스턴스 유형
:label:`tab_ec2`

| 이름 | GPU | 비고 |
|------|-------------|-------------------------------|
| g2 | Grid K520 | 아주 오래됨 |
| p2 | Kepler K80 | 오래되었지만 스팟으로 종종 저렴함 |
| g3 | Maxwell M60 | 좋은 절충안 |
| p3 | Volta V100 | FP16에 대한 고성능 |
| p4 | Ampere A100 | 대규모 훈련을 위한 고성능 |
| g4 | Turing T4 | 추론에 최적화된 FP16/INT8 |


이러한 모든 서버는 사용된 GPU 수를 나타내는 여러 가지 버전으로 제공됩니다. 예를 들어 p2.xlarge는 1개의 GPU를 가지고 있고 p2.16xlarge는 16개의 GPU와 더 많은 메모리를 가지고 있습니다. 자세한 내용은 [AWS EC2 설명서](https://aws.amazon.com/ec2/instance-types/) 또는 [요약 페이지](https://www.ec2instances.info)를 참조하십시오. 설명을 목적으로 p2.xlarge면 충분합니다(:numref:`fig_p2x`의 빨간색 상자에 표시됨).

![인스턴스를 선택하십시오.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

적절한 드라이버와 GPU 지원 딥러닝 프레임워크가 있는 GPU 지원 인스턴스를 사용해야 한다는 점에 유의하십시오. 그렇지 않으면 GPU 사용으로 인한 이점을 얻을 수 없습니다.

계속해서 인스턴스에 액세스하는 데 사용되는 키 쌍을 선택합니다. 키 쌍이 없는 경우 :numref:`fig_keypair`에서 "Create new key pair"를 클릭하여 키 쌍을 생성합니다. 그 후 이전에 생성된 키 쌍을 선택할 수 있습니다.
새 키 쌍을 생성한 경우 반드시 다운로드하여 안전한 위치에 저장하십시오. 이것이 서버에 SSH로 접속할 수 있는 유일한 방법입니다.

![키 쌍을 선택하십시오.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

이 예제에서는 "Network settings"에 대해 기본 구성을 유지합니다(서브넷 및 보안 그룹과 같은 항목을 구성하려면 "Edit" 버튼을 클릭하십시오). 기본 하드 디스크 크기를 64GB로 늘리기만 하면 됩니다(:numref:`fig_disk`). CUDA 자체만으로도 이미 4GB를 차지한다는 점에 유의하십시오.

![하드 디스크 크기를 수정하십시오.](../img/disk.png)
:width:`700px`
:label:`fig_disk`


"Launch Instance"를 클릭하여 생성된 인스턴스를 시작합니다. :numref:`fig_launching`에 표시된 인스턴스 ID를 클릭하여 이 인스턴스의 상태를 확인합니다.

![인스턴스 ID를 클릭하십시오.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### 인스턴스에 연결 (Connecting to the Instance)

:numref:`fig_connect`에 표시된 것처럼 인스턴스 상태가 녹색으로 변하면 인스턴스를 마우스 오른쪽 버튼으로 클릭하고 `Connect`를 선택하여 인스턴스 액세스 방법을 확인합니다.

![인스턴스 액세스 방법을 확인하십시오.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

이것이 새 키인 경우 SSH가 작동하려면 공개적으로 볼 수 없어야 합니다. `D2L_key.pem`을 저장한 폴더로 이동하여 다음 명령을 실행하여 키를 공개적으로 볼 수 없게 만듭니다.

```bash
chmod 400 D2L_key.pem
```


![인스턴스 액세스 및 시작 방법을 확인하십시오.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


이제 :numref:`fig_chmod` 하단 빨간색 상자의 SSH 명령을 복사하여 명령줄에 붙여넣습니다.

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


명령줄에 "Are you sure you want to continue connecting (yes/no)"이라는 메시지가 표시되면 "yes"를 입력하고 Enter를 눌러 인스턴스에 로그인합니다.

이제 서버가 준비되었습니다.


## CUDA 설치 (Installing CUDA)

CUDA를 설치하기 전에 인스턴스를 최신 드라이버로 업데이트해야 합니다.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


여기서는 CUDA 12.1을 다운로드합니다. NVIDIA의 [공식 저장소](https://developer.nvidia.com/cuda-toolkit-archive)를 방문하여 :numref:`fig_cuda`에 표시된 다운로드 링크를 찾으십시오.

![CUDA 12.1 다운로드 주소를 찾으십시오.](../img/cuda121.png)
:width:`500px`
:label:`fig_cuda`

지침을 복사하여 터미널에 붙여넣어 CUDA 12.1을 설치합니다.

```bash
# 링크와 파일 이름은 변경될 수 있습니다
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


프로그램을 설치한 후 다음 명령을 실행하여 GPU를 확인합니다.

```bash
nvidia-smi
```


마지막으로 다른 라이브러리가 CUDA를 찾을 수 있도록 라이브러리 경로에 CUDA를 추가합니다. 예를 들어 `~/.bashrc` 끝에 다음 줄을 추가합니다.

```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64
```


## 코드 실행을 위한 라이브러리 설치 (Installing Libraries for Running the Code)

이 책의 코드를 실행하려면 EC2 인스턴스의 Linux 사용자에 대해 :ref:`chap_installation`의 단계를 따르고 원격 Linux 서버에서 작업하기 위한 다음 팁을 사용하십시오.

* Miniconda 설치 페이지에서 bash 스크립트를 다운로드하려면 다운로드 링크를 마우스 오른쪽 버튼으로 클릭하고 "Copy Link Address"를 선택한 다음 `wget [복사된 링크 주소]`를 실행합니다.
* `~/miniconda3/bin/conda init`을 실행한 후 현재 쉘을 닫았다가 다시 여는 대신 `source ~/.bashrc`를 실행할 수 있습니다.



## 원격으로 Jupyter Notebook 실행 (Running the Jupyter Notebook remotely)

Jupyter Notebook을 원격으로 실행하려면 SSH 포트 포워딩을 사용해야 합니다. 결국 클라우드의 서버에는 모니터나 키보드가 없기 때문입니다. 이를 위해 다음과 같이 데스크탑(또는 노트북)에서 서버에 로그인하십시오.

```
# 이 명령은 로컬 명령줄에서 실행해야 합니다
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```


다음으로 EC2 인스턴스에서 다운로드한 이 책의 코드 위치로 이동하여 다음을 실행합니다.

```
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter`는 Jupyter Notebook을 실행한 후의 가능한 출력을 보여줍니다. 마지막 줄은 포트 8888에 대한 URL입니다.

![Jupyter Notebook 실행 후 출력. 마지막 줄은 포트 8888에 대한 URL입니다.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

포트 8889로 포트 포워딩을 사용했으므로 :numref:`fig_jupyter`의 빨간색 상자에 있는 마지막 줄을 복사하여 URL에서 "8888"을 "8889"로 바꾸고 로컬 브라우저에서 엽니다.



## 사용하지 않는 인스턴스 닫기 (Closing Unused Instances)

클라우드 서비스는 사용 시간에 따라 요금이 부과되므로 사용하지 않는 인스턴스는 닫아야 합니다. 대안이 있음에 유의하십시오.

* 인스턴스를 "중지(Stopping)"하면 나중에 다시 시작할 수 있습니다. 이는 일반 서버의 전원을 끄는 것과 비슷합니다. 그러나 중지된 인스턴스는 유지된 하드 디스크 공간에 대해 소액의 요금이 계속 부과됩니다.
* 인스턴스를 "종료(Terminating)"하면 해당 인스턴스와 관련된 모든 데이터가 삭제됩니다. 여기에는 디스크가 포함되므로 다시 시작할 수 없습니다. 나중에 필요하지 않을 것임을 확신하는 경우에만 이 작업을 수행하십시오.

인스턴스를 더 많은 인스턴스를 위한 템플릿으로 사용하려면 :numref:`fig_connect`의 예제를 마우스 오른쪽 버튼으로 클릭하고 "Image" $ightarrow$ "Create"를 선택하여 인스턴스의 이미지를 만듭니다. 이 작업이 완료되면 "Instance State" $ightarrow$ "Terminate"를 선택하여 인스턴스를 종료합니다. 다음에 이 인스턴스를 사용하고 싶을 때 이 섹션의 단계를 따라 저장된 이미지를 기반으로 인스턴스를 생성할 수 있습니다. 유일한 차이점은 :numref:`fig_ubuntu`에 표시된 "1. Choose AMI"에서 왼쪽의 "My AMIs" 옵션을 사용하여 저장된 이미지를 선택해야 한다는 것입니다. 생성된 인스턴스는 이미지 하드 디스크에 저장된 정보를 유지합니다. 예를 들어 CUDA 및 기타 런타임 환경을 다시 설치할 필요가 없습니다.



## 요약 (Summary)

* 우리는 자신의 컴퓨터를 사고 만들 필요 없이 필요에 따라 인스턴스를 시작하고 중지할 수 있습니다.
* GPU 지원 딥러닝 프레임워크를 사용하기 전에 CUDA를 설치해야 합니다.
* 포트 포워딩을 사용하여 원격 서버에서 Jupyter Notebook을 실행할 수 있습니다.



## 연습 문제 (Exercises)

1. 클라우드는 편의성을 제공하지만 저렴하지는 않습니다. 비용을 절감하는 방법을 알아보기 위해 [스팟 인스턴스(spot instances)](https://aws.amazon.com/ec2/spot/)를 시작하는 방법을 찾아보십시오.
2. 다양한 GPU 서버를 실험해 보십시오. 얼마나 빠릅니까?
3. 멀티 GPU 서버를 실험해 보십시오. 얼마나 잘 확장할 수 있습니까?


[Discussions](https://discuss.d2l.ai/t/423)