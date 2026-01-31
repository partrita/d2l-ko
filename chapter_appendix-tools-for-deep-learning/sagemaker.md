# Amazon SageMaker 사용하기 (Using Amazon SageMaker)
:label:`sec_sagemaker`

Amazon SageMaker는 기계 학습 모델을 신속하게 구축, 학습 및 배포할 수 있게 해주는 완전 관리형 서비스입니다. 
이 섹션에서는 SageMaker를 사용하여 이 책의 코드를 실행하는 방법을 설명합니다.

## 인스턴스 설정 (Setting Up an Instance)

1. AWS 계정에 로그인하고 SageMaker 콘솔로 이동합니다.
2. "Notebook" -> "Notebook instances"를 선택하고 "Create notebook instance"를 클릭합니다.
3. 인스턴스 이름과 유형(예: `ml.t2.medium` 또는 GPU가 필요한 경우 `ml.p2.xlarge`)을 지정합니다.
4. IAM 역할을 생성하거나 기존 역할을 선택하여 필요한 권한을 부여합니다.
5. 인스턴스가 생성되면 "Open Jupyter" 또는 "Open JupyterLab"을 클릭하여 시작합니다.

## 코드 가져오기 및 실행 (Fetching and Running the Code)

Jupyter 터미널을 열고 다음 명령을 실행하여 이 책의 저장소를 클론합니다.

```bash
git clone https://github.com/d2l-ai/d2l-en.git
```

이제 폴더를 탐색하여 원하는 노트북을 열고 실행할 수 있습니다. 
SageMaker는 딥러닝 프레임워크가 사전 설치된 다양한 커널을 제공하므로, 필요에 따라 적절한 커널(예: `conda_mxnet_p36`, `conda_pytorch_p36` 등)을 선택하십시오.

## 인스턴스 중지 (Stopping the Instance)

사용이 끝나면 비용 발생을 방지하기 위해 반드시 인스턴스를 중지하십시오. 
콘솔에서 해당 인스턴스를 선택하고 "Actions" -> "Stop"을 클릭하면 됩니다. 
중지된 상태에서는 데이터는 유지되지만 컴퓨팅 비용은 청구되지 않습니다.