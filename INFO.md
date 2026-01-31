# 빌드 (Building)

## 개발자를 위한 설치

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh  # py3.8의 경우, wget  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b  # py3.8의 경우: sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
. ~/.bashrc
conda create --name d2l python=3.9 -y  # py3.8의 경우: conda create --name d2l python=3.8 -y
conda activate d2l
pip install torch torchvision
pip install d2lbook
git clone https://github.com/d2l-ai/d2l-en.git
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >> ~/.jupyter/jupyter_notebook_config.py
cd d2l-en
pip install -e .  # 소스에서 d2l 라이브러리 설치
jupyter notebook
```

선택 사항: `jupyter_contrib_nbextensions` 사용

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# jupyter nbextension enable execute_time/ExecuteTime
```



## 평가 없이 빌드하기

`config.ini`에서 `eval_notebook = True`를 `eval_notebook = False`로 변경합니다.


## PDF 빌드하기

```
# d2lbook 설치
pip install git+https://github.com/d2l-ai/d2l-book

sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
sudo apt-get install pandoc  # 작동하지 않으면 conda install pandoc

# d2l 임포트를 위해
cd d2l-en
pip install -e .

# PDF 빌드
d2lbook build pdf
```

### PDF용 폰트

```
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh
```


## HTML 빌드하기

```
bash static/build_html.sh
```

## 폰트 설치

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

wget -O Inconsolata.zip https://www.fontsquirrel.com/fonts/download/Inconsolata
unzip Inconsolata -d Inconsolata
sudo mv Inconsolata /usr/share/fonts/opentype/

sudo fc-cache -f -v

```

## 릴리스 체크리스트

### d2l-en

- d2lbook 릴리스
- [선택 사항, 하드카피 도서 또는 파트너 제품 전용]
    - [setup.py](http://setup.py) → requirements 및 static/build.yml에서 라이브러리 버전 고정 (d2lbook 포함)
    - 재평가(re-evaluate)
    - 설치 과정에서 d2l 버전 고정 (아래 pypi에 표시됨)
- d2l.xxx에 대한 docstring 추가
- 프론트페이지 공지 업데이트
- (메이저 전용) 본문에서 수정할 사항이 있는지 확인하기 위해 wa 0.8.0 확인
- d2lbook build lib
- 무작위 colab 테스트
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-en-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- 릴리스 PR 평가
- bahdanau 및 transformer에서 어텐션 무작위성 고정 확인
- config.ini와 build.yml 간의 라이브러리(예: sagemaker 아래) 버전 일관성 확인
- config.ini & d2l/__init__.py의 버전 번호 및 installation.md의 d2l 버전 수정
- 개별 커밋을 유지하며 master를 release로 병합 (머지 커밋 생성)
- git checkout master
- rr -rf d2l.egg-info dist
- d2l을 pypi에 업로드 (팀 계정)
- colab 및 d2l 재테스트
- 릴리스 브랜치에서 git tag 생성
- git checkout master
- 브랜치에서 README 최신 버전 업데이트 후, 복원을 위해 squash merge
- [선택 사항] CloudFront 캐시 무효화
- [선택 사항, 하드카피 도서 전용]
    - config.ini: other_file_s3urls
- [선택 사항, 하드카피 도서 또는 파트너 제품 전용]
    - [setup.py](http://setup.py) → requirements 버전 복원
 
### d2l-zh

- 프론트페이지 공지 업데이트
- (필요 여부 확인) d2lbook build lib
- 무작위 colab 테스트
- static/build.yml을 d2l-en 버전으로 업그레이드
- [http://ci.d2l.ai/computer/(master)/script](http://ci.d2l.ai/computer/(master)/script)
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-zh-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- 릴리스 PR 평가 (bahdanau 및 transformer의 어텐션 무작위성 고정)
- config.ini와 build.yml 간의 라이브러리 버전 일관성 확인
- config.ini & d2l/__init__.py의 버전 번호 수정
- 개별 커밋을 유지하며 master를 release로 병합 (머지 커밋 생성)
- 재테스트 colab
- 릴리스 브랜치에서 git tag 생성
- git checkout master
- 브랜치에서 README 최신 버전 업데이트 후, 복원을 위해 squash merge
- 2.0.0 릴리스 추가 사항
    - s3 콘솔에서
        - [zh-v2.d2l.ai](http://zh-v2.d2l.ai) bucket/d2l-zh.zip을 d2l-webdata bucket/d2l-zh.zip으로 복사
        - d2l-webdata bucket/d2l-zh.zip을 d2l-webdata bucket/d2l-zh-2.0.0.zip으로 이름 변경
        - d2l-zh/release용 CI를 실행하여 config의 other_file_s3urls 트리거
        - 설치 테스트를 위해 cloudfront 캐시 무효화
    - 설치 테스트