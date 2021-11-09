# NIKL-GLUE

## pretrained model & data  
need extract inside each task folder  
boolq : https://drive.google.com/file/d/1k-6W3bTqFVlSBBtsJ42kPcBdBCO7bQkD/view?usp=sharing  
copa : https://drive.google.com/file/d/1PxwwOiYxKb7PUBVByX0LYwnXm9IY6lfh/view?usp=sharing  
cola : https://drive.google.com/file/d/1d-eIMrLZSxreeiE-vcO6lX5hjF70c6zS/view?usp=sharing  
wic : https://drive.google.com/file/d/1DUaUTTl-YAwhZQmTHaLVHsPmA64dyQ75/view?usp=sharing

## Docker inference  
If you have a gpu, run it on a gpu-inference branch and see the branch gpu-inference readme  
1. make docker image  
docker build -t inference-docker -f Dockerfile .  
2. run docker container  
docker run -v "$PWD:/mnt/work" -it inference-docker  
3. do inference  
sh all_inference.sh  

## 평가방법
### CPU inference권장

### 기존 테스트셋으로 inference를 할 때
1. 본 repository를 clone한다

2. 도커 이미지를 빌드한다.
 빌드 명령어: docker build -t inference-docker -f Dockerfile .

3. 도커 이미지를 run한다
 run 명령어: docker run -v "$PWD:/mnt/work" -it inference-docker

4. 기존 테스트셋(대회에서 주어진 testset)으로 inference를 할 때에는 all_inference.sh를 실행한다.
 실행 명령어: sh all_inference.sh

### 새로운 테스트셋(평가set)으로 inference할 때

1. 본 repository를 clone한다

2. 도커를 빌드한다 ///****주의**** -> 재빌드 필수, 기존 도커로 inference 실행불가///
 빌드 명령어: docker build -t inference-docker -f Dockerfile .

3. 도커 이미지를 run한다
 run 명령어: docker run -v "$PWD:/mnt/work" -it inference-docker

4. 새로운 테스트셋(대회에서 주어진 testset)으로 inference를 할 때에는 데이터를 new_test 디렉토리에 다음과 같이 넣어준다
 new_test/BoolQ.tsv
 new_test/CoLA.tsv
 new_test/WiC.tsv
 new_test/CoPA.tsv
주의: 반드시 new_test 디렉토리 내부에 파일 이름을 위와 동일하게 할것

5. new_inference.sh를 실행한다.
 실행 명령어: sh new_inference.sh

## Presentation PDF
https://github.com/NIKL-Team-BC/NIKL-KLUE/blob/main/2021_NIKL_Team_BC.pdf
