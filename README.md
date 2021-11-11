# NIKL-GLUE

## 평가방법
### CPU inference를 권장드립니다

### 기존 테스트셋으로 inference를 할 때
1. 본 repository를 clone합니다

2. 도커 이미지를 빌드합니다
 빌드 명령어: docker build -t inference-docker -f Dockerfile .  

3. 도커 이미지를 run합니다
 run 명령어: docker run -v "$PWD:/mnt/work" -it inference-docker  

4. 기존 테스트셋(대회에서 주어진 testset)으로 inference를 할 때에는 all_inference.sh를 실행합니다 
 실행 명령어: sh all_inference.sh  

### 새로운 테스트셋(평가set)으로 inference할 때  

1. 본 repository를 clone합니다

2. 도커를 빌드합니다 **주의사항 : 다시 재빌드 하지 않고 실행시 오류가 발생할 수 있습니다 재 빌드 해주세요!!**   
 빌드 명령어: docker build -t inference-docker -f Dockerfile .  

3. 도커 이미지를 run합니다
 run 명령어: docker run -v "$PWD:/mnt/work" -it inference-docker  

4. 새로운 테스트셋(새로운 평가 testset)으로 inference를 할 때에는 데이터를 new_test 디렉토리에 다음과 같이 넣어줍니다

   new_test/BoolQ.tsv  
   new_test/CoLA.tsv  
   new_test/WiC.tsv  
   new_test/CoPA.tsv  
 
   **주의: 반드시 new_test 디렉토리 내부에서 위에 명시된 파일 명과 같이 추가해주세요!**

5. new_inference.sh를 실행한다.  
 실행 명령어: sh new_inference.sh  

## Presentation PDF
https://github.com/NIKL-Team-BC/NIKL-KLUE/blob/main/2021_NIKL_Team_BC.pdf  
