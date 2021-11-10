# NIKL-GLUE

## pretrained model & data  
need extract inside each task folder  
boolq : https://drive.google.com/file/d/1OZaN9qVNFlEbk7GIpaoREcEObLttibFk/view?usp=sharing  
copa : https://drive.google.com/file/d/1QB66EebwWP2XhgZFweG00NIu3QIBZ-hh/view?usp=sharing  
cola : https://drive.google.com/file/d/1SqFO4E2M1qIIJHusubUb5r2dklLdvlME/view?usp=sharing  
wic : https://drive.google.com/file/d/1DUaUTTl-YAwhZQmTHaLVHsPmA64dyQ75/view?usp=sharing

## Docker inference-gpu
1. make docker image  
docker build -t inference-docker -f Dockerfile .  
2. run docker container  
docker run --gpus all -v "$PWD:/mnt/work" -it inference-docker  
3. do inference  
sh all_inference.sh  

## Presentation PDF
https://github.com/NIKL-Team-BC/NIKL-KLUE/blob/main/2021_NIKL_Team_BC.pdf
