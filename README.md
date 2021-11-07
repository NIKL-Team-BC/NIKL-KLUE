# NIKL-GLUE

## pretrained model & data  
need extract inside each task folder  
boolq : https://drive.google.com/file/d/1k-6W3bTqFVlSBBtsJ42kPcBdBCO7bQkD/view?usp=sharing  
copa : https://drive.google.com/file/d/1PxwwOiYxKb7PUBVByX0LYwnXm9IY6lfh/view?usp=sharing  
cola : https://drive.google.com/file/d/1DJDch9YquGUrf9GQBwD64X0ecrSAdUbS/view?usp=sharing  
wic : https://drive.google.com/file/d/1DJDch9YquGUrf9GSEJR4X0ecrSAdUbS/view?usp=sharing 

## Docker inference  
1. make docker image  
docker build -t inference-docker -f Dockerfile .  
2. run docker container  
docker run -v "$PWD:/mnt/work" -it inference-docker  
3. do inference  
sh all_inference.sh  
