FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install wget -y
RUN apt-get install -y git

# install pip
RUN apt install -y python3-pip
RUN apt-get install -y python3-setuptools
RUN pip install --upgrade pip
RUN apt install unzip

COPY requirements.txt . 

# install python package
RUN pip install -r requirements.txt
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch --y

RUN pip install gdown

WORKDIR /mnt/work

RUN echo 'build is completed'

