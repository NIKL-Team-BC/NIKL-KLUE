FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install wget -y
RUN apt-get install -y git

# install pip
RUN apt install -y python3-pip
RUN apt-get install -y python3-setuptools
RUN pip install --upgrade pip

COPY requirements.txt . 

# install python package
RUN pip install -r requirements.txt
WORKDIR /mnt/work

RUN echo 'build is completed'