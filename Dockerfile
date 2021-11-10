FROM continuumio/anaconda3:latest

RUN apt-get update
RUN apt-get upgrade -y

RUN apt install unzip
RUN apt install wget

RUN python -m pip install --upgrade pip
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install gdown

WORKDIR /mnt/work
CMD ["/bin/bash"]