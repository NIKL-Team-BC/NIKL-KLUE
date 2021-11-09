FROM jupyter/scipy-notebook

RUN apt-get update
RUN apt-get upgrade -y

RUN apt install unzip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install gdown

WORKDIR /mnt/work

CMD ["/bin/bash"]
