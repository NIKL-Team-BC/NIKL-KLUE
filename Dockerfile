# FROM conda/miniconda3
# VOLUME ["/mnt/c/Users/mello/workspace/NIKL/NIKL-KLUE" "/root/work"]

# WORKDIR /root/work

# COPY conda_requirements.txt .
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # RUN conda create --name test python=3.8 -y
# # RUN conda env create -f conda_requirements.txt

# RUN python submit_copa/inference.py


FROM jupyter/scipy-notebook
# FROM conda/miniconda3

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY Copa/inference.py ./Copa/inference.py

# RUN python3 Copa/inference.py
# VOLUME [".", "/mnt/work"]
WORKDIR /mnt/work

CMD ["/bin/bash"]