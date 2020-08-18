# Pytorch 1.6.0 + cuda 10.1 설치
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# 필요 라이브러리 설치
# RUN pip install ~

# 소스 복사
# work directory: /workspace
COPY ./data_loader/data_loader.py ./data_loader/
COPY ./model/models.py ./model/
COPY train.py ./