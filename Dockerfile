FROM python:3.6.5
ADD . /vision
WORKDIR /vision
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip -i ${PIP_MIRROR}\
    && pip install -r requirements.txt -i ${PIP_MIRROR}\
    && python rcnn_train.py
CMD ["python", "vision_server.py"]