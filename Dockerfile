FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libsm6 libxrender1 libfontconfig1 \
    ffmpeg \
    htop \
    libopenblas-dev \
    libomp-dev \
    tmux \
    python3-pip
# COPY . /py-bottom-up-attention
# RUN echo ""
RUN git clone https://github.com/HimariO/py-bottom-up-attention.git \
    && cd py-bottom-up-attention \
    && git checkout package \
    && cd ..
RUN cd py-bottom-up-attention \
    && pip3 install --upgrade pip \
    && pip3 install scikit-build loguru imgaug albumentations \
    && pip3 install -r requirements.txt \
    && pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && python3 setup.py build develop
WORKDIR /py-bottom-up-attention/demo