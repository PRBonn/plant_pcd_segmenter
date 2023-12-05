ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# Install dependencies

RUN echo
RUN apt-get update
RUN apt-get install -y git openssh-client ffmpeg libsm6 libxext6
RUN apt-get install -y tmux
RUN apt-get clean


RUN apt-get -y install build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \ 
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        wget \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get purge -y imagemagick imagemagick-6-common 

RUN rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade pip

WORKDIR /packages
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install spconv-cu116

RUN pip install git+https://github.com/scikit-learn-contrib/hdbscan.git

RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

RUN pip3 install pyransac3d

RUN pip install tensorboardX

# Switch to same user as host system
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /packages/pcd_leaf_segmenter/src

ENTRYPOINT ["bash", "-c"]