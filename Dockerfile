# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

LABEL symai.image.authors="extensity.ai" \
	symai.image.vendor="ExtensityAI" \
	symai.image.title="SymbolicAI" \
	symai.image.description="SymbolicAI: Compositional Differentiable Programming Library" \
	symai.image.source="https://github.com/ExtensityAI/symbolicai.git" \
	symai.image.revision="${VCS_REF}" \
	symai.image.created="${BUILD_DATE}" \
	symai.image.documentation="https://symbolicai.readthedocs.io/en/latest/README.html"
LABEL symai.dependencies.versions.torch="2.0.1"
LABEL symai.dependencies.versions.cuda="11.7"
ARG DEBIAN_FRONTEND=noninteractive

# NVIDIA key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# Update the base image
RUN apt update && apt upgrade -y

# Install symai

## Install dependencies
RUN apt install -y curl sudo nano git htop netcat wget unzip tmux apt-utils cmake build-essential libgl1-mesa-glx libglib2.0-0

## Upgrade pip
RUN pip3 install --upgrade pip

# Install nvm and pm2
RUN curl -o install_nvm.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh && \
	echo 'fabc489b39a5e9c999c7cab4d281cdbbcbad10ec2f8b9a7f7144ad701b6bfdc7 install_nvm.sh' | sha256sum --check && \
	bash install_nvm.sh

RUN bash -c "source $HOME/.nvm/nvm.sh && \
    # use node 16
    nvm install 16 && \
    # install pm2
    npm install --location=global pm2"

RUN mkdir -p /root/.symai/install
COPY . /root/.symai/install
RUN cd /root/.symai/install && python3 -m pip install ".[all]"

# Increase ulimit to 1,000,000
RUN prlimit --pid=$PPID --nofile=1000000

EXPOSE 8999
