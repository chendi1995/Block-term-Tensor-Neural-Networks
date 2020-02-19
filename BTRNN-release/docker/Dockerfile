FROM nvcr.io/nvidia/tensorflow:18.01-py3

RUN apt-get update && apt-get -y install --no-install-recommends vim tmux zsh \
    openssh-server wget unzip gdb libsm6 libxext6 libfontconfig1 libxrender1 ffpmeg

RUN python3 -m pip install --user --upgrade pip

ADD requirements.txt /requirements.txt
RUN pip3 install cython; pip3 install -r /requirements.txt


RUN sed 's@PermitRootLogin yes@PermitRootLogin prohibit-password@g' -i /etc/ssh/sshd_config


# for remote debug
ARG authorized_keys
RUN mkdir -p /root/.ssh/ && echo "$authorized_keys" > /root/.ssh/authorized_keys

