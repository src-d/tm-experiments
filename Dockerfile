FROM ubuntu:18.04

RUN mkdir /data /package

ARG DEBIAN_FRONTEND=noninteractive 

ENV BBLFSH_HOSTNAME tmexp_bblfshd
ENV BBLFSH_PORT 9432
ENV GITBASE_HOSTNAME tmexp_gitbase
ENV GITBASE_PORT 3306
ENV GITBASE_USERNAME root
ENV GITBASE_PASSWORD ""
ENV ARTM_SHARED_LIBRARY /usr/local/lib/libartm.so

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  apt-utils \
  ca-certificates \
  curl \
  locales \
  && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
  && locale-gen \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://get.docker.com/ | sh \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  libboost-chrono-dev \
  libboost-date-time-dev \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-iostreams-dev \
  libboost-program-options-dev \
  libboost-system-dev \
  libboost-thread-dev \
  libboost-timer-dev \
  make \
  python3-dev \
  python3-pip \
  && ln -s /usr/bin/python3 /usr/local/bin/python \
  && pip3 install numpy scipy pandas protobuf tqdm wheel \
  && git clone --branch v0.10.0 https://github.com/bigartm/bigartm.git /opt/bigartm \
  && mkdir /opt/bigartm/build \
  && cd /opt/bigartm/build \
  && cmake -DINSTALL_PYTHON_PACKAGE=ON -DPYTHON=python3 .. \
  && make -j$(getconf _NPROCESSORS_ONLN) \
  && make install \
  && rm -rf /usr/share/doc /usr/share/man \
  && apt-get autoremove --purge -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /package

RUN pip3 install --no-cache-dir -r /package/requirements.txt

COPY setup.py /package
COPY README.md /package
COPY tmexp /package/tmexp

RUN pip3 install --no-cache-dir /package/.

ENTRYPOINT ["tmexp"]
