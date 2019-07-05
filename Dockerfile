FROM ubuntu:18.04

RUN mkdir /data

ENV FEATURE_DIR=/data/features \
    BOW_DIR=/data/bow \
    TOPICS_DIR=/data/topics \
    VIZ_DIR=/data/visualisations

RUN apt-get update \
  && apt-get install -y dialog apt-utils curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://get.docker.com/ | sh 

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && apt-get remove -y .*-doc .*-man >/dev/null \
  && apt-get autoremove -y \ 
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt package/
COPY setup.py package/
COPY README.md package/
COPY tmexp package/tmexp/

RUN pip3 install --no-cache-dir -r package/requirements.txt
RUN pip3 install --no-cache-dir  package/.
ENTRYPOINT ["tmexp"]
