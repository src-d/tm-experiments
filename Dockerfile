FROM ubuntu:18.04

RUN mkdir /data

ENV BBLFSH_HOSTNAME=tmexp_bblfshd \
  BBLFSH_PORT=9432 \
  GITBASE_HOSTNAME=gitbase \
  GITBASE_PORT=3306 \
  GITBASE_USERNAME=root \
  GITBASE_PASSWORD=

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
  && apt-get autoremove --purge -y \ 
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt package/

RUN pip3 install --no-cache-dir -r package/requirements.txt

COPY setup.py package/
COPY README.md package/
COPY tmexp package/tmexp/

RUN pip3 install --no-cache-dir  package/.
ENTRYPOINT ["tmexp"]