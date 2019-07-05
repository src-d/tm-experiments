BBLFSHD_NAME ?= tmexp_bblfshd
BBLFSHD_PORT ?= 9432
GITBASE_NAME ?= gitbase
GITBASE_PORT ?= 3306
REPOS ?= repos
RESOLVED_REPOS != readlink --canonicalize-missing $(REPOS)

check:
	black --check tmexp setup.py
	mypy tmexp setup.py
	flake8 --count tmexp setup.py

start: bblfsh-start gitbase-start

bblfsh-start:
	if ! docker ps | grep $(BBLFSHD_NAME); then\
		docker run \
			--detach \
			--rm \
			--name $(BBLFSHD_NAME) \
			--privileged \
			--publish $(BBLFSHD_PORT)\:9432 \
			bblfsh/bblfshd:v2.14.0-drivers \
			--log-level DEBUG;\
	fi

gitbase-start:
	if ! docker ps | grep $(GITBASE_NAME); then\
		docker run \
			--detach \
			--rm \
			--name $(GITBASE_NAME) \
			--publish $(GITBASE_PORT):3306 \
			--link $(BBLFSHD_NAME):$(BBLFSHD_NAME) \
			--env BBLFSH_ENDPOINT=$(BBLFSHD_NAME):$(BBLFSHD_PORT) \
			--volume $(RESOLVED_REPOS):/opt/repos \
			srcd/gitbase:v0.22.0;\
	fi

.PHONY: check start bblfsh-start gitbase-start
