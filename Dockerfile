FROM python:3

RUN set -ex; \
    apt-get update -qq; \
    apt-get install -qqy zsh

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["./web"]
