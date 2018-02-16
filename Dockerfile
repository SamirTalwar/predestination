FROM continuumio/anaconda

RUN set -ex; \
    apt-get update -qq; \
    apt-get install -qqy zsh

WORKDIR /app

ENV PATH /opt/conda/envs/app/bin:$PATH
COPY environment.yml ./
RUN conda env create --name=app

COPY . ./

CMD ["./web"]
