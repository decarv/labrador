FROM ubuntu:latest
LABEL authors="decarv"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN mkdir -p /labrador

COPY . /labrador

WORKDIR /labrador

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "-m", "labrador.app.app"]