FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

CMD ["bash"]