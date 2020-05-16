# Build stage image to install python
FROM nvidia/cuda:10.0-cudnn7-runtime as builder

ENV DEBIAN_FRONTEND=noninteractive

# Prerequisites: https://github.com/pyenv/pyenv/wiki/common-build-problems#prerequisites
RUN apt-get update -y
RUN apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN apt-get clean

# Install Pyenv
RUN git clone https://github.com/pyenv/pyenv /opt/pyenv

ENV PYENV_ROOT=/opt/pyenv \
    PATH=${PYENV_ROOT}/bin:${PATH}

# Install Python 3.6.8
RUN /opt/pyenv/bin/pyenv install 3.6.8 \
    && /opt/pyenv/versions/3.6.8/bin/python -m pip install --upgrade pip setuptools wheel

# Runtime image
FROM nvidia/cuda:10.0-cudnn7-runtime

COPY --from=builder /opt/pyenv/versions/ /opt/pyenv/versions

ENV DEBIAN_FRONTEND=noninteractive

ENV PATH=/opt/pyenv/versions/3.6.8/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/pyenv/versions/$python_version/lib

RUN apt-get update && apt-get install -y gcc g++ && apt-get clean

COPY setup.py .
COPY tape ./tape

# Install dependencies
RUN /opt/pyenv/versions/3.6.8/bin/python -m pip install --no-cache-dir -e .

CMD ["/opt/pyenv/versions/3.6.8/bin/python", "tape/server.py"]
