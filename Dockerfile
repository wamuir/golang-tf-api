ARG BAZEL_OPTS="--config=opt"
ARG BAZEL_VERS="3.7.0"
ARG PROTOBUF_VERS="3.14.0"


FROM golang:1.15-buster

ARG BAZEL_OPTS
ARG BAZEL_VERS
ARG PROTOBUF_VERS

RUN apt-get update && apt-get -y install \
    build-essential \
    libpython3-dev \
    openjdk-11-jdk-headless \
    python3 \
    python3-venv \
    swig


# install bazel
RUN curl -fLSs https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERS}/bazel-${BAZEL_VERS}-linux-`uname -m` \
    > /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

# install protoc
RUN curl -fLSs -o /opt/protoc.zip \
    https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERS}/protoc-${PROTOBUF_VERS}-linux-`uname -m`.zip \
    && python -m zipfile -e /opt/protoc.zip /usr/local \
    && chmod +x /usr/local/bin/protoc

# build venv and install numpy 
RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate && pip3 install --upgrade pip wheel && pip3 install numpy

# get tensorflow
RUN go get -d github.com/tensorflow/tensorflow/tensorflow/go || true

# build and install c lib for tensorflow
WORKDIR ${GOPATH}/src/github.com/tensorflow/tensorflow
RUN . /opt/venv/bin/activate && ./configure 
RUN . /opt/venv/bin/activate && bazel build ${BAZEL_OPTS} //tensorflow/tools/lib_package:libtensorflow.tar.gz
RUN tar xz -C /usr/local -f bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz

# link shared libs
RUN ldconfig

# generate and test 
RUN go generate github.com/tensorflow/tensorflow/tensorflow/go/op
RUN go test github.com/tensorflow/tensorflow/tensorflow/go

# get dependencies
RUN go get github.com/go-chi/chi
RUN go get github.com/stretchr/testify
RUN go get github.com/wamuir/go-jsonapi-core

# test and compile app
COPY . ${GOPATH}/src/github.com/wamuir/golang-tf-api
RUN CGO_ENABLED=1 GOOS=linux go test github.com/wamuir/golang-tf-api/classifier
RUN CGO_ENABLED=1 GOOS=linux go build -o /go/bin/golang-tf-api github.com/wamuir/golang-tf-api


FROM debian:buster-slim

# install c lib for tensorflow
COPY --from=0 /go/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz /opt/libtensorflow.tar.gz
RUN tar xz -C /usr/local -f /opt/libtensorflow.tar.gz && rm /opt/libtensorflow.tar.gz

# link shared libs
RUN ldconfig

# copy compiled app
COPY --from=0 /go/bin/golang-tf-api /usr/bin/golang-tf-api

ENTRYPOINT ["/usr/bin/golang-tf-api"]

EXPOSE 5000
