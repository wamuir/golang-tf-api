ARG PROTOBUF_VERS="3.14.0"
ARG TENSORFLOW_VERS="2.4.0"

FROM golang:1.15-buster

ARG PROTOBUF_VERS
ARG TENSORFLOW_VERS

# fetch and install protoc
RUN curl -fLSs -o /opt/protoc.zip \
    https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERS}/protoc-${PROTOBUF_VERS}-linux-`uname -m`.zip \
    && python -m zipfile -e /opt/protoc.zip /usr/local \
    && chmod +x /usr/local/bin/protoc

# fetch and install tensorflow c libs
RUN curl -fLSs -o /opt/libtensorflow.tar.gz \
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-`uname -m`-${TENSORFLOW_VERS}.tar.gz \
    && tar xz -C /usr/local -f /opt/libtensorflow.tar.gz

# link shared libs
RUN ldconfig

# fetch tensorflow source
RUN mkdir -p ${GOPATH}/src/github.com/tensorflow
RUN curl -fLSs -o /opt/tensorflow.tar.gz \
    https://github.com/tensorflow/tensorflow/archive/v${TENSORFLOW_VERS}.tar.gz \
    && tar -xz -C ${GOPATH}/src/github.com/tensorflow -f /opt/tensorflow.tar.gz
RUN mv ${GOPATH}/src/github.com/tensorflow/tensorflow-${TENSORFLOW_VERS} ${GOPATH}/src/github.com/tensorflow/tensorflow

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
COPY --from=0 /opt/libtensorflow.tar.gz /opt/libtensorflow.tar.gz
RUN tar xz -C /usr/local -f /opt/libtensorflow.tar.gz && rm /opt/libtensorflow.tar.gz

# link shared libs
RUN ldconfig

# install compiled app
COPY --from=0 /go/bin/golang-tf-api /usr/bin/golang-tf-api

ENTRYPOINT ["/usr/bin/golang-tf-api"]

EXPOSE 5000
