FROM golang:1.15-buster

RUN mkdir -p /go/src/github.com/wamuir/golang-tf-api

# fetch tensorflow c libs
ENV LIB_TENSORFLOW_VERSION="2.3.0" \
    LIB_TENSORFLOW_SHA1SUM="0e401f0494914bbc0136f280d479fb7b4bd590c7"
RUN curl -fLSs -o /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz \
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz \
    && echo ${LIB_TENSORFLOW_SHA1SUM} /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz | sha1sum --check \
    && tar xz -C /usr/local -f /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz

# build golang package
WORKDIR /go/src/github.com/wamuir/golang-tf-api
ADD . /go/src/github.com/wamuir/golang-tf-api
RUN go mod download && go install github.com/wamuir/golang-tf-api

USER nobody
ENTRYPOINT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib /go/bin/golang-tf-api

EXPOSE 5000
