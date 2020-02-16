FROM golang:1.13-buster

RUN mkdir -p /go/src/gitlab.nps.edu/wamuir1/golang-tf-api

# fetch tensorflow c libs
ENV LIB_TENSORFLOW_VERSION="1.15.0" \
    LIB_TENSORFLOW_SHA1SUM="499b2465a6c37c89631b8ceb3d96e10c143b5d3b"
RUN curl -fLSs -o /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz \
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz \
    && echo ${LIB_TENSORFLOW_SHA1SUM} /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz | sha1sum --check \
    && tar xz -C /usr/local -f /tmp/libtensorflow-cpu-linux-x86_64-${LIB_TENSORFLOW_VERSION}.tar.gz

# fetch exported tensorflow model
ENV TENSORFLOW_CNN_VERSION="y9j79l5lmdbnb5swp750n2mrvd52cz5o" \
    TENSORFLOW_CNN_SHA1SUM="f2e71cc944702bc24dbd10a284a91e2f002861c0"
RUN curl -fLSs -o /tmp/tensorflow-cnn-${TENSORFLOW_CNN_VERSION}.tar.gz \
    https://nps.box.com/shared/static/${TENSORFLOW_CNN_VERSION}.gz \
    && echo ${TENSORFLOW_CNN_SHA1SUM} /tmp/tensorflow-cnn-${TENSORFLOW_CNN_VERSION}.tar.gz | sha1sum --check \
    && tar xz -C /go/src/gitlab.nps.edu/wamuir1/golang-tf-api -f /tmp/tensorflow-cnn-${TENSORFLOW_CNN_VERSION}.tar.gz

# build golang package
WORKDIR /go/src/gitlab.nps.edu/wamuir1/golang-tf-api
ADD . /go/src/gitlab.nps.edu/wamuir1/golang-tf-api
RUN go mod download && go install gitlab.nps.edu/wamuir1/golang-tf-api

USER nobody
ENTRYPOINT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib /go/bin/golang-tf-api

EXPOSE 5000
