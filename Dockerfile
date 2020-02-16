FROM golang:1.13-buster

ADD . /go/src/gitlab.nps.edu/wamuir1/golang-tf-api

# fetch tensorflow c libs
RUN curl -Ls https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz | tar xz -C /usr/local

# fetch exported tensorflow model
RUN curl -Ls /tmp/charCNN.tar.gz https://nps.box.com/shared/static/y9j79l5lmdbnb5swp750n2mrvd52cz5o.gz | tar xz -C /go/src/gitlab.nps.edu/wamuir1/golang-tf-api

# build
WORKDIR /go/src/gitlab.nps.edu/wamuir1/golang-tf-api
RUN go mod download && go install gitlab.nps.edu/wamuir1/golang-tf-api

ENTRYPOINT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib /go/bin/golang-tf-api

EXPOSE 5000
