FROM golang:1.13-buster

ADD . /go/src/gitlab.nps.edu/wamuir1/golang-tf-api

# tensorflow c libs
RUN curl -s https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz | tar xz -C /usr/local

WORKDIR /go/src/gitlab.nps.edu/wamuir1/golang-tf-api
RUN go mod download && go install gitlab.nps.edu/wamuir1/golang-tf-api

ENTRYPOINT LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib /go/bin/golang-tf-api

EXPOSE 5000
