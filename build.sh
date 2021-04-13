#!/bin/sh

docker build -t cancer-instance-train .
docker tag cancer-instance-train carlduke/cancer-instance-train
docker push carlduke/cancer-instance-train