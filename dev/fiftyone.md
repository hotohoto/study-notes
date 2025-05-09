# FiftyOne

https://voxel51.com/docs/fiftyone/

## entities

- Dataset
  -
- Sample
  - (members)
    - id
    - media_type
    - tags
    - metadata
    - filepath
- Field
  - BooleanField
  - IntField
  - FloatField
  - StringField
  - DateField
  - DateTimeField
  - ListField
  - DictField
- DatasetView
- Label
  - Regression
  - Classification
  - Classifications
  - Detections
  - Polylines
  - Keypoints
  - Segmentation
  - Heatmap
  - TemporalDetection
  - GeoLocation

## Scripts

```Dockerfile
FROM python:3.8

RUN pip install fiftyone
EXPOSE 5151
EXPOSE 27017

COPY start.sh start.sh

ENV FIFTYONE_DATABASE_URI=mongodb://localhost

CMD ./start.sh
```

`start.sh`

```bash
#!/bin/bash
/usr/local/lib/python3.8/site-packages/fiftyone/db/bin/mongod \
        --dbpath=/root/.fiftyone/var/lib/mongo \
        --logpath /root/.fiftyone/var/lib/mongo/log/mongo.log &
while ! timeout 1 bash -c "echo > /dev/tcp/localhost/27017"; do echo Waiting... && sleep 3; done
echo Starting...
fiftyone app launch --address=0.0.0.0
```

```bash
# build docker image
docker build -t fiftyone:latest .

# run a docker container
docker run -d \
 --net=host \
 -v /data/hyheo/.fiftyone:/root/.fiftyone \
 -v /data:/data \
 --name hyheo-fiftyone \
 fiftyone:latest

# start a fiftyone session in the host machine
FIFTYONE_DATABASE_URI=mongodb://localhost python my_script.py
```
