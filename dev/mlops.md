https://fastcampus.co.kr/data_online_mlops/
https://fastcampus.co.kr/courses/207488/
https://arxiv.org/abs/2205.02302

## 1. intro

### softwares for each MLOps stack

- data aggregation / pipeline
  - Sqoop
  - Flume
  - Kafka
  - Flink
  - Spark Streaming
  - Airflow
- data storages
  - MySQL
  - Hadoop
  - Amazon S3
  - MinIO
- data management
  - TFDV
  - DVC
  - Feast
  - Amundsen
- Model development
  - Jupyter Hub
  - Docker
  - Kubeflow
  - Optuna
  - Ray
  - katib
- Model version management
  - Git
  - MLflow
  - Github Action
  - Jenkins
- Model training schedule management
  - Grafana
  - Kubernetes
- Model serving
  - Docker
  - Flask
  - FastAPI
  - BentoML
  - Kubeflow
  - TFServing
  - seldon-core
- Model monitoring
  - Prometheus
  - Grafana
  - Thanos
- Pipeline management
  - Kueflow
  - argo workflows
  - Airflow
- All-in-onw
  - AWS SageMaker
  - GCP Vertex AI
  - Azure Machine Learning

## 2. docker and k8s

### docker

- `CMD`
  - the entire command can be replace by another command
- `ENTRYPOINT`
  - by default is not replaced, but the arguments will be used as just extra arguments
  - with `--entrypoint` the command can be replaced
- both `CMD` and `ENTRYPOINT` might be used when we want to provide some default arguments
- `COPY  <src> <dest>`
  - preferred if `ADD` is not required
- `ADD <src> <dest>`
  - does more than `COPY`
  - `<src>` can be URL or a zip file

(How to launch and use a local Docker registry.)

```bash
# Launch a local Docker registry
docker run -d -p 5000:5000 --name registry registry

# Build an image
vi Dockerfile
dokcer build -t myimage:v1.0.0 .

# Tag and push it
docker tag myimage:v1.0.0 localhost:5000/myimage:v1.0.0
docker push localhost:5000/my-mage:v1.0.0

# Validation
curl http://localhost:5000/v2/_catalog
curl http://localhost:5000/v2/myimage/tags/list
```

### YAML

https://rigorous-firefly-d50.notion.site/1-YAML-bdf97d4b9f814acd9b5d3e55ae30ba9f

### Kubernetes

[k8s](./kubernetes.md)

## 3. mlops components

- data and model management
  - [dvc](https://dvc.org/)
  - mlflow
- model serving
  - Flask
  - [Seldon Core](https://github.com/SeldonIO/seldon-core/)
- model monitoring
   - Prometheus
   - Grafana

## 4. mlops with k8s

- automation and moredel research
  - k8s
  - k8s + Katib
- feature store
  - [Feast](https://feast.dev/)
  - Feast server
  - Store and [Minio](https://min.io/)
- CI/CD
  - github action
  - DVC [CML](https://cml.dev/) model metric tracking
  - jenkins
    - CI pipeline with jenkinsfile
- pipeline monitoring
  - FastAPI serving API
  - FastAPI-Prometheus metric gathering
  - Prometheus + Grafana
  - create simulation with Locust
  - ml monitoring with jenkins

## 5. mlops on various cloud platforms

- Amazon SageMaker
  - AutoPilot
- Azure MLOps
  - GitHub action + FastAPI app
- GCP MLOps
  - Feast Feature Store + GCP
  - Feast FastAPI App
- Private Cloud MLOps
  - Private Cloud MLOps
  - Nexus + private docker registry

## 6. Future works

## References

- [머신러닝 서비스 구축을 위한 실전 MLOps 올인원 패키지 Online.](https://fastcampus.co.kr/courses/207488)
  - [MLOps 실습자료](https://sour-source-3a5.notion.site/MLOps-b5c20da66b5a407b83e8097d82329f98)
  - [MLOps 실습자료](https://rigorous-firefly-d50.notion.site/MLOps-486a7bcd320b4e9f93a70b5691b88dd1)

## extra notes

- hydra: configuration management

### tips

- what if there is no feature store?
  - https://towardsdatascience.com/mlops-the-role-of-feature-stores-d30108dedf0
    - duplicate work in creating features again and again
    - may faile to conceive important features which may have been done by senior members earlier
- data scientists need to know better
