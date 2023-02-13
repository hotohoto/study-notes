# Google Cloud Platform

## TODO

try
- IM
- VM
- big query
  - browse public tables
- cloud composer (aiarflow)

## notes

- VM
  - ssh 키 등록해야한다.
    - public key는 GCP의 설정 페이지에 복사해서 붙여넣고
    - private key는 로컬에 가지고 있는다.
- service account key
  - crediential 페이지에서 만든다.
  - role을 owner로 하면 모든 서비스에 다 접근할 수 있다.
  - json 파일로 다운로드 받아서, 서비스를 실행할 곳에 넣어둔다.
  - 개별 API 서비스 종류 별로 enable 눌러 줘야 해당 서비스를 이용할 수 있다.
- python에서 API
  - google-cloud 패키지를 설치하면 모든 서비스에 접근 가능하다.
  - 특정 서비스 FOO 에만 접근할 거면 google-cloud-foo 를 설치하면 된다.

## GKE

```bash
gcloud components install kubectl
```

```bash
gcloud container clusters create-auto my-cluster --region=asia-northeast3  # create a new cluster
```

```bash
gcloud auth list
gcloud auth login --no-browser
gcloud config set account hotohoto.biz@gmail.com
gcloud config set project resolute-tracer-354107  # refer to your projects have already been created
gcloud config set compute/zone us-central1-a
gcloud config set compute/region us-central1
gcloud container clusters get-credentials my-cluster  # update kubectl configurations
gcloud container clusters resize my-cluster --node-pool=default-pool --num-nodes=10
```

```bash
kubectl create deployment hello-server --image=us-docker.pkg.dev/google-samples/containers/gke/hello-app:1.0
kubectl expose deployment hello-server --type LoadBalancer --port 80 --target-port 8080
kubectl get service hello-server
gcloud container clusters delete my-cluster --region=asia-northeast3
```

- region
  - asia-northeast3
- zone
  - asia-northeast3-c

### How to set up persistent volume

- https://devopscube.com/persistent-volume-google-kubernetes-engine/

```bash
gcloud container clusters update my-cluster --update-addons=GcpFilestoreCsiDriver=ENABLED
```

### cluster types

- default
  - more configurable
- autopilot
  - less configurable
  - not allowed to use the privileged mode
    - allowed to use only a part of capabilities
    - seems not possible to set up the pod for a custom nfs server
  - might be better for production rather than development
