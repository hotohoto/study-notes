# GitLab

## Setup with docker-compose

- https://docs.gitlab.com/install/docker/installation/

```yaml
# run with docker registry
gitlab:
    image: 'gitlab/gitlab-ce:latest'
    container_name: gitlab
    restart: always
    hostname: 'gitlab.example.com'
    environment:
      GITLAB_OMNIBUS_CONFIG: |
        external_url 'http://gitlab.example.com:8000'
        nginx['listen_port'] = 8000

        gitlab_rails['registry_enabled'] = true
        registry_external_url 'http://gitlab.example.com:5001'
    ports:
      - '8000:8000'
      - '5001:5001'
    volumes:
      - '/path/to/data/gitlab/config:/etc/gitlab'
      - '/path/to/data/gitlab/logs:/var/log/gitlab'
      - '/path/to/data/gitlab/data:/var/opt/gitlab'
    shm_size: '256m'
```

```bash
docker compose up -d

# change root password
docker exec -it gitlab bash
cd /etc/gitlab
gitlab-rake "gitlab:password:reset[root]"
```

## Setup with Kubernetes

- install kind, kubectl
- run `reset.sh`

`reset.sh`:

```bash
#!/bin/bash

function run {
  echo "$@"
  time eval $(printf '%q ' "$@")
}

run kind delete cluster
run kind create cluster --config kind-config.yaml || exit 1

run kubectl apply -f gitlab-pv.yaml || exit 1

run kubectl apply -f gitlab-deployment.yaml
run kubectl apply -f gitlab-service.yaml

echo "Done"
```

`kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  - hostPath: /data/gitlab
    containerPath: /data/gitlab
  extraPortMappings:
  - hostPort: 8000
    containerPort: 32000
    listenAddress: "0.0.0.0"
```

`gitlab-pv.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-config
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/config"

---

apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-logs
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/logs"

---

apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-data
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/data"

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-gitlab-config
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
  volumeName: pv-gitlab-config

---

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-gitlab-logs
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
  volumeName: pv-gitlab-logs

---

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-gitlab-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
  volumeName: pv-gitlab-data

```

`gitlab-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gitlab
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gitlab
  template:
    metadata:
      labels:
        app: gitlab
    spec:
      containers:
      - name: gitlab
        image: gitlab/gitlab-ce:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITLAB_OMNIBUS_CONFIG
          value: |
            external_url 'http://gitlab.example.com:8000'
            nginx['listen_port'] = 8000
        volumeMounts:
        - name: gitlab-config
          mountPath: /etc/gitlab
        - name: gitlab-logs
          mountPath: /var/log/gitlab
        - name: gitlab-data
          mountPath: /var/opt/gitlab
      volumes:
      - name: gitlab-config
        persistentVolumeClaim:
          claimName: pvc-gitlab-config
      - name: gitlab-logs
        persistentVolumeClaim:
          claimName: pvc-gitlab-logs
      - name: gitlab-data
        persistentVolumeClaim:
          claimName: pvc-gitlab-data

```

`gitlab-service.yaml`:

```
apiVersion: v1
kind: Service
metadata:
  name: gitlab-service
spec:
  selector:
    app: gitlab
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
      nodePort: 32000
  type: NodePort
```

Set the password for the GitLab `root` user.

```bash
kubectl exec -it `kubectl get pods | grep gitlab | awk '{print $1}'` -- gitlab-rake gitlab:password:reset
```

## Setup with Kubernetes and Helm (WIP)

- somehow, it doesn't work to match pv/pvc for the services to launch 🤔
  - TODO:
    - Refer to https://docs.gitlab.com/charts/installation/storage/ for installation.
    - Refer to https://docs.gitlab.com/charts/advanced/persistent-volumes/ for management after installation
- Couldn't figure out yet how to put another service and GitLab together 🤔
- Install kind, kubectl, helm

`kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  - hostPath: /data/gitlab
    containerPath: /data/gitlab
  extraPortMappings:
  - hostPort: 443
    containerPort: 443
    listenAddress: "0.0.0.0"
```

`gitlab-pv.yaml` 🤔:

```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-gitaly
  labels:
    app: gitaly
    release: gitlab
spec:
  capacity:
    storage: 8Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/gitaly"


---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-postgresql
  labels:
    app: postgresql
    release: gitlab
    role: primary
spec:
  capacity:
    storage: 8Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/postgresql"


---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-minio
  labels:
    app: minio
    release: gitlab
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/minio"

---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-gitlab-redis
  labels:
    name: redis
    instance: gitlab
    component: master
spec:
  capacity:
    storage: 8Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: "/data/gitlab/redis"


```

`gitlab-values.yaml` 🤔:

```yaml
# Refer to https://gitlab.com/gitlab-org/charts/gitlab/-/blob/master/values.yaml
global:
  hosts:
    domain: gengen.ai
certmanager-issuer:
  email: "hoyeong.heo@gengen.ai"

# https://docs.gitlab.com/charts/charts/gitlab/gitaly/
gitaly:
  persistence:
    matchLabels:
      app: gitaly
      release: gitlab

# https://docs.gitlab.com/charts/charts/minio/
minio:
  persistence:
    matchLabels:
      app: minio
      release: gitlab

# https://github.com/bitnami/charts/tree/main/bitnami/redis
redis:
  master:
    persistence:
      selector:
        matchLabels:
          instance: gitlab
          name: redis
          component: master

# https://artifacthub.io/packages/helm/bitnami/postgresql
postgresql:
  primary:
    persistence:
      selector:
        matchLabels:
          release: gitlab
          app: postgresql
          role: primary
```

`ingress-nginx-values.yaml`:

```yaml
controller:
  service:
    type: LoadBalancer
    externalTrafficPolicy: Local
    ports:
      gitlab-https: 443
    targetPorts:
      gitlab-https: 443
  ingressClassResource:
    name: nginx
    default: true
  hostNetwork: true
  hostPort:
    enabled: true
```

`metallb-config.yaml`:

```yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: gitlab-pool
  namespace: metallb-system
spec:
  addresses:
  - 172.24.155.163-172.24.155.163  # my laptop address
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: gitlab-l2
  namespace: metallb-system
spec:
  ipAddressPools:
  - gitlab-pool

```

```bash
helm repo add gitlab https://charts.gitlab.io/
helm repo add metallb https://metallb.github.io/metallb
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

kind create cluster --config kind-config.yaml
kubectl cluster-info

helm install gitlab gitlab/gitlab -f gitlab-values.yaml
helm install metallb metallb/metallb --namespace metallb-system --create-namespace
helm install ingress-nginx ingress-nginx/ingress-nginx -f ingress-nginx-values.yaml --namespace ingress-nginx --create-namespace
kubectl apply -f metallb-config.yaml
```

`
