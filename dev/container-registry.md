# Container registry

A container registry consists of container repositories and a searchable catalogue where you manage and deploy images.

## Glossary

- Repository
  - a collection of artifacts
  - e.g.
    - test/postgres
    - test/hello-world
- OCI
  - open container initiative
- OCI image
  - = Docker image
  - the format is opened and called the OCI format

## Docker image registry

```bash
# run a docker registry (hub)
# https://docs.docker.com/registry/deploying/
docker run -d -p 5000:5000 --restart=always --name my-registry registry:2
# Start the registry with basic authentication.
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name registry \
  -v "$(pwd)"/auth:/auth \
  -e "REGISTRY_AUTH=htpasswd" \
  -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  -v "$(pwd)"/certs:/certs \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
  registry:2

# log in to the registry
docker login myregistrydomain.com:5000
```

- https://github.com/Joxit/docker-registry-ui
- https://github.com/andrey-pohilko/registry-cli

## Harbor

- a docker registry implementation

### Install Harbor in kind

- Install docker desktop first.
- Install kubectl
  - https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

- install kind
  - https://kind.sigs.k8s.io/docs/user/quick-start/

```bash
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

- install helm
  - https://helm.sh/docs/intro/install/

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

- make a cluster

```bash
kind create cluster --name my-cluster
kubectl cluster-info
```

- create a namespace

```bash
kubectl create namespace harbor
```

- Add a helm repository

```bash
helm repo add harbor https://helm.goharbor.io
helm repo update
```

- install harbor
  - https://goharbor.io/docs/1.10/working-with-projects/working-with-images/managing-helm-charts/

```bash
 helm install harbor harbor/harbor --namespace harbor
```

- check the status

```bash
kubectl get pods -n harbor
```

```
NAME                                 READY   STATUS    RESTARTS        AGE
harbor-core-6b6c58689c-mn9jq         1/1     Running   2 (4m39s ago)   16m
harbor-database-0                    1/1     Running   0               16m
harbor-jobservice-648f8c587f-dntk6   1/1     Running   0               16m
harbor-portal-7c585d7986-qmfxc       1/1     Running   0               16m
harbor-redis-0                       1/1     Running   0               16m
harbor-registry-55fdb8584f-dljmj     2/2     Running   0               16m
harbor-trivy-0                       1/1     Running   0               16m
```

```bash
kubectl get svc -n harbor
```

```
NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)             AGE
harbor-core         ClusterIP   10.96.43.133    <none>        80/TCP              19m
harbor-database     ClusterIP   10.96.178.10    <none>        5432/TCP            19m
harbor-jobservice   ClusterIP   10.96.186.211   <none>        80/TCP              19m
harbor-portal       ClusterIP   10.96.219.56    <none>        80/TCP              19m
harbor-redis        ClusterIP   10.96.128.60    <none>        6379/TCP            19m
harbor-registry     ClusterIP   10.96.18.153    <none>        5000/TCP,8080/TCP   19m
harbor-trivy        ClusterIP   10.96.123.166   <none>        8080/TCP            19m
```

```bash
nohup kubectl port-forward -n harbor svc/harbor-portal 8080:80 > harbor.log 2>&1 &
```

- uninstall harbor

```
helm uninstall harbor -n harbor

# check remaining resources
kubectl get all -n harbor
```

## Azure container registry (ACR)

- glossary
  - Entra ID
    - https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id
    - identity and access management service
  - tenant
    - an instance of Entra ID for an organization
  - Azure portal
    - a centralized dashboard to manage Azure services
- It provides Azure domain address
- User access is managed at Azure portal
  - Access tokens
    - can be bound with an expiration date
    - can be bound with multiple ACR repositories
    - are one time readable
- migration can be done by Azure CLI

## References

- https://octopus.com/blog/top-8-container-registries
