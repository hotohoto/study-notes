
# Kubernetes (k8s)

## TODO

- [따라하면서 배우는 쿠버네티스](https://youtube.com/playlist?list=PLApuRlvrZKohaBHvXAOhUD-RxD0uQ3z0c)
- [CKAD with tests - Udemy](https://www.udemy.com/share/1013BQ3@KyFBRtRu39Zh5jaqPhk_loeVgnTxhUS4N4XNVRFoLVUGHa-6G-eKEUsOikRO3glo/)
- [CKA with practice tests - Udemy](https://www.udemy.com/share/101WmE3@yCeKwZ6DeUW4PxAP5aO1DloOys0VceW9YF3mQabX5aF5ZI9vMfJqq5SivylJLDId/)

## Playgrounds

- https://labs.play-with-k8s.com/


## architecture

- master component (control-plane)
  - etcd
    - key value storage
  - kube-apiserver
    - REST API Server on the 6433 port
  - kube-scheduler
    - request
  - kube-controller-manager
    - monitoring work nodes
    - and make sure the configurations
  - kublet
    - (only for itself)
- worker node
  - kublet
    - cAdvisor
  - kube-proxy
  - docker
- docker hub
- Container Network Interface (CNI)
  - also called VxLAN or pod network
  - plugins
    - flannel
    - calico
    - waevenet
- add-ons
  - CNI
    - weave, calico, flannel, kube-route
  - core DNS
  - dashboard
  - container resource monitoring
    - cAdvisor
  - cluster logging
    - ELK
      - ElasticSearch
      - Logstash
      - Kibana
    - EFK
      - ElasticSearch
      - Fluentd
      - Kibana
    - DataDog
- we can see kubernetes as an OS

- k8s namespaces

## public clouds

- GKE
  - google
- EKS
  - Amazon
- AKS
  - Azure

## commands

### play with k8s

(control-plane - play with k8s)

```bash
kubeadm init --apiserver-advertise-address $(hostname -i) --pod-network-cidr 10.5.0.0/16
kubectl apply -f https://raw.githubusercontent.com/kubernetes/website/master/content/en/examples/application/nginx-app.yaml
kubectl get nodes -o wide
kubectl get nodes --all-namespaces
kubectl get pod --all-namespaces
```

(worker node - play with k8s)

```bash
kubeadm join 192.168.0.8:6443 --token l12mpj.dbf0df8bewxukysx \
    --discovery-token-ca-cert-hash sha256:2c72286b137951dcc8fec3f49521e2b8f2256b9182bd28c060804ac3cfc743f7
```

### setup k8s on premise

https://github.com/237summit/k8s_core_labs

- install OS on each prepared node
- install docker
- setup for k8s
- install
  - kubeadm
    - the command to bootstrap the cluster
  - kubelet
    - the daemon component that runs on all the machine
  - kubectl
    - the command to talk to your cluster
- setup control-plane
  - kube init
  - install weavnet
- setup woker nodes
- check-up

### setup k3s locally with multipass

https://andreipope.github.io/tutorials/create-a-cluster-with-multipass-and-k3s

(ubuntu)

```bash
sudo snap install multipass --classic --stable
multipass launch --name k3s-master --cpus 1 --mem 1024M --disk 3G
multipass launch --name k3s-node1 --cpus 1 --mem 512M --disk 3G
multipass launch --name k3s-node2 --cpus 1 --mem 512M --disk 3G
multipass exec k3s-master -- /bin/bash -c 'curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh -'
multipass list
K3S_TOKEN=`multipass exec k3s-master -- sudo cat /var/lib/rancher/k3s/server/node-token`
K3S_NODEIP_MASTER="https://$(multipass info k3s-master | grep IPv4 | awk '{print $2}'):6433"
multipass exec k3s-node1 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=${K3S_TOKEN} K3S_URL=${K3S_NODEIP_MASTER} sh -"
multipass exec k3s-node2 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=${K3S_TOKEN} K3S_URL=${K3S_NODEIP_MASTER} sh -"
multipass shell k3s-master
kubectl get nodes
```

(windows powershell)

```shell
multipass launch --name k3s-master --cpus 1 --mem 1024M --disk 3G
multipass launch --name k3s-node1 --cpus 1 --mem 512M --disk 3G
multipass launch --name k3s-node2 --cpus 1 --mem 512M --disk 3G
multipass exec k3s-master -- /bin/bash -c 'curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh -'
multipass list
$Env:K3S_TOKEN=$(multipass exec k3s-master -- sudo cat /var/lib/rancher/k3s/server/node-token)
$Env:K3S_NODEIP_MASTER="https://$(multipass info k3s-master |Select-String ipv4|%{($_ -split "\s+")[1]}):6443"
multipass exec k3s-node1 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=$Env:K3S_TOKEN K3S_URL=$Env:K3S_NODEIP_MASTER sh -"
multipass exec k3s-node2 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=$Env:K3S_TOKEN K3S_URL=$Env:K3S_NODEIP_MASTER sh -"
multipass shell k3s-master
multipass shell k3s-node1
multipass exec k3s-master -- kubectl get nodes
```

### kubectl basics

```bash
# auto completion https://kubernetes.io/docs/tasks/tools/included/optional-kubectl-configs-bash-linux/
echo 'source <(kubectl completion bash)' >>~/.bashrc

# show cluster info
kubectl cluster-info

# show node info
kubectl get nodes
kubectl get nodes -o wide
kubectl describe nodes k3s-master

# show resources and their acronyms
kubectl api-resources
```

### run containers via kubectl

```bash
kubectl run webserver --image=nginx:1.14 --port 80
kubectl get pods
kubectl get pods -o wide
kubectl get pods -o yaml
kubectl get pods -o json
kubectl describe webserver
curl 10.44.0.1  # use the ip address displayed
elinks 10.44.0.1  # use the ip address displayed
kubectl exec webserver -it -- /bin/bash
# check html files in /usr/share/nginx/html
kubectl logs webserver
kubectl port-forward webserver 8080:80  # forward to local 80 of the master node
kubectl run webserver --images=nginx:1.14 --port 80 --dry-run -o yaml > webserver-pod.yaml
kubectl delete pod webserver
kubectl create -f webserver-pod.yaml
kubectl get pods

# run multiple containers
kubectl create deployment mainui --image=httpd --replicas=3
kubectl get deployments.apps
kubectl describe deployments.apps mainui
kubectl get pods
kubectl edit deployments.apps mainui  # we can change the number of replicas at runtime
kubectl delete deployments.apps mainui
kubectl get pods

# install elinks
sudo apt update
sudo apt install elinks

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

# namespace management
kubectl create namespace blue
kubectl get namespaces
kubectl create namespace green --dry-run -o yaml > green-ns.yaml
vim green-ns.yaml
kubectl create -f green-ns.yaml
kubectl get namespace
kubectl delete namespace

# switch default namespace
kubectl config view
kubectl config set-context blue@kubernetes --cluster=kubernetes --user=kubernetes-admin --namespace=blue # @kubernetes is not required
kubectl config set-context orange --cluster=default --user=default --namespace=orange
kubectl config current-context
```

## APIs

developed by CNCF

- Deployment
- Pod
- ReplicaSet
- ReplicationController
- Service
- PersistentVolume

```bash
# Check API versions
kubectl explain pod
kubectl explain namespace
```


## References

- https://github.com/237summit/k8s_core_labs
