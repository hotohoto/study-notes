[TOC]



# Kubernetes

[Official Kubernetes Documentation](https://kubernetes.io/docs/home/)

## Getting started

### Play with k8s

[Online K8S playground](https://labs.play-with-k8s.com/)

(control-plane - play with k8s)

```bash
kubeadm init --apiserver-advertise-address $(hostname -i) --pod-network-cidr 10.5.0.0/16
kubectl apply -f https://raw.githubusercontent.com/cloudnativelabs/kube-router/master/daemonset/kubeadm-kuberouter.yaml
kubeadm token list
kubeadm token delete FOO
kubeadm token create --ttl 1h  # token is deleted in 1h
```

(worker node - play with k8s)

```bash
kubeadm reset
kubeadm join 192.168.0.8:6443 --token l12mpj.dbf0df8bewxukysx \
    --discovery-token-ca-cert-hash sha256:2c72286b137951dcc8fec3f49521e2b8f2256b9182bd28c060804ac3cfc743f7
```

### Install kubectl

- Refer to https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/


```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client

# set up auto-completion
echo 'source <(kubectl completion bash)' >>~/.bashrc
```



### Setup Kubernetes in Docker (kind)

- Install docker desktop first.
- Install kubectl
- Install kind
  - https://kind.sigs.k8s.io/docs/user/quick-start/


```bash
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

**usage:**

```bash
kind create cluster
kind create cluster --name=my-cluster
kubectl cluster-info

kind get clusters

kind delete cluster
kind delete cluster --nmae=my-cluster
```



### Setup Kubernetes on premise

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

### Setup k3s locally with multipass

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
multipass launch --name k3s-master --cpus 2 --mem 1024M --disk 5G
multipass launch --name k3s-node1 --cpus 1 --mem 1024M --disk 4G
multipass launch --name k3s-node2 --cpus 1 --mem 1024M --disk 4G
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

### minikube

```bash
minikube start
minikube start --driver=docker
minikube ip  # print the node ip. Note that minikube has only one node.
minikube ssh  # connect to the node
```

### Options for Highly Available Topology

- https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/ha-topology/
- https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/high-availability/

- nodes
  - control-plane
    - master1
    - master2
    - master3
    - ...
    - LB
  - workers
    - worker1
    - worker2
    - ...

(using kubeadm)

- install docker
- install kubeadm, kubectl, kubelet
- setup load balancer (LB)
  - using either a dedicated hardware or an nginx service
- setup HA cluster
  - master 1: run kubeadm init and register LB
  - master 2,3: join master 1
  - install container netwokr interface (CNI)
  - worker nodes: join master via LB

(Other tools)

- kubespray

## Concepts

### Overview

#### Kubernetes Components

we can see kubernetes as an OS.

(Control Plane Components)

- kube-apiserver
  - REST API Server on the 6433 port
- etcd
  - distributed key value storage
  - this is important when it comes to multiple masters
  - `/var/lib/etcd`
- kube-scheduler
  - request
- kube-controller-manager
  - monitoring work nodes
  - and make sure the configurations
- cloud-controller-manager

(Node Components)

- kublet
- kube-proxy
- Container runtime
  - docker / containerd

(Addons)

- Container Network Interface CNI
  - weavenet, calico, flannel, kube-route
  - also called VxLAN or pod network
    - plugins
      - flannel
      - calico
      - waevenet
- CoreDNS
  - running as a pod
  - `kube-dns` provides the entry address as a service
- Web UI (Dashboard)
- Container Resource Monitoring
  - cAdvisor
- cluster-level logging
  - ELK
    - ElasticSearch
    - Logstash
    - Kibana
  - EFK
    - ElasticSearch
    - Fluentd
    - Kibana
  - DataDog

### Workloads

#### Pods

- how to run
  - `kubectl run myapp --image=myapp:latest`
  - `kubectl create -f pod-myapp.yaml`

- The minimum unit to represeent one or more containers
  - A pod can contain more than one containers 
  - By default those containers have the same IP and can communicate seemlessly
    - e.g.
      - `kubectl exec -it -c centos-container -- /bin/bash`
      - `curl locahost`

- how to check when there are multiple containers in a pod
  - `kubectl exec -it -c nginx-container -- /bin/bash`
  - `kubectl logs -c nginx-container`

(Patterns)

- sidecar

initContainers:
- run as preconditions for running the primary containers

infra container:
- pause
  - this container runs for each pod always

static pod: run/deleted by kublet if you add/remove a pod definition file
- try to add one int the manifest file
- how to restart a kublet daemon

multi container pods patterns

- sidecar
  - single node
  - two containers
    - an application container
    - a sidecar container
      - helps the main application but still in an independent way
        - e.g. logging / monitoring / configuration / networking
- adapter
  - an application container reports info to an adapter container
  - the adapter container is reposnsible for providing those information to another stakeholder
    - [e.g. Prometheus](https://www.magalix.com/blog/the-adapter-pattern)
  - e.g. format logs befores sending it to a central server
- ambassador
  - two containers
    - an application container
      - e.g. connects to a local database
    - an ambassador container
      - works as a proxy
      - introduced since we don't want to modify the application container
      - e.g. forward a connection request to a local database to a proper remote server among dev/test/prod databases
- etc
  - https://docs.microsoft.com/en-us/azure/architecture/patterns/

#### Workload Resources

##### Deployment

- controls replicaset
- supports rolling update/back
  - creates a new replicaset
  - increases the new `replicas` and decreases the old `replicas` one by one
- (without the rolling update/back feature, the behavior might be considered as the same as using just replicasets)
- metadata
  - `annotations`
    - `kubernetes.io/change-cause`: revision history message
- spec
  - `progressDeadlineSeconds`: requires a rolling update to take less than this value.
  - `maxSurge`: the max number of running pods allowed during a rolling update

- blue/green update
  - blue: old version
  - green: new version
- canary update
  - update the new version only for a small part of the replicas
- rolling update
  - update one by one

canary update example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mainui-stable
spec:
  replicas: 20
  selector:
    matchLabels:
      app: mainui
      version: stable
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mainui-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mainui
      version: canary
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mainui-svc
spec:
  selector:
    app: mainui
```

```bash
kubectl create deployment mainui --image=httpd --replicas=3 --record
kubectl get deployments.apps
kubectl describe deployments.apps mainui
kubectl get pods
kubectl edit deployments.apps mainui  # we can change the pod settings - e.g the number of replicas - at runtime
kubectl delete deployments.apps mainui
kubectl delete pod --all
kubectl get pods, rs, deploy

# rolling update
kubectl set image deployment mainui mainui-container=httpd=2.3 --record
kubectl rollout status deployment mainui  # shows live log for the rolling update
kubectl rollout pause deployment mainui
kubectl rollout resume deployment mainui
kubectl rollout history deployment mainui  # shows revision history up to `revisionHistoryLimit` revisions
kubectl rollout undo deployment mainui  # rollback to the previous revision
kubectl rollout undo deployment mainui --to-revision=3  # rollback to the revision 3
```

##### replication controller
- controls pods
- ensures the number of pods running
- also important for rolling update where we update containers one by one without shutting down the entire service
- spec
  - replicas
  - selector
  - template


```bash
# run multiple containers

kubectl edit rc rc-nginx
kubectl scale rc rc-nginx --replicas=2
```

##### ReplicaSet

- controls pods
- almost the same as replication controller
- provides richer label selector options than replication controller
  - key
  - operator: `In`, `NotIn`, `Exists`, `DoesNotExist`
  - value

```bash
# run multiple containers
kubectl scale rs rs-nginx --replicas=2
kubectl delete rs rs-nginx  # delete the replicaset and the pods running
kubectl delete rs rs-nginx --cascade=false  # delete the replicaset without deleting the pods running
```

##### DaemonSet

- It makes sure a single pod is running on each node
  - when a node is added, a new pod for the node is also run
- Rolling update is also supported
  - `kubectl edit daemonset my-daemonset`
  - `kubectl rollout daemonset my-daemonset`
- e.g.
  - can be used for a log importer or a monitoring agent
  - kubeproxy, CNI network

##### statefulset

- It makes sure pod names are deterministic and thier volumnes are persistent
- Useful when pods have their own states
- Rolling update is also supported
- specs
  - `replicas`
  - `serviceName`
  - `podManagementPolicy`
    - `Parallel`
    - `OrderedReady`
- Note that pods are supposed to be stateless by default

###### Jobs

- designed for running a batch
- terminated after the task is done successfully
  - the status is left as `completed` and the pod is not deleted
- restarts the pod if it failed
  - the pod is deleted if it has been failed for `backoffLimits` times
- spec
  - `template`
    - `spec`
      - `restartPolicy`
        - `Never`: creates a new pod when failed
        - `OnFailure`: restarts the pod when failed
  - `backoffLimits`
  - `completions`
    - number of times to run the task
  - `parallelism`
    - number of running pods at most
  - `activeDeadlineSeconds`
    - how many seconds it will wait until the pod is forced to be terminated

##### Cronjob

- controls a job
- spec
  - `schedule`
  - `JobTemplate`
  - `successfulJobHistory`
    - how many pods to keep after completed as history

### Services, Load Balancing, and Networking

#### Service

```bash

# https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#expose
# Create a service for a pod valid-pod, which serves on port 444 with the name "frontend"
kubectl expose pod valid-pod --port=444 --name=frontend
```

- You may check if a port is actually open. `netstat -napt | grep 30200`

- types
  - `ClusterIP`
    - the most basic option
    - creates a virtual IP and redirects requests to a random pod among the pods selected by labels
    - but the virtual IP is accessible only from inside the cluster
  - `NodePort`
    - each node opens a port and redirects requests to a random pod among the pods selected by labels
    - Other than that, it's the same as ClusterIP
  - `LoadBalancer`
    - uses an external load balancer provided by a pulblic cloud e.g. GCP, AWS, Azure
    - setup `NodePort` and let a load balancer to have that connection information
    - you may install MetalLB
  - `ExternalName`
    - provides DNS
    - forward `my_external_svc_name.my_external_namespace_name.svc.clutser.local` to the specified address e.g. `google.com`
    - anologous to /etc/hosts

(Headless Services)

- the single entry point has no IP
- but the endpoints are registered to CoreDNS
  - pod-ip-addr.namespace.pod.cluster.local
- useful for statefulSet
- spec
  - `type: ClusterIP`
  - `clusterIP: None`


```bash
kubectl exec -it ... -- /bin/bash
cat /etc/rsolve.conf  # check which DNS service is in use
curl 10-36-0-1.default.pod.cluster.local
```

(kube-proxy)

- the backend of service
  - manipulate forwarding mechanism
    - modes
      - userspace (seems not in use anymore)
      - iptables (by default)
      - IPVS (L4 loadbalancer supported by linux kernel)
  - listening to node ports

##### An example

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  clusterIP: 10.43.0.100  # this is normally not required
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 80
```

```bash
kubectl get pods -o wide
```

```
NAME                               READY   STATUS    RESTARTS      AGE   IP           NODE         NOMINATED NODE   READINESS GATES
nginx-deployment-9456bbbf9-kdrlc   1/1     Running   1 (14h ago)   24h   10.42.0.45   k3s-master   <none>           <none>
nginx-deployment-9456bbbf9-2whfp   1/1     Running   1 (14h ago)   24h   10.42.0.49   k3s-master   <none>           <none>
nginx-deployment-9456bbbf9-tngkb   1/1     Running   0             14h   10.42.2.16   k3s-node2    <none>           <none>
```

- nodes
  - `k3s-master`: 172.29.219.55
  - `k3s-node1`: 172.29.221.104
  - `k3s-node2`: 172.29.221.225
- `kube-dns`: 10.43.0.10 (service)
  - `CoreDNS`: 10.42.0.47 (pod)
- `my-service`: 10.43.0.100 (service)
  - `nginx-deployment-9456bbbf9-kdrlc`: 10.42.0.45 (pod, master)
  - `nginx-deployment-9456bbbf9-2whfp`: 10.42.0.49 (pod, master)
  - `nginx-deployment-9456bbbf9-tngkb`: 10.42.2.16 (pod, node2)

```bash
iptables -t nat -S
```

```bash
...
-N KUBE-SERVICES
-A PREROUTING -m comment --comment "kubernetes service portals" -j KUBE-SERVICES
-A OUTPUT -m comment --comment "kubernetes service portals" -j KUBE-SERVICES
...
-N KUBE-SEP-GHMRCEVUEFEBIU3U  # nginx pod tngkb
-N KUBE-SEP-IYQJ2XIYX64WA745  # nginx pod kdrlc
-N KUBE-SEP-WFIAUFYDSPQC5HJ7  # nginx pod 2whfp
...
-N KUBE-SVC-FXIYY6OHUSNBITIX  # my-service
...
-A KUBE-SEP-GHMRCEVUEFEBIU3U -s 10.42.2.16/32 -m comment --comment "default/my-service" -j KUBE-MARK-MASQ
-A KUBE-SEP-GHMRCEVUEFEBIU3U -p tcp -m comment --comment "default/my-service" -m tcp -j DNAT --to-destination 10.42.2.16:80
-A KUBE-SEP-IYQJ2XIYX64WA745 -s 10.42.0.45/32 -m comment --comment "default/my-service" -j KUBE-MARK-MASQ
-A KUBE-SEP-IYQJ2XIYX64WA745 -p tcp -m comment --comment "default/my-service" -m tcp -j DNAT --to-destination 10.42.0.45:80
-A KUBE-SEP-WFIAUFYDSPQC5HJ7 -s 10.42.0.49/32 -m comment --comment "default/my-service" -j KUBE-MARK-MASQ
-A KUBE-SEP-WFIAUFYDSPQC5HJ7 -p tcp -m comment --comment "default/my-service" -m tcp -j DNAT --to-destination 10.42.0.49:80
...
-A KUBE-SERVICES -d 10.43.0.100/32 -p tcp -m comment --comment "default/my-service cluster IP" -m tcp --dport 8080 -j KUBE-SVC-FXIYY6OHUSNBITIX
-A KUBE-SVC-FXIYY6OHUSNBITIX ! -s 10.42.0.0/16 -d 10.43.0.100/32 -p tcp -m comment --comment "default/my-service cluster IP" -m tcp --dport 8080 -j KUBE-MARK-MASQ
...
-A KUBE-SVC-FXIYY6OHUSNBITIX -m comment --comment "default/my-service" -m statistic --mode random --probability 0.33333333349 -j KUBE-SEP-IYQJ2XIYX64WA745
-A KUBE-SVC-FXIYY6OHUSNBITIX -m comment --comment "default/my-service" -m statistic --mode random --probability 0.50000000000 -j KUBE-SEP-WFIAUFYDSPQC5HJ7
-A KUBE-SVC-FXIYY6OHUSNBITIX -m comment --comment "default/my-service" -j KUBE-SEP-GHMRCEVUEFEBIU3U
...
```

`/etc/resolve.conf` in `nginx-deployment-9456bbbf9-kdrlc`

```conf
search default.svc.cluster.local svc.cluster.local cluster.local mshome.net
nameserver 10.43.0.10
options ndots:5
```

With in a pod, we can access to the service using a domain name as follows.

```bash
curl my-service:8080
curl my-service.default.svc.cluster.local:8080
```

With in a pod, we can access to a pod as follows

```bash
curl 10-42-2-16.default.pod.cluster.local
```

#### Ingress

e.g. https://github.com/237summit/k8s_core_labs/blob/main/8/ingress3.yaml

- can redirect to a ClusterIP service registered depending on the request path
  - note that ClusterIP provides only an internal entry address
- also supports virtual hosts
- One of the open project controllers can be used.
  - e.g. NGINX Ingress
- seems to get the external IP address by querying by the ingress hostname
  - so ingress is not suitable for development environment


https://stackoverflow.com/questions/45079988/ingress-vs-load-balancer

### Storage

#### Volumes

#### Persistent Volumes

- Pod -> PVC -> storageclass -> PV (-> Host machine)
- a PVC and a PV are going to be bound

```bash
kubectl describe storageclass
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myclaim
spec:
  accessModes:
    - ReadWriteMany # the volume can be mounted as read-write by many nodes.
  volumeMode: Filesystem
  resources:
    requests:
      storage: 10Mi # storage size
  storageClassName: "standard-rwx" # the name of storageclass
```

```bash
kubectl apply -f myclaim.yaml
kubectl get pvc,pv
```

### Configuration


#### ConfigMaps

- can be passed as
  - environment variables
  - arguments
  - files in a volume mounted


```bash
# kubectl create configmap NAME [--from-file=source] [--from-literal=key1=value1]
kubectl create configmap myconfig --from-file=key1  # the content is value1
kubectl create configmap myconfig --from-literal=key1=value1 --from-literal=key2=value2
kubectl create configmap myconfig --from-file=/path/to/config/dir

kubectl get configmaps myconfig
kubectl describe configmaps myconfig
kubectl edit configmaps myconfig
```

(import key1)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myweb
spec:
  containers:
  - image: nginx
    name: nginx-container
    env:
    - name: key1
      valueFrom:
        configMapKeyRef:
          name: myconfigmap
          key: key1
```

(import all configs)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myweb
spec:
  containers:
  - image: nginx
    name: nginx-container
    envFrom:
    - configMapRef:
      name: myconfigmap
```

(mount configs in a config map)

- https://github.com/237summit/k8s_core_labs/blob/main/10/fortune-volume-cm.yaml

#### Secrets

```bash
echo -n 'mypw' | base64  # bXlwdw==
echo -n 'bXlwdw==' | base64 --decode  # mypw
kubectl get secrets
kubectl create secret tls mysecret --cert=path/to/cert/file --key=path/to/key/file
kubectl create secret docker-registry mysecret --docker-username=tiger --docker-password=pass --docker-email=tiger@example.com
kubectl create secret generic mysecret --from-literal=key1=value1 --from-file=path/to/secret/dir  # encoded in base64 but not encrypted
kubectl create secret generic mysecret --from-literal=key1=value1 --from-literal=key2=value2  # encoded in base64 but not encrypted
kubectl describe secret mysecret
kubectl describe secret mysecret -o yaml
```

types:
- Opaque
  - arbtrary user defined data
- kubernetes.io/service-account-token
- kubernetes.io/dockercfg
  - serialized `~/dockercfg` file
- kubernetes.io/dockerconfigjson
  - serialized `~/.docker/config.json`
- kubernetes.io/basic-auth
  - credentials for authentication
- kubernetes.io/ssh-auth
- kubernetes.io/tls
- bootstrap.kubernetes.io/token

(import key1)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myweb
spec:
  containers:
  - image: nginx
    name: nginx-container
    env:
    - name: key1
      valueFrom:
        secretKeyRef:
          name: mysecret
          key: key1
```

(import all configs)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myweb
spec:
  containers:
  - image: nginx
    name: nginx-container
    envFrom:
    - secretRef:
      name: mysecret
```

(mount configs in a secret)

- https://github.com/237summit/k8s_core_labs/blob/main/10/fortune-volume-cm-secret.yaml

#### Resource Management for Pods and Containers

resources per container
- `requests`
  - if the requested resources are not available, the pod is not scheduled and stays as pending
  - if requests and limits are not defined explicitly, the default requests are used
    - the default requests can be set by LimitRange object under the namespace
      - https://kubernetes.io/docs/tasks/administer-cluster/manage-resources/memory-default-namespace/
      - https://kubernetes.io/docs/tasks/administer-cluster/manage-resources/cpu-default-namespace/
- `limits`
  - if the container exceeds the cpu limit it will get throttled
  - if the container exceeds the memory limit it will get killed and restarted
  - if requests are not defined explicitly, requests are also set the same as limits

### Security

- Authentication: "Is that really you?"
- Authorization: "Do you have the permission?"
- Admission Control: "Is it fine to do that?"
- users and groups: accounts for human
- service accounts: accounts for pods

#### Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-pod
specs:
  containers:
  - name: ubuntu
    image: ubuntu
    command: ["sleep", "3600"]
    securityContext:
      runAsUser: 1000
      capabilities:
        add: ["MAC_ADMIN"]
```

(Docker security)

- The default user for containers is `root` but it's not the same as `root` in the host machine
- By the `ps aux` command, a container can see only the processes within the container
- If we want to run a container with it a user can be set by the methods as follow
  - `docker run --user=1000 ...`
  - `USER 1000` (in Dockerfile)
- You can change the capability of `root` within a container
  - `docker run --privileged ...`
    - enables all privileges
  - `docker run --cap-add=MAC_ADMIN ...`
  - `docker run --cap-drop=KILL ...`

#### Pod Security Admission

#### Pod Security Policies (deprecated)

(apllied to the entire namespaces)

- ClusterRole
- ClusterRoleBinding

(applied to a specific namespace)

- Role
- RoleBinding

### Scheduling, Preemption and Eviction

#### Assigning Pods to Nodes

(nodeSelector)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myweb
  annotations:
    builder: "Foo Bar (baz@example.com)
    imageregistry: "https://hub.docker.com"
spec:
  containers:
  - image: nginx
    name: nginx-container
  - image: centos:7
    name: centos-container
    # "command" overrides ENTRYPOINT
    command:
    - sleep
    - "10000"
  nodeSelector:
    key1: value1
    key2: value2
    gpu: "true"
    disk: ssd
```

(Affinity and anti-affinity)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myweb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: blue
  template:
    metadata:
      labels:
        app: blue
    spec:
      containers:
      - image: nginx
        name: nginx-container
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:  # Choose among only the nodes matches. (Options under this work the same as nodeSelector)
            nodeSelectorTerms:
            - matchExpressions:
              - key: disk
                operator: Exists
          preferredDuringSchedulingIgnoredDuringExecution:  # Choose the best node
          - weight: 10  # gives 10 points for each expression matches
            preference:
              matchExpressions:
              - key: gpu
                operator: In
                values:
                - true
              - key: disk
                operator: In
                values:
                - ssd
        podAffinity:  # wants to pick a node following the selected pods
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: security
                operator: In
                values:
                - S1
            topologyKey: topology.kubernetes.io/zone  # requires all the selected nodes are in the same zone
        podAntiAffinity:  # wants to pick a node which is different from the selected pods
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: security
                  operator: In
                  values:
                  - S2
              topologyKey: topology.kubernetes.io/zone
```

- podAffinity:
- podAntiAffiity:
- toplogyKey

#### Taints and Tolerations

https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/

```bash
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  containers:
  - name: nginx-container
    images: nginx
  tolerations:
  - key: "key1"
    operator: "Equal"
    value: "value1"
    effect: "NoSchedule"
```

```bash
# don't schedule any pods onto node1 unless there a matching toleration in the pod spec for each taint set
kubectl taint nodes node1 key1=value1:NoSchedule
kubectl taint nodes node1 key1=value1:NoSchedule-
kubectl taint nodes controlplane node-role.kubernetes.io/master:NoSchedule-
```

```yaml
# pod spec - I can be scheduled onto a node even though the node has one of these taint defined
tolerations:
- key: "key1"
  operator: "Equal"
  value: "value1"
  effect: "NoSchedule"
- key: "key1"
  operator: "Equal"
  value: "value1"
  effect: "NoExecute"
```

- effects
  - `NoSchedule`
    - if the pod doesn't have this toleration, don't schedule it on to the node
  - `PreferNoSchedule`
    - if the pod doesn't have this toleration, preferred not to schedule it on to the node
  - `NoExecute`
    - if the pod doesn't have this toleration, stop executing the running pod

## Tasks

### Administer a Cluster

#### Safely Drain a Node

- cordon
  - prevent pods from being scheduled on to the node
- uncordon
  - allow pods to be scheduled on to the node
- drain
  - cordon and delete pods running on the node

```bash
kubectl cordon node2
kubectl drain node2 --ignore-daemonsets
kubectl drain node2 --ignore-daemonsets --force
kubectl drain node2 --ignore-daemonsets --force --deete-empty-dir
kubectl uncordon node2
```

#### Operating etcd clusters for Kubernetes

(backup)

```bash
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=<trusted-ca-file> \
  --cert=<cert-file> \
  --key=<key-file> \
  snapshot save <backup-file-location>
```

### Configure Pods and Containers

#### Configure Service Accounts for Pods

```bash
kubectl create serviceaccount dashboard-sa
kubectl get serviceaccount
kubectl describe serviceaccount dashboard-sa
kubectl describe secret dashboard-sa-token-kbbdm
curl https://192.168.56.70:6443/api -insecure --header "Authorization Bearer: eyJhbG..."
```

- In the service account created you can see the token and the secret reference.
- The secret object refered by it contains actual token.
- The token is used for an application to access K8S API with
- A serviceaccount would be mounted at `/var/run/secrets/kubernetes.io/serviceaccount` within the containers of a pod.
- By default, `default` service account is mounted to a pod.
  - you can disable that by specifying `automountServiceAccountToken: false` in the pod spec section.
- To change the serivce account being mounted, specify a different servic account as follows.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-kubernetes-dashboard
spec:
  containers:
  - name: my-kubernetes-dashboard
    images: my-kubernetes-dashboard
  serviceAccountName: dashboard-sa
```

#### Configure Liveness, Readiness and Startup Probes

(self-healing)

https://youtu.be/-NeJS7wQu_Q

- can set up on a container and it may restart the container (not the pod)

```yaml
# check if the status code is 200
livenessProbe:
  httpGet:
    path: /
    port: 80

# check if it can be connected
livenessProbe:
  tcpSocket:
    port: 22

# check if the exit code is zero
livenessProbe:
  exec:
    command:
    - ls
    - /data/file
```

  - there are more options you can set - to see the default values try to describe the running pod
    - e.g.
      - `initialDelaySeconds: 15`
      - `periodSeconds: 20`
      - `timeoutSecons: 1`
      - `successThreshold: 1`
      - `failureThreshold: 3`

(yaml example)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: myapp
spec:
  containers:
  - image: busybox
    name: myapp-container
    # "args" overrides CMD
    args:
    - sh
    - -c
    - touch /tmp/healthy; sleep 30; rm -f touch /tmp/healthy; sleep 10000
    livenessProbe:
      exec:
        command:
        - ls
        - /tmp/healthy
```

### Inject Data Into Applications

#### Define Environment Variables for a Container

- `env`
  - defined as a list of maps with name and value
  - it can define a new environment variable
  - it can overwrite an existing environment variable

### Run Applications

#### Horizontal Pod Autoscaling

Install metric servers

```bash
git clone https://github.com/237summit/kubernetes-metrics-server.git
cd kubernetes-metrics-server
kubectl apply -f .  # applies all the yaml files in the current folder

kubectl top nodes
kubectl top pods -A
kubectl top pods -l foo=bar --sort-by=cpu
```

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-web
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deploy-web
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 50  # with respect to the CPU resource requested by each pod
```

```bash
# apply it
kubectl apply -f hpa-web.yaml

# or equivalently
kubectl autoscale deployment hpa-web --cpu-percent=50 --min=1 --max=10

# check if it's set
kubectl get hpa
```

## References

### API Access Control

#### Certificate Signing Requests

(Normal User)

```bash
# create myuser.key
openssl genrsa -out myuser.key 2048

# create myuser.csr which is a request for a certificate
openssl req -new -key myuser.key -out myuser.csr
openssl req -new -key myuser.key -out myuser.csr -subj "/CN=myuser"

# make it to be a k8s csr
cat myuser.csr |base64| tr -d "\n"
vi csr-myuser.yaml
```

```yaml
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: myuser
spec:
  request: LS0tL...o= #
  signerName: kubernetes.io/kube-apiserver-client
  expirationSeconds: 86400  # one day
  usages:
  - client auth
```

```bash
kubectl apply -f csr-myuser.yaml
kubectl get csr
kubectl certificate approve myuser
kubectl get csr
kubectl get csr/myuser -o yaml
kubectl get csr myuser -o jsonpath='{.status.certificate}'| base64 -d > myuser.crt
kubectl config set-credentials myuser --client-key=myuser.key --client-certificate=myuser.crt --embed-certs=true
kubectl config set-context myuser --cluster=kubernetes --user=myuser
kubectl config use-context myuser

kubectl create role developer --verb=create --verb=get --verb=list --verb=update --verb=delete --resource=pods
kubectl create rolebinding developer-binding-myuser --role=developer --user=myuser
```

```yaml
# kubectl create clusterrole developer --verb=create,list,get,update,delete --resource=deployment,pod,service
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: developer
rules:
- apiGroups:
  - ""
  resources:
  - pods
  - services
  verbs:
  - create
  - list
  - get
  - update
  - delete
- apiGroups:
  - apps
  resources:
  - deployments
  verbs:
  - create
  - list
  - get
  - update
  - delete
---
# kubectl create clusterrolebinding developer-binding-myuser --clusterrole=developer --user=myuser
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  creationTimestamp: null
  name: developer-binding-myuser
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: developer
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: myuser
```

## Appendix

### Autoscaling

- cluster level
  - Adds more nodes if there are pending pods due to the lack of resources
- pod level
  - Horizontal Pod Autoscaler (HPA)
    - Adds more pods if resources are in short
    - Delete pods in 5 min if resources are not in short
    - scale in/out
  - Vertical Pod Autoscaler (VPA)
    - Allows more resources and restarts the pods one by one
    - scale up/down
- metric server
  - monitors how much resources are in use by pods for each node

### public clouds

- GKE
  - google
- EKS
  - Amazon
- AKS
  - Azure

### basic kubectl commands

```bash
# auto completion https://kubernetes.io/docs/tasks/tools/included/optional-kubectl-configs-bash-linux/
echo 'source <(kubectl completion bash)' >>~/.bashrc

# show cluster info
kubectl cluster-info

# show resources and their acronyms
kubectl api-resources

# show descriptions for a certain resource
kubectl explain pod
kubectl explain pod --recursive | grep envFrom -A10
kubectl explain pod --recursive | less
kubectl explain namespace

# show node info
kubectl get all
kubectl get nodes
kubectl get nodes -o wide
kubectl describe nodes k3s-master

kubectl label node node{2,3}.example.com gpu=true
kubectl label node node2.example.com disk=ssd
kubectl label node node2.example.com gpu-
kubectl get nodes -o wide
kubectl get nodes --all-namespaces
kubectl get nodes --show-labels
kubectl get nodes -L disk,gpu  # show only the speified labels in a separate columns which are `disk` and `gpu`
kubectl get pods --all-namespaces

kubectl run python --image=python:3.7.13 -- tail -f /dev/null
kubectl run testpod --image=centos:7 --comand sleep 5  # kubernetes restarts this pod after it's completed
kubectl run webserver --image=nginx:1.14 --port 80
kubectl get pods
kubectl get pods -o wide
watch kubectl get pods -o wide
watch kubectl get pods -o wide --watch  # open a new terminal and run this to watch what's going on
kubectl get pods -o yaml
kubectl get pods -o json
kubectl get pods --show-labels
kubectl get pods -l name=mainui
kubectl get pods --selector name=mainui  # the same as above
kubectl label pod testpod rel=qa name=payment
kubectl label pod testpod rel=stable --overwrite
kubectl label pod testpod name-  # delete the label `name`
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
kubectl delete -f webserver-pod.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/website/master/content/en/examples/application/nginx-app.yaml

# namespace management
kubectl create namespace blue
kubectl get namespaces
kubectl create namespace green --dry-run -o yaml > green-ns.yaml
vim green-ns.yaml
kubectl create -f green-ns.yaml
kubectl get namespace
kubectl delete namespace

# switch default namespace
kubectl config current-context
kubectl config view
kubectl config view --minify
kubectl config set-context new_context_name --cluster=kubernetes --user=kubernetes-admin --namespace=blue
kubectl config set-context orange --cluster=default --user=default --namespace=orange
kubectl config use-context new_context_name

# cp files from/to a pod
kubectl cp path/to/src pod:/path/to/destination
kubectl cp pod:/path/to/src ./
```

### Other commands

```bash
# install elinks
sudo apt update
sudo apt install elinks
```

### ~/.vimrc

https://dev.to/marcoieni/ckad-2021-tips-vimrc-bashrc-and-cheatsheet-hp3

```
set nu
set tabstop=2 shiftwidth=2 expandtab
set ai
```

## External Links

- [Kubernetes The Hard Way On VirtualBox](https://github.com/mmumshad/kubernetes-the-hard-way)
- [Debugging kubectl](https://www.shellhacks.com/kubectl-debug-increase-verbosity/)

(course links)

- [CKAD with tests - Udemy](https://www.udemy.com/course/certified-kubernetes-application-developer/)
  - [Labs](https://kodekloud.com/courses/labs-certified-kubernetes-application-developer/)
  - [Challenges](https://kodekloud.com/courses/kubernetes-challenges/)
- [CKA with practice tests - Udemy](https://www.udemy.com/course/certified-kubernetes-administrator-with-practice-tests/)
  - [Labs]()
- [따라하면서 배우는 쿠버네티스](https://youtube.com/playlist?list=PLApuRlvrZKohaBHvXAOhUD-RxD0uQ3z0c)
  - [실습코드 - 따라하면서 배우는 쿠버네티스](https://github.com/237summit/k8s_core_labs)
- [따라하면서 배우는 쿠버네티스 (심화)](https://youtube.com/playlist?list=PLApuRlvrZKohLYdvfX-UEFYTE7kfnnY36)

(certificates)

- [Certified Kubernetes Administrator (CKA)](https://www.cncf.io/certification/cka/)
  - https://training.linuxfoundation.org/certification/certified-kubernetes-administrator-cka/
- [Certified Kubernetes Application Developer (CKAD)](https://www.cncf.io/certification/ckad/)
  - https://training.linuxfoundation.org/certification/certified-kubernetes-application-developer-ckad/
- [Exam Curriculum (Topics)](https://github.com/cncf/curriculum)
- [Candidate Handbook](https://www.cncf.io/certification/candidate-handbook)
- [Exam Tips](http://training.linuxfoundation.org/go//Important-Tips-CKA-CKAD)