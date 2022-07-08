# Kubernetes

[Official Kubernetes Documentation](https://kubernetes.io/docs/home/)

## Getting started

### play with k8s

[Online K8S playground](https://labs.play-with-k8s.com/)

(control-plane - play with k8s)

```bash
kubeadm init --apiserver-advertise-address $(hostname -i) --pod-network-cidr 10.5.0.0/16
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
- core DNS
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
- ambassador
  - two containers
    - an application container
    - an ambassador container
      - works as a proxy
      - introduced since we don't want to modify the application container
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

- You may check if a port is actually open. `netstat -napt | grep 30200`

- types
  - `ClusterIP`
    - the most basic option
    - creates a virtual IP and redirects requests to a random pod among the pods selected by labels
    - but the virtual IP is accessible only from inside the cluster
  - `NodePort`
    - each node opens a port and redirects requests to a random pod among the pods selected by labels
    - Other than that, it's the same as CluterIP
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

```bash
iptables -t nat -S | grep 80
```
#### Ingress

e.g. https://github.com/237summit/k8s_core_labs/blob/main/8/ingress3.yaml

- One of the open project controllers can be used.
  - e.g. NGINX Ingress
- can redirect to a ClusterIP service registered depending on the request path
  - note that ClusterIP provides only an internal entry address
- also supports virtual hosts

### Storage

#### Persistent Volumes

Pod -> PVC -> PV -> Host machine

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
    - ReadWriteMany # ReadWriteOnce or ReadWriteMany
  volumeMode: Filesystem
  resources:
    requests:
      storage: 10Mi # storage size
  storageClassName: "local-path" # the name of storageclass
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
- `limits`
  - if exceeded the container gets killed and restarted
  - if requests are not defined explicitly, requests are also set the same as limits

### Scheduling, Preemption and Eviction

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
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:  # Choose among only the nodes matches. (Options under this work the same as nodeSelector)
        nodeSelectorTerms:
        - matchExpressions:
          - {key: disk, operator: Exists}
      preferredDuringSchedulingIgnoredDuringExecution:  # Cchoose the best node
      - weight: 10  # gives 10 points for each expression matches
        preference:
        - matchExpressions:
          - {key: gpu, operator: In, values: ["true"]}
          - {key: disk, operator: In, values: ["ssd"]}
    podAffinity:  # wants to pick a node following the selected pods
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - {key: security, operator: In, values: ["S1"]}
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

- podAffinity: better to
- podAntiAffinity: better
- toplogyKey

## Tasks

### Configure Pods and Containers

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

## Appendix

### public clouds

- GKE
  - google
- EKS
  - Amazon
- AKS
  - Azure

#### kubectl basic commands

```bash
# auto completion https://kubernetes.io/docs/tasks/tools/included/optional-kubectl-configs-bash-linux/
echo 'source <(kubectl completion bash)' >>~/.bashrc

# show cluster info
kubectl cluster-info

# show resources and their acronyms
kubectl api-resources

# show descriptions for a certain resource
kubectl explain pod
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
```

#### Other commands

```bash
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
```

## References

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
