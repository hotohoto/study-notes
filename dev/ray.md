# Ray

## Getting started

### Ray core quickstart

https://docs.ray.io/en/latest/ray-overview/getting-started.html#ray-core-quickstart

```bash
pip install -U ray
```

```py
import ray
ray.init()

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(4)]
print(ray.get(futures)) # [0, 1, 4, 9]
```

ray.init()

- Check if Ray cluster exists
  - if not, create one
- Connect to the Ray cluster



### Ray cluster quick start

https://docs.ray.io/en/latest/ray-overview/getting-started.html#ray-cluster-quickstart

```bash
ray submit cluster.yaml example.py --start
```

- (TODO)
  - https://github.com/ray-project/ray/blob/master/doc/yarn/example.py
  - https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-minimal.yaml
  - https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/local/example-minimal-automatic.yaml
  - https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/local/example-full.yaml
- I don't know how to make the cluster configuration that can be shared in GitHub
  - It has a SSH username in the configuration.
  - A possible workaround is to use a shared user with the expiration date for the project




## Ray core

.get()

- sync

.remote()

- async

.wait()

- 

### User guides

#### Design patterns and anti patterns

##### Pattern: Using nested tasks to achieve nested parallelism

https://docs.ray.io/en/latest/ray-core/patterns/nested-tasks.html

##### Pattern: Using generators to reduce heap memory usage

https://docs.ray.io/en/latest/ray-core/patterns/generators.html

##### Pattern: Using ray.wait to limit the number of pending tasks

https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html

```py
MAX_NUM_PENDING_TASKS = 100
result_refs = []
for _ in range(NUM_TASKS):
    if len(result_refs) > MAX_NUM_PENDING_TASKS:
        ready_refs, result_refs = ray.wait(result_refs, num_returns=1)  # returns a single task immediately
        ray.get(ready_refs)  # wait until a single task is done

    result_refs.append(actor.heavy_compute.remote())  # request a remote task asynchronously

ray.get(result_refs)  # wait until all tasks are done
```





## Ray cluster

https://docs.ray.io/en/master/cluster/getting-started.html

https://docs.ray.io/en/master/cluster/key-concepts.html

- a ray cluster needs to be deployed to use multiple nodes



###### Ray cluster

- has worker nodes
- has a head node

###### Head node

- https://docs.ray.io/en/master/cluster/key-concepts.html#cluster-head-node
- can be a worker node as well
- responsible for cluster management
  - autoscaler
  - GCS
  - the Ray driver processes which run Ray jobs

###### Worker node

- connected to a head node

###### Autoscaling

- a process
- can try to add or remove worker nodes
- decided by the number of requests
  - (not by physical resources)

###### Ray jobs

- a single application
- the collection of Ray tasks, objects, and actors that originated from the same script
- `driver`
  - the worker that runs the python script
- ray job ≠ RayJob

### Ray cluster management API

#### Cluster management CLI

https://docs.ray.io/en/latest/cluster/cli.html

- ray start
  - start ray processes manually on the local machine
  - ray start --head
  - ray start --address=HEAD_NODE_IP:HEAD_NODE_PORT
- ray stop 
  - stop ray processes
- ray up CLUSTER_CONFIG_FILE
  - create or update a Ray cluster
- ray down CLUSTER_CONFIG_FILE
  - tear down a Ray cluster
- ray exec CLUSTER_CONFIG_FILE CMD
  - execute a command via SSH on a Ray cluster
- ray submit CLUSTER_CONFIG_FILE SCRIPT [SCRIPT_ARGS]...
- ray attach CLUSTER_CONFIG_FILE
  - create or attach to a SSH session
- ray get_head_ip CLUSTER_CONFIG_FILE
- ray monitor CLUSTER_CONFIG_FILE
  - tails the autoscaler logs of a Ray cluster



## Remarks

### An example ray node `dict`

```python
# ray.nodes()[0]
{
    "NodeID": "b3d3c5f49578205c32afe6d09753c05cd792791ba442a42bc13c8aa7",
    "Alive": True,
    "NodeManagerAddress": "172.24.167.174",
    "NodeManagerHostname": "mypc",
    "NodeManagerPort": 35215,
    "ObjectManagerPort": 42791,
    "ObjectStoreSocketName": "/tmp/ray/session_2025-02-17_10-38-12_546977_17166/sockets/plasma_store",
    "RayletSocketName": "/tmp/ray/session_2025-02-17_10-38-12_546977_17166/sockets/raylet",
    "MetricsExportPort": 51953,
    "NodeName": "172.24.167.174",
    "RuntimeEnvAgentPort": 56808,
    "DeathReason": 0,
    "DeathReasonMessage": "",
    "alive": True,
    "Resources": {
        "CPU": 20.0,
        "node:__internal_head__": 1.0,
        "GPU": 1.0,
        "memory": 3556218471.0,
        "node:172.23.156.163": 1.0,
        "object_store_memory": 1778109235.0,
    },
    "Labels": {
        "ray.io/node_id": "b3d3c5f49578205c32afe6d09753c05cd792791ba442a42bc13c8aa7"
    },
}
```

