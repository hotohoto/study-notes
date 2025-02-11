# Ray

## Ray Core

.get()

- sync

.remote()

- async

.wait()

- 

### User Guides

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





