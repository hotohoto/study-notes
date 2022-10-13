# memory management

## Linux memory stats

- RSS or RES
  - Resident Set Size
  - allocated to that process in RAM
  - not includes swapped out memory
  - includes memory from shared libraries
    -  as long as the pages from those libraries are actually in memory
 -  includes all stack and heap memory
- VSZ
  - Virtual Memory Size
  - includes all memory that the process can access
    - memory swapped out
    - memory allocated, but not used
    - memory that is from shared libraries

### htop

- shows threads as well by default
  - press H to toggle user process threads

### time

```sh
/usr/bin/time -v command_to_run
```

- shows max RSS as well taking account for all the child/descendant processes
  - but note that it's not the accumulated sum from all the processes

### stats for docker containers

- `docker stats`
- `htop` in the container
  - It seems to show the stats taking other containers into account.
    - - So it's hard to get stats for each container relying on it.
  - Per process stats looks fine.

### profiling a python program

- define a test case
- try memray
  - memray outputs a bin file
  - memray can create a report from a bin file
    - but it cannot create a report larger than around 20G
  - a long test may generate a too large bin file
  - the generated flamegraph represents the snapshot of the python call frames
    - along with the size stats of allocated memory
    - the snapshot is taken when the largest memory was in use
      - it's not corresponding to a function, but to frames
        - e.g. similar codes of creating a Tensor can be in many places. and the memory size stats are all different depending on the call stack
  - analyze codes to find out the reason why strange memory blocks consuming a large size
    - focus mainly on the memory blocks that are growing
  - fix the code and try it the experiment again to see if it's optimized
  - for a long resource/time consuming test case, we can run in the live mode
    - but not more convenient/efficient than the default mode

```sh
memray run -o my_test_case1.bin -m pytest test_my_test_cases.py::TestMyTestCases::test_my_test_case1
memray flamegraph -f my_test_case1.bin

memray run --live -m pytest test_my_test_cases.py::TestMyTestCases::test_my_test_case1
htop
sudo py-spy dump --locals --pid 1234567 |grep FOO
```

### tensorflow

- try `tf.config.experimental.get_memory_info`
  - check documents for dumping tf debug info
    - https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info
      - https://www.tensorflow.org/tensorboard/debugger_v2

## References

- https://stackoverflow.com/a/21049737/1874690

