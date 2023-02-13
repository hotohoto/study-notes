# Python intermediate

TODO:

- logging
- profiling
- threading
  - https://realpython.com/intro-to-python-threading/
- async io
- socket
- multi-processing
- lock
- design pattern
  - https://python-patterns.guide/
  - https://www.toptal.com/python/python-design-patterns
- regex
  - https://regexr.com/
  - find and replace in IDE
- pandas
  - https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f

## 숙제 (복습)

## python memory management

[Python Memory Management](https://realpython.com/python-memory-management/)

- `sys.getrefcount(object)`
- `gc.collect()`

## Coding tips

- make constants
  - prevents typo
  - easy to change later
- ask and follow coding conventions
  - `pep8`, `black`
  - use spaces rather than tab
  - remove trailing white spaces
  - put trailing new line at the end of file
- avoid circular dependencies
- name by what rather than how
  - easy to change implementation
- keep it simple and stupid
  - make it stateless if possible (avoid stateful behaviors)
    - a stateless function might be called pure function
  - single responsibility
- reuse codes as much as possible
- separation of concern
  - e.g. Model-View-Controller
- clean code is better than clever code usually
  - avoid comments and let codes speak
- write tests
  - helps code be refactored easily
- don't leave commented out codes
  - use git instead to see history

## Pytest

- why test?
  - helps code be refactored easily
  - ci test
- run with path
- `-k`: run finding test with keyword
- `-s`: run printing messages
- `pytest -s --log-cli-level=DEBUG`
- mock and patch

## References

- Fluent Python
- https://python-patterns.guide/
- https://www.toptal.com/python/python-design-patterns
