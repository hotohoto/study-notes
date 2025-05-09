# uv



- `dependency-groups`
  - https://packaging.python.org/en/latest/specifications/dependency-groups/
  - e.g.
    - dev
      - `uv add --dev pytest`
- `project.optional-dependencies`
  - they are also called "extras"
  - WARNING:
    - all the optional dependencies might be installed if this project is a uv member of another. 🤔
  - e.g.
    - train
      - uv add diffusers --optional train
    - carla
      - 
    - ...
- don't specify source index in a shared library
  - e.g.
    - torch index server

- add a sub project as a member
  - may require all the optional dependencies of the member 🤔
- add a sub project as an editable package
  - seems like this is preferred to making it a member 🤔