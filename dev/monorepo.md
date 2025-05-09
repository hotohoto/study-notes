# Monorepo



- goals
  - trunk based development
  - source code level internal dependency management
  - (use the same external dependency internally)
- pros
  - change multiple packages at the same time via atomic commits
  - easy to use shared modules
  - everybody can focus on HEAD revision
- Google, Facebook, Netflix, and Uber are known to use monorepo and trunk based development
- build system types
  - DAG based
    - the root build settings manage the entire system
    - e.g. Bazel, Buck, Pants
  - recursive based
    - each module has its own build system
    - e.g.  Make, CMake, Gradle



## References

- https://trunkbaseddevelopment.com/monorepos/
- https://trunkbaseddevelopment.com/expanding-contracting-monorepos/
  - https://monorepo.tools/