# Git

## Git sparse-checkout

- decide what to show
- cone mode
  - show all the descendants from a folder recursively
- commands
  - git sparse-checkout init
    - turn on cone mode
  - git sparse-checkout disable
    - show all files again
  - git sparse-checkout set
    - set the list of files to show
    - hide all the others
  - git sparse-checkout add
    - add some files to the list of files to show
  - git sparse-checkout list
    - list the files to show

## Git LFS

### Client

Install the client

- https://git-lfs.com/
  - made by GitHub

Basic usage

```bash
# validates git lfs
git lfs install  # no message expected

git lfs track path/to/file/or/pattern
git add .gitattributes
git add path/to/file
git commit
git push
```

Download a large file directly (FIXME)

```bash
# out of project
git lfs clone --include=”path/to/file” --exclude=”*”

# within project
git lfs pull path/to/file
```

Setup git lfs server

```bash
# check the current entrypoint
git lfs env

# set url
git config lfs.url "http://localhost:9999"
```



### Servers

- GitHub
  - 2GB file size limit 👎
  - total file size limit
  - download bandwidth limit
  - no upload bandwidth limit
- (the other servers)
  - https://github.com/git-lfs/lfs-test-server
  - https://github.com/git-lfs/git-lfs/wiki/Implementations



## GitHub



## How to generate a key

```shell
ssh-keygen -t ed25519 -C "your_email@example.com"
```