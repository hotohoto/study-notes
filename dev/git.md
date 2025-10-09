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

## Authentication via git credential helper

- configs
    - `credential.helper`
        - the git config that points to a script poops out username and passwords.
        - (the script is called a "git credential helper")
- commands
    - (interfaces - https://git-scm.com/docs/git-credential/2.32.0)
        - `git credential fill`
            - invoke the current git credential helper to receive credentials
        - `git credential approve`
            - invoke the current git credential helper to keep the credentials
        - `git credential reject`
            - invoke the current git credential helper to remove the credentials
    - (built-in git credential helpers)
        - `git credential-cache`
            - keeps the password in memory.
        - `git credential-store`
            - keeps the password in a plain text file.
            - DO NOT use this.

Example usage:

```bash
git config --global credential.helper 'cache --timeout=86400'  # 1 day
git config --global credential.helper 'cache --timeout=2592000'  # 1 month
git config --global credential.helper 'cache --timeout=31536000'  # 1 year
git config --global credential.helper 'cache --timeout=315360000'  # 10 years

git fetch
```

```bash
git config --global credential.helper store  # save credentials globally as plain text
```

## Shared local git repository

```
git config --global --add safe.directory /path/to/shared/repo

cd /path/to/shared/repo

sudo groupadd git
sudo usermod -aG git username1
sudo usermod -aG git username2

sudo chgrp -R git .git
sudo chmod -R 770 .git
```

- You may need to restart your terminal

## Change editor

```bash
git config --global core.editor "nano"
```

## References

    - https://git-scm.com/docs/gitcredentials
    - https://git-scm.com/docs/git-credential
    - https://git-scm.com/doc/credential-helpers
    - https://tylercipriani.com/blog/2024/07/31/git-as-a-password-prompt/
