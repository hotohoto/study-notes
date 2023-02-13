# SSH

- create a key using `ssh-keygen`
- register your public key to the server by either
    - using GUI
    - or adding it `~/.ssh/authorized_keys` into the server manually
    - or running `ssh-copy-id` in the client

## Example

(windows client)

```powershell
cd .ssh
ssh-keygen.exe -t rsa -b 4096
notepad id_rsa.pub  # copy the public key text
ssh my_server_user@server_host
# ssh my_server_user@server_host -i C:\\Users\\my_local_user\\.ssh\\id_rsa
```
(server)

```bash
echo ~/.ssh/authorized_keys
cat >> ~/.ssh/authorized_keys
# paste the public key text and press CTRL+D
```

## References

- https://amazelimi.tistory.com/6
