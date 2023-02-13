
## nameserver issue

- check the DNS server address

## Reset wsl network

(cmd as admin)

```
wsl --shutdown
netsh winsock reset
netsh int ip reset all
netsh winhttp reset proxy
ipconfig /flushdns
```

https://github.com/microsoft/WSL/issues/5336#issuecomment-984995551


## BLUEMAX Client

- uninstall bluemax client
- check the DNS server in Windows

```
ipconfig /all
```

- set the DNS server in WSL

```bash
sudo vi /etc/wsl.conf  #  nameserver x.x.x.x
```

- install bluemax client
- don't use auto login
