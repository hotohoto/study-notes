
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
