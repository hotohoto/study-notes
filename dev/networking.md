# Networking

## Pinpoint process using a specific port

```bash
lsof -i :8080
```

## VPN using tailscale

Depending on the firewall set up it may not work.

```bash
tailscale status  #
tailscale ping my_device_name  # works
ping ip_addr_to_my_device  # doesn't work
```

Refer to the links below.

- https://tailscale.com/kb/1023/troubleshooting/
- https://tailscale.com/kb/1181/firewalls/

## ngrok

Even though tailscale doesn't work ngrok can work.


```bash
# go to https://ngrok.com/ and sign in
# get your authentication token
ngrok config add-authtoken <token>
ngrok http 8080
```

## SSH port forwarding

```bash
# ssh user@server -L local_port1:dest_ip:remote_port1 -L local_port2:dest_ip:remote_port2 -- command_to_run
ssh myuser@10.90.1.11 -L 8000:127.0.0.1:8000 -- python3 -m http.server
```
