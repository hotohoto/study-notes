# RustDesk



## In the remote Ubuntu machine

```bash
wget https://github.com/rustdesk/rustdesk/releases/download/1.3.9/rustdesk-1.3.9-x86_64.deb
sudo apt install -fy ./rustdesk-1.3.9-x86_64.deb

# stop and disable the service (FIXME)
sudo service rustdesk stop
sudo systemctl disable rustdesk

# delete `enc_id` and set `id` and `password`
# id = '012345678'
# password = 'my_password'
sudo nano /root/.config/rustdesk/RustDesk.toml

# make rendezvous_server empty for security reason
# rendezvous_server = ''
sudo nano /root/.config/rustdesk/RustDesk2.toml

# restart and apply the new settings
sudo service rustdesk restart

# check if id and password are now encoded
sudo cat /root/.config/rustdesk/RustDesk.toml
sudo journalctl -u rustdesk.service
sudo ss -tulnp | grep rustdesk

# uninstall rustdsk
sudo apt remove rustdesk
```



## In the local machine

- Download and install a windows client
  - https://github.com/rustdesk/rustdesk/releases/tag/1.3.9
- Enter the remote machine ip address instead of `id`
  - Password might be still needed
- Enter the `id` and the `password` you set above in the remote machine
- asdfkjfs0323jfkslkdfj0asfjGaskfsdf
- 821084405718