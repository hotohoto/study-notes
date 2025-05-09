## Mount a Linux drive

- you can mount a Linux drive following https://learn.microsoft.com/en-us/windows/wsl/wsl2-mount-disk
- I just format the drive in WSL as if I'm on the plain Ubuntu system

  https://phoenixnap.com/kb/linux-format-disk
- but it needs to be mounted when ever you restart the system
- there's a workaround: https://github.com/microsoft/WSL/issues/6073#issuecomment-1774085064



Run `setup.bat` as Administrator

```bat
@echo off
setlocal

:: Set your physical drive variable ⭐
set PHYSICAL_DRIVE=PHYSICALDRIVE1

:: Create the task
schtasks /create /tn "mount-wsl-disks" /tr "C:\Windows\System32\wsl.exe --mount \\.\%PHYSICAL_DRIVE% --bare" /sc onstart /ru "HOTOHOTO-GGAI\hotohoto" /rl highest /f

:: Note:
:: /tn specifies the name of the task
:: /tr specifies the command to run
:: /sc specifies the schedule, in this case, "onstart" for when the system starts
:: /ru specifies the user context under which the task should run ⭐
:: /rl specifies to run with highest privileges
:: /f forces the creation of the task and overwrites if exists

endlocal
```



`/mount.sh`

```bash
#!/bin/bash

# Task name
task_name="mount-wsl-disks"

# Windows system path
windows_sys_path="/mnt/c/Windows/system32"

# Command to execute task
task_command="schtasks.exe /run /tn"

# Timeout and interval in seconds
timeout=10
interval=1

# Drives to check and mount ⭐
candidate_target_drives=(
    "/dev/sdc"
    "/dev/sdd"
)

target_drive=""

for drive in "${candidate_target_drives[@]}"; do
    if [ ! -e $drive ]; then
        # Execute the task
        $windows_sys_path/$task_command "$task_name"
        target_drive=$drive

        # Initialize elapsed time counter
        elapsed=0

        # Wait for the target drive to be ready
        while [ ! -e $target_drive ]; do
            if [ $elapsed -ge $timeout ]; then
                echo "Timed out waiting for $target_drive to be ready."
                exit 1
            fi

            echo "Waiting for $target_drive to be ready..."
            sleep $interval
            elapsed=$((elapsed + interval))
        done
        break
    fi

    fdisk -l $drive | grep "Disk model: SHPP41-2000GM" > /dev/null
    if [ $? -ne 0 ]; then
        continue
    fi
    fdisk -l $drive | grep "2000398934016 bytes" > /dev/null
    if [ $? -ne 0 ]; then
        continue
    fi
    target_drive=$drive
    break
done

# Mount point ⭐
mount_point="/data"

# Mount the target drive
mount $target_drive $mount_point
```



`/boot.sh`

```bash
#!/bin/bash
/bin/bash /mount.sh > /mount.log 2>&1
```



`/etc/wsl.conf`

```
[boot]
command="bash /boot.sh"
```



If you need to wait a folder within the drive to be mounted you may use this script.

```bash
FOLDER=/data/hyheo/.pyenv

for i in {1..10}; do
  if [ -d "$FOLDER" ]; then
    break
  else
    echo "Folder $FOLDER is not available, waiting 1 second"
    sleep 1
  fi
done
```





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
