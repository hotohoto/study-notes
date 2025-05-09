# Obsidian





- Using vault in a WSL path doesn't work
  - https://forum.obsidian.md/t/support-for-vaults-in-windows-subsystem-for-linux-wsl/8580/69
- Using vault in a windows path looks slow.

## Install in WSL

- download `.deb` file from https://github.com/obsidianmd/obsidian-releases/releases

```
sudo dkpg -i obsidian_1.8.10_amd64.deb
```

- Korean are broken 🫤

- If the taskbar window caption shows [WARN: COPY MODE]
  - https://github.com/microsoft/wslg/discussions/312

```
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt-get update && sudo apt upgrade
```

