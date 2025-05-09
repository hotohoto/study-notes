# Obsidian

## Settings

- Appearance
	- Turn off `Show inline title`
- Editor
	- Turn off `Show indentation guide`

## Community addons to install

- linter
	- Lint on save.
	- Inserts the file name as a H1 heading if no H1 heading exists.
	- Heading levels should only increment by one level at a time.
	- Remove two or more consecutive spaces. Ignore spaces at the beginning and ending of the line.
	- There should be at most one consecutive blank line.
	- All headings have one blank line before and after
		- Bottom
		- Empty line between YAML and Header
	- Ensures that there is exactly one line break at the end of a document if the note is not empty.
	- There should not be any empty lines between list markers and checklists.
	- Remove extra spaces after every line

## WSL issues

- Using vault in a WSL path doesn't work
	- https://forum.obsidian.md/t/support-for-vaults-in-windows-subsystem-for-linux-wsl/8580/69
- Using vault in a windows path looks slow.
- Install obsidian in WSL is possible (seems not recommended though)
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
