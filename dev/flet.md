# Flet

(Tested in WSL)

```bash
uv init try-flet
cd try-flet
uv add flet
uv run flet create
uv run flet run
uv run flet build apk
```

- You can find your apk file in `./build/flutter/build/app/outputs/flutter-apk/*`

## Trouble shooting

```
~/.flet/client/flet-desktop-light-0.80.5/flet/flet: error while loading shared libraries: libsecret-1.so.0: cannot open shared object file: No such file or directory
```

```bash
sudo apt install libsecret-1-0
```
