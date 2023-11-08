# Blender



## Scripting

```sh
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:savoury1/blender

sudo apt update
sudo apt install blender

blender scene.blend --background --python script.py
blender scene.blend --background --python script.py --python-use-system-env
blender scene.blend --background --python script.py --python-use-system-env -- 1 2 3
```



- `--python-use-system-env` makes packages installed by `pip` available
  - e.g. you may want to install `numpy` in your system python and use it in the blender
- you may pass python arguments after `--`
  - https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script



## Notes

- make numpad keys available for tenkeyless keyboards
  - `Edit` > `Preferences` > `Input` > `Keyboard` > `Emulate Numpad`
- make the background transparent
  - `Render` > `Film` > `Transparent`
- make an object a transparent mask
  - requires the background to be transparent first
  - (2.83)
    - Add Material Slot
    - Set `Surface` as `Holdout`
  - (3.6)
    - `Object` > `Visibility` > `Mask` > `Holdout`
- attach camera to the current view
  - `N` > `View` > `Lock` > `Camera to View `
- render the scene
  - `F12`
- save the rendered image when rendering the scene
  - Compositing
    - Check `Use Nodes`
    - `Add` > `Output` > `File Output`
      - set the folder name
    - Connect from the image of `Rendered Layers` to the image of `File Output`
- make an object partially transparent
  - Set `Material` > `Settings` > `Blend Mode` as `Alpha Blend`
  - Modify `Material` > `Surface` > `Alpha`to be less than 1
- group objects
  - change it to `Object Mode`
  - shift+a > `Empty` > `Plain Axes`
  - select children objects at first and the parent object lastly
  - ctrl+p > `Object`
- 