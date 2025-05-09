# Unreal engine

## TODO

- camera distortion

  - https://dev.epicgames.com/documentation/en-us/unreal-engine/camera-lens-calibration-quick-start-for-unreal-engine

- shadow catcher

  - https://dev.epicgames.com/community/learning/tutorials/ByW6/unreal-engine-shadow-catcher-render-pass
  - https://youtu.be/JqoUg5hFFF4?si=wD9gg5w7joO1zyQ6

- putting an image at the back

- save as mask

  - https://youtu.be/PiQ_JLJKi0M?si=ePFP-xpLre1YgZD5

- adding a dynamic asset

  - https://github.com/TriAxis-Games/RealtimeMeshComponent
  - https://github.com/timdecode/UE4RuntimeMeshComponent
  - https://docs.unrealengine.com/4.26/en-US/API/Runtime/Engine/Engine/UAssetManager/AddDynamicAsset/
    - ??

## Class naming

Prefixes

- `A`: Objects that inherit `AActor` e.g. `ACharacter`, `APlayerController`
- `U`: Objects that don't inherit `AActor` but `UObject`. e.g.  `UActorComponent`, `UUserWidget`
- `F`: general class or struct  that doesn't inherit `UObject`. e.g.: `FVector`, `FRotator`
- `I`: Interface classes. e.g. `IInterface`
- `T`: Template classes. e.g. `TArray`, `TMap`

## Tips

- change sun direction
  - hold ctrl + L
- install python packages
  - Engine/ThridParty/Binaries/Python3/Win64/python.exe -m pip install ...
- texture streaming
  - edit "C:\Program Files (x86)\Epic Games\Launcher\Engine\Config\Windows\WindowsEngine.ini"

## Command line

- https://dev.epicgames.com/documentation/en-us/unreal-engine/command-line-arguments-in-unreal-engine
- useful arguments
  - -log
  - -windowed
  - -ResX
  - -ResY

## Trouble shooting

### Texture streaming pool over budget:

`r.Streaming.PoolSize 3000`