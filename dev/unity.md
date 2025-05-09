# Unity

## Software Architecture

- Scenes
- GameObjects
  - have coordinates
  - serializable
- Components
  - defines behavior of game objects
  - members
    - OnEnable()
    - OnDisable()
    - Start()
    - OnDestroy()
    - Update()
  - descendents
    - Behavior
      - MonoBehavior
        - for any scripts including C#
        - serializable
  - serializable
- Assets
  - prefab
    - special assets for game objects that are often reused so saved in the storage

## Other terminologies



## References

(Tips for programmers)
- https://blog.eyas.sh/2020/10/unity-for-engineers-pt1-basic-concepts/
- https://unity.com/how-to/programming-unity
- https://answers.unity.com/questions/547496/how-do-i-dynamically-generate-scenes.html
- https://answers.unity.com/questions/12003/instantiate-a-prefab-through-code-in-c.html

(assets)
- https://assetstore.unity.com/templates
