# IDE

## Cursor

### Migration from VSCode

- can delete the key binding `ctrl+shift+k` since it's duplicate to `ctrl+k`
- can import settings from vscode in the cursor settings menu
- can change theme to `Dark+` which is similar to vscode
- can toggle right side bar by `ctrl+alt+b`
- enable partial accept in cursor settings
    - can use it by pressing `ctrl+right
    - but it seems to be annoying when I want to move the cursor to the right of the current word. (So I turned it off.)
trouble shooting
- cannot connect to the remote dev container
    - "In order to use Anysphere Remote Containers, `ms-vscode-remote.remote-containers` must be uninstalled" shows up
    - Clicking `Uninstall & Restart Cursor` doesn't resolve the problem
        - Seems like it cannot be uninstalled while in use
    - Open a local project and remove `ms-vscode-remote.remote-containers` manually in Cursor
    - Try to reopen the remote dev container project again
