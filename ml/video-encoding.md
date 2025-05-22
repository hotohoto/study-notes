# video-encoding

## Make a video from the blender output images

```bash
ffmpeg -f image2 -start_number 1 -i "outputs/Image%04d.png" -framerate 24 -c:v mpeg4 -q:v 2 video.avi
```

    - `-start_number`
        - starting file index number
    - `-c:v`
        - codec
    - `-q:v`
        - video quality
