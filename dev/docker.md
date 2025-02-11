# Docker



## Tips and notes



- Save and send a docker image

```bash
docker save name:tag | ssh user@remote-server 'docker load'
```

- ADD vs COPY
  - ADD has more functionality
  - COPY is simpler and more secure
  - 👉 Use COPY unless ADD is required.
- map ports within a range

```bash
-p 6100-6200:6100-6200
```



### An example of building an image from a .tar.gz file including all the contents and the Dockerfile

(files)

- Dockerfile
- src
  - hello.py

```py
print("Hello, World!")
```

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY src/ /app/src/
CMD ["python", "src/hello.py"]
```


```bash
tar cvzf ../hello.tar.gz .
docker build -t hello-world - < ../hello.tar.gz
docker run hello-world
```

