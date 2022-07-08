# dvc

(Initialize)

```bash
git init
dvc init
git commit -m "Initialized DVC"
```

(data versioning)

```bash
dvc add data/data.xml
git add data/data.xml.dvc data/.gitignore
git commit -m "Add raw data"

# storing and sharing
dvc remote add -d storage s3://mybucket/dvcstore
git add .dvc/config
git commit -m "Configure remote storage"
dvc push

# retrieving
dvc pull

# making changes
dvc add data/data.xml
git commit -m data/data.xml.dvc -m "Dataset updates"
dvc push

# switching between versions
git checkout <...>
dvc checkout
```

(Share a cache for big datasets)

```bash
# preparation
mkdir -p /home/shared/dvc-cache

# transfer existing cache (optional)
mv .dvc/cache/* /home/shared/dvc-cache
sudo find /home/shared/dvc-cache -type d -exec chmod 0775 {} \;
sudo find /home/shared/dvc-cache -type f -exec chmod 0444 {} \;
sudo chown -R myuser:ourgroup /home/shared/dvc-cache/

# configure the shared cache
dvc cache dir /home/shared/dvc-cache
dvc config cache.shared group
dvc config cache.type symlink

git add .dvc/config
git commit -m "config external/shared DVC cache"
```
