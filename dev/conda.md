## Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /path/to/miniconda3
conda env create --file environment.yaml
conda init

# after reopen the current shell

conda activate environment_name
conda install xformers::xformers  # https://anaconda.org/xformers/xformers
```

