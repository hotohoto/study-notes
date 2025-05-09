# Helm

## Glossary

https://helm.sh/docs/glossary/

- helm
  - the package manager for Kubernetes
- chart
  - a helm package
- release
  - an installed instance of a chart

## Install helm

- Refer to https://helm.sh/docs/intro/install/

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh

sudo sh -c "helm completion bash > /etc/bash_completion.d/helm"
```



## Add a repository

```bash
helm repo add harbor https://helm.goharbor.io
helm repo update
```



## Install a chart

```
helm install my-harbor harbor/harbor
helm get values
```
