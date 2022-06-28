# Google Cloud Platform

## TODO

try
- IM
- VM
- big query
  - browse public tables
- cloud composer (aiarflow)
- GKE
- container registry
  - https://cloud.google.com/container-registry

## notes

- VM
  - ssh 키 등록해야한다.
    - public key는 GCP의 설정 페이지에 복사해서 붙여넣고
    - private key는 로컬에 가지고 있는다.
- service account key
  - crediential 페이지에서 만든다.
  - role을 owner로 하면 모든 서비스에 다 접근할 수 있다.
  - json 파일로 다운로드 받아서, 서비스를 실행할 곳에 넣어둔다.
  - 개별 API 서비스 종류 별로 enable 눌러 줘야 해당 서비스를 이용할 수 있다.
- python에서 API
  - google-cloud 패키지를 설치하면 모든 서비스에 접근 가능하다.
  - 특정 서비스 FOO 에만 접근할 거면 google-cloud-foo 를 설치하면 된다.
