## Engineers shouldn't write ETL

https://multithreaded.stitchfix.com/blog/2016/03/16/engineers-shouldnt-write-etl/

(전통적인 business intelligence 조직)

- 구성
  - ETL developers (doers)
  - report developers (doers)
  - DBA (doers)

(전형적인 data scientist 조직)

- 구성
  - Data Scientists (thinkers)
  - Data Engineers (doers)
  - Infrastructure Engineers (plumbers)
- 문제점
  - 데이터 엔지니어를 둘 정도로 데이터가 크지 않다.
  - 모든 사람이 thinker 가 되려고 한다. ETL 코드만 작성하는 적당한 사람이 되고 싶지 않다.
- 해결
  - 👉 업무 범위를 조정하여 모든 사람이 일을 재미있게 하고 성장할 수 있게 해야 한다.

(제안하는 조직)

- 구성
  - Data Scientists
    - 가용한 레고 블럭으로 새로운 것을 만든다
  - Engineers
    - 필요할 것 같은 레고 블럭들을 미리 만든다
      - Data Scientists 들이 이상항 블록으로 억지로 뭔가 만들려고 하기 전에 먼저 움직일 수 있게 해야 한다.

## Fullstack data science generalists

https://multithreaded.stitchfix.com/blog/2019/03/11/FullStackDS-Generalists/

비지니스 도메인, 인프라, DS까지 vertical 하게 다 할 수 있는 데이터 사이언티스트 되는 것도 꽤 괜찮다.

- 장점
  - 다른 사람 때문에 내가 성공하지 못했다고 하게 하지 말자.
  - 성장하는 재미를 누리게 하자.
    - 비지니스를 위한 처음부터 끝까지를 다 다루는 내공을 쌓는 것도 재미있지 않은가.
    - 나중에 CDO 나 CTO 가 될 수 있을 듯.
  - 회사 전체에 큰 임팩트를 줄 수 있다.
- 단점
  - DS에 집중하지 못할 수 있다.

## Data science tools

- containerization
- infrastructure abstraction
  - Kubeflow, Metaflow
  - helps use the code locally and in production
- workflow orchestration
  - Airflow, Argo, Perfect

## data warehouse designing

- data mart
  - a subject-oriented database
  - often a partitioned segment of an enterprise data warehouse
  - aligned with a particular business unit like sales, finance, or marketing
  - easy to gain actionable insights quickly
- data warehouse
  - consists of data marts

- The Kimball methodology
  - fast to construct
  - star schema
- The Inmon methodology
  - normalized
    - unified source of truth
    - low data redundancy
  - higher complexity
  - data marts are created after the creation of the data warehouse.

## references

- https://outerbounds.com/blog/the-modern-stack-of-ml-infrastructure
- https://teamtopologies.com/key-concepts
