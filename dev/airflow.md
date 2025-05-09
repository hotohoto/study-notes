# Airflow tuor

- task orchestration with DAG

## setup

- password is printed in the log
  - `xGhgz4baNnznzAXT`

## concepts

- dag
  - task
- dag run
  - task instance
- logical date

## commands

- `airflow tasks run example_bash_operator 2015-01-01`
  - if there is a dag run failed/suceeded on 2015-01-01 we can run it again
  - you may need to run `airflow tasks backfill` first if there's no dag run for the date
- `airflow tasks backfill example_bash_operator --start-date=2015-01-01 --end-date=2015-02-01`
  - create dag runs covering the rage
- `airflow tasks test dag_id task_id execution_date_or_run_id`
  - run only the task as a unit test

## variables

- make your variables to be in a json file
- import the file in the web UI
- load it in the DAG file as follows

```py
from airflow.models import Variable
dag_config = Variable.get("example_variables_config", deserialize_json=True)
var1 = dag_config["var1"]
```

- or you can use them directly with Airflow's Jinja template features

```py
t3 = BashOperator(
  task_d="get_variable_json",
  bash_commands="echo {{var.json.example_variable_config.var3}}",
  dag=dag
)
```
- you can get/set a variable value by running a CLI command

```bash
airflow variables --get var1
airflow variables --set var4 value4
airflow variables --import /usr/loca/airflow/dags/config/example_variable.json
```

## questions.

- bigquery operator
  - options
    - use_legacy_sql
      - https://cloud.google.com/bigquery/docs/reference/standard-sql/migrating-from-legacy-sql
    - write_disposition
    - destination_dataset_table

## References

- https://www.talend.com/resources/what-is-data-mart/
- https://youtube.com/playlist?list=PLYizQ5FvN6pvIOcOd6dFZu3lQqc6zBGp2
