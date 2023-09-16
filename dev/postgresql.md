# postgres cli

## run as a docker container

- https://hub.docker.com/_/postgres

```bash
# docker run --name some-postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres
docker run --name postgres -e POSTGRES_PASSWORD=pass -d postgres -p 5432:5432
```





## basic commands

```bash
#psql -U username -h localhost -p 5432 dbname
psql -U postgres -h localhost -p 5432 postgres
```

```sql

SELECT current_database();

-- list databases
\l

-- list schema in the current database
\dn

-- list tables from all schemas
-- (if `*.*` is omitted will only show SEARCH_PATH ones)
\dt *.*

-- list tables in the given schema
\dt schema_name.*

-- describe a table

\d schema_name.table_name

-- list users
\du

-- show all configurations
SHOW ALL;

-- don't forget to put semicolon at the end
select * from schema_name.table_name;

-- quit
\q
```

## Remote access from all hosts

- https://stackoverflow.com/a/3278835/1874690
- check the location of the configuration file being used
  - `SHOW ALL;`
- edit `/etc/postgresql/10/main/pg.`
  - `host all all 0.0.0.0/0 trust`
    - Add this at the end of the file
    - It trusts hosts without password because of `trust`
- edit `/etc/postgresql/10/main/postgresql.conf`
  - `listen_addresses='*'`
- apply configuration by running either
  - `SELECT pg_reload_conf();` (?)
  - `/etc/init.d/postgresql restart` (?)
  - or `docker restart your_postgres_container`

## dump schema queries

```
docker exec -it my_postgresql_container bash
su postgres
pg_dump -s postgres
```

## references

- https://gist.github.com/Kartones/dd3ff5ec5ea238d4c546
