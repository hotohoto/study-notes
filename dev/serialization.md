# Serialization

Sorted by personal preference

## JSON

- pros
- cons
  - slow for big objects

## yaml

- pros
- cons
  - slow for big objects

## msgpack

- how it works
  - json like, but binary and efficient
- pros
  - much faster than JSON or yaml
  - One or more CLI tools are available that can convert them to JSON
- cons
  - a bit worse readability than text formats

## pickle

- pros
  - easy to use when it comes to python
- cons
  - works only with python
  - too flexible and hard to manage unless it was used in a proper way

## FlatBuffers

- how it works
  - generates codes
- pros
- cons
  - not self-describing
  - bad at version management
    - https://google.github.io/flatbuffers/flatbuffers_guide_writing_schema.html#autotoc_md34
    - guides
      - don't remove fields but deprecate
      - add new fields at the end only

## Avro

- how it works
  - there is a schema and it is embedded in the data
  - serializes data into a compact binary format





