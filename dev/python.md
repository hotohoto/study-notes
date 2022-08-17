# Python

## module, package, import

- module
  - python file
- sub-module
- package
  - directory containing `__index__.py`
- sub-package
- import
  - `import package.or.module`
  - `from package.or.module import sub_package_or_sub_module_or_anything`

TL;DR

- Put your main script in the root folder of the project
- If a script belongs to a module it can refer to its siblings by using relative path like `from .`.

Detailed description

- Running a python script directly, it cannot access to the parent folder or sibling folders.
- Running a python script directly, it cannot import any sibling codes as a module.
- Any module can refer to the sibling codes either by using relative `from` clause with `.`.
- Using absolute path any codes in sub-directory can access to any codes in the project. In this case
  - the target script to run should be at the root folder of the project.
  - otherwise sys.path should contain the root folder of the project.

You may refer to [definite guide to python import statements](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html).

## Debugging

```bash
python -m pdb buggy.py
```

```py
import pdb

pdb.set_trace()
pdb.pm()
```

(commands)

- `l` or `list`
  - show codes
  - `l 20`
- `n` or `next`
- `c` or `cont`
  - continue
- `s` or `step`
  - step in
- `r` or `return`
- `b` or `break`
  - `b 21`
- `cl`
  - clear break point
- `w`

(VSCode)

- you can debug external modules as well
  - set "justMyCode" as false in `luanch.json`

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "my_test_case1",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["test_my_test_cases.py", "-k", "my_test_case1"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

## Coding Convention

- put trailing commas
- 4 spaces instead of tab
- 2 spaces before an inline comment
- inherit from object explicitly although it can be done by default

- No 2 statements in single line if possible.
- No too much reflection
- _ is a prefix convention for marking "internal" fields in class
- `is` is for the same object references
- `==` is for the same values
- for long sentences

```py
my_very_big_string = (
    "For a long time I used to go to bed early. Sometimes, "
    "when I had put out my candle, my eyes would close so quickly "
    "that I had not even time to say "I’m going to sleep."
)

from some.deep.module.inside.a.module import (
    a_nice_function, another_nice_function, yet_another_nice_function)
```

## Cautions

- avoid mutable default argument, or it will be reused.

```py
def append_to(element, to=[]):
    to.append(element)
    return to

my_list = append_to(12)
print(my_list) # [12]

my_other_list = append_to(42)
print(my_other_list) # [12][42]
```

## Pandas

`.ix`

- support reference by index
  - `df.ix[0:3, 1:3]`
    - get 4 rows from 0 to 3
    - get 2 columns from 1 to 2
- support reference by column label
  - `df.ix[:, "A"]`

`.loc`

- support reference by column label
  - `df.ix[:, "A"]`

`.iloc`

- support reference by index (a bit differently from `.ix`)
  - `df.ix[0:3, 1:3]`
    - get 3 rows from 0 to 2
    - get 2 columns from 1 to 2

## Flyweight pattern

```py
# Instances of CheeseBrand will be the Flyweights
class CheeseBrand(object):
    def __init__(self, brand, cost):
        self.brand = brand
        self.cost = cost
        self._immutable = True   # Disables future attributions

    def __setattr__(self, name, value):
        if getattr(self, '_immutable', False):  # Allow initial attribution
            raise RuntimeError('This object is immutable')
        else:
            super(CheeseBrand, self).__setattr__(name, value)


class CheeseShop(object):
    menu = {}  # Shared container to access the Flyweights

    def __init__(self):
        self.orders = {}  # per-instance container with private attributes

    def stock_cheese(self, brand, cost):
        cheese = CheeseBrand(brand, cost)
        self.menu[brand] = cheese   # Shared Flyweight

    def sell_cheese(self, brand, units):
        self.orders.setdefault(brand, 0)
        self.orders[brand] += units   # Instance attribute

    def total_units_sold(self):
        return sum(self.orders.values())

    def total_income(self):
        income = 0
        for brand, units in self.orders.items():
            income += self.menu[brand].cost * units
        return income


shop1 = CheeseShop()
shop2 = CheeseShop()

shop1.stock_cheese('white', 1.25)
shop1.stock_cheese('blue', 3.75)
# Now every CheeseShop have 'white' and 'blue' on the inventory
# The SAME 'white' and 'blue' CheeseBrand

shop1.sell_cheese('blue', 3)    # Both can sell
shop2.sell_cheese('blue', 8)    # But the units sold are stored per-instance

assert shop1.total_units_sold() == 3
assert shop1.total_income() == 3.75 * 3

assert shop2.total_units_sold() == 8
assert shop2.total_income() == 3.75 * 8
```

## CPython specific details

- object
  - `.id()` returns the memory address the object is saved at
  - https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    - integers in [-5, 256] are pre-allocated in memory
    - the other numbers will be created as objects when each number is required
- gc
  - using reference counting scheme
  - circular referenced objects may not be garbage-collected

## VS Code

```json
"python.linting.pylintEnabled": true,
"python.linting.pylintArgs": [
    "--enable=W0611"
],
```

Refer to https://docs.pylint.org/en/1.6.0/features.html

## Etc

- kwargs : keyword arguments that is named arguments

## Class

- immutable object
  - str
  - int
  - tuple
- mutable object
  - list
  - set
  - dict
- containers
  - list
  - tuple
  - set
  - dict

### The standard type hierarchy

- numbers.Number
  - numbers.Integral
    - Integers(int)
    - Booleans(bool)
  - numbers.Real(float)
  - numbers.Complex(complex)
- Sequences
  - Immutable sequences
    - Strings
    - Tuples
    - Bytes
  - mutable sequences
    - Lists
    - Byte Arrays
- Set types
  - Sets
  - Frozen sets
- Mappings
  - Dictionaries
- Callable types
  - User-defined functions
    - attributes
      - `__doc__`
      - `__name__`
      - `__qualname__`
      - `__module__`
      - `__defaults__`
        - 디폴트 파라미터를 볼 수 있음.
      - `__code__`
        - 소스 코드가 어디있는지 찾을 수 있음.
      - `__globals__`
      - `__dict__`
        - 맴버 변수를 볼 수 있음.
        - 이름이 `__` 로 시작하는 맴버 변수 등
      - `__closure__`
      - `__annotations__`
      - `__kwdefaults__`
        - default values of variables following a variable length positional args
  - Instance methods
    - `__doc__`
    - `__func__`
    - `__name__`
    - ...
  - Generator functions
  - Coroutine functions
  - Asynchronous generator functions
  - Built-in functions
  - Built-in methods
  - Classes
    - `__new__`
    - `__init__`
  - Class Instances with the `__call__` method defined
- Modules
- Custom classes
- Class instances
- I/O objects (a.k.a file objects)
- Internal types
  - Code objects
  - Frame objects
  - Traceback objects
  - Slice objects
  - Static method objects
  - Class method objects

Notes:

- numbers.Booleans(bool) 은 numbers.Integral 의 subclass이다. True + True = 2
- python numbers.Real(float) 은 double-precision floating point number (64bit) 이다. 32bit 는 지원하지 않는다.

### Special method names

- `object.__new__()`
  - create an object
- `object.__init__()`
  - customize the object
- `object.__del__()`
  - called when it's about to be destroyed
- `object.__repr__()`
  - for debugging
- `object.__bytes__()`
- `object.__str__()`
- `object.__lt__(self, other)`
- `object.__le__(self, other)`
- `object.__eq__(self, other)`
- `object.__ne__(self, other)`
- `object.__gt__(self, other)`
- `object.__ge__(self, other)`
- `object.__hash__(self, other)`
  - get a hash value to find hash bucket with
  - to be hashable a user-defined class object has to implment none or both of `__eq__` and `__hash__` methods
- `object.__getattr__(self, name)`
  - fallback
- `object.__getattribute__(self, name)`
  - unconditional

## type codes

- `h`: short signed integer
- `B`: unsigned char
- ...

[array — Efficient arrays of numeric values](https://docs.python.org/3/library/array.html)

## more containers

- array.array
  - supports c level types
  - slicing copies the part first
- memoryview
  - can wrap an array
  - slicing doesn't copy anything
- named tuple
  - specialized tuple
  - provides key like naming for each element
- dict
  - keys must be hashable
  - `(1,2,[3,4])` is not hashable whereas `(1,2,(3,4))` is hashable.
  - Since Python 3.7 the insertion order of dict is also kept as LIFO like `collections.OrderedDict`.
    - https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6/39980744
- collections.OrderedDict
- collections.ChainMap
  - makes multiple dicts searchable
  - `import builtins; pylookup = ChainMap(locals(), globals(), vars(builtins))`
- collections.Counter
- bisect.insort
  - keeps sorted by value
- bisect.bisect
- set
  - implemented with hash table
  - elements must be hashable
  - has a significant memory overhead ?? TODO
- frozenset
  - implemented with hash table

## static typing

- it is recommended to use as the code base is getting larger
- we can use `mypy` package for it
- [dropbox' migration case](https://blogs.dropbox.com/tech/2019/09/our-journey-to-type-checking-4-million-lines-of-python/)

## threading

- In most of Python3 implementations not actually executing at the same time because of GIL

- https://realpython.com/intro-to-python-threading/
- https://www.guru99.com/python-multithreading-gil-example.html

## asyncio

## socket programming

## logging

- logger
  - attributes
    - qualname
      - required
      - if the logger name is `a.b.c` and the `qualname` is `a` then its messages will be handled by this logger
    - handler
      - required
  - The root logger accepts any messages which have not been handled by any loggers
  - once a logger was instantiated it is not affected by `logging.config.fileConfig(...)`
- handler
  - there is some example code of custom handler but I'm not sure if it's recommended way of customizing it
- formatter
- adapter
  - to replace messages with a new message including context information

## virtual subclass

- when we want a 3rd party class to inherit our own class. we could do that by registering the 3rd party class within our own class.
- `isinstance` and `issubclass` will returns `True`.
- OurOwnClass.mro() will NOT show the 3rd party class as an ancestor.
- https://stackoverflow.com/a/54049994/1874690

## Extend python with C/C++

`mymodule.c`

```c
#include <Python.h>

static PyObject *method_fputs(PyObject *self, PyObject *args) {

    char *str, *filename = NULL;
    int bytes_copied = -1;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "ss", &str, &filename)) {
        return NULL;
    }

    FILE *fp = fopen(filename, "w");
    bytes_copied = fputs(str, fp);
    fclose(fp);

    return PyLong_FromLong(bytes_copied);
}

static PyMethodDef MyModuleMethods[] = {
    {"fputs", method_fputs, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule",
    "Python interface for C library functions",
    -1,
    MyModuleMethods
};

PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymodule);
}
```

`setup.py`

```py
from distutils.core import setup, Extension

def main():
    setup(name="mymodule",
          version="1.0.0",
          description="Python interface for C library functions!",
          author="Hoyeong Heo",
          author_email="hotohoto82@gmail.com",
          ext_modules=[Extension("mymodule", ["mymodule.c"])])

if __name__ == "__main__":
    main()
```

`test.py`

```py
import mymodule

print(mymodule.__doc__)
print(mymodule.__name__)
mymodule.fputs("Hello World!", "out.txt")
with open("out.txt") as f:
    print(f.read())
```

```bash
python3 setup.py install
pip show mymodule
python3 test.py

# Check and deleted installed module.
# WARNING: Make sure to check the files before deleting them!
find ~/.pyenv/versions/myvenv/ |grep mymodule
rm `find ~/.pyenv/versions/myvenv/ |grep mymodule`
```

References:

- https://realpython.com/build-python-c-extension-module/
- https://docs.python.org/3/extending/extending.html
