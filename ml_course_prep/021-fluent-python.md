# Fluent python summary

## 16

- priming
  - call next() to make it ready to receive a value from `send()`
- getgeneratorstate()
  - https://docs.python.org/3/library/inspect.html#inspect.getgeneratorstate
  - GEN_CREATED
  - GEN_RUNNING
  - GEN_SUSPENDED
  - GEN_CLOSED
  - since python 3.2
- getcoroutinestate()
  - CORO_CREATED
  - CORO_RUNNING
  - CORO_SUSPENDED
  - CORO_CLOSED
  - since python 3.5

## 17

- concurrent.futures
  - runs on a thread or process pool
- asyncio
  - runs in a single thread
- https://idolstarastronomer.com/two-futures.html

## 18

- awaitables
  - coroutine
  - `async def xxx()` function
  - asyncio.Task
  - subclass of Future
  - asyncio.Future
  - low level
  - https://docs.python.org/3/library/asyncio-task.html#awaitables
- 네트워크나 파일 입출력시 asyncio 가 잘 지원되지 않는다.
  - https://stackoverflow.com/questions/33824359/read-file-line-by-line-with-asyncio
  - 직접 쓰레드를 써서 구현할수도 있다.
  - aiohttp 나 aiofiles 같은 라이브러리를 쓰는 방법도 있다.
- async / await
  - introduced in python 3.5
  - https://snarky.ca/how-the-heck-does-async-await-work-in-python-3-5/

## 19

Dynamic Attributes and Properties. In programming languages like Java, it’s considered bad practice to let clients directly access a class public attributes. In Python, this is actually a good idea thanks to properties and dynamic attributes that can control attribute access.

- `__getattr__`
- `__setattr__`
- `__new__`
- `@property`
- `@xxx.setter`
- `@xxx.deleter`

## 20

Chapter 20: Attribute Descriptors. Descriptors are like properties since they let you define access logic for attributes; however, descriptors let you generalize and reuse the access logic across multiple attributes.

- `__get__`
- `__set__`
- `__delete__`

```py
class RevealAccess(object):
    """A data descriptor that sets and returns values
       normally and prints a message logging their access.
    """

    def __init__(self, initval=None, name='var'):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print('Retrieving', self.name)
        return self.val

    def __set__(self, obj, val):
        print('Updating', self.name)
        self.val = val

class MyClass(object):
    x = RevealAccess(10, 'var "x"')
    y = 5

m = MyClass()
m.x
```

Shortcomings

- There is no way to get the managed class' attribute name programmatically unless we make use of a class decorator or a metaclass.

## 21

Chapter 21: Class Metaprogramming. Metaprogramming in Python means creating or customizing classes at runtime. Python allows you to do this by creating classes with functions, inspecting or changing classes with class decorators, and using metaclasses to create whole new categories of classes.

```py
class MyMeta(type):
    def __new__(meta, name, bases, dct):
        print '-----------------------------------'
        print "Allocating memory for class", name
        print meta
        print bases
        print dct
        return super(MyMeta, meta).__new__(meta, name, bases, dct)
    def __init__(cls, name, bases, dct):
        print '-----------------------------------'
        print "Initializing class", name
        print cls
        print bases
        print dct
        super(MyMeta, cls).__init__(name, bases, dct)

class MyKlass(object, metaclass=MyMeta):

    def foo(self, param):
        pass

    barattr = 2
```
