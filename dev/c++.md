# C++



```c++
server = new ::rpc::server(RPCPort);
```

- :: 로 시작하는건 현재 namespace가 아니라 전역 namespace에서 rpc::server 를 찾겠다는 의미.



```c++
using namespace std::chrono_literals;
/// Update port number and try again.
std::this_thread::sleep_for(500ms);
```

- 숫자 뒤에 500ms 처럼 숫자 뒤에 저런 suffix literal 을 사용하고 있음.
- 이런 문법은 C++14 에서 도입되었음.



```c++
class MyClass {
public:
    MyClass(int value) : value_(value) {}

private:
    int value_;
};

MyClass obj = 123
```

- 멤버 변수 value_ 를 가짐.
- 생성자는 멤버 변수만 세팅하고 다른 내용은 비어 있음.
- 이게 되네..



```c++
class MyClass {
public:
    explicit MyClass(int value) : value_(value) {}

private:
    int value_;
};
```

- explicit 이 있어서 `MyClass obj = 123` 이거 안됨
- 대신 아래 처럼 해야함.



```c++
MyClass obj(123);
```

- obj 라는 인스턴스 변수를 만듦.
- 타입은 `MyClass`이고
- `MyClass`의 생성자를 부름.