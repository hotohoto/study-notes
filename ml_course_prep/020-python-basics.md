# Python basics

## Visual Studio Code setup

extensions

- `pyright`
- `python`

packages

- `pylint`
- `black`

set `python formatting provider` as `black`

## Debugging

- run / step over / step into / step out / restart / stop
- watch variable
- run codes interactively

`launch.json` example

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "main",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal"
        }
    ]
}
```

## 자습 (예습)

- https://tutorial.djangogirls.org/ko/python_introduction/
- what is the virtual environment for python?

## 수업

- ml_course repo
- http://www.jollybus.kr/dev/2019/03/05/python-intro/

## 숙제 (복습)

### 1부터 100까지의 합 구하고 화면에 출력하기

### 리스트 관리자 만들기

- 1-4 까지의 명령 번호를 받는다 (루프를 통해 계속해서 입력받을 수 있게 한다)
- 1 append (뒤에 추가)
  - 사용자 입력을 받아서 리스트 맨 마지막에 넣는다
- 2 remove (찾아서 삭제)
  - 사용자 입력을 받아서 리스트에서 찾은 뒤 모두 지운다
- 3 show (현재 항목 출력)
- 4 exit (나가기)
  - 인사하고 나간다
- 그 밖의 입력을 받으면 선택가능한 명령 목록을 출력한다

예시:

```text
select action: 0
1: append, 2: remove, 3: show, 4: exit
select action:
1: append, 2: remove, 3: show, 4: exit
select action: 1
item name: cat
select action: 1
item name: dog
select action: 3
cat
dog
select action: 2
item name: cat
select action: 3
dog
select action: 2
horse
no such item!
select action: 4
bye~
```

### Set 관리자 만들기

- 1-4 까지의 명령 번호를 받는다 (루프를 통해 계속해서 입력받을 수 있게 한다)
- 1 append (뒤에 추가)
  - 사용자 입력을 받아서 Set 에 넣는다
- 2 remove (찾아서 삭제)
  - 사용자 입력을 받아서 Set에서 지운다
- 3 show (현재 항목 출력)
- 4 exit (나가기)
  - 인사하고 나간다
- 그 밖의 입력을 받으면 선택가능한 명령 목록을 출력한다

예시:

```text
1: append, 2: remove, 3: show, 4: exit
select action: 0
1: append, 2: remove, 3: show, 4: exit
select action:
1: append, 2: remove, 3: show, 4: exit
select action: 1
item name: cat
select action: 1
item name: dog
select action: 3
cat
dog
select action: 2
item name: cat
select action: 3
dog
select action: 2
horse
no such item!
select action: 4
bye~
```

### 피보나치(Fibonacci) 수열 출력하기

- 화면에 50번째 까지의 피보나치 수열을 출력한다.
- 재귀함수를 사용한다
- 성능 개선을 위해 캐시를 사용한다

```text
1: 1
2: 1
3: 2
4: 3
5: 5
6: 8
7: 13
8: 21
...
50: 12586269025
```

### 현재 폴더 파일 목록 출력하기

```bash
python ls.py
```

결과:

```text
a.txt
b.txt
c.txt
...
```

### 특정 파일 읽어서 출력하기

```bash
python read.py abc.txt
```

결과:

```text
Hello World from abc.txt
```

### 파일 목록 출력하고 사용자가 선택한 파일 읽어서 내용 출력하기

- 현재 폴더의 파일 목록을 알파벳 순서대로 번호와 함께 출력한다.
- 디렉터리는 표시하지 않는다.
- 사용자 입력을 받아 파일을 선택한다.
- 선택된 파일의 내용을 화면에 출력한다.

### 파일 목록 출력하고 사용자가 선택한 파일 읽어서 내용 출력하기 MVC 버전

- 현재 폴더의 일반 파일 및 디렉터리 목록을 알파벳 순서대로 번호와 함께 출력한다.
- 사용자 입력을 받아 일반 파일 이나 디렉터리를 선택한다.
- 디렉터리를 선택하면 다른 디렉터리로 이동할수 있다.
- 일반 파일을 선택하면 내용을 화면에 출력한다.

- [MVC 패턴](https://www.guru99.com/mvc-tutorial.html) 참고
  - `main.py`
    - 각 모듈들 연결
    - 기타 필요한 초기화
    - view 의 루프 시작
  - `view.py`
    - 루프에서 메뉴를 화면에 출력하고 사용자 입력을 받도록 함.
    - 메뉴를 그리기 위해 모델에 있는 데이터를 받아 올 수 있음.
    - `현재 폴더`나 `현재 읽을 대상파일` 등의 상태를 변경해야 하는 경우 controller 에 요청한다.
  - `controller.py`
    - 회원가입이라는 단일 비지니스 로직 함수가 있다면 그 안에서 회원가입 가능 여부 확인, 가입 메일 발송, db에 상태 변경 등 여러가지 작업을 하게 된다. controller는 비지니스 로직별 해야할 일들을 총괄한다고 할 수 있다. 이 과제에서는 단순히 몇 가지 상태 변경 요청을 모델에 전달해주도록 한다.
    - `현재 폴더` 변경 원하면 요청 model에서 수정하기
    - `현재 읽을 대상 파일` 변경 원하면 요청을 model에서 수정하기
  - `model.py`
    - 파일 목록 받아주기
    - 파일 내용 읽어주기
    - `현재 폴더` 변수로 가지고 있기
    - `현재 읽을 대상 파일` 변수로 가지고 있기
- 사용할 수 있는 libraries
  - `open()`
    - 파일을 읽거나 쓰기위해 연다.
  - `os.path.join()`
    - 여러개의 경로 문자열(str)들을 모아서 하나의 경로 문자열을 만들어 리턴한다.
  - `os.path.isfile()`
    - 입력인자 경로에 해당하는 파일이 디렉터리나 링크 같은 특별한 파일이 아니라 일반적인 파일인지 확인하고 True 또는 False를 리턴한다.
  - `os.path.isdir()`
    - 입력인자 경로에 해당하는 파일이 디렉터리인지 확인하고 True 또는 False를 리턴한다.
  - `os.path.listdir()`
    - 입력인자 경로에 있는 파일 이름들을 문자열로 표현한 리스트를 리턴한다.
  - `os.path.abspath()`
    - 입력인자 경로를 시스템의 절대 경로 형태로 바꾼다. 유닉스 리눅스 시스템의 경우 `/` 로 시작하는 경로가 절대 경로이며 윈도우즈의 경우 `C:\` 나 `D:\` 등으로 시작하는 경로가 절대 경로이다.

```text
1. hello.txt
2. world.txt
select a file to read: 2

This is the contents of worlds file.
```

### 파일에 1부터 100까지 쓰기

- numbers.txt 파일에 1부터 100까지를 출력한다.

### 특정 파일 읽어서 줄 단위로 출력하는 iterator 만들기

(1단계)

- 내용은 한번에 읽어서 따로 저장해두기
- iterator 클래스 만들기. __getitem__ 과 __len__ 사용
- for문 사용해서 한줄씩 혹은 뒤에서부터 한줄씩 잘 꺼내주는지 확인하기

(2단계)

- 동일한 내용을 __iter__ 사용해서 구현하기.

(3단계)

- 파일 읽고 닫는 부분을 with clause 사용해서 같은 클래스로 처리하기.

### 걸린 시간 측정해보기

1. 시간을 체크한다.
2. 원하는 동작을 한다.
3. 시간을 체크해서 처음 시간을 뺀 후 출력한다..
4. `with` 사용해서 하기. __enter__ __exit__ 사용해서 해본다.
5. decorator `@` 사용해서 해본다.

## Class

### mro

- http://www.srikanthtechnologies.com/blog/python/mro.aspx

## References

- https://www.w3schools.com/python/
- [python quick basic in Korean](https://tutorial.djangogirls.org/ko/python_introduction/)
- [python code visualization](http://pythontutor.com/)
- https://docs.python.org/3/tutorial/
- https://docs.python.org/ko/3/tutorial/index.html
- Fluent Python
- https://cscircles.cemc.uwaterloo.ca/visualize
