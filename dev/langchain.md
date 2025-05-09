# Langchain



## Libraries



- langchain
  - modules
  - LCEL
    -  `Runnable` interface???
  - composition
    - LLM or Chatmodel
    - prompt template
    - Output parser
  - components
    - BaseMessage
      - content
      - role
        - human/ai/system/outputs...
  - etc.
    - https://python.langchain.com/docs/modules/model_io/chat/streaming
- langsmith
  - remote LLM service??
- langserve
  - langchain API provider for applications



## External libraries

- llama-cpp-python
  - uses cpu by default

```
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```





## Example codes

```py
# https://python.langchain.com/docs/integrations/chat/llama2_chat
# 

from os.path import expanduser

from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_experimental.chat_models import Llama2Chat

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)


model_path = expanduser("~/models/llama-2-13b-chat.Q5_K_M.gguf")

model = Llama2Chat(
    llm=LlamaCpp(
        model_path=model_path,
        streaming=False,
    )
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

print(
    chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
)

print(chain.run(text="Tell me more about it."))
```

