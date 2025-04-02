# AI-agents



(external links)

- https://huggingface.co/docs/smolagents/en/index
  - https://github.com/huggingface/smolagents
- https://huggingface.co/agents-course
  - https://huggingface.co/learn/agents-course
  - https://github.com/huggingface/agents-course
- https://huggingface.co/blog/smolagents



(TODO)

- deadline: 2025 May 1st
- 



## 1 Introduction to agents

https://huggingface.co/learn/agents-course/unit1/introduction

- agent
- actions
  - can use multiple tools
- tools



### Messages and special tokens

https://huggingface.co/learn/agents-course/unit1/messages-and-special-tokens

- chat-templates

  - (includes)

    - system messages

      - list of tools

      - how to format actions
      - guides for the thought process

    - conversations

      - user messages
      - assistant Messages

  - e.g.

    - chatml
      - https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md

  - references

    - https://huggingface.co/docs/transformers/main/en/chat_templating

### What are tools?

https://huggingface.co/learn/agents-course/unit1/tools

```py
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

Then, the tool description like below is provided to LLM as a part of the system message.

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```



- https://modelcontextprotocol.io/
- https://github.com/modelcontextprotocol/python-sdk

### Understanding AI agents through the thought-action-observation cycle

https://huggingface.co/learn/agents-course/unit1/agent-steps-and-structure

TODO



## 1.5 Fine-tuning an LLM for function-calling



## 2 Frameworks for AI agents



## 2.1 The smolagents framework



## 2.2 The LlamaIndex framework



## 2.3 LangGraph



## 3 Use cases



## 4 Final assessment with benchmark

