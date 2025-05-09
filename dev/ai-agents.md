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

- Query
  - Loop
    - Thought
      - LLM decides what the next step should be
    - Action
      - The agent calls the tools with the associated arguments
    - Observation
      - The model reflects on the response of the tool

Guided to do so by the system messages

The method called ReAct. See below.



### Thought: Internal Reasoning and the ReAct Approach

https://huggingface.co/learn/agents-course/unit1/thoughts



Chain-of-Though (CoT)

- Generate intermediate texts for the reasoning inference steps
- may use special tokens `<think>` and `</think>`
  - which is not zero-shot
  - e.g.
    - Deepseek R1
    - OpenAI's o1

ReAct

- **Re**asoning with **Act**ing
- Zero-shot-CoT
  - Let's think step by step
- https://arxiv.org/pdf/2210.03629
- ICLR 2023



### Actions: Enabling the Agent to Engage with Its Environment

https://huggingface.co/learn/agents-course/unit1/actions

- type of agent
  - JSON agent
    - Specify an action in JSON format
  - code agent
    - Writes a code to run
  - function-calling agent
    - a JSON agent
    - finetuned to generate a new message for each action
- type of action
  - information gathering
  - tool usage
  - environment interaction
  - communication
    - engaging with users via chat
    - or collaborating with other agent
- Stop generating new token when an action is "complete".
  - and take the action
- parse
  - 



## 1.5 Fine-tuning an LLM for function-calling



## 2 Frameworks for AI agents



## 2.1 The smolagents framework



## 2.2 The LlamaIndex framework



## 2.3 LangGraph



## 3 Use cases



## 4 Final assessment with benchmark

