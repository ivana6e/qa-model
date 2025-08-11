# qa-model

## Embedding Model
* BAAI/bge-m3

## LLM Model
* qwen:7b-chat
* llama3
* gpt-oss

## LLM Server 
* distiller: test
* gpt2: test
* ollama: using
* roberta: test

### ollama:
create container and rum llm model

```
docker pull ollama/ollama:latest
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama bash
ollama run qwen:7b-chat 
ollama run llama3
ollama run gpt-oss
```
## Run Example
```
py qwen_rag.py
```

**You should build docker first to run qwen_rag.py**
