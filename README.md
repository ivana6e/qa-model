# qa-model

- distilbert: test
- gpt2: test
- ollama: **USE THIS**
- roberta: test

### Docker cmd
```
docker pull ollama/ollama:latest
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama bash
ollama run qwen:7b-chat
```

**You should build docker first to run qwen_rag.py**
