# Local LLMs - DSPY w/ Llama3 & Qwen2
This repository contains code that is a simple test of Llama3 and Qwen 2.

## Prereqs
1. Install Ollama - https://ollama.com/download

    - Download Qwen2 `ollama run qwen2` ** This requires 4.4GB of vram. If you need lower resources visit https://ollama.com/library/qwen2 to find the command for lower resource variants.
    
    - Download Llama3 8b - `ollama run llama3`
    
    ** You also need to update the code Line 33 with the new model name **

## Getting Started
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python start.py`


Youtube Guide: https://youtu.be/ZgkKsPuJHdo

Thanks to contributors for testing additional combinations of prompts/models:
https://github.com/hjvogel