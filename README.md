## Executing the Code
First, you need to install the requirements and load ollama models (make sure ollama is installed on the computer and running): 
```bash 
pip install -r requirements.txt 
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Then you can run the actual script and follow the instructions to ask question about cats.
```bash 
python demo.py
``` 

## Reflection 
### Observed Limitations
- some questions could not be answered as the information could not be retrieved (probably not in the information) like "How many wild cat breeds exist?"
- sometimes the model does not answer the actual question but only repeats something from the information that is somewhat related but not an answer to the question

### Improvements 
- larger dataset could help to allow more answers also to more specific questions 
- more advanced models would understand the question better and could be used to augment the information in the dataset (with the increased risk of hallucination)

