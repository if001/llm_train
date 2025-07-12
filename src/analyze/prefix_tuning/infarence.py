from transformers import pipeline

pipe = pipeline("text-generation", model="./prefix-tuned-gpt2", tokenizer="./prefix-tuned-gpt2")
print(pipe("Once upon a time,", max_new_tokens=50))