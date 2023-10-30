import streamlit as st
# transformers library is used for loading LLM models
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch
# getting input for context
val1 = st.text_input('Context')
# getting input for question
val2 = st.text_input('Question')
# loading google flan t5 xl model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
# get GPU availablity
use_cuda = torch.cuda.is_available()
# check if GPU is present
if use_cuda:
    # use GPU
    device='cuda:0'
else:
    # use CPU
    device='cpu'
model.to(device)
# creating prompt with 20 tokens
def query_from_list(query, options):
    t5query = f"""Question: {query}". Context: {options}"""
    inputs = tokenizer(t5query, return_tensors="pt")
    inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    # return model generated answers
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
if val1 and val2 and st.button('Go'):
    result = query_from_list(val1,val2)
    # return answer
    st.write(result[0])
