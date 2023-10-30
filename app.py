
# pandas is used for read and write csv
import pandas
# transformers library is used for loading LLM models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tqdm is used to generate status bar
from tqdm.auto import tqdm
import torch
# get GPU availablity
use_cuda = torch.cuda.is_available()
# loading google flan t5 xl model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
# check if GPU is present
if use_cuda:
    # use GPU
    device='cuda:0'
else:
    # use CPU
    device='cpu'
model.to(device)
results=[]
# reading the test data from test.csv file
dfs=pandas.read_csv('./test.csv')

# creating prompt with 20 tokens
def query_from_list(query, options):
    t5query = f"""Question: {query}". Context: {options}"""
    inputs = tokenizer(t5query, return_tensors="pt")
    inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    # return model generated answers
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# loading the test data from csv, iterating through the data and saving the generated in array
for z,(i,j) in tqdm(enumerate((dfs.iterrows()))):
    result = query_from_list(j['Story'], j['Question'])
    results.append(result[0])

df=pandas.DataFrame(list())
df['Answer']=results
# writing answers to submission.csv file
df.to_csv('./submission.csv',index=False)
