from transformers import BertModel, BertTokenizer
import re
import pandas as pd

def get_prot_bert_features(sequence: str|list[str]):
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    if type(sequence)==str:
        sequence=[sequence]
    sequence=[seq.upper() for seq in sequence] # Upper case
    sequence=[" ".join(seq) for seq in sequence] # Separate by spaces
    sequence = [re.sub(r"[UZOB]", "X", seq) for seq in sequence] # Replace rare AAs with X
    encoded_input = tokenizer(sequence, return_tensors='pt')
    output = model(**encoded_input)
    return output
    
def get_prot_bert_df(sequence: str|list[str]):
    features=get_prot_bert_features(sequence)
    df=pd.DataFrame(features[0].flatten().detach().numpy())
    df.columns=[f"prot_bert_{i}" for i in range(df.shape[1])]
    return df
