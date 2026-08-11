from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import pipeline

from adc_developability.utils.features import featurizer, count_tokens
import pandas as pd 

TOKENIZER = AutoTokenizer.from_pretrained("juIm/smiles_bert", do_lower_case=False)
MODEL = AutoModel.from_pretrained("juIm/smiles_bert")
MODEL_MLM = AutoModelForMaskedLM.from_pretrained("juIm/smiles_bert")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_smiles_bert_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame([features[i].flatten().detach().numpy() for i in range(len(features))])
    df.columns=[f"smiles_bert_{i}" for i in range(df.shape[1])]
    return df

def smiles_bert_count_tokens(smiles_list):
    return count_tokens(smiles_list, TOKENIZER)