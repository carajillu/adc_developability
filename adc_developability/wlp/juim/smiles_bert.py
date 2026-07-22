from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import pipeline

from adc_developability.wlp.utils.features import featurizer
import pandas as pd 

TOKENIZER = AutoTokenizer.from_pretrained("juIm/smiles_bert", do_lower_case=False)
MODEL = AutoModel.from_pretrained("juIm/smiles_bert")
MODEL_MLM = AutoModelForMaskedLM.from_pretrained("juIm/smiles_bert")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_smiles_bert_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame(features[0].flatten().detach().numpy()).transpose()
    df.columns=[f"smiles_bert_{i}" for i in range(df.shape[1])]
    return df