from transformers import AutoModel
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForMaskedLM
from transformers import pipeline

from adc_developability.wlp.utils.features import featurizer
import pandas as pd 

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/chembert_chembl_pretrained", do_lower_case=False)
MODEL = AutoModel.from_pretrained("jonghyunlee/chembert_chembl_pretrained")
MODEL_MLM = AutoModelForMaskedLM.from_pretrained("jonghyunlee/chembert_chembl_pretrained")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_chembert_chembl_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame(features[0].flatten().detach().numpy()).transpose()
    df.columns=[f"chembert_chembl_{i}" for i in range(df.shape[1])]
    return df