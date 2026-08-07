from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import pipeline

from adc_developability.utils.features import featurizer
import pandas as pd 

TOKENIZER = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM", do_lower_case=False)
MODEL = RobertaModel.from_pretrained("DeepChem/ChemBERTa-100M-MLM")
MODEL_MLM = RobertaForMaskedLM.from_pretrained("DeepChem/ChemBERTa-100M-MLM")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_chemberta_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame([features[i].flatten().detach().numpy() for i in range(len(features))])
    df.columns=[f"chemberta_{i}" for i in range(df.shape[1])]
    return df