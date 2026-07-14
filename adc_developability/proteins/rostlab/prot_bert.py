from transformers import BertModel, BertTokenizer, BertForMaskedLM, pipeline
from adc_developability.proteins.utils.sequence_prep import seq_prep
from adc_developability.proteins.utils.masking import masker, unmasker
from adc_developability.proteins.utils.features import featurizer
import pandas as pd

TOKENIZER = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
MODEL = BertModel.from_pretrained("Rostlab/prot_bert")
MODEL_MLM = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_prot_bert_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame(features[0].flatten().detach().numpy()).transpose()
    df.columns=[f"prot_bert_{i}" for i in range(df.shape[1])]
    return df

def prot_bert_unmask(sequence:str|list[str],mask_fraction:float=0.15,seed:int=42):
    sequence=seq_prep(sequence)
    masked=pd.concat([masker(sequence[i], mask_str="[MASK]", mask_fraction=mask_fraction, seed=seed) for i in range(0,len(sequence))],ignore_index=True)
    unmasked_df=unmasker(masked.masked_sequence.tolist(), PIPELINE)
    df=pd.concat([masked,unmasked_df],axis=1)
    return df