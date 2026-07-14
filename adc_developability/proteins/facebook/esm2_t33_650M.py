from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM, pipeline
from adc_developability.proteins.utils.sequence_prep import seq_prep
from adc_developability.proteins.utils.masking import masker, unmasker
from adc_developability.proteins.utils.features import featurizer
import pandas as pd

TOKENIZER = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)
MODEL = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
MODEL_MLM = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
PIPELINE = pipeline('fill-mask', model=MODEL_MLM, tokenizer=TOKENIZER)

def get_esm2_df(sequence: str|list[str]):
    features=featurizer(sequence, TOKENIZER, MODEL)
    df=pd.DataFrame(features[0].flatten().detach().numpy()).transpose()
    df.columns=[f"esm2_{i}" for i in range(df.shape[1])]
    return df

def esm2_unmask(sequence:str|list[str],mask_fraction:float=0.15,seed:int=42):
    sequence=seq_prep(sequence)
    masked=pd.concat([masker(sequence[i], mask_str="<mask>", mask_fraction=mask_fraction, seed=seed) for i in range(0,len(sequence))],ignore_index=True)
    unmasked_df=unmasker(masked.masked_sequence.tolist(), PIPELINE)
    df=pd.concat([masked,unmasked_df],axis=1)
    return df