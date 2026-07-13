from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM, pipeline
import re
import pandas as pd
import numpy as np
import random
import torch

def esm2_prep(sequence: str|list[str]) -> list[str]:
    if type(sequence)==str:
        sequence=[sequence]
    sequence=[seq.upper() for seq in sequence] # Upper case
    sequence = [re.sub(r"[BXZJUO]", "X", seq) for seq in sequence] # Replace rare AAs with X (not sure if relevant for this model)
    return sequence

def get_esm2_features(sequence: str|list[str]):
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    sequence=esm2_prep(sequence)
    encoded_input = tokenizer(sequence, return_tensors='pt')
    output = model(**encoded_input)
    return output
    
def get_esm2_df(sequence: str|list[str]):
    features=get_esm2_features(sequence)
    df=pd.DataFrame(features[0].flatten().detach().numpy()).transpose()
    df.columns=[f"esm2_{i}" for i in range(df.shape[1])]
    return df

def masker(seq_str: str, mask_fraction: float=0.15, seed: int=42) -> pd.DataFrame:
    seq_lst=" ".join(seq_str).split()
    if mask_fraction==1:
        mask_positions=range(len(seq_lst))
    else:
        n_mask=max(1, int(len(seq_lst) * mask_fraction))
        mask_positions = random.sample(range(len(seq_lst)),n_mask)
    masked_seqs=[]
    for position in mask_positions:
        masked_seqs.append(
            {"masked_position":position,
             "masked_residue":seq_lst[position],
             "masked_sequence":" ".join(["<mask>" if i == position else aa for i, aa in enumerate(seq_lst)]),
             }
        )
    return pd.DataFrame(masked_seqs)


def esm2_unmask(sequence:str|list[str],mask_fraction:float=0.15,seed:int=42):
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False )
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    sequence=esm2_prep(sequence)
    df=masker(sequence[0],mask_fraction=mask_fraction,seed=seed)
    for i in range(1,len(sequence)):
        df=pd.concat([df,masker(sequence[0],mask_fraction=mask_fraction,seed=seed)])
    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    unmasked_lst=df.masked_sequence.apply(unmasker)
    unmasked_df=[]
    for preds in unmasked_lst:
        row = {}
        for j, pred in enumerate(preds):
            row[f"pred_{j}"] = pred["token_str"]
            row[f"score_{j}"] = pred["score"]
        unmasked_df.append(row)
    unmasked_df=pd.DataFrame(unmasked_df)
    unmasked_df=pd.concat([df,unmasked_df],axis=1)
    return unmasked_df