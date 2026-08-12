import torch
from torch import tensor
from antiberty import AntiBERTyRunner
import pandas as pd

RUNNER = AntiBERTyRunner()

def get_embeddings(sequences: str|list[str]):
    if type(sequences)==str:
        sequences=[sequences]
    embeddings = RUNNER.embed(sequences)
    return embeddings

def get_df(sequence: str|list[str]):
    embeddings_list=get_embeddings(sequence)
    features=[]
    for element in embeddings_list:
        features.append(element.mean(dim=0).detach().numpy())
    df=pd.DataFrame(features)
    df.columns=[f"AntiBERTy_{i}" for i in range(df.shape[1])]
    return df
    