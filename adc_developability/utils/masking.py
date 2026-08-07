import random
import pandas as pd 

def masker(seq_str: str, mask_str: str, mask_list: list[int]|None = None, mask_fraction: float=0.15, seed: int=42) -> pd.DataFrame:
    seq_lst=" ".join(seq_str).split()
    if mask_list is not None:
        mask_positions = mask_list
    elif mask_fraction == 1:
        mask_positions = range(len(seq_lst))
    else:
        n_mask=max(1, int(len(seq_lst) * mask_fraction))
        mask_positions = random.sample(range(len(seq_lst)),n_mask)
    masked_seqs=[]
    for position in mask_positions:
        masked_seqs.append(
            {"masked_position":position,
             "masked_residue":seq_lst[position],
             "masked_sequence":" ".join([mask_str if i == position else aa for i, aa in enumerate(seq_lst)]),
             }
        )
    return pd.DataFrame(masked_seqs)

def unmasker(masked_seq: str|list[str], unmasker_pipeline) -> pd.DataFrame:
    unmasked_lst=unmasker_pipeline(masked_seq)
    unmasked_df=[]
    for element in unmasked_lst:
        row = {}
        for j, pred in enumerate(element):
            row[f"pred_{j}"] = pred["token_str"]
            row[f"score_{j}"] = pred["score"]
        unmasked_df.append(row)
    return pd.DataFrame(unmasked_df)