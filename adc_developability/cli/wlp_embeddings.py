import argparse
import pandas as pd
from copy import deepcopy

def parse():
    parser = argparse.ArgumentParser(description="Calculate embeddings for WLPs using the specified models\n \
                                                  calculate cosine similarity between pairs of embeddings\n \
                                                  calculate Tanimoto similarity between pairs of SMILES strings\n \
                                                  calculate correlation between cosine similarity and Tanimoto similarity\n \
                                                  for each model.")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--debug",action="store_true", default=False, help="Use only the first row of the dataset")
    parser.add_argument("--chemberta",action="store_true", default=False, help="Calculate ChemBERTa embeddings")
    parser.add_argument("--chembert_chembl",action="store_true", default=False, help="Calculate ChemBERT-ChEMBL embeddings")
    parser.add_argument("--smiles_bert",action="store_true", default=False, help="Calculate SMILES-BERT embeddings")
    parser.add_argument("--bert_base_smiles",action="store_true", default=False, help="Calculate BERT-base-SMILES embeddings")
    parser.add_argument("--output", type=str, default="wlp_embeddings", help="Output file prefix (model name will be appended)")
    return parser.parse_args()

MAX_TOKENS = 510
DF_KEYS=["adc_id","adc_status","antibody_HC","antibody_LC","adc_wlpsmiles","adc_meanDAR"]

if __name__ == "__main__":
    args=parse()
    
    # Load data sets and extract constant regions
    df=pd.read_csv(args.input)
    df=df.dropna(subset=DF_KEYS).reset_index(drop=True)
    if args.debug:
       df=df.head(10)

    print(df)

    # Calculate embeddings
    if args.chemberta:
       from adc_developability.wlp.deepchem.chemberta import get_chemberta_df, chemberta_count_tokens
       df_chemberta=deepcopy(df)
       df_chemberta["n_tokens"]=chemberta_count_tokens(df_chemberta.adc_wlpsmiles.tolist())
       df_chemberta=df_chemberta[df_chemberta.n_tokens<=MAX_TOKENS].reset_index(drop=True)
       print(f"Calculating ChemBERTa embeddings for {len(df_chemberta)} WLPs")
       embeddings=get_chemberta_df(sequence=df_chemberta.adc_wlpsmiles.tolist())
       embeddings.columns=[f"chemberta_{i}" for i in range(embeddings.shape[1])]
       df_chemberta=pd.concat([df_chemberta,embeddings],axis=1)
       df_chemberta.to_csv(f"chemberta_{args.output}.csv",index=False)
       del(df_chemberta)

    if args.smiles_bert:
       from adc_developability.wlp.juim.smiles_bert import get_smiles_bert_df, smiles_bert_count_tokens
       df_smilesBERT=deepcopy(df)
       df_smilesBERT["n_tokens"]=smiles_bert_count_tokens(df_smilesBERT.adc_wlpsmiles.tolist())
       df_smilesBERT=df_smilesBERT[df_smilesBERT.n_tokens<=MAX_TOKENS].reset_index(drop=True)
       print(f"Calculating SMILES-BERT embeddings for {len(df_smilesBERT)} WLPs")
       embeddings=get_smiles_bert_df(sequence=df_smilesBERT.adc_wlpsmiles.tolist())
       embeddings.columns=[f"smilesBERT_{i}" for i in range(embeddings.shape[1])]
       df_smilesBERT=pd.concat([df_smilesBERT,embeddings],axis=1)
       df_smilesBERT.to_csv(f"smilesBERT_{args.output}.csv",index=False)
       del(df_smilesBERT)

    if args.bert_base_smiles:
       from adc_developability.wlp.unikei.bert_base_smiles import get_bert_base_smiles_df, bert_base_smiles_count_tokens
       df_bert_base_smiles=deepcopy(df)
       df_bert_base_smiles["n_tokens"]=bert_base_smiles_count_tokens(df_bert_base_smiles.adc_wlpsmiles.tolist())
       df_bert_base_smiles=df_bert_base_smiles[df_bert_base_smiles.n_tokens<=MAX_TOKENS].reset_index(drop=True)
       print(f"Calculating BERT-base-SMILES embeddings for {len(df_bert_base_smiles)} WLPs")
       embeddings=get_bert_base_smiles_df(sequence=df_bert_base_smiles.adc_wlpsmiles.tolist())
       embeddings.columns=[f"bert_base_smiles_{i}" for i in range(embeddings.shape[1])]
       df_bert_base_smiles=pd.concat([df_bert_base_smiles,embeddings],axis=1)
       df_bert_base_smiles.to_csv(f"bert_base_smiles_{args.output}.csv",index=False)
       del(df_bert_base_smiles)