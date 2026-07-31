import argparse
import pandas as pd





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

if __name__ == "__main__":
    args=parse()
    
    # Load data sets and extract constant regions
    if not args.debug:
       df=pd.read_csv(args.input)
    else:
       df=pd.read_csv(args.input).head(10)
    print(df)

    # Calculate embeddings
    if args.chemberta:
       from adc_developability.wlp.deepchem.chemberta import get_chemberta_df
       embeddings=get_chemberta_df(sequence=df.adc_wlpsmiles.tolist())
       embeddings.columns=[f"chemberta_{i}" for i in range(embeddings.shape[1])]
       df_chemberta=pd.concat([df,embeddings],axis=1)
       df_chemberta.to_csv(f"chemberta_{args.output}.csv",index=False)
       del(df_chemberta)

    if args.chembert_chembl:
       from adc_developability.wlp.jonghyunlee.chembert_chembl import get_chembert_chembl_df
       embeddings=get_chembert_chembl_df(sequence=df.adc_wlpsmiles.tolist())
       embeddings.columns=[f"chembert_chembl_{i}" for i in range(embeddings.shape[1])]
       df_chembert_chembl=pd.concat([df,embeddings],axis=1)
       df_chembert_chembl.to_csv(f"chembert_chembl_{args.output}.csv",index=False)
       del(df_chembert_chembl)

    if args.smiles_bert:
       from adc_developability.wlp.juim.smiles_bert import get_smiles_bert_df
       embeddings=get_smiles_bert_df(sequence=df.adc_wlpsmiles.tolist())
       embeddings.columns=[f"smilesBERT_{i}" for i in range(embeddings.shape[1])]
       df_smilesBERT=pd.concat([df,embeddings],axis=1)
       df_smilesBERT.to_csv(f"smilesBERT_{args.output}.csv",index=False)
       del(df_smilesBERT)

    if args.bert_base_smiles:
       from adc_developability.wlp.unikei.bert_base_smiles import get_bert_base_smiles_df
       embeddings=get_bert_base_smiles_df(sequence=df.adc_wlpsmiles.tolist())
       embeddings.columns=[f"bert_base_smiles_{i}" for i in range(embeddings.shape[1])]
       df_bert_base_smiles=pd.concat([df,embeddings],axis=1)
       df_bert_base_smiles.to_csv(f"bert_base_smiles_{args.output}.csv",index=False)
       del(df_bert_base_smiles)