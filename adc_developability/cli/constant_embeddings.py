"""
This program calculates embeddings for antibody constant regions using the Protein Language models described in the paper "Something Something Developability of ADCs":
- Rostlab/prot_bert (420M parameters)
- facebook/esm2_t33_650M_UR50D (650M parameters)
- yarongef/DistilProtBert (230M parameters)
- zjunlp/OntoProtein (Undisclosed parameters)
"""
import argparse
import pandas as pd

from adc_developability.proteins.utils.sequence_prep import extract_region, pad_sides
from adc_developability.proteins.facebook.esm2_t33_650M import get_esm2_df
from adc_developability.proteins.zjunlp.onto_protein import get_onto_protein_df
from adc_developability.proteins.rostlab.prot_bert import get_prot_bert_df
from adc_developability.proteins.yarongef.distillprotbert import get_distill_prot_bert_df

def parse():
    parser = argparse.ArgumentParser(description="ADC Developability Quality CLI")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--debug",action="store_true", default=False, help="Use only the first row of the dataset")
    parser.add_argument("--lightchain_key", type=str, default="antibody_LC", help="Light chain key")
    parser.add_argument("--heavychain_key", type=str, default="antibody_HC", help="Heavy chain key")
    parser.add_argument("--esm2",action="store_true", default=False, help="Calculate ESM-2 embedddings")
    parser.add_argument("--onto_protein",action="store_true", default=False, help="Calculate onto_protein embedddings")
    parser.add_argument("--prot_bert",action="store_true", default=False, help="Calculate prot_bert embedddings")
    parser.add_argument("--distill_prot_bert",action="store_true", default=False, help="Calculate distil_prot_bert embedddings")
    parser.add_argument("--output", type=str, default="embeddings", help="Output file prefix (model name will be appended)")
    return parser.parse_args()


if __name__=="__main__":
    args=parse()
    
    # Load data sets and extract constant regions
    if not args.debug:
       df=pd.read_csv(args.input)
    else:
       df=pd.read_csv(args.input).head(1)
    print(df)

    df["constant_region_LC"] = df[args.lightchain_key].apply(lambda x: extract_region(x, variable=False, force=True))
    df["constant_region_HC"] = df[args.heavychain_key].apply(lambda x: extract_region(x, variable=False, force=True))
    df=df.dropna(subset=["constant_region_LC", "constant_region_HC"]).reset_index(drop=True)

    # Pad constant regions on the right so that they all have the same length
    max_len=max(df["constant_region_LC"].apply(len))
    print(f"padding light chains to a length of {max_len}")
    df["constant_region_LC"]=df["constant_region_LC"].apply(lambda x: pad_sides(sequence=x,target_length=max_len,variable=False,pad_char="X"))

    max_len=max(df["constant_region_HC"].apply(len))
    print(f"padding heavy chains to a length of {max_len}")
    df["constant_region_HC"]=df["constant_region_HC"].apply(lambda x: pad_sides(sequence=x,target_length=max_len,variable=False,pad_char="X"))

    # Calculate embeddings
    if args.prot_bert:
       embeddings_lc=get_prot_bert_df(sequence=df["constant_region_LC"])
       embeddings_hc=get_prot_bert_df(sequence=df["constant_region_HC"])
       embeddings_lc.columns=[f"prot_bert_lc_{i}" for i in range(embeddings_lc.shape[1])]
       embeddings_hc.columns=[f"prot_bert_hc_{i}" for i in range(embeddings_hc.shape[1])]
       df_prot_bert=pd.concat([df,embeddings_lc,embeddings_hc],axis=1)
       df_prot_bert.to_csv(f"prot_bert_{args.output}.csv",index=False)
       del(df_prot_bert)

    if args.onto_protein:
       embeddings_lc=get_onto_protein_df(sequence=df["constant_region_LC"])
       embeddings_hc=get_onto_protein_df(sequence=df["constant_region_HC"])
       embeddings_lc.columns=[f"onto_protein_lc_{i}" for i in range(embeddings_lc.shape[1])]
       embeddings_hc.columns=[f"onto_protein_hc_{i}" for i in range(embeddings_hc.shape[1])]
       df_onto_protein=pd.concat([df,embeddings_lc,embeddings_hc],axis=1)
       df_onto_protein.to_csv(f"onto_protein_{args.output}.csv",index=False)
       del(df_onto_protein)

    if args.distill_prot_bert:
       embeddings_lc=get_distill_prot_bert_df(sequence=df["constant_region_LC"])
       embeddings_hc=get_distill_prot_bert_df(sequence=df["constant_region_HC"])
       embeddings_lc.columns=[f"distill_prot_bert_lc_{i}" for i in range(embeddings_lc.shape[1])]
       embeddings_hc.columns=[f"distill_prot_bert_hc_{i}" for i in range(embeddings_hc.shape[1])]
       df_distill_prot_bert=pd.concat([df,embeddings_lc,embeddings_hc],axis=1)
       df_distill_prot_bert.to_csv(f"distill_prot_bert_{args.output}.csv",index=False)
       del(df_distill_prot_bert)
    
    if args.esm2:
       embeddings_lc=get_esm2_df(sequence=df["constant_region_LC"])
       embeddings_hc=get_esm2_df(sequence=df["constant_region_HC"])
       embeddings_lc.columns=[f"esm2_lc_{i}" for i in range(embeddings_lc.shape[1])]
       embeddings_hc.columns=[f"esm2_hc_{i}" for i in range(embeddings_hc.shape[1])]
       df_esm2=pd.concat([df,embeddings_lc,embeddings_hc],axis=1)
       df_esm2.to_csv(f"esm2_{args.output}.csv",index=False)
       del(df_esm2)




   