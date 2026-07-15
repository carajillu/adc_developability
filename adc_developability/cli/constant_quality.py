"""
This program assesses the performance of the Protein Language models used to describe antibody constant regions in the paper "Something Something Developability of ADCs":
- Rostlab/prot_bert (420M parameters)
- facebook/esm2_t33_650M_UR50D (650M parameters)
- yarongef/DistilProtBert (230M parameters)
- zjunlp/OntoProtein (Undisclosed parameters)

1) Load the dataset in pandas
2) Separate constant and variable regions for the light and heavy chains
3) For each model, compute the following properties:
- "n_masked_positions",
-  "top1_accuracy",
-  "top5_accuracy", # f
-  "mean_true_probability", # float if present in top5, NaN if not
-  "MRR"
"""
import argparse
import pandas as pd
import random
from adc_developability.proteins.utils.sequence_prep import extract_region, pad_sides
from adc_developability.proteins.utils.stat_metrics import evaluate_unmask_df
from adc_developability.proteins.facebook.esm2_t33_650M import esm2_unmask
from adc_developability.proteins.zjunlp.onto_protein import onto_protein_unmask
from adc_developability.proteins.rostlab.prot_bert import prot_bert_unmask
from adc_developability.proteins.yarongef.distillprotbert import distill_prot_bert_unmask
from adc_developability.proteins.utils.stat_metrics import evaluate_unmask_df


def parse_args():
    parser = argparse.ArgumentParser(description="ADC Developability Quality CLI")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--lightchain_key", type=str, default="antibody_LC", help="Light chain key")
    parser.add_argument("--heavychain_key", type=str, default="antibody_HC", help="Heavy chain key")
    parser.add_argument("--mask_fraction", type=float, default=0.15, help="Fraction of residues to mask for evaluation")
    parser.add_argument("--mask_seed", type=int, default=42, help="Seed for reproducible masking")
    parser.add_argument("--output", type=str, default="quality", help="Output file prefix (model name will be appended)")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    # Load the dataset
    df=pd.read_csv(args.input)

    # Extract constant regions for light and heavy chains
    df["constant_region_LC"] = df[args.lightchain_key].apply(lambda x: extract_region(x, variable=False, force=True))
    df["constant_region_HC"] = df[args.heavychain_key].apply(lambda x: extract_region(x, variable=False, force=True))
    df=df.dropna(subset=["constant_region_LC", "constant_region_HC"]).reset_index(drop=True)

    print(">>> Processing Heavy Chains <<<")

    print(">>> Running ESM-2 model <<<")
    unmasked_heavy_esm2 = esm2_unmask(sequence=df.constant_region_HC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_heavy_esm2)
    quality_metrics_esm2 = evaluate_unmask_df(unmasked_heavy_esm2, model_name="ESM-2")

    print(">>> Running OntoProtein model <<<")
    unmasked_heavy_onto = onto_protein_unmask(sequence=df.constant_region_HC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_heavy_onto)
    quality_metrics_onto = evaluate_unmask_df(unmasked_heavy_onto, model_name="OntoProtein")

    print(">>> Running ProtBERT model <<<")
    unmasked_heavy_prot_bert = prot_bert_unmask(sequence=df.constant_region_HC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_heavy_prot_bert)
    quality_metrics_prot_bert = evaluate_unmask_df(unmasked_heavy_prot_bert, model_name="ProtBERT")

    print(">>> Running Distill ProtBERT model <<<")
    unmasked_heavy_distill_prot_bert = distill_prot_bert_unmask(sequence=df.constant_region_HC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_heavy_distill_prot_bert)
    quality_metrics_distill_prot_bert = evaluate_unmask_df(unmasked_heavy_distill_prot_bert, model_name="Distill ProtBERT")

    quality_metrics_heavy=pd.DataFrame([quality_metrics_esm2, quality_metrics_onto, quality_metrics_prot_bert, quality_metrics_distill_prot_bert])
    quality_metrics_heavy.to_csv(f"{args.output}_heavy.csv", index=False)

    print(">>> Processing Light Chains <<<")

    print(">>> Running ESM-2 model <<<")
    unmasked_light_esm2 = esm2_unmask(sequence=df.constant_region_LC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_light_esm2)
    quality_metrics_esm2 = evaluate_unmask_df(unmasked_light_esm2, model_name="ESM-2")

    print(">>> Running OntoProtein model <<<")
    unmasked_light_onto = onto_protein_unmask(sequence=df.constant_region_LC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_light_onto)
    quality_metrics_onto = evaluate_unmask_df(unmasked_light_onto, model_name="OntoProtein")

    print(">>> Running ProtBERT model <<<")
    unmasked_light_prot_bert = prot_bert_unmask(sequence=df.constant_region_LC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_light_prot_bert)
    quality_metrics_prot_bert = evaluate_unmask_df(unmasked_light_prot_bert, model_name="ProtBERT")

    print(">>> Running Distill ProtBERT model <<<")
    unmasked_light_distill_prot_bert = distill_prot_bert_unmask(sequence=df.constant_region_LC.tolist(), mask_fraction=args.mask_fraction, seed=args.mask_seed)
    print(unmasked_light_distill_prot_bert)
    quality_metrics_distill_prot_bert = evaluate_unmask_df(unmasked_light_distill_prot_bert, model_name="Distill ProtBERT")

    quality_metrics_light=pd.DataFrame([quality_metrics_esm2, quality_metrics_onto, quality_metrics_prot_bert, quality_metrics_distill_prot_bert])
    quality_metrics_light.to_csv(f"{args.output}_light.csv", index=False)