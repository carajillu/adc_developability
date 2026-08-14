import argparse
import pandas as pd
import importlib
import sys
from adc_developability.utils.sequence_prep import extract_region, pad_sides

DF_KEYS=["adc_id","adc_status","antibody_HC","antibody_LC","adc_wlpsmiles","adc_meanDAR"]

def parse():
    """
    Parse command line arguments for the full embedding CLI.
    Arguments will be the general PLM for constant regions (--plm),
    the antibody-specific PLM for variable regions (--abplm),
    the small molecule model for WLPs (--smllm),
    the input file path (--input),
    and the output file path (--output).
    
    Example command line usage:
        python full_embedding.py 
        --input .data/ADCdb_processed.csv
        --plm adc_developability.proteins.rostlab.prot_bert 
        --abplm adc_developability.antibody_Fv.antiberty_utils 
        --smllm adc_developability.wlp.deepchem.chemberta
        --output output.csv
    """
    parser = argparse.ArgumentParser(description="Full embedding CLI for ADC developability.")
    parser.add_argument("--debug", action="store_true", default=False, help="Use only the first 10 rows of the dataset for debugging.")
    parser.add_argument("--plm", type=str, required=True, help="General PLM for constant regions.")
    parser.add_argument("--abplm", type=str, required=True, help="Antibody-specific PLM for variable regions.")
    parser.add_argument("--smllm", type=str, required=True, help="Small molecule model for WLPs.")
    parser.add_argument("--input", type=str, required=True, help="Input file path.")
    parser.add_argument("--scheme", type=str, required=False, default="chothia", help="Numbering scheme to use for antibody sequences (e.g., 'chothia', 'imgt').")
    parser.add_argument("--pad_char", type=str, required=False, default="_", help="Character to use for padding gaps and sides in antibody sequences.")
    parser.add_argument("--output", type=str, required=True, help="Output file path.")

    args = parser.parse_args()
    return args

def import_function(module_pythonpath: str, function_name: str):
    module = importlib.import_module(module_pythonpath)
    return getattr(module, function_name)

def get_len(sequence: str|None) -> int|None:
    """
    If sequence is str, return its length. If sequence is None, return None.
    """
    if sequence is None:
        return None
    else:
        return len(sequence)

if __name__ == "__main__":
    args = parse()

    # Load models
    get_constant_df = import_function(args.plm, "get_df")
    get_variable_df = import_function(args.abplm, "get_df")
    get_wlp_df = import_function(args.smllm, "get_df")
    get_wlp_token_count = import_function(args.smllm, "token_counter")

    # Load input data
    df = pd.read_csv(args.input).dropna(subset=DF_KEYS).reset_index(drop=True)[DF_KEYS]
    if args.debug:
        df = df.sample(n=10, random_state=42).reset_index(drop=True)

    # Split constant and variable regions
    df["antibody_HC_constant"] = df["antibody_HC"].apply(lambda x: extract_region(x, variable=False, scheme=args.scheme, pad_char=args.pad_char,force=True))
    df["antibody_HC_variable"] = df["antibody_HC"].apply(lambda x: extract_region(x, variable=True, scheme=args.scheme, pad_char=args.pad_char,force=True))
    df["antibody_LC_constant"] = df["antibody_LC"].apply(lambda x: extract_region(x, variable=False, scheme=args.scheme, pad_char=args.pad_char,force=True))
    df["antibody_LC_variable"] = df["antibody_LC"].apply(lambda x: extract_region(x, variable=True, scheme=args.scheme, pad_char=args.pad_char,force=True))
    REGION_KEYS=["antibody_HC_constant","antibody_HC_variable","antibody_LC_constant","antibody_LC_variable"]
    df = df.dropna(subset=REGION_KEYS).reset_index(drop=True)

    # Discard ADCs with WLPs that exceed the token limit of the small molecule model
    df["wlp_token_count"] = get_wlp_token_count(df["adc_wlpsmiles"].tolist())
    max_token_limit = 512  # This should be set according to the small molecule model's specifications
    df = df[df["wlp_token_count"] <= max_token_limit].reset_index(drop=True)
    print(f"After filtering, {df.shape[0]} ADCs remain with WLP token counts within the limit of {max_token_limit}.")

    # pad sides of variable regions to the max length in the dataset
    max_hc_var_len = int(df["antibody_HC_variable"].apply(get_len).max())
    max_lc_var_len = int(df["antibody_LC_variable"].apply(get_len).max())
    print(f"padding sides of HC variable region to length {max_hc_var_len} and LC variable region to length {max_lc_var_len}")
    df["antibody_HC_variable"] = df["antibody_HC_variable"].apply(lambda x: pad_sides(x, max_hc_var_len, variable=True, pad_char=args.pad_char))
    df["antibody_LC_variable"] = df["antibody_LC_variable"].apply(lambda x: pad_sides(x, max_lc_var_len, variable=True, pad_char=args.pad_char))
    print(f"DataFrame length after padding variable regions: {df.shape[0]}")

    # Pad sides of constant regions to the max length in the dataset
    max_hc_const_len = int(df["antibody_HC_constant"].apply(get_len).max())
    max_lc_const_len = int(df["antibody_LC_constant"].apply(get_len).max())
    print(f"padding sides of HC constant region to length {max_hc_const_len} and LC constant region to length {max_lc_const_len}")
    df["antibody_HC_constant"] = df["antibody_HC_constant"].apply(lambda x: pad_sides(x, max_hc_const_len, variable=False, pad_char=args.pad_char))
    df["antibody_LC_constant"] = df["antibody_LC_constant"].apply(lambda x: pad_sides(x, max_lc_const_len, variable=False, pad_char=args.pad_char))
    print(f"DataFrame length after padding constant regions: {df.shape[0]}")


    # Constant region embeddings
    print("Generating constant region embeddings...")
    constant_HC_df=get_constant_df(df.antibody_HC_constant.tolist())
    constant_HC_df.columns=[f"constant_hc_{i}" for i in range(constant_HC_df.shape[1])]
    print(f"Constant region embeddings shape: {constant_HC_df.shape}")
    constant_LC_df=get_constant_df(df.antibody_LC_constant.tolist())
    constant_LC_df.columns=[f"constant_lc_{i}" for i in range(constant_HC_df.shape[1],constant_HC_df.shape[1]+constant_LC_df.shape[1])]
    print(f"Constant region embeddings shape: {constant_LC_df.shape}")
    constant_df=pd.concat([constant_HC_df,constant_LC_df],axis=1)
    print(f"Constant region embeddings shape: {constant_df.shape}")
    del constant_HC_df, constant_LC_df

    # Variable region embeddings
    print("Generating variable region embeddings...")
    variable_HC_df=get_variable_df(df.antibody_HC_variable.tolist())
    variable_HC_df.columns=[f"variable_hc_{i}" for i in range(variable_HC_df.shape[1])]
    print(f"Variable region embeddings shape: {variable_HC_df.shape}")
    variable_LC_df=get_variable_df(df.antibody_LC_variable.tolist())
    variable_LC_df.columns=[f"variable_lc_{i}" for i in range(variable_HC_df.shape[1],variable_HC_df.shape[1]+variable_LC_df.shape[1])]
    print(f"Variable region embeddings shape: {variable_LC_df.shape}")
    variable_df=pd.concat([variable_HC_df,variable_LC_df],axis=1)
    print(f"Variable region embeddings shape: {variable_df.shape}")
    del variable_HC_df, variable_LC_df

    # WLP embeddings
    print("Generating WLP embeddings...")
    wlp_df=get_wlp_df(df.adc_wlpsmiles.tolist())
    wlp_df.columns=[f"wlp_{i}" for i in range(wlp_df.shape[1])]
    print(f"WLP embeddings shape: {wlp_df.shape}")

    # Combine all embeddings
    print("Combining all embeddings...")
    df=pd.concat([df,constant_df,variable_df,wlp_df],axis=1)
    print(f"Final df shape: {df.shape}")

    # Save to output
    df.to_csv(args.output, index=False)

    
    
