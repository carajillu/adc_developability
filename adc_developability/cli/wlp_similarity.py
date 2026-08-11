
import argparse
import pandas as pd
import numpy as np
from adc_developability.utils.similarity import get_cosine_similarity, get_tanimoto_coefficient

def parse():
    parser = argparse.ArgumentParser(description="Calculate cosine similarity between pairs of WLP embeddings \n \
                                                  calculate Tanimoto similarity between pairs of SMILES strings\n \
                                                  calculate correlation between cosine similarity and Tanimoto similarity\n \
                                                  for each model.")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--debug",action="store_true", default=False, help="Use only the first row of the dataset")
    parser.add_argument("--key", type=str, required=True, help="Embedding key pattern")
    parser.add_argument("--output", type=str, default="wlp_embeddings", help="Output file prefix (model name will be appended)")
    return parser.parse_args()

def calculate_tanimoto_matrix(smiles_list):
    n=len(smiles_list)
    tanimoto_matrix=pd.DataFrame(index=range(n),columns=range(n))
    for i in range(n):
        for j in range(i,n):
            tanimoto_matrix.iloc[i,j]=get_tanimoto_coefficient(smiles_list[i],smiles_list[j])
            tanimoto_matrix.iloc[j,i]=tanimoto_matrix.iloc[i,j]
    return tanimoto_matrix

def calculate_cosine_matrix(embeddings):
    n=embeddings.shape[0]
    cosine_matrix=pd.DataFrame(index=range(n),columns=range(n))
    for i in range(n):
        for j in range(i,n):
            cosine_matrix.iloc[i,j]=get_cosine_similarity(embeddings.iloc[i,:],embeddings.iloc[j,:])
            cosine_matrix.iloc[j,i]=cosine_matrix.iloc[i,j]
    return cosine_matrix

def plot_correlation(cosine_matrix, tanimoto_matrix, output_file):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr

    mask = np.triu(np.ones(cosine_matrix.shape, dtype=bool), k=1)
    cosine_values = cosine_matrix.values[mask].flatten().tolist()
    tanimoto_values = tanimoto_matrix.values[mask].flatten().tolist()
    #print(f"Cosine values: {cosine_values}")
    #print(f"Tanimoto values: {tanimoto_values}")

    similarity_df = pd.DataFrame({'Cosine Similarity': cosine_values, 'Tanimoto Similarity': tanimoto_values})
    similarity_df.to_csv(output_file.replace('.png', '_similarity_values.csv'), index=False)

    correlation, _ = pearsonr(cosine_values, tanimoto_values)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=cosine_values, y=tanimoto_values)
    plt.title(f'Cosine vs Tanimoto Similarity (Pearson r={correlation:.2f})')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Tanimoto Similarity')
    plt.savefig(output_file)

if __name__ == "__main__":
    args=parse()
    
    # Load data sets and extract constant regions
    if not args.debug:
       df=pd.read_csv(args.input)
    else:
       df=pd.read_csv(args.input).head(10)
    print(df)

    tanimoto_matrix=calculate_tanimoto_matrix(df.adc_wlpsmiles.tolist())
    cosine_df=df.filter(like=args.key)
    if cosine_df.empty:
        raise ValueError(f"No columns found with key '{args.key}' in the input dataset.")
    cosine_matrix=calculate_cosine_matrix(cosine_df)
    plot_correlation(cosine_matrix, tanimoto_matrix, f"{args.output}_correlation.png")


