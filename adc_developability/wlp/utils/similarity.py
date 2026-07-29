import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

def get_cosine_similarity(vec1: list[float]|np.ndarray, vec2: list[float]|np.ndarray):
    """
    Calculate the cosine similarity between two vectors.
    Args:
        vec1 (list or np.array): First vector.
        vec2 (list or np.array): Second vector.
    Returns:
        float: Cosine similarity between vec1 and vec2.
    """

    # Convert lists to numpy arrays if they are not already
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Calculate the dot product and norms
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    # Calculate cosine similarity
    cos_sim = dot_product / (norm_vec1 * norm_vec2)

    # Normalise similarity to be between 0 and 1
    return (cos_sim + 1) / 2

def get_tanimoto_coefficient(smiles1: str, smiles2: str):
    """
    Calculate the Tanimoto coefficient between two SMILES strings.
    Uses extended connectivity fingerprints (ECFP) to compute the similarity.
    Args:
        smiles1 (str): First SMILES string.
        smiles2 (str): Second SMILES string.
    Returns:
        float: Tanimoto coefficient between smiles1 and smiles2.
    """

    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    #Create fingerprint generator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    # Generate ECFP fingerprints
    fp1 = mfpgen.GetFingerprint(mol1)
    fp2 = mfpgen.GetFingerprint(mol2)

    # Calculate Tanimoto coefficient
    return DataStructs.TanimotoSimilarity(fp1, fp2)
    