from adc_developability.embeddings import get_embeddings

SMILES="CCO"

def test_ChemBERTa_77M_MLM() -> None:
    try:
        model="DeepChem/ChemBERTa-77M-MLM"
        bd=get_embeddings(SMILES,model)
        target_shape=(384,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)

def test_ChemBERT_CHEMBL_pretrained() -> None:
    try:
        model="jonghyunlee/ChemBERT_CHEMBL_pretrained"
        bd=get_embeddings(SMILES,model)
        target_shape=(256,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)
    
def test_SMILES_BERT() -> None:
    try:
        model="juIm/SMILES_BERT"
        bd=get_embeddings(SMILES,model)
        target_shape=(768,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)
    
def test_BERT_base_SMILES() -> None:
    try:
        model="unikei/bert-base-smiles"
        bd=get_embeddings(SMILES,model)
        target_shape=(768,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)
