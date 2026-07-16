
from adc_developability.proteins.utils.sequence_prep import seq_prep
import torch

def featurizer(sequence: str|list[str],tokenizer,model):
    sequence=seq_prep(sequence)
    encoded_input = tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    return output