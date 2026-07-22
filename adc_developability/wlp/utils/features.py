import torch

def featurizer(sequence: str|list[str],tokenizer,model):
    encoded_input = tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    return output