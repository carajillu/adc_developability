import torch

def featurizer(sequence: str|list[str],tokenizer,model):
    encoded_input = tokenizer(sequence, return_tensors='pt',padding=True)
    with torch.no_grad():
        output = model(**encoded_input).last_hidden_state
    return output