import torch

def featurizer(sequence: str|list[str],tokenizer,model):
    encoded_input = tokenizer(sequence, return_tensors='pt',padding=True)
    with torch.no_grad():
        output = model(**encoded_input).last_hidden_state # shape: (batch_size, seq_len, hidden_size)

    mask = encoded_input["attention_mask"].unsqueeze(-1).float() # shape: (batch_size, seq_len, 1)
    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1) # shape: (batch_size, hidden_size)
    return pooled

def count_tokens(smiles_list, tokenizer):
    encodings = tokenizer(smiles_list,add_special_tokens=True,truncation=False)
    return [len(ids) for ids in encodings["input_ids"]]