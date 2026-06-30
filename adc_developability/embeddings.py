import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

"""
def get_embeddings(txt_str: str, model_id: str, max_length: int = 512) -> np.ndarray:

    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(
        txt_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state              # [batch, seq, hidden]
    attention_mask = inputs["attention_mask"].unsqueeze(-1)   # [batch, seq, 1]

    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)

    mean_pooled = summed / counts                             # [batch, hidden]
    return mean_pooled.squeeze(0).cpu().numpy()
"""

def get_embeddings(txt_str: str, model_id: str, max_length: int = 512, device_str: str = "") -> np.ndarray:
    """
    Return a single vector embedding for one SMILES/sequence string.
    use the GPU (cuda or mps) if available

    Output:
        numpy.ndarray of shape [hidden_size]
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    model = AutoModel.from_pretrained(model_id)

    if device_str=="":
       if torch.cuda.is_available():
          device = torch.device("cuda")
       elif torch.backends.mps.is_available():
          device = torch.device("mps")
       else:
          device = torch.device("cpu")
    else:
       device=torch.device(device_str)
    
    model.to(device)
    model.eval()

    inputs = tokenizer(
        txt_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if getattr(model.config, "is_encoder_decoder", False):   # for ProtT5
            outputs = model.encoder(**inputs)
        else:                           # for ProtBERT
            outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)

    mean_pooled = summed / counts

    return mean_pooled.squeeze(0).cpu().numpy()