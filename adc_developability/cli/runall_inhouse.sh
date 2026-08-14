python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.rostlab.prot_bert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.deepchem.chemberta  --output protbert_chemberta.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.rostlab.prot_bert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.juim.smiles_bert  --output protbert_smiles-bert.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.rostlab.prot_bert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.unikei.bert_base_smiles  --output protbert_bert-base-smiles.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.facebook.esm2_t33_650M  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.deepchem.chemberta  --output esm2_chemberta.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.facebook.esm2_t33_650M  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.juim.smiles_bert  --output esm2_smiles-bert.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.facebook.esm2_t33_650M  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.unikei.bert_base_smiles  --output esm2_bert-base-smiles.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.yarongef.distillprotbert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.deepchem.chemberta  --output distillprotbert_chemberta.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.yarongef.distillprotbert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.juim.smiles_bert  --output distillprotbert_smiles-bert.csv --scheme chothia 

python adc_developability/cli/full_embedding.py  --input .data/inhouse.csv  --plm adc_developability.proteins.yarongef.distillprotbert  --abplm adc_developability.antibody_Fv.antiberty_utils  --smllm adc_developability.wlp.unikei.bert_base_smiles  --output distillprotbert_bert-base-smiles.csv --scheme chothia 