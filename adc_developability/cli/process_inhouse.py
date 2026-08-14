import argparse
import pandas as pd

DF_KEYS=["adc_id","mAb","Linker","Payload","Mean %Mon","Mean %HMwS","Mean %LMwS","Mean DAR"]

TRASTUZUMAB_HC=("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRY"
                "ADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
                "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
                "GLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGG"
                "PSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
                "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREE"
                "MTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRW"
                "QQGNVFSCSVMHEALHNHYTQKSLSLSPGK")
TRASTUZUMAB_LC=("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPS"
                "RFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPP"
                "SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT"
                "LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC")

DATOPOTAMAB_HC=("QVQLVQSGAEVKKPGASVKVSCKASGYTFTTAGMQWVRQAPGQGLEWMGWINTHSGVPKY"
                "AEDFKGRVTISADTSTSTAYLQLSSLKSEDTAVYYCARSGFGSSYWYFDVWGQGTLVTVS"
                "SASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQS"
                "SGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLG"
                "GPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQY"
                "NSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRE"
                "EMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSR"
                "WQQGNVFSCSVMHEALHNHYTQKSLSLSPGK")
DATOPOTAMAB_LC=("DIQMTQSPSSLSASVGDRVTITCKASQDVSTAVAWYQQKPGKAPKLLIYSASYRYTGVPS"
                "RFSGSGSGTDFTLTISSLQPEDFAVYYCQQHYITPLTFGQGTKLEIKRTVAAPSVFIFPP"
                "SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT"
                "LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC")

CETUXIMAB_HC=("QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYN"
              "TPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSAA"
              "STKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSG"
              "LYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSPKSCDKTHTCPPCPAPELL"
              "GGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQ"
              "YNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSR"
              "DELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKS"
              "RWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK")
CETUXIMAB_LC=("DILLTQSPVILSVSPGERVSFSCRASQSIGTNIHWYQQRTNGSPRLLIKYASESISGIPS"
              "RFSGSGSGTDFTLSINSVESEDIADYYCQQNNNWPTTFGAGTKLELKRTVAAPSVFIFPP"
              "SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT"
              "LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGA")


def parse():
    parser = argparse.ArgumentParser(description="Full embedding CLI for ADC developability.")
    parser.add_argument("--debug", action="store_true", default=False, help="Use only the first")
    parser.add_argument("--input", type=str, required=True, help="Input file path.")
    parser.add_argument("--output", type=str, required=True, help="Output file path.")
    
    args = parser.parse_args()
    return args

def locate_wlp(linker: str, payload: str,wlp_data:pd.DataFrame):
    if (not pd.isna(linker)) and (not pd.isna(payload)):
       smiles=wlp_data.SMILES[(wlp_data.linker==linker) & (wlp_data.payload==payload)]
       print(f"Neither is not NaN: {linker}, {payload}")
    elif not pd.isna(linker):
        print(f"linker is not NaN: {linker}, {payload}")
        smiles=wlp_data.SMILES[(wlp_data.linker==linker)]
    elif not pd.isna(payload):
        print(f"payload is not NaN: {linker}, {payload}")
        smiles=wlp_data.SMILES[(wlp_data.payload==payload)]
    try:
          return smiles.iloc[0]
    except Exception as e:
          return None

def add_antibody_sequence():
    """
    takes a dataframe containing a column called "mAb"
    adds two columns to the dataframe called "antibody_HC" and "antibody_LC"
    which contain the heavy and light chain sequences for the antibodies in the "mAb" column
    extracted from the hardcoded sequences for trastuzumab, datopotamab, and cetuximab
    """
    
    return

if __name__ == "__main__":
    args = parse()
    wlp=pd.read_excel(args.input,sheet_name="WLP data")
    qc=pd.read_excel(args.input,sheet_name="QC",skiprows=1)
    qc["adc_id"]=qc.index+1
    qc=qc.dropna(subset=DF_KEYS).reset_index(drop=True)
    qc["adc_wlpsmiles"]=qc.apply(lambda x: locate_wlp(x.Linker,x.Payload,wlp),axis=1)
    qc["antibody_HC"] = qc["mAb"].apply(lambda x: TRASTUZUMAB_HC if x.lower() == "trastuzumab" else (DATOPOTAMAB_HC if x.lower() == "datopotamab" else (CETUXIMAB_HC if x.lower() == "cetuximab" else None)))
    qc["antibody_LC"] = qc["mAb"].apply(lambda x: TRASTUZUMAB_LC if x.lower() == "trastuzumab" else (DATOPOTAMAB_LC if x.lower() == "datopotamab" else (CETUXIMAB_LC if x.lower() == "cetuximab" else None)))
    qc=qc.dropna(subset=["adc_wlpsmiles"]).reset_index(drop=True)[DF_KEYS+["adc_wlpsmiles","antibody_HC","antibody_LC"]]
    qc = qc.rename(columns={"Mean DAR": "adc_meanDAR"})
    qc["adc_status"]=["inhouse"]*qc.shape[0]
    qc.to_csv(args.output,index=False)
