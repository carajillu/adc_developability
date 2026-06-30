from adc_developability.embeddings import get_embeddings

SEQ = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSRYIINWVRQAPGQGLEWMGRIIPILGVENYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARKDWFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

def test_protBERT() -> None:
    try:
        model="rostlab/prot_bert"
        bd=get_embeddings(SEQ,model)
        target_shape=(1024,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)

def test_prostT5() -> None:
    try:
        model="rostlab/prostT5"
        bd=get_embeddings(SEQ,model)
        target_shape=(1024,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)

def test_esm2() -> None:
    try:
        model="facebook/esm2_t36_3B_UR50D"
        bd=get_embeddings(SEQ,model)
        target_shape=(2560,)
        assert bd.shape==target_shape, f"Shape of embedding matrix should be {target_shape}. I am getting {bd.shape}"
    except Exception as e:
        raise(e)