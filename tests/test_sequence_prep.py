from adc_developability.proteins.utils.sequence_prep import *

SEQ = "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSRYIINWVRQAPGQGLEWMGRIIPILGVENYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAVYYCARKDWFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

def test_extract_region()-> None:
    constant=extract_region(sequence=SEQ, tool="imgt", variable=False, pad_char=None)
    variable=extract_region(sequence=SEQ, tool="imgt", variable=True, pad_char=None)
    assert variable+constant==SEQ, "extract_region() is failing to recover the sample sequence."
    return

def test_pad_sides()-> None:
    variable=extract_region(sequence=SEQ, tool="imgt", variable=True, pad_char=None)
    variable_len=150
    variable_pad=pad_sides(sequence=variable,target_length=variable_len,variable=True,pad_char="_")
    assert len(variable_pad)==variable_len, "Padding variable region is failing to achieve the correct length"
    assert variable_pad[0]=="_", "Variable region might not be being padded on the correct side (left)"
    constant=extract_region(sequence=SEQ, tool="imgt", variable=False, pad_char=None)
    constant_len=350
    constant_pad=pad_sides(sequence=constant,target_length=constant_len,variable=False,pad_char="_")
    assert len(constant_pad)==constant_len, "Padding constant region is failing to achieve the correct length"
    assert constant_pad[-1]=="_", "Constsnt region might not be being padded on the correct side (right)"
