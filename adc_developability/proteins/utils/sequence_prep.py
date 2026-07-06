def extract_region(sequence: str, variable: bool=True):
    """
    Verifies that sequence is an antibody sequence.
    Splits sequence into variable region and the rest (constant+hinge)
    if variable is True, returns the variable region. Otherwise, returns the rest
    """
    return
def pad_constant(sequence: str, ref_sequence: str, target_length: int, pad_character: str="X"):
    """
    Verifies that sequence corresponds to an antibody constant region.
    Aligns a costant region with a reference sequence (of anothe constant region)
    Fills the gaps with the padding character
    Pads begining and ending with the padding character to reach a target length
    Throws an error if the length without side padding is higher than target
    """
    return
def pad_variable(sequence: str, target_length: int, pad_character: str="X"):
    """
    Verifies that sequence is a variable region sequence of an antibody
    Uses anarcii to number and pad sequence up to a certain length
    throws an error if the length without padding is higher than the target
    """
    return