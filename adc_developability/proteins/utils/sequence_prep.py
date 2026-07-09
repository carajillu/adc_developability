import typing
from typing import Set
from anarci import anarci

# Global amino acid set (extend as needed)
AA1LCODES: Set[str] = set("ACDEFGHIKLMNPQRSTVWYBXZJUOB")

def extract_region(sequence: str, tool: str = "imgt", variable: bool = True, pad_char: str|None = "-") -> str:
    
    """
    Extract the variable or constant region of an antibody sequence using
    IMGT numbering via ANARCI, with optional control over alignment padding.

    This function:
    1. Validates the input sequence using a global amino acid set (AA1LCODES),
       allowing both standard and non-standard residues.
    2. Uses ANARCI with the IMGT scheme to identify and number the antibody
       variable domain.
    3. Reconstructs the variable region from IMGT-numbered alignment output.
       - If padding is present ('-' in ANARCI output), handling depends on `pad_char`:
         * If pad_char is None → gap characters are removed (returns contiguous sequence).
         * If pad_char is a string → gap characters are replaced with pad_char
           (preserves IMGT positional alignment).
    4. Extracts the constant (and hinge) region as the remaining portion of the
       original sequence following the variable domain.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of an antibody chain.
    tool : str, optional
        Must be "imgt". Any other value raises NotImplementedError.
    variable : bool, optional
        If True, returns the reconstructed variable region.
        If False, returns the constant/hinge region.
    pad_char : str or None, optional
        Character used to replace alignment gaps ('-') in the IMGT-numbered
        variable region:
        - None → remove gaps entirely (default behaviour for modelling/ML use)
        - str  → replace gaps with this character (useful for aligned outputs)

    Returns
    -------
    str
        The extracted sequence region:
        - Variable region (IMGT-defined, optionally padded), or
        - Constant + hinge region (unaltered, contiguous sequence)

    Raises
    ------
    NotImplementedError
        If tool is not "imgt".
    ImportError
        If ANARCI is not installed.
    ValueError
        If the sequence is invalid, contains unsupported characters, or cannot
        be IMGT-numbered as an antibody.

    Notes
    -----
    - The variable region is reconstructed from ANARCI's IMGT-numbered alignment,
      not directly sliced from the raw sequence.
    - Gap handling is explicitly controlled via `pad_char`, enabling either:
        * biologically contiguous sequences (no gaps), or
        * positionally aligned IMGT sequences (with padding).
    - The constant region is always taken directly from the original sequence and
      is never padded or modified.
    - This function assumes a single antibody domain and uses the first detected
      domain from ANARCI output.

    Typical use cases include antibody sequence preprocessing, alignment-aware
    modelling, and IMGT-based region extraction workflows.
    """
    if tool.lower() != "imgt":
        raise NotImplementedError("Only IMGT-based extraction is implemented.")

    if anarci is None:
        raise ImportError(
            "ANARCI is required for IMGT numbering. Install with `pip install anarci`."
        )

    seq = sequence.strip().upper()

    # Validate sequence using global AA1LCODES
    if not seq or not all(residue in AA1LCODES for residue in seq):
        raise ValueError("Sequence contains invalid amino acid characters.")

    # Run ANARCI
    results, alignment_details, hit_tables = anarci(
        [("query", seq)], scheme="imgt"
    )

    if not results or not results[0]:
        raise ValueError("Sequence could not be IMGT-numbered as an antibody.")

    #  domain
    domain_tuple = results[0][0]

    try:
        numbering_list, seqstart_index, seqend_index = domain_tuple
    except (ValueError, TypeError):
        raise ValueError("Unexpected ANARCI output format.")

    if numbering_list is None:
        raise ValueError("No numbering information returned.")

    # reconstruct variable region
    variable_region = "".join(
        residue if residue != "-" else ("" if pad_char is None else pad_char)
        for (_, residue) in numbering_list
    )

    # Constant region from original sequence
    constant_region = seq[seqend_index+1:]

    return variable_region if variable else constant_region

def pad_sides(sequence: str, target_length: int, variable: bool = True, pad_char: str = "-") -> str:
    """
    Pad a string to a specified total length using a chosen character.

    Parameters
    ----------
    sequence : str
        The input string to be padded.
    target_length : int
        The desired total length after padding.
    variable : bool, optional
        If True, padding is added to the beginning (left side).
        If False, padding is added to the end (right side).
        Default is True.
    pad_char : str, optional
        The character used for padding. Must be a single character.
        Default is "-".

    Returns
    -------
    str
        The padded string.

    Raises
    ------
    ValueError
        If the input string is longer than the target length.
        If pad_char is not a single character.
    """
    if len(sequence) > target_length:
        raise ValueError(f"Sequence length of {len(sequence)} exceeds target length of {target_length}")
    
    if len(pad_char) != 1:
        raise ValueError("pad_char must be a single character.")

    padding_length = target_length - len(sequence)
    padding = pad_char * padding_length

    if variable:
        return padding + sequence
    else:
        return sequence + padding