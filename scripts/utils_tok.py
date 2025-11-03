import torch
from transformers import GPT2TokenizerFast
from pathlib import Path
from typing import List

# --- Constants ---
TOKENIZER_DIR = Path("tokenizer")

def get_tokenizer() -> GPT2TokenizerFast:
    """
    Loads the frozen GPT-2 tokenizer from the local 'tokenizer' directory.
    """
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found at {TOKENIZER_DIR}\n"
            "Please run 'scripts/freeze_tokenizer.py' first."
        )
    
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    except Exception as e:
        print(f"Failed to load tokenizer from {TOKENIZER_DIR}: {e}")
        print("Ensure the directory contains 'tokenizer.json', 'vocab.json', etc.")
        raise
        
    # Per your spec, we *only* want EOS, not BOS.
    # We also added <|pad|> in the freeze script.
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id. "
            "Did you run 'scripts/freeze_tokenizer.py'?"
        )
        
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer is missing an EOS token.")
        
    return tokenizer

def encode_forward(tokenizer: GPT2TokenizerFast, line: str) -> List[int]:
    """
    Encodes a string in the forward direction.
    Rule: tok.encode(s, add_special_tokens=False) + [eos]
    """
    # We don't add special tokens, as we are manually appending our *only*
    # special token (EOS) at the end.
    ids = tokenizer.encode(line, add_special_tokens=False)
    return ids + [tokenizer.eos_token_id]

def reverse_tokens(ids: List[int], eos_token_id: int) -> List[int]:
    """
    Reverses a list of token IDs, keeping the final EOS token in place.
    Rule: reverse(ids[:-1]) + [eos]
    """
    if not ids:
        raise ValueError("Cannot reverse an empty list of IDs.")
        
    if ids[-1] != eos_token_id:
        raise ValueError(
            f"Expected final token to be EOS ({eos_token_id}), but got {ids[-1]}"
        )
    
    content_ids = ids[:-1]
    eos_id = [ids[-1]]
    
    # [::-1] creates a reversed copy
    return content_ids[::-1] + eos_id

def encode_reversed(tokenizer: GPT2TokenizerFast, line: str) -> List[int]:
    """
    Encodes a string in the reversed token direction.
    """
    # 1. Encode forward to get the canonical token IDs
    forward_ids = encode_forward(tokenizer, line)
    
    # 2. Reverse the token IDs, keeping EOS at the end
    reversed_ids = reverse_tokens(forward_ids, tokenizer.eos_token_id)
    
    return reversed_ids

# --- Self-Test Main Block ---
if __name__ == "__main__":
    print("--- 🧪 Running Sanity Checks for utils_tok.py ---")
    
    try:
        tok = get_tokenizer()
    except Exception as e:
        print(f"\nCRITICAL: Failed to load tokenizer.")
        print("This script cannot be tested. Exiting.")
        exit(1)

    print(f"Tokenizer loaded (Vocab size: {tok.vocab_size})")
    print(f"EOS ID: {tok.eos_token_id}, PAD ID: {tok.pad_token_id}")

    # --- Test Case ---
    s = "This is a simple test."
    print(f"\nTest string: '{s}'")

    # --- Test 1: Forward Encoding ---
    f_ids = encode_forward(tok, s)
    print(f"\nForward IDs ({len(f_ids)}): {f_ids}")
    print(f"Decoded F: '{tok.decode(f_ids)}'")
    assert f_ids[-1] == tok.eos_token_id, "Forward IDs missing EOS at end."
    print("✅ Test 1 (Forward) PASSED")

    # --- Test 2: Reversed Encoding ---
    r_ids = encode_reversed(tok, s)
    print(f"\nReversed IDs ({len(r_ids)}): {r_ids}")
    print(f"Decoded R: '{tok.decode(r_ids)}'")
    assert r_ids[-1] == tok.eos_token_id, "Reversed IDs missing EOS at end."
    assert len(f_ids) == len(r_ids), "F and R IDs have different lengths."
    assert f_ids[0] != r_ids[0], "Reversed IDs look same as forward."
    print("✅ Test 2 (Reversed) PASSED")

    # --- Test 3: Reversal Invariant (The E1 Sanity Check) ---
    # Re-reverse the reversed IDs
    rere_ids = reverse_tokens(r_ids, tok.eos_token_id)
    
    print(f"\nRe-reversed IDs ({len(rere_ids)}): {rere_ids}")
    print(f"Decoded Re-R: '{tok.decode(rere_ids)}'")
    
    assert f_ids == rere_ids, "Reversal invariant FAILED: rev(rev(F)) != F"
    print("✅ Test 3 (Reversal Invariant) PASSED")

    # --- Test 4: Edge Cases ---
    s_empty = ""
    f_empty = encode_forward(tok, s_empty)
    assert f_empty == [tok.eos_token_id], "Empty string encode failed."
    r_empty = encode_reversed(tok, s_empty)
    assert r_empty == [tok.eos_token_id], "Empty string reverse failed."
    print("\n✅ Test 4 (Edge Cases) PASSED")

    print("\n--- ✨ All Sanity Checks Passed ---")