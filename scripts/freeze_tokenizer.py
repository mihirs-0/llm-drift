import hashlib
import transformers
from transformers import GPT2TokenizerFast
from pathlib import Path

# --- Constants ---
TOKENIZER_NAME = "gpt2"
LOCAL_DIR = Path("tokenizer")
PAD_TOKEN = "<|pad|>"

def main():
    print(f"--- 🥶 Freezing Tokenizer ---")
    print(f"Using transformers version: {transformers.__version__}")
    
    LOCAL_DIR.mkdir(exist_ok=True)

    # --- 1. Load, add pad token, and save ---
    print(f"Loading '{TOKENIZER_NAME}' tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you have an internet connection and 'transformers' is installed.")
        return

    # Add the pad token (as per your E1 spec)
    print(f"Adding pad token: '{PAD_TOKEN}'")
    
    # We use add_special_tokens to ensure it's a single token
    # and not split by the BPE model.
    special_tokens_dict = {'pad_token': PAD_TOKEN}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Save to local directory
    try:
        tokenizer.save_pretrained(str(LOCAL_DIR))
        print(f"Tokenizer saved to: {LOCAL_DIR}")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        print(f"Check permissions for directory: {LOCAL_DIR}")
        return

    # --- 2. Create checksum.txt ---
    tokenizer_file = LOCAL_DIR / "tokenizer.json"
    checksum_file = LOCAL_DIR / "checksum.txt"

    # Calculate SHA256 of the tokenizer.json
    try:
        with open(tokenizer_file, 'rb') as f:
            data = f.read()
            tokenizer_sha256 = hashlib.sha256(data).hexdigest()
        
        print(f"SHA256(tokenizer.json): {tokenizer_sha256}")
        
        # Write the checksum file
        with open(checksum_file, 'w', encoding='utf-8') as f:
            f.write(f"tokenizer.json_sha256: {tokenizer_sha256}\n")
            f.write(f"transformers_version: {transformers.__version__}\n")
        
        print(f"Checksum file written to: {checksum_file}")

    except FileNotFoundError:
        print(f"ERROR: Could not find {tokenizer_file} to calculate checksum.")
    except Exception as e:
        print(f"Error writing checksum: {e}")

    print("\n--- Tokenizer freeze complete ---")

if __name__ == "__main__":
    main()