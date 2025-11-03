scripts/build_corpus.py
A robust, streaming data-processing pipeline for cleaning and filtering Hugging Face datasets (e.g., OpenWebText) for language modeling.

This script is designed for reproducibility, robustness, and debuggability. It streams data to avoid large local downloads, handles interrupts gracefully, and provides detailed logs and samples for both kept and dropped data.

🚀 Core Features
Streaming: Processes datasets line-by-line from Hugging Face, requiring minimal disk space.

Robust Filtering: A multi-stage filtering pipeline that removes boilerplate, code, non-English text, and other noise.

Deduplication: Performs both exact-match and near-match deduplication.

Graceful Shutdown: Handles Ctrl-C (SIGINT) and SIGTERM to flush all buffers and save progress before exiting.

Resumeable: Can be stopped and resumed. Persists exact-match hashes to disk (seen_exact.sha1.txt) to maintain deduplication state across runs.

Configurable: Nearly every filter, threshold, and I/O setting is exposed as a CLI argument.

Debuggable:

Generates a stats.json with detailed counters for all drop reasons.

Generates a drop_examples.json with samples of text that failed each filter.

Generates a kept_samples.txt with the first 100 kept lines for quality-checking.

Generates a manifest.json to record the exact arguments and git revision for a run.

📋 Requirements
datasets
langid
tqdm
Usage
Example 1: Quick Smoke Test
This is the recommended first command. It runs on a small sample, processes quickly, and writes to a separate data_test directory so you can inspect the outputs.

Bash

python scripts/build_corpus.py \
    --target_kept 5000 \
    --sample_run 20000 \
    --out_dir data_test
Example 2: Full Production Run (Fresh)
This runs the full pipeline to build a 200k-line corpus, gzipping the output for space.

Bash

python scripts/build_corpus.py \
    --out_dir data_raw \
    --target_kept 200000 \
    --gzip
Example 3: Resuming a Halted Run
If a run is stopped (or crashes), you can resume it. The script will load stats.json and seen_exact.sha1.txt from the output directory and continue.

Note: ---resume is incompatible with ---gzip.

Bash

python scripts/build_corpus.py \
    --out_dir data_raw \
    --target_kept 200000 \
    --resume
Example 4: Tuned Run (e.g., C4 dataset, allow emojis)
This example targets the c4 dataset, allows emojis, and uses a much larger target.

Bash

python scripts/build_corpus.py \
    --dataset "c4" \
    --dataset_config "en" \
    --data_key "text" \
    --out_dir data_c4 \
    --target_kept 500000 \
    --allow_emojis \
    --gzip
(Note: For C4, you may need to add --dataset_config "en" and --data_key "text" to the load_dataset call in the script, as C4's structure differs from OpenWebText.)

🎛️ CLI Arguments
Core I/O & Run Control
--dataset (default: openwebtext): Hugging Face dataset name to stream.

--split (default: train): Dataset split to use.

--out_dir (default: data_raw): Output directory for all data and logs.

--target_kept (default: 20000): Total number of kept lines to collect.

--batch_size (default: 10000): Lines to buffer in memory before writing to disk.

--sample_run (default: 0): If > 0, stop after processing this many seen documents (for testing).

Mode Toggles
--resume: Resume from stats.json and seen_exact.sha1.txt in out_dir. Incompatible with --gzip.

--gzip: Write output corpus.txt.gz as a gzipped file. Incompatible with --resume.

--allow_emojis: Expands the bad_chars filter to allow Unicode "Symbol, other" (So) category (i.e., emojis).

--lower_exact_dedup: Lowercase text before exact-match hashing (merges case-variant duplicates).

Filter Toggles (Disable Filters)
--no_social_filter: Disable the filter for high @ or # density.

--no_symbol_filter: Disable the filter for high ~^*|\ density.

--no_url_filter: Disable the filter for high URL density.

--no_boilerplate_filter: Disable the filter for high-density boilerplate (e.g., "privacy policy").

Filter Thresholds
--min_len (default: 20): Drop lines shorter than this.

--max_len (default: 2000): Drop lines longer than this.

--url_rate_threshold (default: 0.02): Drop if (http:// + https://) / len(line) > this.

--boilerplate_density_threshold (default: 0.02): Drop if boilerplate heuristic score is > this.

--boilerplate_token_len (default: 14): Assumed average length of a boilerplate token.

--char_class_threshold (default: 0.6): Drop if ratio of "good" chars ([A-Za-z0-9 ,.;:'"!?\-\(\)/]) is < this.

--code_meta_threshold (default: 0.2): Drop if ratio of code/meta chars ({<>#=_}) is > this.

--social_threshold (default: 0.08): Drop if ratio of social chars (@#) is > this.

--symbol_threshold (default: 0.08): Drop if ratio of symbol chars (~^*|\) is > this.

--langid_min_score (default: 0.90): Drop if langid.classify score for en is < this (unless backoff rule applies).

Performance & Memory
--in_memory_hash_limit (default: 5000000): Max exact-match hashes to keep in RAM. Clears set if exceeded (persisted hashes are unaffected).

--near_dup_limit (default: 3000000): Max near-match hashes to keep in RAM. Clears set if exceeded.

--max_loaded_hashes (default: 10000000): Max exact-match hashes to load from disk on --resume.

💾 Outputs
When run with ---out_dir data_my_run, the script produces:

data_my_run/corpus.txt (or .txt.gz): The final, cleaned, one-line-per-document corpus.

data_my_run/stats.json: A JSON file of counters for kept, seen, and all drop(...) reasons.

data_my_run/drop_examples.json: A JSON dictionary mapping drop reasons to a list of up to 5 example lines that failed the filter.

data_my_run/kept_samples.txt: The first 100 lines that passed all filters, for a quick sanity check.

data_my_run/manifest.json: A record of the CLI arguments, thresholds, and git commit used for this run.

data_my_run/seen_exact.sha1.txt: A newline-separated list of all exact-match hashes. This file is appended to during checkpoints and used to load the seen_exact set on --resume.

⚠️ Design Notes
--gzip vs. --resume: You cannot resume a gzipped run. Appending to a .gz file is unreliable. The script will exit if both flags are used.

Exact-Match Deduplication: This is stateful and persistent. Hashes are saved to seen_exact.sha1.txt and reloaded on --resume, ensuring deduplication across multiple sessions.

Near-Match Deduplication: This is ephemeral. The seen_near set lives only in memory and is not saved. It is useful for filtering duplicates within a single run but is reset on --resume.