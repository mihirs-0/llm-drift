import argparse
import json
import re
import unicodedata
import hashlib
import os
import random
import subprocess
import time
from pathlib import Path
from collections import Counter, defaultdict, deque
import gzip
import signal
import sys

# --- Imports from v8 ---
from datasets import load_dataset
import langid
from tqdm import tqdm
import unicodedata as ud

# --- Signal Handling ---
STOP_REQUESTED = False

def handle_signal(signum, frame):
    """Request a graceful stop on SIGINT/SIGTERM."""
    global STOP_REQUESTED
    if STOP_REQUESTED:
        print(f"\n--- Second signal ({signum}) received. Forcing stop. ---")
    else:
        print(f"\n--- Signal {signum} received, requesting graceful stop... (Press again to force) ---")
    STOP_REQUESTED = True

# --- Reproducibility ---
def set_seed(seed=1337):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed set to {seed}")

# --- Drop Reason Logging ---
EXAMPLES_PER_REASON = 5
drop_examples = defaultdict(lambda: deque(maxlen=EXAMPLES_PER_REASON))

def drop(reason, s, stats_counter):
    stats_counter[reason] += 1
    if len(drop_examples[reason]) < EXAMPLES_PER_REASON:
        drop_examples[reason].append(s[:240])
    if reason == 'non_printable' and stats_counter[reason] <= 3:
        try:
            fails = [c for c in s if not c.isprintable()][:5]
            codes = [hex(ord(c)) for c in fails]
            tqdm.write(f"QoL: 'non_printable' drop. Sample chars: {fails} (Codes: {codes})")
        except Exception:
            pass
    return None

# --- LangID Warmup ---
try:
    langid.set_languages(['en'])
    print("langid constrained to 'en' only.")
except AttributeError:
    print("Could not constrain langid.")

# --- Default Thresholds (CLI overridable) ---
DEFAULT_CHAR_CLASS_THRESHOLD = 0.6
DEFAULT_CODE_META_THRESHOLD = 0.2
DEFAULT_SOCIAL_THRESHOLD = 0.08
DEFAULT_SYMBOL_THRESHOLD = 0.08
DEFAULT_URL_RATE_THRESHOLD = 0.02
DEFAULT_BOILERPLATE_DENSITY_THRESHOLD = 0.02
DEFAULT_BOILERPLATE_TOKEN_LEN = 14
DEFAULT_LANGID_MIN_SCORE = 0.90
DEFAULT_MIN_LEN = 20
DEFAULT_MAX_LEN = 2000
DEFAULT_TARGET_KEPT = 20000 
DEFAULT_IN_MEMORY_HASH_LIMIT = 5_000_000
DEFAULT_NEAR_DUP_LIMIT = 3_000_000
DEFAULT_MAX_LOADED_HASHES = 10_000_000

# --- Set-based filters (static) ---
ALNUM_PUNC = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.;:'\"!?-()/")
CODE_META_SET = set("{}<>#=_`")
SOCIAL_SET = {'@', '#'}
SYMBOL_SET = set("~^*|\\")

# --- Regex patterns (precompiled) ---
WHITESPACE_RE = re.compile(r'\s+')
EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
NONWORD_RE = re.compile(r'\W+')
# Fix 1: Separate © from \b group
BOILERPLATE_RE = re.compile(
    r"\b(cookie|consent|subscribe|newsletter|sign up|accept all|privacy policy)\b|©",
    re.IGNORECASE
)

# --- Gzip Helper Function ---
def get_open_func(is_gzip):
    """Returns the correct open function (gzip or standard)."""
    if is_gzip:
        return lambda p, m: gzip.open(p, f"{m}t", encoding='utf-8')
    else:
        return lambda p, m: open(p, m, encoding='utf-8')

# --- Schema for Stats ---
STATS_SCHEMA = [
    'seen', 'kept', 'dup_exact', 'dup_near', 'non_en', 'short', 'long',
    'bad_chars', 'code_meta', 'non_printable', 'boilerplate_high_density',
    'too_many_urls_rate', 'symbol_noise', 'social_sludge', 'langid_backoff_keep',
    'langid_skipped_long_clean', 'near_dup_resets', 'in_memory_hash_resets',
    'err_unicode', 'err_langid', 'empty', 'empty_lang_prep', 'empty_near_dup'
]


def clean_and_filter(doc_text, stats_counter, seen_exact, seen_near, seen_exact_buffer, args):
    stats_counter['seen'] += 1
    
    try:
        s = unicodedata.normalize('NFC', doc_text)
        s = s.replace("\x00", " ")
        s = WHITESPACE_RE.sub(' ', s).strip()
    except Exception:
        return drop('err_unicode', doc_text, stats_counter)

    if not s:
        return drop('empty', s, stats_counter)

    s_len = len(s)
    if s_len < args.min_len:
        return drop('short', s, stats_counter)
    if s_len > args.max_len:
        return drop('long', s, stats_counter)

    printable_ratio = sum(c.isprintable() for c in s) / s_len
    if printable_ratio < 0.95:
        return drop('non_printable', s, stats_counter)

    if args.allow_emojis:
        char_fraction = sum((c in ALNUM_PUNC) or (ud.category(c).startswith('So')) for c in s) / s_len
    else:
        char_fraction = sum(c in ALNUM_PUNC for c in s) / s_len
    if char_fraction < args.char_class_threshold:
        return drop('bad_chars', s, stats_counter)

    code_meta_fraction = sum(c in CODE_META_SET for c in s) / s_len
    if code_meta_fraction > args.code_meta_threshold:
        return drop('code_meta', s, stats_counter)

    if not args.no_boilerplate_filter:
        matches = BOILERPLATE_RE.findall(s)
        if matches and (len(matches) * args.boilerplate_token_len / s_len) > args.boilerplate_density_threshold:
            return drop('boilerplate_high_density', s, stats_counter)
    
    # Fix 4: Calculate URL hits once
    url_hits = s.count("http://") + s.count("https://")
    if not args.no_url_filter and (url_hits / s_len) > args.url_rate_threshold:
        return drop('too_many_urls_rate', s, stats_counter)

    if not args.no_symbol_filter:
        if (sum(c in SYMBOL_SET for c in s) / s_len) > args.symbol_threshold:
            return drop('symbol_noise', s, stats_counter)

    if not args.no_social_filter:
        if (sum(c in SOCIAL_SET for c in s) / s_len) > args.social_threshold:
            return drop('social_sludge', s, stats_counter)

    if s_len > 1200 and char_fraction > 0.95:
        stats_counter['langid_skipped_long_clean'] += 1
    else:
        try:
            s_for_lang = EMAIL_RE.sub(' ', s)
            s_for_lang = s_for_lang.replace("http://", " ").replace("https://", " ")
            words = s_for_lang.split()
            if len(words) > 250:
                s_for_lang = ' '.join(words[:250])
            else:
                s_for_lang = ' '.join(words)
            
            if not s_for_lang.strip():
                return drop('empty_lang_prep', s, stats_counter)
                
            lang, score = langid.classify(s_for_lang)
            if lang != "en" or score < args.langid_min_score:
                # Fix 5: Tighten lang backoff
                ascii_alpha_ratio = sum(c.isalpha() and ord(c) < 128 for c in s) / s_len
                if lang == "en" and char_fraction > 0.9 and s_len > 200 and ascii_alpha_ratio > 0.6:
                    stats_counter['langid_backoff_keep'] += 1
                else:
                    return drop('non_en', s, stats_counter)
        except Exception:
            return drop('err_langid', s, stats_counter)

    s_for_hash = s.lower() if args.lower_exact_dedup else s
    h_exact = hashlib.sha1(s_for_hash.encode('utf-8')).hexdigest()
    if h_exact in seen_exact:
        return drop('dup_exact', s, stats_counter)
    seen_exact.add(h_exact)
    seen_exact_buffer.append(h_exact)
    
    s_near = NONWORD_RE.sub('', s).lower()
    s_near = s_near[:2000]
    if not s_near:
        return drop('empty_near_dup', s, stats_counter)
        
    h_near = hashlib.sha1(s_near.encode('utf-8')).hexdigest()
    if h_near in seen_near:
        return drop('dup_near', s, stats_counter)
    seen_near.add(h_near)

    stats_counter['kept'] += 1
    return s

def main():
    set_seed(1337)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    parser = argparse.ArgumentParser(description="Build a clean text corpus from streaming HF dataset.")
    # --- Core Params ---
    parser.add_argument('--dataset', default='openwebtext', help="HF dataset to stream.")
    parser.add_argument('--split', default='train', help="Dataset split to use.")
    parser.add_argument('--out_dir', default='data_raw', type=Path, help="Output directory for data and logs.")
    parser.add_argument('--target_kept', type=int, default=DEFAULT_TARGET_KEPT, help="Target *total* number of lines to keep.")
    parser.add_argument('--batch_size', type=int, default=10000, help="Number of lines to buffer before writing.")
    parser.add_argument('--sample_run', type=int, default=0, help="For a quick test, process only this many *seen* documents.")
    
    # --- Mode Toggles ---
    # Fix 8: Add help text for incompatibility
    parser.add_argument('--resume', action='store_true', help="Resume from existing stats.json. (Incompatible with --gzip)")
    parser.add_argument('--gzip', action='store_true', help="Write output corpus as .txt.gz. (Incompatible with --resume)")
    parser.add_argument('--allow_emojis', action='store_true', help="Allow emojis (Unicode 'So' category) in char_fraction check.")
    parser.add_argument('--lower_exact_dedup', action='store_true', help="Lowercase text before exact-match hashing for dedup.")
    
    # --- Filter Toggles ---
    parser.add_argument('--no_social_filter', action='store_true')
    parser.add_argument('--no_symbol_filter', action='store_true')
    parser.add_argument('--no_url_filter', action='store_true')
    parser.add_argument('--no_boilerplate_filter', action='store_true')

    # --- Filter Thresholds ---
    parser.add_argument('--min_len', type=int, default=DEFAULT_MIN_LEN)
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument('--url_rate_threshold', type=float, default=DEFAULT_URL_RATE_THRESHOLD)
    parser.add_argument('--boilerplate_density_threshold', type=float, default=DEFAULT_BOILERPLATE_DENSITY_THRESHOLD)
    parser.add_argument('--boilerplate_token_len', type=int, default=DEFAULT_BOILERPLATE_TOKEN_LEN)
    parser.add_argument('--char_class_threshold', type=float, default=DEFAULT_CHAR_CLASS_THRESHOLD)
    parser.add_argument('--code_meta_threshold', type=float, default=DEFAULT_CODE_META_THRESHOLD)
    parser.add_argument('--social_threshold', type=float, default=DEFAULT_SOCIAL_THRESHOLD)
    parser.add_argument('--symbol_threshold', type=float, default=DEFAULT_SYMBOL_THRESHOLD)
    parser.add_argument('--langid_min_score', type=float, default=DEFAULT_LANGID_MIN_SCORE)
    parser.add_argument('--in_memory_hash_limit', type=int, default=DEFAULT_IN_MEMORY_HASH_LIMIT)
    parser.add_argument('--near_dup_limit', type=int, default=DEFAULT_NEAR_DUP_LIMIT)
    parser.add_argument('--max_loaded_hashes', type=int, default=DEFAULT_MAX_LOADED_HASHES)

    args = parser.parse_args()

    # Fix 3: Validate CLI ranges up front
    if not (0 <= args.char_class_threshold <= 1):
        sys.exit(f"ERROR: --char_class_threshold must be in [0, 1], got {args.char_class_threshold}")
    if args.min_len <= 0 or args.max_len <= args.min_len:
        sys.exit(f"ERROR: Invalid lengths, must have 0 < min_len < max_len, got min={args.min_len}, max={args.max_len}")
    for name in ["url_rate_threshold", "boilerplate_density_threshold", "code_meta_threshold",
                 "social_threshold", "symbol_threshold", "langid_min_score"]:
        val = getattr(args, name)
        if val < 0:
            sys.exit(f"ERROR: --{name} must be >= 0, got {val}")

    if args.gzip and args.resume:
        print("ERROR: --resume and --gzip cannot be used together.", file=sys.stderr)
        print("Appending to a .gz file is not reliable. Please run a fresh, gzipped job or resume a non-gzipped one.")
        sys.exit(1)

    # --- File Targets ---
    args.out_dir.mkdir(exist_ok=True)
    open_func = get_open_func(args.gzip)
    corpus_file = args.out_dir / ("corpus.txt.gz" if args.gzip else "corpus.txt")
    stats_file = args.out_dir / "stats.json"
    examples_file = args.out_dir / "drop_examples.json"
    manifest_file = args.out_dir / "manifest.json"
    kept_samples_file = args.out_dir / "kept_samples.txt"
    seen_exact_file = args.out_dir / "seen_exact.sha1.txt"
    KEPT_SAMPLE_TARGET = 100

    # --- Resume Logic ---
    stats = Counter()
    for k in STATS_SCHEMA:
        _ = stats[k]
        
    seen_exact = set()
    seen_exact_buffer = [] 
    seen_near = set()
    open_mode = 'w'

    if args.resume:
        if stats_file.exists() and corpus_file.exists():
            print(f"Resuming... loading stats from {stats_file}")
            try:
                # Fix 2: Use 'with' to ensure file handle is closed
                with open(stats_file, 'r', encoding='utf-8') as f_stats:
                    loaded_stats = Counter(json.load(f_stats))
                stats.update(loaded_stats)
                print(f"Loaded stats. Kept so far: {stats['kept']}. Seen: {stats['seen']}.")
                open_mode = 'a'
            except json.JSONDecodeError:
                print(f"Warning: {stats_file} is corrupt. Starting fresh.")
        
        if seen_exact_file.exists():
            print(f"Resuming... loading exact dedup hashes from {seen_exact_file} (limit: {args.max_loaded_hashes})...")
            try:
                with open(seen_exact_file, 'r', encoding='utf-8') as f_hashes:
                    for i, line in enumerate(f_hashes):
                        if i >= args.max_loaded_hashes:
                            print(f"--- Hit load limit of {args.max_loaded_hashes} hashes. ---")
                            break
                        seen_exact.add(line.strip())
                print(f"Loaded {len(seen_exact)} exact hashes.")
            except Exception as e:
                print(f"Warning: could not load hashes: {e}. Starting with empty set.")
        if open_mode == 'a':
            tqdm.write("\n*** WARNING: Resuming. Near-dedup is not persisted and will reset. ***\n")

    # --- Run Manifest ---
    print("Generating run manifest...")
    args_dict = vars(args).copy()
    args_dict['out_dir'] = str(args.out_dir)
    manifest = {
        "timestamp_start": time.time(),
        "args": args_dict,
        "git_rev": subprocess.getoutput("git rev-parse --short HEAD").strip() or "git_not_found",
        "langid_languages_constrained": ['en'],
    }
    with open(manifest_file, "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_file}")

    # --- Load Dataset ---
    print(f"Loading {args.dataset} (split={args.split}) stream...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    
    line_buffer = []
    kept_sample_buffer = []

    print(f"Starting corpus build. Target: {args.target_kept} *total* kept lines.")
    
    try:
        with open_func(corpus_file, open_mode) as f_corpus:
            total = args.sample_run if args.sample_run > 0 else None
            
            for doc in tqdm(ds, total=total, desc="Processing docs", initial=stats['seen']):
                if STOP_REQUESTED:
                    tqdm.write("\nStop request acknowledged, breaking loop...")
                    break
                
                doc_text = doc.get('text')
                if not doc_text:
                    continue

                cleaned_line = clean_and_filter(
                    doc_text, stats, seen_exact, seen_near, seen_exact_buffer, args
                )

                if cleaned_line:
                    line_buffer.append(cleaned_line + '\n')
                    if open_mode == 'w' and stats['kept'] <= KEPT_SAMPLE_TARGET:
                        kept_sample_buffer.append(cleaned_line + '\n')
                    if stats['kept'] in {1, 10, 100, 1000, 10000}:
                        tqdm.write(f"\n--- KEPT SAMPLE {stats['kept']} ---\n{cleaned_line[:160]}\n")

                if stats['seen'] % args.batch_size == 0:
                    f_corpus.writelines(line_buffer)
                    line_buffer = []
                
                if stats['seen'] % (args.batch_size * 5) == 0:
                    if line_buffer:
                        f_corpus.writelines(line_buffer)
                        line_buffer = []
                    
                    if seen_exact_buffer:
                        tqdm.write(f"\nCheckpoint: Flushing {len(seen_exact_buffer)} new hashes...")
                        with open(seen_exact_file, 'a', encoding='utf-8') as f_hashes:
                            f_hashes.writelines(h + '\n' for h in seen_exact_buffer)
                        seen_exact_buffer.clear()
                        
                        try:
                            hash_file_size = seen_exact_file.stat().st_size
                            if hash_file_size > 1_073_741_824: # 1 GB
                                tqdm.write(f"\n--- WARNING: Hash file {seen_exact_file} is now > 1GB ({hash_file_size // 1024**2} MB) ---")
                        except FileNotFoundError:
                            pass 
                    
                    if len(seen_exact) > args.in_memory_hash_limit:
                        tqdm.write(f"\n--- Checkpoint: In-memory exact hash set > {args.in_memory_hash_limit}. Clearing... ---")
                        seen_exact.clear()
                        stats['in_memory_hash_resets'] = stats.get('in_memory_hash_resets', 0) + 1
                    
                    if len(seen_near) > args.near_dup_limit:
                        tqdm.write(f"\n--- Checkpoint: Near-dup hash set > {args.near_dup_limit}. Clearing... ---")
                        seen_near.clear()
                        stats['near_dup_resets'] = stats.get('near_dup_resets', 0) + 1

                    kept_pct = (stats['kept'] / max(1, stats['seen'])) * 100
                    top_drops = {k: stats.get(k, 0) for k in ('short', 'boilerplate_high_density', 'bad_chars', 'code_meta', 'non_en')}
                    tqdm.write(
                        f"\n--- CHECKPOINT @ {stats['seen']} --- "
                        f"Kept={stats['kept']} ({kept_pct:.2f}%) | "
                        f"TopDrops={top_drops}\n"
                    )
                    with open(stats_file, 'w', encoding='utf-8') as f_stats:
                        json.dump(stats, f_stats, indent=2)

                if stats['kept'] >= args.target_kept:
                    tqdm.write(f"\nTarget of {args.target_kept} total kept lines reached.")
                    break
                if args.sample_run > 0 and stats['seen'] >= args.sample_run:
                    tqdm.write(f"\nSample run limit of {args.sample_run} seen docs reached.")
                    break
    
    finally:
        if STOP_REQUESTED:
            print("--- Graceful stop finalized. Saving remaining data... ---")

        if line_buffer:
            print(f"Flushing final {len(line_buffer)} lines...")
            with open_func(corpus_file, 'a') as f_corpus:
                f_corpus.writelines(line_buffer)
        
        if seen_exact_buffer:
            print(f"Flushing final {len(seen_exact_buffer)} exact hashes...")
            with open(seen_exact_file, 'a', encoding='utf-8') as f_hashes:
                f_hashes.writelines(h + '\n' for h in seen_exact_buffer)
            seen_exact_buffer.clear()
            
        print("\nCorpus build complete.")
        
        print("--- Final Stats ---")
        print(json.dumps(stats, indent=2))
        with open(stats_file, 'w', encoding='utf-8') as f_stats:
            json.dump(stats, f_stats, indent=2)
        
        print("--- Saving Drop Examples ---")
        drop_examples_list = {k: list(v) for k, v in drop_examples.items()}
        with open(examples_file, 'w', encoding='utf-8') as f_examples:
            json.dump(drop_examples_list, f_examples, indent=2)

        if kept_sample_buffer and open_mode == 'w':
            print(f"--- Saving {len(kept_sample_buffer)} Kept Samples ---")
            with open(kept_samples_file, 'w', encoding='utf-8') as f_kept:
                f_kept.writelines(kept_sample_buffer)
        
        if stats.get('near_dup_resets', 0) > 0:
            print(f"\n*** WARNING: Near-dup set was reset {stats['near_dup_resets']} time(s). ***")
        if stats.get('in_memory_hash_resets', 0) > 0:
            print(f"\n*** WARNING: In-memory exact-dup hash set was cleared {stats['in_memory_hash_resets']} time(s). ***")

        print(f"\nCorpus saved to: {corpus_file}")
        print(f"Stats saved to: {stats_file}")
        print(f"Drop examples saved to: {examples_file}")
        print(f"Kept samples saved to: {kept_samples_file}")
        print(f"Manifest saved to: {manifest_file}")

if __name__ == "__main__":
    main()