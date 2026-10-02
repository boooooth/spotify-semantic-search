from pathlib import Path
import pandas as pd

# Dynamically resolve project root (one directory up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "data" / "spotify_millsongdata.csv"
if not csv_path.exists():
    csv_path = PROJECT_ROOT / "spotify_millsongdata.csv"

print(f"Reading: {csv_path} ...")
df = pd.read_csv(csv_path)

print(f"Total Rows: {len(df):,}\n")
print(f"{'Column':<10} | {'Max Chars':<10} | {'Max Words':<10} | {'Avg Chars':<10} | {'Sample Longest Entry'}")
print("-" * 80)

for col in ["artist", "song", "link", "text"]:
    if col in df.columns:
        series = df[col].fillna("").astype(str)
        
        # Character lengths
        char_lens = series.str.len()
        max_char = char_lens.max()
        avg_char = int(char_lens.mean())
        
        # Word counts
        word_counts = series.str.split().str.len()
        max_words = word_counts.max()
        
        # Longest item preview
        longest_idx = char_lens.idxmax()
        longest_val = series.iloc[longest_idx]
        preview = (longest_val[:40] + "...") if len(longest_val) > 40 else longest_val
        
        print(f"{col:<10} | {max_char:<10} | {max_words:<10} | {avg_char:<10} | {preview}")

print("-" * 80)
