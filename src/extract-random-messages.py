import pandas as pd
import sys

def extract_random_messages(csv_path, output_path, column_name='content.cleaned_text', n=50):
    df = pd.read_csv(csv_path)
    # Allow column_name to be a string or a list of strings
    if isinstance(column_name, str):
        columns = [column_name]
    else:
        columns = column_name
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in CSV.")
            sys.exit(1)
    sampled = df[columns].dropna().sample(n=min(n, len(df)), random_state=42)
    sampled.to_csv(output_path, index=False, header=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract-random-messages.py <input_csv> <output_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    extract_random_messages(input_csv, output_csv, column_name=['metadata.message_id', 'content.cleaned_text'])