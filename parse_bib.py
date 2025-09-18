import re
import csv

def clean_html_text(text):
    """Clean HTML tags and special characters from text"""
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up LaTeX/HTML entities
    text = re.sub(r'\\textit\s*\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textbf\s*\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textless([^\\]*?)\\textgreater', r'<\1>', text)
    text = re.sub(r'\\textgreater', '>', text)
    text = re.sub(r'\\textless', '<', text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_field_improved(text, field_name):
    """Improved field extraction that handles complex BibTeX format"""
    # Look for field = { or field = "
    pattern1 = rf'{field_name}\s*=\s*{{'
    pattern2 = rf'{field_name}\s*=\s*"'

    start_match = re.search(pattern1, text, re.IGNORECASE)
    if not start_match:
        start_match = re.search(pattern2, text, re.IGNORECASE)
        if not start_match:
            return None

    start_pos = start_match.end()
    quote_char = '"' if '"' in start_match.group() else '{'

    if quote_char == '{':
        # Handle nested braces
        brace_count = 1
        pos = start_pos
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            return text[start_pos:pos-1].strip()
    else:
        # Handle quoted strings
        pos = start_pos
        while pos < len(text) and text[pos] != '"':
            pos += 1
        if pos < len(text):
            return text[start_pos:pos].strip()

    return None

def parse_bib_file(filepath, label):
    """Parse BibTeX file with improved error handling"""
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found")
        return entries

    # Split by @ to find entries
    parts = content.split('@')
    print(f"Found {len(parts)-1} potential entries in {filepath}")

    for i, part in enumerate(parts[1:], 1):  # skip first empty
        if 'article' in part.lower():
            try:
                entry = {}

                # Extract fields with improved parsing
                title_raw = extract_field_improved(part, 'title')
                abstract_raw = extract_field_improved(part, 'abstract')
                keywords_raw = extract_field_improved(part, 'keywords')

                # Clean the extracted text
                entry['title'] = clean_html_text(title_raw) if title_raw else ''
                entry['abstract'] = clean_html_text(abstract_raw) if abstract_raw else ''
                entry['keywords'] = clean_html_text(keywords_raw) if keywords_raw else ''
                entry['label'] = label

                # Only add if we have both title and abstract (required fields)
                if entry['abstract'] and entry['title']:
                    entries.append(entry)
                    print(f"  ✓ Entry {i}: {entry['title'][:50]}...")
                else:
                    print(f"  ✗ Entry {i}: Missing required fields (title: {bool(entry['title'])}, abstract: {bool(entry['abstract'])})")

            except Exception as e:
                print(f"  ✗ Entry {i}: Error parsing - {str(e)}")
                continue

    print(f"Successfully parsed {len(entries)} entries from {filepath}")
    return entries

def main():
    print("Starting improved BibTeX parsing...")

    dual_entries = parse_bib_file('datasets/dual.bib', 1)
    solo_entries = parse_bib_file('datasets/solo.bib', 0)
    all_entries = dual_entries + solo_entries

    print(f"\nTotal entries parsed: {len(all_entries)}")
    print(f"Dual lifestyle entries: {len(dual_entries)}")
    print(f"Solo lifestyle entries: {len(solo_entries)}")

    if len(all_entries) == 0:
        print("ERROR: No entries found! Check your BibTeX files.")
        return

    # Write to CSV (use temp file to avoid permission issues)
    temp_filename = 'abstracts_temp.csv'
    final_filename = 'abstracts.csv'

    with open(temp_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['abstract_text', 'label', 'title', 'keywords']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_entries:
            writer.writerow({
                'abstract_text': entry['abstract'],
                'label': entry['label'],
                'title': entry['title'],
                'keywords': entry['keywords']
            })

    # Try to replace the original file
    try:
        import os
        if os.path.exists(final_filename):
            os.remove(final_filename)
        os.rename(temp_filename, final_filename)
        print(f"Saved {len(all_entries)} entries to abstracts.csv")
    except Exception as e:
        print(f"Could not rename temp file: {e}")
        print(f"Data saved to {temp_filename} instead")
        final_filename = temp_filename

    # Load and split
    import pandas as pd
    df = pd.read_csv('abstracts.csv')
    print(f"DataFrame shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # Use 60/40 split as requested
    train_df = df.sample(frac=0.6, random_state=42)
    test_df = df.drop(train_df.index)

    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print("✅ Parsing completed successfully!")
    print(f"📊 Final dataset: {len(df)} total abstracts")
    print(f"🎯 Test set now has {len(test_df)} samples")

if __name__ == '__main__':
    main()
