#!/usr/bin/env python3
"""
JSONL Text Field Statistics Tool

This script recursively finds all jsonl files in a given directory,
extracts the "text" field from each line, and provides comprehensive statistics.

Usage:
    python stat_jsonl.py <directory_path>
    python stat_jsonl.py <directory_path> --model <model_file>  # with sentencepiece
    
Example:
    python stat_jsonl.py ./data
    python stat_jsonl.py ./data --model ./m.model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from collections import Counter
import statistics
import re


def find_jsonl_files(directory: str) -> List[str]:
    """
    Recursively find all jsonl files in the given directory.
    
    Args:
        directory: The root directory to search
        
    Returns:
        List of paths to jsonl files
    """
    jsonl_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    
    # Recursively find all .jsonl files
    for jsonl_file in directory_path.rglob("*.jsonl"):
        jsonl_files.append(str(jsonl_file))
    
    # Also find files with .jsonl extension variations (e.g., .jsonl.gz)
    for pattern in ["*.jsonl.*", "*.jl"]:
        for jsonl_file in directory_path.rglob(pattern):
            if str(jsonl_file) not in jsonl_files:
                jsonl_files.append(str(jsonl_file))
    
    return sorted(jsonl_files)


def load_sentencepiece(model_path: Optional[str] = None):
    """
    Load SentencePiece model for tokenization.
    
    Args:
        model_path: Path to the sentencepiece model file
        
    Returns:
        SentencePieceProcessor instance or None
    """
    if model_path is None:
        return None
    
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model_path)
        print(f"Loaded SentencePiece model: {model_path}")
        print(f"Vocab size: {sp.get_piece_size()}")
        return sp
    except ImportError:
        print("Warning: sentencepiece not installed. Token statistics will be skipped.")
        return None
    except Exception as e:
        print(f"Warning: Failed to load model '{model_path}': {e}")
        return None


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def calculate_distribution_stats(data: List[float]) -> Dict:
    """Calculate distribution statistics for a list of values."""
    if not data:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'p50': 0,
            'p75': 0,
            'p90': 0,
            'p95': 0,
            'p99': 0
        }
    
    return {
        'min': min(data),
        'max': max(data),
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'std': statistics.stdev(data) if len(data) > 1 else 0,
        'p50': calculate_percentile(data, 50),
        'p75': calculate_percentile(data, 75),
        'p90': calculate_percentile(data, 90),
        'p95': calculate_percentile(data, 95),
        'p99': calculate_percentile(data, 99)
    }


def count_words(text: str) -> int:
    """Count words in text (split by whitespace)."""
    if not text:
        return 0
    return len(text.split())


def analyze_char_types(text: str) -> Dict:
    """Analyze character types in text."""
    if not text:
        return {
            'alpha_count': 0,
            'digit_count': 0,
            'punct_count': 0,
            'space_count': 0,
            'chinese_count': 0,
            'other_count': 0,
            'utf8_bytes': 0
        }
    
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    punct_count = sum(1 for c in text if c in '.,!?;:\'"()[]{}—–-')
    space_count = sum(1 for c in text if c.isspace())
    # Chinese character detection (CJK Unicode range)
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
    other_count = len(text) - alpha_count - digit_count - punct_count - space_count - chinese_count
    
    # UTF-8 byte count
    utf8_bytes = len(text.encode('utf-8'))
    
    return {
        'alpha_count': alpha_count,
        'digit_count': digit_count,
        'punct_count': punct_count,
        'space_count': space_count,
        'chinese_count': chinese_count,
        'other_count': other_count,
        'utf8_bytes': utf8_bytes
    }


def count_sentences(text: str) -> int:
    """Count sentences in text (naive approach)."""
    if not text:
        return 0
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    return sum(1 for s in sentences if s.strip())


def process_jsonl_file(file_path: str, sp_model=None) -> Dict:
    """
    Process a single jsonl file and extract text statistics.
    
    Args:
        file_path: Path to the jsonl file
        sp_model: Optional SentencePieceProcessor for tokenization
        
    Returns:
        Dictionary containing statistics for this file
    """
    stats = {
        'file': file_path,
        'total_lines': 0,
        'valid_lines': 0,
        'empty_lines': 0,
        'error_lines': 0,
        'missing_text_field': 0,
        'empty_text_field': 0,
        'total_chars': 0,
        'total_tokens': 0,
        'total_words': 0,
        'total_sentences': 0,
        'total_utf8_bytes': 0,
        'char_counts': [],
        'token_counts': [],
        'word_counts': [],
        'sentence_counts': [],
        'utf8_byte_counts': [],
        'texts': [],
        'all_fields': set(),
        'field_counts': Counter(),
        # Character type aggregations
        'total_alpha': 0,
        'total_digit': 0,
        'total_punct': 0,
        'total_space': 0,
        'total_chinese': 0,
        'total_other': 0,
        # Word frequency (sampled)
        'word_freq': Counter(),
        'line_stats': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    stats['empty_lines'] += 1
                    continue
                
                stats['total_lines'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # Track all fields in the JSON
                    for key in data.keys():
                        stats['all_fields'].add(key)
                    stats['field_counts'].update(data.keys())
                    
                    text = data.get('text', None)
                    
                    if text is None:
                        # text field is missing
                        stats['missing_text_field'] += 1
                        stats['valid_lines'] += 1
                        stats['line_stats'].append({
                            'line_num': line_num,
                            'char_count': 0,
                            'token_count': 0,
                            'word_count': 0,
                            'sentence_count': 0,
                            'utf8_bytes': 0,
                            'has_text': False
                        })
                        continue
                    
                    if not isinstance(text, str):
                        # Handle case where text might be a list or other type
                        text = str(text) if text else ''
                    
                    # Check if text field is empty string
                    if not text:
                        stats['empty_text_field'] += 1
                    
                    # Character count
                    char_count = len(text)
                    stats['total_chars'] += char_count
                    stats['char_counts'].append(char_count)
                    
                    # Word count
                    word_count = count_words(text)
                    stats['total_words'] += word_count
                    stats['word_counts'].append(word_count)
                    
                    # Word frequency (sample first 1000 lines per file)
                    if len(stats['texts']) < 1000:
                        words = text.lower().split()
                        stats['word_freq'].update(words)
                    
                    # Sentence count
                    sentence_count = count_sentences(text)
                    stats['total_sentences'] += sentence_count
                    stats['sentence_counts'].append(sentence_count)
                    
                    # Character type analysis
                    char_types = analyze_char_types(text)
                    stats['total_alpha'] += char_types['alpha_count']
                    stats['total_digit'] += char_types['digit_count']
                    stats['total_punct'] += char_types['punct_count']
                    stats['total_space'] += char_types['space_count']
                    stats['total_chinese'] += char_types['chinese_count']
                    stats['total_other'] += char_types['other_count']
                    stats['total_utf8_bytes'] += char_types['utf8_bytes']
                    stats['utf8_byte_counts'].append(char_types['utf8_bytes'])
                    
                    # Token count (using sentencepiece if available)
                    token_count = 0
                    if sp_model and text:
                        try:
                            ids = sp_model.encode(text, return_type=int)
                            token_count = len(ids)
                            stats['total_tokens'] += token_count
                            stats['token_counts'].append(token_count)
                        except Exception as e:
                            pass
                    
                    stats['valid_lines'] += 1
                    stats['texts'].append(text)
                    stats['line_stats'].append({
                        'line_num': line_num,
                        'char_count': char_count,
                        'token_count': token_count,
                        'word_count': word_count,
                        'sentence_count': sentence_count,
                        'utf8_bytes': char_types['utf8_bytes'],
                        'has_text': bool(text)
                    })
                    
                except json.JSONDecodeError:
                    stats['error_lines'] += 1
                except Exception as e:
                    stats['error_lines'] += 1
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
    
    return stats


def aggregate_statistics(all_stats: List[Dict]) -> Dict:
    """
    Aggregate statistics from all processed files.
    
    Args:
        all_stats: List of statistics dictionaries from each file
        
    Returns:
        Aggregated statistics dictionary
    """
    total_files = len(all_stats)
    total_lines = sum(s['total_lines'] for s in all_stats)
    valid_lines = sum(s['valid_lines'] for s in all_stats)
    empty_lines = sum(s['empty_lines'] for s in all_stats)
    error_lines = sum(s['error_lines'] for s in all_stats)
    missing_text_field = sum(s['missing_text_field'] for s in all_stats)
    empty_text_field = sum(s['empty_text_field'] for s in all_stats)
    total_chars = sum(s['total_chars'] for s in all_stats)
    total_tokens = sum(s['total_tokens'] for s in all_stats)
    total_words = sum(s['total_words'] for s in all_stats)
    total_sentences = sum(s['total_sentences'] for s in all_stats)
    total_utf8_bytes = sum(s['total_utf8_bytes'] for s in all_stats)
    
    # Character type totals
    total_alpha = sum(s['total_alpha'] for s in all_stats)
    total_digit = sum(s['total_digit'] for s in all_stats)
    total_punct = sum(s['total_punct'] for s in all_stats)
    total_space = sum(s['total_space'] for s in all_stats)
    total_chinese = sum(s['total_chinese'] for s in all_stats)
    total_other = sum(s['total_other'] for s in all_stats)
    
    # Collect all counts for distribution
    all_char_counts = []
    all_token_counts = []
    all_word_counts = []
    all_sentence_counts = []
    all_utf8_byte_counts = []
    all_texts = []
    all_fields = set()
    total_field_counts = Counter()
    total_word_freq = Counter()
    
    for s in all_stats:
        all_char_counts.extend(s['char_counts'])
        all_token_counts.extend(s['token_counts'])
        all_word_counts.extend(s['word_counts'])
        all_sentence_counts.extend(s['sentence_counts'])
        all_utf8_byte_counts.extend(s['utf8_byte_counts'])
        all_texts.extend(s['texts'])
        all_fields.update(s['all_fields'])
        total_field_counts.update(s['field_counts'])
        total_word_freq.update(s['word_freq'])
    
    # Calculate averages
    valid_text_lines = valid_lines - missing_text_field - empty_text_field
    avg_chars_per_line = total_chars / valid_text_lines if valid_text_lines > 0 else 0
    avg_tokens_per_line = total_tokens / valid_text_lines if valid_text_lines > 0 else 0
    avg_words_per_line = total_words / valid_text_lines if valid_text_lines > 0 else 0
    avg_sentences_per_line = total_sentences / valid_text_lines if valid_text_lines > 0 else 0
    avg_utf8_bytes_per_line = total_utf8_bytes / valid_text_lines if valid_text_lines > 0 else 0
    
    # Calculate distribution statistics
    char_distribution = calculate_distribution_stats(all_char_counts)
    word_distribution = calculate_distribution_stats(all_word_counts)
    sentence_distribution = calculate_distribution_stats(all_sentence_counts)
    utf8_byte_distribution = calculate_distribution_stats(all_utf8_byte_counts)
    token_distribution = calculate_distribution_stats(all_token_counts) if all_token_counts else None
    
    # Duplicate and unique text statistics
    text_counter = Counter(all_texts)
    duplicate_lines = sum(1 for count in text_counter.values() if count > 1)
    unique_lines = len(text_counter)
    
    # Top words
    top_words = dict(total_word_freq.most_common(20))
    
    # Calculate character type percentages
    total_chars_all = total_alpha + total_digit + total_punct + total_space + total_chinese + total_other
    char_type_percentages = {
        'alpha_pct': (total_alpha / total_chars_all * 100) if total_chars_all > 0 else 0,
        'digit_pct': (total_digit / total_chars_all * 100) if total_chars_all > 0 else 0,
        'punct_pct': (total_punct / total_chars_all * 100) if total_chars_all > 0 else 0,
        'space_pct': (total_space / total_chars_all * 100) if total_chars_all > 0 else 0,
        'chinese_pct': (total_chinese / total_chars_all * 100) if total_chars_all > 0 else 0,
        'other_pct': (total_other / total_chars_all * 100) if total_chars_all > 0 else 0,
    }
    
    return {
        'total_files': total_files,
        'total_lines': total_lines,
        'valid_lines': valid_lines,
        'empty_lines': empty_lines,
        'error_lines': error_lines,
        'missing_text_field': missing_text_field,
        'empty_text_field': empty_text_field,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'total_words': total_words,
        'total_sentences': total_sentences,
        'total_utf8_bytes': total_utf8_bytes,
        'avg_chars_per_line': avg_chars_per_line,
        'avg_tokens_per_line': avg_tokens_per_line,
        'avg_words_per_line': avg_words_per_line,
        'avg_sentences_per_line': avg_sentences_per_line,
        'avg_utf8_bytes_per_line': avg_utf8_bytes_per_line,
        'char_distribution': char_distribution,
        'word_distribution': word_distribution,
        'sentence_distribution': sentence_distribution,
        'utf8_byte_distribution': utf8_byte_distribution,
        'token_distribution': token_distribution,
        'duplicate_lines': duplicate_lines,
        'unique_lines': unique_lines,
        'total_unique_texts': len(text_counter),
        'all_fields': list(all_fields),
        'field_counts': dict(total_field_counts),
        # Character type stats
        'char_types': {
            'alpha': total_alpha,
            'digit': total_digit,
            'punct': total_punct,
            'space': total_space,
            'chinese': total_chinese,
            'other': total_other,
            'percentages': char_type_percentages
        },
        # Top words
        'top_words': top_words
    }


def print_statistics(all_stats: List[Dict], aggregated: Dict, show_details: bool = False):
    """
    Print the statistics in a formatted way.
    
    Args:
        all_stats: List of statistics from each file
        aggregated: Aggregated statistics
        show_details: Whether to show detailed statistics for each file
    """
    print("\n" + "=" * 70)
    print("                    JSONL Text Field Statistics                      ")
    print("=" * 70)
    
    # Basic counts
    print(f"\n{'─' * 70}")
    print("                          Basic Counts                                ")
    print(f"{'─' * 70}")
    print(f"  Total files found:          {aggregated['total_files']:>15,}")
    print(f"  Total lines:               {aggregated['total_lines']:>15,}")
    print(f"  Valid lines (with text):   {aggregated['valid_lines']:>15,}")
    print(f"  Empty lines:               {aggregated['empty_lines']:>15,}")
    print(f"  Error lines (parse fail):  {aggregated['error_lines']:>15,}")
    print(f"  Missing text field:         {aggregated['missing_text_field']:>15,}")
    print(f"  Empty text field:           {aggregated['empty_text_field']:>15,}")
    
    # Total counts
    print(f"\n{'─' * 70}")
    print("                          Total Counts                               ")
    print(f"{'─' * 70}")
    print(f"  Total characters:          {aggregated['total_chars']:>15,}")
    print(f"  Total words:               {aggregated['total_words']:>15,}")
    print(f"  Total sentences:           {aggregated['total_sentences']:>15,}")
    print(f"  Total tokens:              {aggregated['total_tokens']:>15,}")
    print(f"  Total UTF-8 bytes:         {aggregated['total_utf8_bytes']:>15,}")
    
    # Average counts
    print(f"\n{'─' * 70}")
    print("                         Average per Line                            ")
    print(f"{'─' * 70}")
    print(f"  Avg chars per line:        {aggregated['avg_chars_per_line']:>15.2f}")
    print(f"  Avg words per line:        {aggregated['avg_words_per_line']:>15.2f}")
    print(f"  Avg sentences per line:    {aggregated['avg_sentences_per_line']:>15.2f}")
    print(f"  Avg tokens per line:       {aggregated['avg_tokens_per_line']:>15.2f}")
    print(f"  Avg UTF-8 bytes per line:  {aggregated['avg_utf8_bytes_per_line']:>15.2f}")
    
    # Character type distribution
    print(f"\n{'─' * 70}")
    print("                      Character Type Distribution                   ")
    print(f"{'─' * 70}")
    ct = aggregated['char_types']
    print(f"  Alphabetic characters:    {ct['alpha']:>15,} ({ct['percentages']['alpha_pct']:>5.1f}%)")
    print(f"  Digit characters:         {ct['digit']:>15,} ({ct['percentages']['digit_pct']:>5.1f}%)")
    print(f"  Punctuation characters:   {ct['punct']:>15,} ({ct['percentages']['punct_pct']:>5.1f}%)")
    print(f"  Whitespace characters:    {ct['space']:>15,} ({ct['percentages']['space_pct']:>5.1f}%)")
    print(f"  Chinese characters:        {ct['chinese']:>15,} ({ct['percentages']['chinese_pct']:>5.1f}%)")
    print(f"  Other characters:          {ct['other']:>15,} ({ct['percentages']['other_pct']:>5.1f}%)")
    
    # Character count distribution
    print(f"\n{'─' * 70}")
    print("                    Character Count Distribution                    ")
    print(f"{'─' * 70}")
    cd = aggregated['char_distribution']
    print(f"  Min:                       {cd['min']:>15,}")
    print(f"  Max:                       {cd['max']:>15,}")
    print(f"  Mean:                      {cd['mean']:>15.2f}")
    print(f"  Median:                    {cd['median']:>15.2f}")
    print(f"  Std Dev:                   {cd['std']:>15.2f}")
    print(f"  Percentiles:")
    print(f"    P50:                     {cd['p50']:>15.2f}")
    print(f"    P75:                     {cd['p75']:>15.2f}")
    print(f"    P90:                     {cd['p90']:>15.2f}")
    print(f"    P95:                     {cd['p95']:>15.2f}")
    print(f"    P99:                     {cd['p99']:>15.2f}")
    
    # Word count distribution
    print(f"\n{'─' * 70}")
    print("                      Word Count Distribution                       ")
    print(f"{'─' * 70}")
    wd = aggregated['word_distribution']
    print(f"  Min:                       {wd['min']:>15,}")
    print(f"  Max:                       {wd['max']:>15,}")
    print(f"  Mean:                      {wd['mean']:>15.2f}")
    print(f"  Median:                    {wd['median']:>15.2f}")
    print(f"  Std Dev:                   {wd['std']:>15.2f}")
    print(f"  Percentiles:")
    print(f"    P50:                     {wd['p50']:>15.2f}")
    print(f"    P75:                     {wd['p75']:>15.2f}")
    print(f"    P90:                     {wd['p90']:>15.2f}")
    print(f"    P95:                     {wd['p95']:>15.2f}")
    print(f"    P99:                     {wd['p99']:>15.2f}")
    
    # Sentence count distribution
    print(f"\n{'─' * 70}")
    print("                    Sentence Count Distribution                    ")
    print(f"{'─' * 70}")
    sd = aggregated['sentence_distribution']
    print(f"  Min:                       {sd['min']:>15,}")
    print(f"  Max:                       {sd['max']:>15,}")
    print(f"  Mean:                      {sd['mean']:>15.2f}")
    print(f"  Median:                    {sd['median']:>15.2f}")
    print(f"  Std Dev:                   {sd['std']:>15.2f}")
    print(f"  Percentiles:")
    print(f"    P50:                     {sd['p50']:>15.2f}")
    print(f"    P75:                     {sd['p75']:>15.2f}")
    print(f"    P90:                     {sd['p90']:>15.2f}")
    print(f"    P95:                     {sd['p95']:>15.2f}")
    print(f"    P99:                     {sd['p99']:>15.2f}")
    
    # UTF-8 byte distribution
    print(f"\n{'─' * 70}")
    print("                    UTF-8 Byte Count Distribution                   ")
    print(f"{'─' * 70}")
    bd = aggregated['utf8_byte_distribution']
    print(f"  Min:                       {bd['min']:>15,}")
    print(f"  Max:                       {bd['max']:>15,}")
    print(f"  Mean:                      {bd['mean']:>15.2f}")
    print(f"  Median:                    {bd['median']:>15.2f}")
    print(f"  Std Dev:                   {bd['std']:>15.2f}")
    print(f"  Percentiles:")
    print(f"    P50:                     {bd['p50']:>15.2f}")
    print(f"    P75:                     {bd['p75']:>15.2f}")
    print(f"    P90:                     {bd['p90']:>15.2f}")
    print(f"    P95:                     {bd['p95']:>15.2f}")
    print(f"    P99:                     {bd['p99']:>15.2f}")
    
    # Token distribution (if available)
    if aggregated['token_distribution']:
        print(f"\n{'─' * 70}")
        print("                     Token Count Distribution                     ")
        print(f"{'─' * 70}")
        td = aggregated['token_distribution']
        print(f"  Min:                       {td['min']:>15,}")
        print(f"  Max:                       {td['max']:>15,}")
        print(f"  Mean:                      {td['mean']:>15.2f}")
        print(f"  Median:                    {td['median']:>15.2f}")
        print(f"  Std Dev:                   {td['std']:>15.2f}")
        print(f"  Percentiles:")
        print(f"    P50:                     {td['p50']:>15.2f}")
        print(f"    P75:                     {td['p75']:>15.2f}")
        print(f"    P90:                     {td['p90']:>15.2f}")
        print(f"    P95:                     {td['p95']:>15.2f}")
        print(f"    P99:                     {td['p99']:>15.2f}")
    
    # Data quality
    print(f"\n{'─' * 70}")
    print("                         Data Quality                                ")
    print(f"{'─' * 70}")
    print(f"  Unique text lines:         {aggregated['unique_lines']:>15,}")
    print(f"  Duplicate lines:           {aggregated['duplicate_lines']:>15,}")
    
    # Top words
    if aggregated['top_words']:
        print(f"\n{'─' * 70}")
        print("                      Top 20 Most Common Words                     ")
        print(f"{'─' * 70}")
        for i, (word, count) in enumerate(aggregated['top_words'].items(), 1):
            print(f"  {i:>2}. {word:<20} {count:>10,}")
    
    # Field statistics
    if aggregated['all_fields']:
        print(f"\n{'─' * 70}")
        print("                      Field Statistics                             ")
        print(f"{'─' * 70}")
        print(f"  All fields found:         {', '.join(aggregated['all_fields'])}")
        print(f"\n  Field occurrence counts:")
        for field, count in sorted(aggregated['field_counts'].items(), key=lambda x: -x[1]):
            print(f"    {field}: {count:,}")
    
    # Per-file statistics
    if show_details:
        print(f"\n{'=' * 70}")
        print("                       Per-File Statistics                        ")
        print(f"{'=' * 70}")
        for stats in all_stats:
            print(f"\n  File: {stats['file']}")
            print(f"    Lines: {stats['total_lines']} (valid: {stats['valid_lines']}, empty: {stats['empty_lines']}, error: {stats['error_lines']}, missing: {stats['missing_text_field']}, empty_text: {stats['empty_text_field']})")
            print(f"    Characters: {stats['total_chars']:,}")
            print(f"    Words: {stats['total_words']:,}")
            print(f"    Sentences: {stats['total_sentences']:,}")
            print(f"    Tokens: {stats['total_tokens']:,}")
            print(f"    UTF-8 bytes: {stats['total_utf8_bytes']:,}")
            if stats['valid_lines'] > 0:
                print(f"    Avg chars/line: {stats['total_chars']/stats['valid_lines']:.2f}")
                print(f"    Avg words/line: {stats['total_words']/stats['valid_lines']:.2f}")
                print(f"    Avg sentences/line: {stats['total_sentences']/stats['valid_lines']:.2f}")
                print(f"    Avg tokens/line: {stats['total_tokens']/stats['valid_lines']:.2f}")


def save_results(aggregated: Dict, all_stats: List[Dict], output_file: str):
    """
    Save statistics results to a JSON file.
    
    Args:
        aggregated: Aggregated statistics
        all_stats: Per-file statistics
        output_file: Path to output file
    """
    # Convert Counter to dict for JSON serialization
    output_data = {
        'aggregated': {
            'total_files': aggregated['total_files'],
            'total_lines': aggregated['total_lines'],
            'valid_lines': aggregated['valid_lines'],
            'empty_lines': aggregated['empty_lines'],
            'error_lines': aggregated['error_lines'],
            'missing_text_field': aggregated['missing_text_field'],
            'empty_text_field': aggregated['empty_text_field'],
            'total_chars': aggregated['total_chars'],
            'total_words': aggregated['total_words'],
            'total_sentences': aggregated['total_sentences'],
            'total_tokens': aggregated['total_tokens'],
            'total_utf8_bytes': aggregated['total_utf8_bytes'],
            'avg_chars_per_line': aggregated['avg_chars_per_line'],
            'avg_words_per_line': aggregated['avg_words_per_line'],
            'avg_sentences_per_line': aggregated['avg_sentences_per_line'],
            'avg_tokens_per_line': aggregated['avg_tokens_per_line'],
            'avg_utf8_bytes_per_line': aggregated['avg_utf8_bytes_per_line'],
            'char_distribution': aggregated['char_distribution'],
            'word_distribution': aggregated['word_distribution'],
            'sentence_distribution': aggregated['sentence_distribution'],
            'utf8_byte_distribution': aggregated['utf8_byte_distribution'],
            'token_distribution': aggregated['token_distribution'],
            'duplicate_lines': aggregated['duplicate_lines'],
            'unique_lines': aggregated['unique_lines'],
            'total_unique_texts': aggregated['total_unique_texts'],
            'all_fields': aggregated['all_fields'],
            'field_counts': aggregated['field_counts'],
            'char_types': aggregated['char_types'],
            'top_words': aggregated['top_words']
        },
        'files': []
    }
    
    for stats in all_stats:
        file_info = {
            'file': stats['file'],
            'total_lines': stats['total_lines'],
            'valid_lines': stats['valid_lines'],
            'empty_lines': stats['empty_lines'],
            'error_lines': stats['error_lines'],
            'missing_text_field': stats['missing_text_field'],
            'empty_text_field': stats['empty_text_field'],
            'total_chars': stats['total_chars'],
            'total_words': stats['total_words'],
            'total_sentences': stats['total_sentences'],
            'total_tokens': stats['total_tokens'],
            'total_utf8_bytes': stats['total_utf8_bytes']
        }
        
        # Add distribution stats for each file
        if stats['char_counts']:
            file_info['char_distribution'] = calculate_distribution_stats(stats['char_counts'])
        if stats['word_counts']:
            file_info['word_distribution'] = calculate_distribution_stats(stats['word_counts'])
        if stats['sentence_counts']:
            file_info['sentence_distribution'] = calculate_distribution_stats(stats['sentence_counts'])
        if stats['utf8_byte_counts']:
            file_info['utf8_byte_distribution'] = calculate_distribution_stats(stats['utf8_byte_counts'])
        if stats['token_counts']:
            file_info['token_distribution'] = calculate_distribution_stats(stats['token_counts'])
            
        output_data['files'].append(file_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Recursively find jsonl files and extract text field statistics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory to search for jsonl files (default: current directory)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to SentencePiece model file for token counting'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file to save results'
    )
    
    parser.add_argument(
        '--details',
        action='store_true',
        help='Show detailed statistics for each file'
    )
    
    args = parser.parse_args()
    
    # Find all jsonl files
    print(f"Searching for jsonl files in: {args.directory}")
    jsonl_files = find_jsonl_files(args.directory)
    
    if not jsonl_files:
        print("No jsonl files found.")
        return
    
    print(f"Found {len(jsonl_files)} jsonl file(s)")
    
    # Load SentencePiece model if provided
    sp_model = load_sentencepiece(args.model)
    
    # Process each file
    print("\nProcessing files...")
    all_stats = []
    
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"  [{i}/{len(jsonl_files)}] Processing: {jsonl_file}")
        stats = process_jsonl_file(jsonl_file, sp_model)
        all_stats.append(stats)
    
    # Aggregate statistics
    aggregated = aggregate_statistics(all_stats)
    
    # Print results
    print_statistics(all_stats, aggregated, args.details)
    
    # Save results if output file specified
    if args.output:
        save_results(aggregated, all_stats, args.output)


if __name__ == '__main__':
    main()
