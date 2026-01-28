import os
import pandas as pd
import re
import unicodedata
import string
import numpy as np
from collections import defaultdict
from config import CLEANING_CONFIG


class StockTwitsDataCleaner:
    def __init__(self, config):
        self.config = config
        self.statistics = {}  # Store statistical information

    def load_data(self, file_path):
        """Load CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded file: {file_path}, data volume: {len(df)}")
            return df
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            return None

    def calculate_text_length_stats(self, df, text_column='Text'):
        """Calculate text length statistics"""
        if df is None or len(df) == 0:
            return {'mean': 0, 'median': 0, 'word_count': []}

        # Calculate character count and word count separately
        char_counts = df[text_column].str.len()
        word_counts = df[text_column].str.split().str.len()

        return {
            'char_mean': char_counts.mean(),
            'char_median': char_counts.median(),
            'word_mean': word_counts.mean(),
            'word_median': word_counts.median(),
            'char_count': char_counts.tolist(),
            'word_count': word_counts.tolist()
        }

    def calculate_char_ratio(self, text):
        """Calculate proportion of English letters and digits"""
        if pd.isna(text) or len(str(text)) == 0:
            return {'english_ratio': 0, 'digit_ratio': 0}

        text_str = str(text)
        total_chars = len(text_str.replace(' ', ''))

        if total_chars == 0:
            return {'english_ratio': 0, 'digit_ratio': 0}

        english_chars = sum(1 for char in text_str if char in string.ascii_letters)
        digit_chars = sum(1 for char in text_str if char.isdigit())

        return {
            'english_ratio': english_chars / total_chars if total_chars > 0 else 0,
            'digit_ratio': digit_chars / total_chars if total_chars > 0 else 0
        }

    def count_keyword_frequency(self, text, keyword_list):
        """Count keyword frequency"""
        if pd.isna(text):
            return 0

        text_lower = str(text).lower()
        count = 0
        for keyword in keyword_list:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        return count

    def count_direction_keywords(self, text):
        """Count direction-related keywords (positive/negative)"""
        if pd.isna(text):
            return {'positive': 0, 'negative': 0}

        text_lower = str(text).lower()
        positive_count = 0
        negative_count = 0

        for keyword in self.config['direction_keywords']['positive']:
            positive_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))

        for keyword in self.config['direction_keywords']['negative']:
            negative_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))

        return {'positive': positive_count, 'negative': negative_count}

    def count_numbers_in_text(self, text):
        """Count entries containing numbers"""
        if pd.isna(text):
            return False
        return bool(re.search(r'\d', str(text)))

    def initialize_asset_stats(self, asset_name):
        """Initialize asset statistics"""
        self.statistics[asset_name] = {
            'original_count': 0,
            'cleaned_count': 0,
            'filter_rate': 0.0,
            # Correction: Store character and word count statistics separately before and after cleaning
            'length_before': {'char_mean': 0, 'char_median': 0, 'word_mean': 0, 'word_median': 0},
            'length_after': {'char_mean': 0, 'char_median': 0, 'word_mean': 0, 'word_median': 0},
            'truncated_count': 0,
            'char_ratio_before': {'english': 0, 'digit': 0},
            'char_ratio_after': {'english': 0, 'digit': 0},
            'keyword_freq_before': 0,
            'keyword_freq_after': 0,
            'direction_count_before': {'positive': 0, 'negative': 0},
            'direction_count_after': {'positive': 0, 'negative': 0},
            'number_count_before': 0,
            'number_count_after': 0
        }

    def remove_empty_entries(self, df, text_column='Text', asset_name=''):
        """Remove empty entries and record statistics"""
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        removed_count = initial_count - len(df)
        print(f"Removed empty content: {removed_count} entries")

        if asset_name:
            self.statistics[asset_name]['original_count'] = initial_count
        return df

    def unicode_normalization(self, text):
        """Unicode normalization"""
        if pd.isna(text):
            return text
        return unicodedata.normalize('NFKC', str(text))

    def remove_web_content(self, text):
        """Remove useless web information"""
        if pd.isna(text):
            return text

        text = str(text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        for keyword in self.config['web_keywords']:
            text = text.replace(keyword, '')

        return text

    def normalize_whitespace(self, text):
        """Compress multiple spaces"""
        if pd.isna(text):
            return text
        return re.sub(r'\s+', ' ', str(text)).strip()

    def clean_special_chars(self, text):
        """Keep English letters, digits, and common symbols"""
        if pd.isna(text):
            return text

        text = str(text)
        allowed_chars_pattern = f"[{re.escape(self.config['allowed_chars'])}]"
        cleaned_text = ''.join(char for char in text if re.match(allowed_chars_pattern, char))

        return cleaned_text

    def is_english(self, text):
        """Detect if content is in English"""
        if pd.isna(text) or len(str(text).strip()) < 10:
            return False

        try:
            text_str = str(text)
            english_chars = sum(1 for char in text_str if char in string.ascii_letters + string.whitespace)
            total_chars = len(text_str.replace(' ', ''))

            if total_chars == 0:
                return False

            english_ratio = english_chars / total_chars
            return english_ratio >= self.config['english_threshold']

        except Exception:
            return False

    def remove_short_texts(self, df, text_column='Text', asset_name=''):
        """Remove texts shorter than 5 words"""
        initial_count = len(df)
        df = df[df[text_column].str.split().str.len() >= self.config['min_words']]
        removed_count = initial_count - len(df)
        print(f"Removed short texts: {removed_count} entries")
        return df

    def check_letter_ratio(self, text):
        """Check proportion of letters"""
        if pd.isna(text):
            return False

        text_str = str(text)
        if len(text_str) == 0:
            return False

        letters = sum(1 for char in text_str if char.isalpha())
        total_chars = len(text_str.replace(' ', ''))

        if total_chars == 0:
            return False

        letter_ratio = letters / total_chars
        return letter_ratio >= self.config['letter_ratio_threshold']

    def calculate_tag_ratio(self, text):
        """Calculate proportion of tags in text"""
        if pd.isna(text):
            return 0

        text_str = str(text)
        words = text_str.split()

        if len(words) == 0:
            return 0

        tag_count = 0
        for word in words:
            if re.match(self.config['tag_pattern'], word):
                tag_count += 1

        tag_ratio = tag_count / len(words)
        return tag_ratio

    def remove_high_tag_ratio_texts(self, df, text_column='Text', asset_name=''):
        """Remove data with tag ratio greater than 90%"""
        initial_count = len(df)
        tag_ratios = df[text_column].apply(self.calculate_tag_ratio)
        df = df[tag_ratios <= self.config['tag_ratio_threshold']]
        removed_count = initial_count - len(df)
        print(f"Removed content with excessive tag ratio: {removed_count} entries")
        return df

    def truncate_text(self, text, asset_name=''):
        """Truncate text to 500 words and record statistics"""
        if pd.isna(text):
            return text

        words = str(text).split()
        if len(words) > self.config['max_words']:
            truncated = ' '.join(words[:self.config['max_words']])
            if asset_name:
                self.statistics[asset_name]['truncated_count'] += 1
            return truncated
        return text

    def format_output(self, text, output_prefix=None):
        """Format output"""
        if pd.isna(text):
            return text
        prefix = output_prefix or self.config['output_prefix']
        return f"{prefix} {text}"

    def remove_duplicates(self, df, text_column='Text', asset_name=''):
        """Remove duplicates"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_column])
        removed_count = initial_count - len(df)
        print(f"Removed duplicates: {removed_count} entries")
        return df

    def collect_statistics(self, df, text_column='Text', asset_name='', stage='before'):
        """Collect statistical information"""
        if df is None or len(df) == 0:
            return

        stats = self.statistics[asset_name]

        if stage == 'before':
            # Before cleaning statistics
            length_stats = self.calculate_text_length_stats(df, text_column)
            stats['length_before'] = {
                'char_mean': length_stats['char_mean'],
                'char_median': length_stats['char_median'],
                'word_mean': length_stats['word_mean'],
                'word_median': length_stats['word_median']
            }

            # Character proportion statistics
            char_ratios = df[text_column].apply(self.calculate_char_ratio)
            stats['char_ratio_before']['english'] = char_ratios.apply(lambda x: x['english_ratio']).mean()
            stats['char_ratio_before']['digit'] = char_ratios.apply(lambda x: x['digit_ratio']).mean()

            # Keyword frequency statistics
            stats['keyword_freq_before'] = df[text_column].apply(
                lambda x: self.count_keyword_frequency(x, self.config['finance_keywords'])
            ).mean()

            # Direction keyword statistics
            direction_counts = df[text_column].apply(self.count_direction_keywords)
            stats['direction_count_before']['positive'] = direction_counts.apply(lambda x: x['positive']).sum()
            stats['direction_count_before']['negative'] = direction_counts.apply(lambda x: x['negative']).sum()

            # Number of entries containing numbers
            stats['number_count_before'] = df[text_column].apply(self.count_numbers_in_text).sum()

        elif stage == 'after':
            # After cleaning statistics
            length_stats = self.calculate_text_length_stats(df, text_column)
            stats['length_after'] = {
                'char_mean': length_stats['char_mean'],
                'char_median': length_stats['char_median'],
                'word_mean': length_stats['word_mean'],
                'word_median': length_stats['word_median']
            }

            stats['cleaned_count'] = len(df)
            stats['filter_rate'] = stats['cleaned_count'] / stats['original_count'] if stats[
                                                                                           'original_count'] > 0 else 0

            # Character proportion statistics
            char_ratios = df[text_column].apply(self.calculate_char_ratio)
            stats['char_ratio_after']['english'] = char_ratios.apply(lambda x: x['english_ratio']).mean()
            stats['char_ratio_after']['digit'] = char_ratios.apply(lambda x: x['digit_ratio']).mean()

            # Keyword frequency statistics
            stats['keyword_freq_after'] = df[text_column].apply(
                lambda x: self.count_keyword_frequency(x, self.config['finance_keywords'])
            ).mean()

            # Direction keyword statistics
            direction_counts = df[text_column].apply(self.count_direction_keywords)
            stats['direction_count_after']['positive'] = direction_counts.apply(lambda x: x['positive']).sum()
            stats['direction_count_after']['negative'] = direction_counts.apply(lambda x: x['negative']).sum()

            # Number of entries containing numbers
            stats['number_count_after'] = df[text_column].apply(self.count_numbers_in_text).sum()

    def clean_data(self, df, text_column='Text', asset_name='', output_prefix=None):
        """Execute full data cleaning pipeline"""
        if df is None:
            return None

        print("Starting data cleaning process...")

        # Initialize asset statistics
        self.initialize_asset_stats(asset_name)

        # Collect pre-cleaning statistics
        self.collect_statistics(df, text_column, asset_name, 'before')

        # 1. Remove empty entries
        df = self.remove_empty_entries(df, text_column, asset_name)

        # 2. Unicode normalization
        df[text_column] = df[text_column].apply(self.unicode_normalization)

        # 3. Remove web content
        df[text_column] = df[text_column].apply(self.remove_web_content)

        # 4. Normalize whitespace
        df[text_column] = df[text_column].apply(self.normalize_whitespace)

        # 5. Clean special characters
        df[text_column] = df[text_column].apply(self.clean_special_chars)

        # 6. Language filtering
        initial_count = len(df)
        df = df[df[text_column].apply(self.is_english)]
        removed_count = initial_count - len(df)
        print(f"Removed non-English content: {removed_count} entries")

        # 7. Remove duplicates
        df = self.remove_duplicates(df, text_column, asset_name)

        # 8. Remove texts shorter than 5 words
        df = self.remove_short_texts(df, text_column, asset_name)

        # 9. Remove texts with letter ratio less than 70%
        initial_count = len(df)
        df = df[df[text_column].apply(self.check_letter_ratio)]
        removed_count = initial_count - len(df)
        print(f"Removed content with insufficient letter ratio: {removed_count} entries")

        # 10. Remove data with tag ratio greater than 90%
        df = self.remove_high_tag_ratio_texts(df, text_column, asset_name)

        # 11. Truncate to 500 words
        df[text_column] = df[text_column].apply(lambda x: self.truncate_text(x, asset_name))

        # 12. Format output
        df[text_column] = df[text_column].apply(
            lambda x: self.format_output(x, output_prefix=output_prefix)
        )

        # Collect post-cleaning statistics
        self.collect_statistics(df, text_column, asset_name, 'after')

        print(f"Cleaning completed, remaining data: {len(df)} entries")
        return df

    def generate_summary_report(self):
        """Generate overall statistical report"""
        if not self.statistics:
            return "No statistical information available"

        report = []
        report.append("=" * 80)
        report.append("Data Cleaning Statistical Report")
        report.append("=" * 80)

        # Overall statistics
        total_original = sum(stats['original_count'] for stats in self.statistics.values())
        total_cleaned = sum(stats['cleaned_count'] for stats in self.statistics.values())
        overall_filter_rate = total_cleaned / total_original if total_original > 0 else 0

        report.append(f"Overall Statistics:")
        report.append(f"  Total entries crawled: {total_original:,}")
        report.append(f"  Total entries after cleaning: {total_cleaned:,}")
        report.append(f"  Overall filtering rate: {overall_filter_rate:.2%}")
        report.append("")

        # Detailed statistics for each asset
        for asset_name, stats in self.statistics.items():
            report.append(f"Asset: {asset_name}")
            report.append(f"  Original entries: {stats['original_count']:,}")
            report.append(f"  Entries after cleaning: {stats['cleaned_count']:,}")
            report.append(f"  Filtering rate: {stats['filter_rate']:.2%}")

            # Correction: Display character and word count statistics separately before and after cleaning
            report.append(f"  Text length statistics - Before cleaning:")
            report.append(
                f"    Characters: mean {stats['length_before']['char_mean']:.1f}, median {stats['length_before']['char_median']:.1f}")
            report.append(
                f"    Words: mean {stats['length_before']['word_mean']:.1f}, median {stats['length_before']['word_median']:.1f}")

            report.append(f"  Text length statistics - After cleaning:")
            report.append(
                f"    Characters: mean {stats['length_after']['char_mean']:.1f}, median {stats['length_after']['char_median']:.1f}")
            report.append(
                f"    Words: mean {stats['length_after']['word_mean']:.1f}, median {stats['length_after']['word_median']:.1f}")

            report.append(f"  Truncated entries: {stats['truncated_count']:,}")
            report.append(
                f"  Character ratio - Before: English {stats['char_ratio_before']['english']:.2%}, Digits {stats['char_ratio_before']['digit']:.2%}")
            report.append(
                f"  Character ratio - After: English {stats['char_ratio_after']['english']:.2%}, Digits {stats['char_ratio_after']['digit']:.2%}")
            report.append(f"  Finance keyword frequency - Before: average {stats['keyword_freq_before']:.2f} per entry")
            report.append(f"  Finance keyword frequency - After: average {stats['keyword_freq_after']:.2f} per entry")
            report.append(
                f"  Direction keywords - Before: Positive {stats['direction_count_before']['positive']:,}, Negative {stats['direction_count_before']['negative']:,}")
            report.append(
                f"  Direction keywords - After: Positive {stats['direction_count_after']['positive']:,}, Negative {stats['direction_count_after']['negative']:,}")
            report.append(f"  Entries containing numbers - Before: {stats['number_count_before']:,}")
            report.append(f"  Entries containing numbers - After: {stats['number_count_after']:,}")
            report.append("")

        return "\n".join(report)

    def save_statistics_to_csv(self, output_file='cleaning_statistics.csv'):
        """Save statistics to CSV file"""
        if not self.statistics:
            print("No statistical information to save")
            return False

        try:
            # Prepare data
            rows = []
            for asset_name, stats in self.statistics.items():
                row = {
                    'asset': asset_name,
                    'original_count': stats['original_count'],
                    'cleaned_count': stats['cleaned_count'],
                    'filter_rate': stats['filter_rate'],
                    # Correction: Save character and word count statistics separately before and after cleaning
                    'char_length_before_mean': stats['length_before']['char_mean'],
                    'char_length_before_median': stats['length_before']['char_median'],
                    'word_length_before_mean': stats['length_before']['word_mean'],
                    'word_length_before_median': stats['length_before']['word_median'],
                    'char_length_after_mean': stats['length_after']['char_mean'],
                    'char_length_after_median': stats['length_after']['char_median'],
                    'word_length_after_mean': stats['length_after']['word_mean'],
                    'word_length_after_median': stats['length_after']['word_median'],
                    'truncated_count': stats['truncated_count'],
                    'english_ratio_before': stats['char_ratio_before']['english'],
                    'digit_ratio_before': stats['char_ratio_before']['digit'],
                    'english_ratio_after': stats['char_ratio_after']['english'],
                    'digit_ratio_after': stats['char_ratio_after']['digit'],
                    'keyword_freq_before': stats['keyword_freq_before'],
                    'keyword_freq_after': stats['keyword_freq_after'],
                    'positive_keywords_before': stats['direction_count_before']['positive'],
                    'negative_keywords_before': stats['direction_count_before']['negative'],
                    'positive_keywords_after': stats['direction_count_after']['positive'],
                    'negative_keywords_after': stats['direction_count_after']['negative'],
                    'number_count_before': stats['number_count_before'],
                    'number_count_after': stats['number_count_after']
                }
                rows.append(row)

            # Create DataFrame and save
            stats_df = pd.DataFrame(rows)
            stats_df.to_csv(output_file, index=False, encoding=self.config['encoding'])
            print(f"Statistics saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Failed to save statistics: {e}")
            return False

    def process_file(self, input_file, output_file, text_column='Text'):
        """Process a single file"""
        print(f"\nProcessing file: {input_file}")

        # Extract asset name from file name
        file_name = os.path.basename(input_file)
        asset_name = os.path.splitext(file_name)[0].replace('_merged', '')

        # Load data
        df = self.load_data(input_file)
        if df is None:
            return False
        if text_column not in df.columns:
            print(f"Missing text column '{text_column}', available columns: {list(df.columns)}")
            return False

        # Clean data
        normalized_path = os.path.normpath(input_file).lower()
        news_marker = os.path.normpath(os.path.join('data', 'news')).lower()
        news_marker_alt = os.path.normpath(os.path.join('data', 'textual_data', 'news')).lower()
        output_prefix = '[NEWS]' if (news_marker in normalized_path or news_marker_alt in normalized_path) else None
        cleaned_df = self.clean_data(
            df,
            text_column=text_column,
            asset_name=asset_name,
            output_prefix=output_prefix
        )
        if cleaned_df is None or len(cleaned_df) == 0:
            print("No valid data after cleaning")
            return False

        # Save result
        try:
            cleaned_df.to_csv(output_file, index=False, encoding=self.config['encoding'])
            print(f"Result saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save file: {e}")
            return False


def main():
    """Main function"""
    # File list
    """files_to_process = [
        'COMPQ_merged.csv',
        'DIA_merged.csv', 
        'GLD_merged.csv',
        'SLV_merged.csv',
        'SPX_merged.csv',
        'USO_merged.csv'
    ]"""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_dir = os.path.join(repo_root, 'data', 'textual_data', 'news')
    files_to_process = [
        'DJI_2017-2024_all.csv',
        'GOLD_2017-2024_all.csv',
        'GSPC_2017-2024_all.csv',
        'IXIC_2017-2024_all.csv',
        'OIL_2017-2024_all.csv',
        'SILVER_2017-2024_all.csv',
    ]

    # Create cleaner instance
    cleaner = StockTwitsDataCleaner(CLEANING_CONFIG)

    # Process each file
    success_count = 0
    for input_file in files_to_process:
        input_path = os.path.join(base_dir, input_file)
        output_file = f"cleaned_{input_file}"
        output_path = os.path.join(base_dir, output_file)
        if cleaner.process_file(input_path, output_path, text_column='title'):
            success_count += 1

    # Generate statistical report
    print("\n" + "=" * 80)
    print("Generating statistical report...")
    print("=" * 80)

    # Print detailed report
    report = cleaner.generate_summary_report()
    print(report)

    # Save statistics to CSV
    cleaner.save_statistics_to_csv('cleaning_statistics_summary.csv')

    print(f"\nProcessing completed! Successfully processed {success_count}/{len(files_to_process)} files")


if __name__ == "__main__":
    main()
