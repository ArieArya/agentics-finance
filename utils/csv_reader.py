"""
Utility for reading split CSV files as if they were a single file.
Handles the case where merged_data.csv is split into multiple files.
"""

import os
import pandas as pd
import csv
import sys
from typing import List, Optional
from pathlib import Path


def get_merged_data_paths(data_dir: Optional[str] = None) -> List[str]:
    """
    Get paths to merged data files (either single file or split files).

    Args:
        data_dir: Directory containing data files. If None, uses default data directory.

    Returns:
        List of file paths to read (in order)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Check for single merged_data.csv file (backward compatibility)
    single_file = os.path.join(data_dir, "merged_data.csv")
    if os.path.exists(single_file):
        return [single_file]

    # Check for split files (merged_data_part1.csv, merged_data_part2.csv, etc.)
    split_files = []
    for i in range(1, 5):  # Support up to 4 parts
        part_file = os.path.join(data_dir, f"merged_data_part{i}.csv")
        if os.path.exists(part_file):
            split_files.append(part_file)
        else:
            break  # Stop if we find a gap

    if split_files:
        return split_files

    # If neither exists, return empty list
    return []


def read_merged_data_csv(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Read merged data from either a single file or split files.

    Args:
        data_dir: Directory containing data files. If None, uses default data directory.

    Returns:
        pd.DataFrame: Combined dataframe from all parts
    """
    file_paths = get_merged_data_paths(data_dir)

    if not file_paths:
        raise FileNotFoundError(
            "No merged data files found. Expected either merged_data.csv "
            "or merged_data_part1.csv, merged_data_part2.csv, etc."
        )

    # Read all parts and concatenate
    dataframes = []
    header = None

    for i, file_path in enumerate(file_paths):
        if i == 0:
            # First file: read with header to get column names
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            header = list(df.columns)
        else:
            # Subsequent files: skip header row and use column names from first file
            df = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=1, names=header)
        dataframes.append(df)

    # Concatenate all parts
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def read_merged_data_header(data_dir: Optional[str] = None) -> List[str]:
    """
    Read just the header row from merged data files.

    Args:
        data_dir: Directory containing data files. If None, uses default data directory.

    Returns:
        List of column names
    """
    file_paths = get_merged_data_paths(data_dir)

    if not file_paths:
        return []

    # Read header from first file (all should have same header)
    try:
        # Increase CSV field size limit
        try:
            csv.field_size_limit(10 * 1024 * 1024)  # 10MB
        except OverflowError:
            csv.field_size_limit(sys.maxsize)

        with open(file_paths[0], 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            return header
    except Exception:
        return []


def get_merged_data_file_for_agentics(data_dir: Optional[str] = None) -> str:
    """
    Get a file path for Agentics AG.from_csv().
    If files are split, creates a temporary combined file.

    Args:
        data_dir: Directory containing data files. If None, uses default data directory.

    Returns:
        Path to a single CSV file (either original or temporary combined file)
    """
    file_paths = get_merged_data_paths(data_dir)

    if not file_paths:
        raise FileNotFoundError(
            "No merged data files found. Expected either merged_data.csv "
            "or merged_data_part1.csv, merged_data_part2.csv, etc."
        )

    # If single file, return it directly
    if len(file_paths) == 1:
        return file_paths[0]

    # If split files, create a temporary combined file
    # This is needed because AG.from_csv() expects a single file
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    temp_file = os.path.join(data_dir, "merged_data_combined_temp.csv")

    # Check if temp file exists and is newer than all parts
    if os.path.exists(temp_file):
        temp_mtime = os.path.getmtime(temp_file)
        all_newer = all(os.path.getmtime(f) < temp_mtime for f in file_paths)
        if all_newer:
            return temp_file

    # Create combined file
    print(f"📦 Combining {len(file_paths)} split files into temporary combined file...")
    combined_df = read_merged_data_csv(data_dir)
    combined_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    print(f"✅ Created temporary combined file: {temp_file}")

    return temp_file

