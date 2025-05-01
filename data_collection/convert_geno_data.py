#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_to_npy(input_file, output_file, chunksize=1000, dtype=np.float32):
    """
    Convert a large text file to NumPy format using chunked processing to minimize memory usage.

    Args:
        input_file: Path to the input text file (tab-separated)
        output_file: Path to the output .npy file
        chunksize: Number of rows to process at once
        dtype: NumPy data type for the output array
    """
    print(f"Starting conversion of {input_file} to {output_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False

    # Count total rows (this takes time but is necessary for pre-allocation)
    print("Counting rows in input file...")
    nrows = 0
    with open(input_file) as f:
        for _ in tqdm(f):
            nrows += 1
    print(f"File contains {nrows} rows")

    # Get number of columns from first row
    print("Determining number of columns...")
    with open(input_file) as f:
        first_line = f.readline()
        ncols = len(first_line.strip().split("\t"))
    print(f"File contains {ncols} columns")

    # Calculate expected memory usage
    estimated_size_gb = (nrows * ncols * np.dtype(dtype).itemsize) / (1024**3)
    print(f"Expected output file size: {estimated_size_gb:.2f} GB")

    # Create empty memory-mapped array (pre-allocate on disk)
    print("Creating memory-mapped output file...")
    output_shape = (nrows, ncols)
    data_array = np.lib.format.open_memmap(output_file, mode="w+", dtype=dtype, shape=output_shape)

    # Process in chunks
    print(f"Processing file in chunks of {chunksize} rows...")
    for i in tqdm(range(0, nrows, chunksize)):
        end = min(i + chunksize, nrows)
        # Read chunk from text file
        chunk = pd.read_csv(
            input_file, sep="\t", header=None, skiprows=i, nrows=end - i, dtype=dtype, engine="c"
        )

        # Write chunk to memory-mapped array
        data_array[i:end, :] = chunk.values

        # Free memory
        del chunk

    # Flush to ensure all data is written
    data_array.flush()

    print(f"Conversion complete! Output saved to {output_file}")
    return True


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a large text file to NumPy format.")
    parser.add_argument(
        "--input",
        "-i",
        default="data/merged_geno_data.txt",
        help="Path to the input text file (tab-separated)",
    )
    parser.add_argument(
        "--output", "-o", default="data/merged_geno_data.npy", help="Path to the output .npy file"
    )
    parser.add_argument(
        "--chunksize", "-c", type=int, default=1000, help="Number of rows to process at once"
    )

    args = parser.parse_args()

    print("Starting conversion with the following parameters:")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Chunk size: {args.chunksize} rows")
    print("This process may take a while for large files...")

    convert_to_npy(args.input, args.output, args.chunksize)
