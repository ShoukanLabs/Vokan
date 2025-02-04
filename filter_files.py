import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path

def filter_files(input_file, max_duration=9.0, min_duration=1.0, max_amplitude=0.95):
    # Create output filename with _filtered suffix
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    # Read input file and filter lines
    filtered_lines = []
    total_lines = sum(1 for _ in open(input_file, 'r'))
    
    # Keep track of why lines were filtered out
    skipped_too_long = 0
    skipped_too_short = 0
    skipped_mutable_token = 0
    skipped_too_loud = 0
    skipped_error = 0
    
    print(f"Processing {input_file}...")
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=total_lines):
            # Skip if line contains MutableToken
            if "MutableToken" in line:
                skipped_mutable_token += 1
                continue
                
            # Get wav path (everything before first |)
            wav_path = line.split('|')[0].strip()
            
            try:
                # Check audio duration and amplitude
                data, samplerate = sf.read(wav_path)
                duration = len(data) / samplerate
                
                # Skip if duration is outside bounds
                if duration > max_duration:
                    skipped_too_long += 1
                    continue
                if duration < min_duration:
                    skipped_too_short += 1
                    continue
                
                # Check amplitude
                max_abs_amplitude = np.max(np.abs(data))
                if max_abs_amplitude > max_amplitude:
                    skipped_too_loud += 1
                    continue
                    
                # If we get here, the line passed all filters
                filtered_lines.append(line)
            except Exception as e:
                print(f"Warning: Could not process {wav_path}: {e}")
                skipped_error += 1
    
    # Write filtered lines to output file
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"\nResults for {input_path.name}:")
    print(f"Original lines: {total_lines}")
    print(f"Filtered lines: {len(filtered_lines)}")
    print(f"Skipped files:")
    print(f"  Too long (>{max_duration}s): {skipped_too_long}")
    print(f"  Too short (<{min_duration}s): {skipped_too_short}")
    print(f"  Too loud (>{max_amplitude:.2f}): {skipped_too_loud}")
    print(f"  Contains MutableToken: {skipped_mutable_token}")
    print(f"  Error processing: {skipped_error}")
    print(f"Output written to: {output_file}")

def main():
    data_dir = Path("data_paths")
    
    # Process all .txt files in the Data directory
    for file in data_dir.glob("*.txt"):
        if not file.name.endswith("_filtered.txt"):  # Skip already filtered files
            filter_files(file)

if __name__ == "__main__":
    main()
