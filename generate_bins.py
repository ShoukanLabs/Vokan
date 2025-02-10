import os
import librosa
import concurrent.futures
from tqdm import tqdm

# === Configuration Parameters - (CALIBRATED FOR 48GB of VRAM) ===
batch_size = 9  # Number of primary buckets.
max_seconds = 16  # Maximum duration (seconds) for threshold calculations.
duration_bin_step = 0.2  # Allowed duration window (in seconds) within each sub-bucket.

# Specify the output directory where the batch text files will be written.
output_dir = "data_paths/"  # Change this to your desired directory path.
os.makedirs(output_dir, exist_ok=True)

# === Primary Bucketing Based on FLOORS ===
FLOORS = [max_seconds / (i + 2) for i in range(batch_size)]
FLOORS[-1] = 0  # Ensure the last bucket catches very short files
grouped_entries = [[] for _ in FLOORS]

# Path to the input text file
wavs_txt_path = "data_paths/train_data_filtered.txt"


def process_entry(entry):
    """Process a single entry with error handling and duration calculation."""
    try:
        entry = entry.strip()
        if not entry:
            return None

        parts = entry.split("|")
        if len(parts) < 3:
            print(f"Skipping invalid entry: {entry}")
            return None

        file_path = parts[0].strip()
        transcript = parts[1].strip()
        speaker_id = parts[2].strip()
        full_path = os.path.join(os.path.dirname(wavs_txt_path), file_path)

        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            return None

        duration = librosa.get_duration(filename=full_path)
        return {
            "file_path": file_path,
            "transcript": transcript,
            "speaker_id": speaker_id,
            "duration": duration
        }

    except Exception as e:
        print(f"Error processing entry: {e}")
        return None


# Read and process entries with multithreading
try:
    with open(wavs_txt_path, "r", encoding="utf-8") as f:
        entries = f.read().strip().split("\n")
except Exception as e:
    print(f"Error reading {wavs_txt_path}: {e}")
    exit(1)

# Process entries in parallel with progress tracking
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_entry, entry) for entry in entries]
    processed_entries = []

    for future in tqdm(concurrent.futures.as_completed(futures),
                       total=len(futures),
                       desc="Processing audio files"):
        result = future.result()
        if result:
            processed_entries.append(result)

# Distribute processed entries into primary buckets
for entry in processed_entries:
    try:
        bucket_index = next(i for i, floor in enumerate(FLOORS)
                            if entry["duration"] >= floor)
    except StopIteration:
        bucket_index = batch_size - 1
    grouped_entries[bucket_index].append(entry)

# === Sub-Bucketing and Export ===
sub_grouped_entries = []
for bucket in tqdm(grouped_entries, desc="Organizing buckets"):
    bucket_sorted = sorted(bucket, key=lambda x: x["duration"])
    sub_buckets = []
    current_sub_bucket = []

    if bucket_sorted:
        current_sub_bucket_min = bucket_sorted[0]["duration"]
        for entry in bucket_sorted:
            if entry["duration"] - current_sub_bucket_min <= duration_bin_step:
                current_sub_bucket.append(entry)
            else:
                sub_buckets.append(current_sub_bucket)
                current_sub_bucket = [entry]
                current_sub_bucket_min = entry["duration"]
        if current_sub_bucket:
            sub_buckets.append(current_sub_bucket)
    sub_grouped_entries.append(sub_buckets)

# Write output files with progress tracking
file_counter = 0
for primary_idx, sub_buckets in enumerate(tqdm(sub_grouped_entries,
                                               desc="Writing batches")):
    for sub_idx, sub_bucket in enumerate(sub_buckets):
        if not sub_bucket:
            continue

        file_name = f"wavs_batch[{primary_idx}]_{len(sub_bucket)}_{sub_idx}.txt"
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in sub_bucket:
                line = f"{entry['file_path']}|{entry['transcript']}|{entry['speaker_id']}\n"
                f.write(line)
        file_counter += 1

print(f"\nSuccessfully exported {file_counter} batch files to: {output_dir}")