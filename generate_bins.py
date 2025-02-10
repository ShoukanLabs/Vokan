import os
import librosa

# === Configuration Parameters - (CALIBRATED FOR 48GB of VRAM) ===
batch_size = 9  # Number of primary buckets.
max_seconds = 16  # Maximum duration (seconds) for threshold calculations.
duration_bin_step = 0.2  # Allowed duration window (in seconds) within each sub-bucket.

# Specify the output directory where the batch text files will be written.
output_dir = "data_paths/"  # Change this to your desired directory path.
os.makedirs(output_dir, exist_ok=True)

# === Primary Bucketing Based on FLOORS ===
# Compute thresholds ("FLOORS") for each primary bucket.
FLOORS = [max_seconds / (i + 2) for i in range(batch_size)]
FLOORS[-1] = 0  # Ensure the last bucket catches very short files.

# Create an empty list for each primary bucket.
grouped_entries = [[] for _ in FLOORS]

# Path to the input text file.
wavs_txt_path = "data_paths/train_data_filtered.txt"

# Read and process the entire contents of wavs.txt.
# The file is expected to have entries separated by "~" where each entry is:
#    relative_path_to_file|phoneme_transcript|speakerID
try:
    with open(wavs_txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
except Exception as e:
    print(f"Error reading {wavs_txt_path}: {e}")
    exit(1)

# Split the content into individual entries using "~" as the delimiter.
entries = content.split("\n")

# Process each entry.
for entry in entries:
    entry = entry.strip()
    if not entry:
        continue

    parts = entry.split("|")
    if len(parts) < 3:
        print(f"Skipping invalid entry: {entry}")
        continue

    file_path = parts[0].strip()  # Relative path to the audio file.
    transcript = parts[1].strip()  # Phoneme transcript.
    speaker_id = parts[2].strip()  # Speaker ID.

    full_path = os.path.join(os.path.dirname(wavs_txt_path), file_path)

    # Verify that the audio file exists.
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
        continue

    # Compute the duration (in seconds) using librosa.
    try:
        duration = librosa.get_duration(filename=full_path)
    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        continue

    # Assign this entry to the first bucket whose floor is <= duration.
    try:
        bucket_index = next(i for i, floor in enumerate(FLOORS) if duration >= floor)
    except StopIteration:
        # Fallback: assign to the last bucket.
        bucket_index = batch_size - 1

    grouped_entries[bucket_index].append({
        "file_path": file_path,
        "transcript": transcript,
        "speaker_id": speaker_id,
        "duration": duration
    })

# === Second Pass: Sub-Bucketing Within Each Primary Bucket ===
# For each primary bucket, further split the entries into sub-buckets so that every
# entry in a sub-bucket is within `duration_bin_step` seconds of the shortest entry.
sub_grouped_entries = []  # This will hold lists of sub-buckets per primary bucket.

for bucket in grouped_entries:
    # Sort entries by duration (ascending order).
    bucket_sorted = sorted(bucket, key=lambda x: x["duration"])
    sub_buckets = []  # List to hold sub-buckets for this primary bucket.
    current_sub_bucket = []

    if bucket_sorted:
        # Initialize the current sub-bucket using the shortest entry's duration.
        current_sub_bucket_min = bucket_sorted[0]["duration"]
        for entry in bucket_sorted:
            if entry["duration"] - current_sub_bucket_min <= duration_bin_step:
                current_sub_bucket.append(entry)
            else:
                # Current entry is too far from the minimum;
                # finish this sub-bucket and start a new one.
                sub_buckets.append(current_sub_bucket)
                current_sub_bucket = [entry]
                current_sub_bucket_min = entry["duration"]
        # Append any remaining entries.
        if current_sub_bucket:
            sub_buckets.append(current_sub_bucket)
    sub_grouped_entries.append(sub_buckets)

# === Exporting Each Sub-Bucket as a Text File ===
# Each file will contain lines in the format:
#    relative_path_to_wav_from_.txt|phonemes|speakerID
file_counter = 0

for primary_index, sub_buckets in enumerate(sub_grouped_entries):
    for sub_bucket_index, sub_bucket in enumerate(sub_buckets):
        # Determine the number of entries in this sub-bucket.
        count = len(sub_bucket)
        # Create the file name using the desired format.
        file_name = f"wavs_batch[{count}]_{primary_index}_{sub_bucket_index}.txt"
        output_file_path = os.path.join(output_dir, file_name)

        with open(output_file_path, "w", encoding="utf-8") as out_file:
            for entry in sub_bucket:
                # Write each entry in the expected format.
                line = f"{entry['file_path']}|{entry['transcript']}|{entry['speaker_id']}\n"
                out_file.write(line)
        file_counter += 1

print(f"Exported {file_counter} batch files to directory: {output_dir}")
