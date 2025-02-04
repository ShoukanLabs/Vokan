import os
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("parler-tts/libritts_r_filtered", name="clean", split="train.clean.100")

# Print dataset information
print("\nDataset Features:")
print(dataset.features)
print("\nFirst example:")
first_example = dataset[0]
print(first_example)
print("\nPath from first example:")
print(first_example['path'])

# The hash we're looking for (from the actual dataset path)
target_hash = "38ec98f3d30a82fb2d9844979481e0c2be8bf568f3496b79e3b1a07b9808a110"

output_dir = "data"  # Output directory for restored files
os.makedirs(output_dir, exist_ok=True)

# Function to verify paths
def verify_paths(dataset_split):
    print("\nVerifying paths...")
    count_valid = 0
    total_items = len(dataset_split)
    
    for i in tqdm(range(total_items), desc="Checking paths"):
        try:
            item = dataset_split[i]
            if target_hash in item['path']:
                count_valid += 1
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
    
    if count_valid == 0:
        raise ValueError(f"No paths containing the target hash '{target_hash}' were found!")
    
    print(f"\nFound {count_valid} matching paths out of {total_items} items ({(count_valid/total_items)*100:.2f}%)")

# Verify paths in the dataset
verify_paths(dataset)

entries = []

# Process dataset entries
for i in tqdm(range(len(dataset)), desc="Processing entries"):
    example = dataset[i]
    audio_data = example["audio"]
    
    # Get just the filename part from the audio path
    filename = os.path.basename(example['path'])
    # Get the speaker and chapter directories
    speaker_id = example['speaker_id']
    chapter_id = example['chapter_id']
    
    # Create the directory structure
    speaker_dir = os.path.join(output_dir, speaker_id)
    chapter_dir = os.path.join(speaker_dir, chapter_id)
    os.makedirs(chapter_dir, exist_ok=True)
    
    # Create the full path for the audio file
    audio_path = os.path.join(chapter_dir, filename)
    relative_path = os.path.relpath(audio_path, output_dir)
    
    if not os.path.exists(audio_path):
        try:
            # Save audio file as WAV
            sf.write(
                audio_path,
                audio_data["array"],
                audio_data["sampling_rate"],
                subtype="PCM_16"
            )
            #print(f"Saved: {audio_path}")
        except Exception as e:
            print(f"Error writing {audio_path}: {str(e)}")
            continue
    
    entries.append(f"{relative_path}|{example.get('text_normalized', '')}|{example.get('speaker_id', '')}")

print(f"\nProcessed {len(entries)} entries")
print(f"Audio files saved to: {output_dir}")