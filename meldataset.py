#coding: utf-8
import os
import os.path as osp
import re
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from utils import get_data_path_list

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}


def preprocess(wave, sr, hop, win, nfft):
    wave_tensor = torch.from_numpy(wave).float()
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=nfft, win_length=win, hop_length=hop, sample_rate=sr
    )
    mean, std = -4, 4
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 hop=300,
                 win=1200,
                 nfft=2048,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.hop = hop
        self.win = win
        self.nfft = nfft

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave, self.sr, self.hop, self.win, self.nfft).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # get OOD text
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)
        
        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave, self.sr, self.hop, self.win, self.nfft).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    
    # Verify all files exist
    missing_files = []
    for i, item in enumerate(dataset.data_list):
        wav_path = item[0].strip() if os.path.isabs(item[0].strip()) else os.path.join(root_path, item[0].strip())
        if not os.path.exists(wav_path):
            missing_files.append(wav_path)
    
    if missing_files:
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"First missing file: {missing_files[0]}")
        print(f"Root path: {root_path}")
        first_item = dataset.data_list[0][0].strip()
        print(f"First item path part: {first_item}")
        print(f"Is first item absolute path? {os.path.isabs(first_item)}")
        print(f"Full constructed path: {first_item if os.path.isabs(first_item) else os.path.join(root_path, first_item)}")
        print(f"Does root path exist? {os.path.exists(root_path) if root_path else 'Root path is empty'}")
        raise FileNotFoundError(f"Found {len(missing_files)} missing files out of {len(dataset)} total files")
    
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader


def create_batched_dataloaders(train_dir,
                               val_path,
                               root_path,
                               OOD_data,
                               min_length,
                               val_batch_size=2,
                               num_workers_train=2,
                               num_workers_val=0,
                               device="cpu",
                               dataset_config=None):
    """
    Create training and validation dataloaders based on a directory of batch files and a validation file.

    Training files are expected to be named in the format:

        wavs_batch[<batch_size>]_<primary_bucket_index>_<sub_bucket_index>.txt

    The <batch_size> extracted from the filename is automatically passed to the dataloader builder.
    If the batch size cannot be extracted, a default batch size of 2 will be used.

    Parameters:
      - train_dir (str): Directory containing training text files.
      - val_path (str): Path to the validation text file.
      - root_path (str): Root directory to prepend to the relative paths in the data lists.
      - OOD_data (Any): Out-of-distribution data specification (passed to build_dataloader).
      - min_length (float): Minimum length requirement (passed to build_dataloader).
      - val_batch_size (int): Batch size to use for the validation dataloader (default: 2).
      - num_workers_train (int): Number of workers for the training dataloaders (default: 2).
      - num_workers_val (int): Number of workers for the validation dataloader (default: 0).
      - device (str or torch.device): Device to use (e.g., "cpu" or "cuda").
      - dataset_config (dict): Optional dictionary with additional dataset configuration.

    Returns:
      - train_dataloaders (list): A list containing a dataloader for each training batch file.
      - val_dataloader: A single validation dataloader.

    Note:
      This function assumes that get_data_path_list() and build_dataloader() are defined elsewhere.
    """

    # Ensure we have a valid (possibly empty) configuration dictionary.
    if dataset_config is None:
        dataset_config = {}

    train_dataloaders = []

    # Loop over each file in the training directory.
    for filename in os.listdir(train_dir):
        # We only care about .txt files that follow the naming convention.
        if filename.endswith(".txt") and "wavs_batch" in filename:
            # Extract the batch size from the filename.
            match = re.search(r"wavs_batch_(\d+)_", filename)
            if match:
                local_batch_size = int(match.group(1))
            else:
                # If no match is found, default to batch size of 2.
                print(
                    f"Warning: Could not determine batch size from filename '{filename}'. Using default batch size 2.")
                local_batch_size = 2

            # Full path to the current training file.
            train_file_path = os.path.join(train_dir, filename)

            # Get the training data list from the file.
            # Here we call get_data_path_list with the training file; the second argument is unused so we pass None.
            train_list, _ = get_data_path_list(train_file_path, None)

            # Build the dataloader for this training file.
            if local_batch_size > 0:
                dataloader = build_dataloader(
                    train_list,
                    root_path,
                    OOD_data=OOD_data,
                    min_length=min_length,
                    batch_size=local_batch_size,
                    num_workers=num_workers_train,
                    dataset_config=dataset_config,
                    device=device
                )

                train_dataloaders.append(dataloader)

    # --- Build the Validation Dataloader ---
    # We assume that when calling get_data_path_list with train_path as None,
    # it returns (None, val_list).
    _, val_list = get_data_path_list(None, val_path)
    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=val_batch_size,
        validation=True,
        num_workers=num_workers_val,
        device=device,
        dataset_config=dataset_config
    )

    return train_dataloaders, val_dataloader
