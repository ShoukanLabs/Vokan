print("NLTK")
import nltk
nltk.download('punkt')
print("SCIPY")
from scipy.io.wavfile import write
print("TORCH STUFF")
import torch
print("START")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# print(torch.cuda.device_count())
import IPython.display as ipd
import os
os.environ['CUDA_HOME'] = '/home/ubuntu/miniconda3/envs/respair/lib/python3.11/site-packages/torch/lib/include/cuda'
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
from text_utils import TextCleaner
textclenaer = TextCleaner()


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask




import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from Modules.KotoDama_sampler import tokenizer_koto_prompt, tokenizer_koto_text
from utils import *

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

from konoha import SentenceTokenizer


sent_tokenizer = SentenceTokenizer()

# %matplotlib inline
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style_through_clip(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def Kotodama_Prompter(model, text, device):
    
    with torch.no_grad():
        style = model.KotoDama_Prompt(**tokenizer_koto_prompt(text, return_tensors="pt").to(device))['logits']
    return style

def Kotodama_Sampler(model, text, device):
    
    with torch.no_grad():
        style = model.KotoDama_Text(**tokenizer_koto_text(text, return_tensors="pt").to(device))['logits']
    return style


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = yaml.safe_load(open("Configs/config_kanade.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)


KotoDama_Prompter = load_KotoDama_Prompter(path="Utils/KTD/prompt_enc/checkpoint-73285")
KotoDama_TextSampler = load_KotoDama_TextSampler(path="Utils/KTD/text_enc/checkpoint-22680")

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert, KotoDama_Prompter, KotoDama_TextSampler)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth", map_location='cpu')
params = params_whole['net']


for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])


_ = [model[key].eval() for key in model]


from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
diffusion_sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(text=None, ref_s=None, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, rate_of_speech=1.):

    tokens = textclenaer(text)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)

        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
        


        s_pred = diffusion_sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)
        
  

        x = model.predictor.lstm(d)
        x_mod =  model.predictor.prepare_projection(x) 
        duration = model.predictor.duration_proj(x_mod) 


        duration = torch.sigmoid(duration).sum(axis=-1) / rate_of_speech
        
        pred_dur = torch.round(duration.squeeze()).clamp(min=1) 



        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))



        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))


        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-50] 


def Longform(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1, rate_of_speech=1.0):
    

    tokens = textclenaer(text)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = diffusion_sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, 
                                             num_steps=diffusion_steps).squeeze(1)
        
        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        
        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

        x = model.predictor.lstm(d)
        x_mod =  model.predictor.prepare_projection(x) # 640 -> 512
        duration = model.predictor.duration_proj(x_mod)

        duration = torch.sigmoid(duration).sum(axis=-1) / rate_of_speech
        pred_dur = torch.round(duration.squeeze()).clamp(min=1) 


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-100], s_pred 



def trim_long_silences(wav_data, sample_rate=24000, silence_threshold=0.01, min_silence_duration=0.8):

    
    min_silence_samples = int(min_silence_duration * sample_rate)
    
    
    envelope = np.abs(wav_data)
    
    
    silence_mask = envelope < silence_threshold
    
    
    silence_changes = np.diff(silence_mask.astype(int))
    silence_starts = np.where(silence_changes == 1)[0] + 1
    silence_ends = np.where(silence_changes == -1)[0] + 1
    
    
    if silence_mask[0]:
        silence_starts = np.concatenate(([0], silence_starts))
    if silence_mask[-1]:
        silence_ends = np.concatenate((silence_ends, [len(wav_data)]))
    
    
    if len(silence_starts) == 0 or len(silence_ends) == 0:
        return wav_data
    
    processed_segments = []
    last_end = 0
    
    for start, end in zip(silence_starts, silence_ends):
        
        processed_segments.append(wav_data[last_end:start])
        
        
        silence_duration = end - start
        
        if silence_duration > min_silence_samples:
            
            silence_segment = np.zeros(min_silence_samples)
          
            fade_samples = min(1000, min_silence_samples // 4)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            silence_segment[:fade_samples] *= fade_in
            silence_segment[-fade_samples:] *= fade_out
            processed_segments.append(silence_segment)
        else:
           
            processed_segments.append(wav_data[start:end])
        
        last_end = end
    

    if last_end < len(wav_data):
        processed_segments.append(wav_data[last_end:])
    
  
    return np.concatenate(processed_segments)


def merge_short_elements(lst):
    i = 0
    while i < len(lst):
        if i > 0 and len(lst[i]) < 10:
            lst[i-1] += ' ' + lst[i]
            lst.pop(i)
        else:
            i += 1
    return lst


def merge_three(text_list, maxim=2):

    merged_list = []
    for i in range(0, len(text_list), maxim):
        merged_text = ' '.join(text_list[i:i+maxim])
        merged_list.append(merged_text)
    return merged_list


def merging_sentences(lst):
    return merge_three(merge_short_elements(lst))


import os


from openai import OpenAI


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "Respair/Japanese_Phoneme_to_Grapheme_LLM"


def p2g(param):

    chat_response = client.chat.completions.create(

        model=model_name,
        max_tokens=512,
        temperature=0.1,


        messages=[
            
            {"role": "user", "content": f"convert this pronunciation back to normal japanese if you see one, otherwise copy the same thing: {param}"}]
    )   
    
    result = chat_response.choices[0].message.content
    # if " 　" in result:
    #     result = result.replace(" 　"," ")

    return result.lstrip()