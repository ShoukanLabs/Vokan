from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from text_utils import TextCleaner
textclenaer = TextCleaner()


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask




device = 'cuda' if torch.cuda.is_available() else 'cpu'


# tokenizer_koto_prompt = AutoTokenizer.from_pretrained("google/mt5-small", trust_remote_code=True)
tokenizer_koto_prompt = AutoTokenizer.from_pretrained("ku-nlp/deberta-v3-base-japanese", trust_remote_code=True)
tokenizer_koto_text = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)

class KotoDama_Prompt(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        
        self.backbone = AutoModel.from_config(config)

        self.output = nn.Sequential(nn.Linear(config.hidden_size, 512),
                                    nn.LeakyReLU(0.2), 
                                    nn.Linear(512, config.num_labels))

                                    

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


        sequence_output = outputs.last_hidden_state[:, 0, :]
        outputs = self.output(sequence_output)

        # if labels, then we are training
        loss = None
        if labels is not None:

            loss_fn = nn.MSELoss()
            # labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels)
            
        return {
            "loss": loss,
            "logits": outputs
        }


class KotoDama_Text(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        
        self.backbone = AutoModel.from_config(config)

        self.output = nn.Sequential(nn.Linear(config.hidden_size, 512),
                                    nn.LeakyReLU(0.2), 
                                    nn.Linear(512, config.num_labels))

                                    

    def forward(
        self,
        input_ids,
        attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
        )


        sequence_output = outputs.last_hidden_state[:, 0, :]
        outputs = self.output(sequence_output)

        # if labels, then we are training
        loss = None
        if labels is not None:

            loss_fn = nn.MSELoss()
            # labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels)
            
        return {
            "loss": loss,
            "logits": outputs
        }


def inference(model, diffusion_sampler, text=None, ref_s=None, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, rate_of_speech=1.):

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


def Longform(model, diffusion_sampler, text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1, rate_of_speech=1.0):

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
