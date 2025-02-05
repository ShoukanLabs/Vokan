import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, WhisperConfig, WhisperPreTrainedModel
import whisper

from transformers.models.whisper.modeling_whisper import WhisperEncoder


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()
    
    


# #####################
# MIXED PRECISION


class WhisperEncoderOnly(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)

    def forward(self, input_features, attention_mask=None):
        return self.encoder(input_features, attention_mask)



class WavLMLoss(torch.nn.Module):
    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
    
        config = WhisperConfig.from_pretrained("Respair/Whisper_Large_v2_Encoder_Block")

        # this will load the full model and keep only the encoder
        full_model = WhisperEncoderOnly.from_pretrained("openai/whisper-large-v2", config=config, device_map='auto',torch_dtype=torch.bfloat16)
        model = WhisperEncoderOnly(config)
        model.encoder.load_state_dict(full_model.encoder.state_dict())
        del full_model

        
        self.wavlm = model.to(torch.bfloat16)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, wav,  y_rec, generator=False, discriminator=False, discriminator_forward=False):
        
        if generator:
            y_rec = y_rec.squeeze(1)
            

            y_rec = whisper.pad_or_trim(y_rec)
            y_rec = whisper.log_mel_spectrogram(y_rec)

            with torch.no_grad():
                y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_df_hat_g = self.wd(y_rec_embeddings.to(torch.float32))
            loss_gen = torch.mean((1-y_df_hat_g)**2)
            
            return loss_gen.to(torch.float32)
        
        elif discriminator:
            
            wav = wav.squeeze(1)
            y_rec = y_rec.squeeze(1)

            wav = whisper.pad_or_trim(wav)
            wav = whisper.log_mel_spectrogram(wav)

            y_rec = whisper.pad_or_trim(y_rec)
            y_rec = whisper.log_mel_spectrogram(y_rec)

            with torch.no_grad():
                wav_embeddings = self.wavlm.encoder(wav.to(torch.bfloat16), output_hidden_states=True).hidden_states
                y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states

                y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
                y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            y_d_rs = self.wd(y_embeddings.to(torch.float32))
            y_d_gs = self.wd(y_rec_embeddings.to(torch.float32))
            
            y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
            
            r_loss = torch.mean((1-y_df_hat_r)**2)
            g_loss = torch.mean((y_df_hat_g)**2)
            
            loss_disc_f = r_loss + g_loss
                            
            return loss_disc_f.mean().to(torch.float32)
        
        
        
        elif discriminator_forward:
            # Squeeze the channel dimension if it's unnecessary
            wav = wav.squeeze(1) # Adjust this line if the channel dimension is not at dim=1


            with torch.no_grad():
                
                wav_16 = self.resample(wav)
                wav_16 = whisper.pad_or_trim(wav_16)
                wav_16 = whisper.log_mel_spectrogram(wav_16)
                
                wav_embeddings = self.wavlm.encoder(wav_16.to(torch.bfloat16) , output_hidden_states=True).hidden_states
                y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            y_d_rs = self.wd(y_embeddings.to(torch.float32))
            
            return y_d_rs
        
        else:
        
            wav = wav.squeeze(1)
            y_rec = y_rec.squeeze(1)

            wav = whisper.pad_or_trim(wav)
            wav = whisper.log_mel_spectrogram(wav)

            y_rec = whisper.pad_or_trim(y_rec)
            y_rec = whisper.log_mel_spectrogram(y_rec)

            with torch.no_grad():
                wav_embeddings = self.wavlm.encoder(wav.to(torch.bfloat16), output_hidden_states=True).hidden_states
        
                y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states


            floss = 0
            for er, eg in zip([e.to(torch.float32) for e in wav_embeddings], [e.to(torch.float32) for e in y_rec_embeddings]):
                floss += torch.mean(torch.abs(er - eg))
            
            return floss.mean()



    def generator(self, y_rec):
        
        y_rec = y_rec.squeeze(1)
        

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec)

        with torch.no_grad():
            y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings.to(torch.float32))
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen.to(torch.float32)

    def discriminator(self, wav, y_rec):
        
        wav = wav.squeeze(1)
        y_rec = y_rec.squeeze(1)

        wav = whisper.pad_or_trim(wav)
        wav = whisper.log_mel_spectrogram(wav)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec)

        with torch.no_grad():
            wav_embeddings = self.wavlm.encoder(wav.to(torch.bfloat16), output_hidden_states=True).hidden_states
            y_rec_embeddings = self.wavlm.encoder(y_rec.to(torch.bfloat16), output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings.to(torch.float32))
        y_d_gs = self.wd(y_rec_embeddings.to(torch.float32))
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean().to(torch.float32)
    
    


    def discriminator_forward(self, wav):
        # Squeeze the channel dimension if it's unnecessary
        wav = wav.squeeze(1) # Adjust this line if the channel dimension is not at dim=1


        with torch.no_grad():
            
            wav_16 = self.resample(wav)
            wav_16 = whisper.pad_or_trim(wav_16)
            wav_16 = whisper.log_mel_spectrogram(wav_16)
            
            wav_embeddings = self.wavlm.encoder(wav_16.to(torch.bfloat16) , output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings.to(torch.float32))
        
        return y_d_rs
    