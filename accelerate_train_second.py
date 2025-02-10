# load packages
import random

import accelerate
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings

warnings.simplefilter('ignore')
# from autoclip.torch import QuantileClip
from meldataset import create_batched_dataloaders

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm, ProjectConfiguration

try:
    import wandb
except ImportError:
    wandb = None

# from Utils.fsdp_patch import replace_fsdp_state_dict_type

# replace_fsdp_state_dict_type()

import logging

from accelerate.logging import get_logger
from logging import StreamHandler

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    save_freq = config.get('save_freq', 2)
    save_iter = 10000
    log_interval = 10
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    hop = config['preprocess_params']["spect_params"].get('hop_length', 300)
    win = config['preprocess_params']["spect_params"].get('win_length', 1200)
    nfft = config['preprocess_params']["spect_params"].get('nfft', 2048)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)

    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch

    optimizer_params = Munch(config['optimizer_params'])

    tracker = config.get("tracker", "tensorboard")

    try:
        tracker = config.get("tracker", "tensorboard")
    except KeyError:
        tracker = "mlflow"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    configAcc = ProjectConfiguration(project_dir=log_dir, logging_dir=log_dir)
    accelerator = Accelerator(log_with=tracker,
                              project_config=configAcc,
                              split_batches=True,
                              kwargs_handlers=[ddp_kwargs],
                              )

    accelerator.init_trackers(project_name="Vokan-Second-Stage",
                              config=config if tracker == "wandb" else None)

    train_list, val_list = train_path, val_path

    device = accelerator.device

    train_dataloaders, val_dataloader = create_batched_dataloaders(
        train_dir=train_path,
        val_path=val_path,
        root_path=root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        val_batch_size=batch_size,
        num_workers_train=0,
        num_workers_val=0,
        device=device,
        dataset_config={"sr": sr, "hop": hop, "win": win, "nfft": nfft}
    )

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # build model
    config['model_params']["sr"] = sr

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model_params["sr"] = sr
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # # # DP
    # for key in model:
    #     if key != "mpd" and key != "msd" and key != "wd":
    #         model[key] = accelerator.prepare(model[key])

    # for k in model:
    #     model[k] = nn.SyncBatchNorm.convert_sync_batchnorm(model[k])

    for k in model:
        model[k] = accelerator.prepare(model[k])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)

    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            accelerator.print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model,
                                                           None,
                                                           first_stage_path,
                                                           load_only_params=True,
                                                           ignore_modules=['bert', 'bert_encoder', 'predictor',
                                                                           'predictor_encoder', 'msd', 'mpd', 'wd',
                                                                           'diffusion'])  # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            model.style_encoder.train()
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model,
                   model.wd,
                   sr,
                   model_params.slm.sr).to(device)

    gl = accelerator.prepare(gl)
    dl = accelerator.prepare(dl)
    wl = accelerator.prepare(wl)
    wl = wl.eval()

    sampler = DiffusionSampler(
        model.diffusion.module.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
        clamp=False
    )

    scheduler_params = {
        "max_lr": optimizer_params.lr * accelerator.num_processes,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": sum([len(dataloader) for dataloader in train_dataloaders]),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=scheduler_params_dict,
                                lr=optimizer_params.lr * accelerator.num_processes)

    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4

    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                                                               load_only_params=config.get('load_only_params', True))

    n_down = model.text_aligner.module.n_down

    # for k in model:
    #     model[k] = accelerator.prepare(model[k])

    best_loss = float('inf')  # best test loss
    iters = 0

    criterion = nn.L1Loss()  # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss(sr=sr).to(device)

    accelerator.print('BERT', optimizer.optimizers['bert'])
    accelerator.print('decoder', optimizer.optimizers['decoder'])

    start_ds = False

    running_std = []

    slmadv_params = Munch(config['slmadv_params'])

    slmadv = SLMAdversarialLoss(model, wl, sampler,
                                slmadv_params.min_len,
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter,
                                sig=slmadv_params.sig
                                )

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    val_dataloader = accelerator.prepare(val_dataloader)

    for idx, i in enumerate(train_dataloaders):
        train_dataloaders[idx] = accelerator.prepare(i)

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.text_aligner.train()
        model.text_encoder.train()

        model.predictor.train()
        model.predictor_encoder.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()
        model.wd.train()

        if epoch >= diff_epoch:
            start_ds = True

        log_step = 0

        for _ in range(len(train_dataloaders)):
            with accelerator.main_process_first():
                random_choice = random.randrange(len(train_dataloaders)) if accelerator.is_main_process() else 0

                number_tensor = torch.tensor(random_choice).to(device)
                number_tensor = accelerate.utils.broadcast(number_tensor, 0)

                # Convert the broadcasted tensor back to a Python int.
                random_choice = int(number_tensor.item())

            for i, batch in enumerate(train_dataloaders[random_choice]):
                if accelerator.is_main_process():
                    log_step += len(batch[0])
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    mel_mask = length_to_mask(mel_input_length).to(device)
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    try:
                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)
                    except:
                        continue

                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                    # encode
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    asr = (t_en @ s2s_attn_mono)

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # compute reference styles
                    if multispeaker and epoch >= diff_epoch:
                        ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                        ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref = torch.cat([ref_ss, ref_sp], dim=1)

                # compute the style of the entire utterance
                # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
                ss = []
                gs = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item())
                    mel = mels[bib, :, :mel_input_length[bib]]
                    s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                    ss.append(s)
                    s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                    gs.append(s)

                s_dur = torch.stack(ss).squeeze(1)  # global prosodic styles
                gs = torch.stack(gs).squeeze(1)  # global acoustic styles
                s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser

                bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                # denoiser training
                if epoch >= diff_epoch:
                    num_steps = np.random.randint(3, 5)

                    if model_params.diffusion.dist.estimate_sigma_data:
                        model.diffusion.module.diffusion.sigma_data = s_trg.std(
                            axis=-1).mean().item()  # batch-wise std estimation
                        running_std.append(model.diffusion.module.diffusion.sigma_data)

                    if multispeaker:
                        s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=1,
                                          features=ref,  # reference from the same speaker as the embedding
                                          embedding_mask_proba=0.1,
                                          num_steps=num_steps).squeeze(1)
                        loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur,
                                                    features=ref).mean()  # EDM loss
                        loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
                    else:
                        s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=1,
                                          embedding_mask_proba=0.1,
                                          num_steps=num_steps).squeeze(1)
                        loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1),
                                                                     embedding=bert_dur).mean()  # EDM loss
                        loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
                    # print(loss_sty)
                else:
                    # print("here")
                    loss_sty = 0
                    loss_diff = 0

                d, p = model.predictor(d_en, s_dur,
                                       input_lengths,
                                       s2s_attn_mono,
                                       text_mask)

                # mel_len = int(mel_input_length.min().item() / 2 - 1)

                mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
                mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])

                mel_len_st = int(mel_input_length.min().item() / 2 - 1)
                en = []
                gt = []
                st = []
                p_en = []
                wav = []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start + mel_len])
                    p_en.append(p[bib, :, random_start:random_start + mel_len])
                    gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

                    y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                    wav.append(torch.from_numpy(y).to(device))

                    # style reference (better to be different from the GT)
                    random_start = np.random.randint(0, mel_length - mel_len_st)
                    st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])

                wav = torch.stack(wav).float().detach()

                en = torch.stack(en)
                p_en = torch.stack(p_en)
                gt = torch.stack(gt).detach()
                st = torch.stack(st).detach()

                if gt.size(-1) < 80:
                    continue

                s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
                s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

                with torch.no_grad():
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2])

                    asr_real = model.text_aligner.module.get_feature(gt)

                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec_gt = wav.unsqueeze(1)
                    y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

                    if epoch >= joint_epoch:
                        # ground truth from recording
                        wav = y_rec_gt  # use recording since decoder is tuned
                    else:
                        # ground truth from reconstruction
                        wav = y_rec_gt_pred  # use reconstruction since decoder is fixed

                F0_fake, N_fake = model.predictor(texts=p_en, style=s_dur, f0=True)

                y_rec = model.decoder(en, F0_fake, N_fake, s)

                loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
                loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

                if start_ds:
                    optimizer.zero_grad()
                    d_loss = dl(wav.detach(), y_rec.detach()).mean()
                    accelerator.backward(d_loss)
                    accelerator.clip_grad_norm_(model.msd.parameters(), max_norm=1.0)
                    accelerator.clip_grad_norm_(model.mpd.parameters(), max_norm=1.0)
                    optimizer.step('msd')
                    optimizer.step('mpd')
                else:
                    d_loss = 0

                # generator loss
                optimizer.zero_grad()

                loss_mel = stft_loss(y_rec, wav)
                if start_ds:
                    loss_gen_all = gl(wav, y_rec).mean()
                else:
                    loss_gen_all = 0
                loss_lm = wl(wav.detach().squeeze(1), y_rec.squeeze(1)).mean()

                loss_ce = 0
                loss_dur = 0
                for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                    _s2s_pred = _s2s_pred[:_text_length, :]
                    _text_input = _text_input[:_text_length].long()
                    _s2s_trg = torch.zeros_like(_s2s_pred)
                    for p in range(_s2s_trg.shape[0]):
                        _s2s_trg[p, :_text_input[p]] = 1
                    _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                    loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1],
                                          _text_input[1:_text_length - 1])
                    loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

                loss_ce /= texts.size(0)
                loss_dur /= texts.size(0)

                g_loss = loss_params.lambda_mel * loss_mel + \
                         loss_params.lambda_F0 * loss_F0_rec + \
                         loss_params.lambda_ce * loss_ce + \
                         loss_params.lambda_norm * loss_norm_rec + \
                         loss_params.lambda_dur * loss_dur + \
                         loss_params.lambda_gen * loss_gen_all + \
                         loss_params.lambda_slm * loss_lm + \
                         loss_params.lambda_sty * loss_sty + \
                         loss_params.lambda_diff * loss_diff

                running_loss += accelerator.gather(loss_mel).mean().item()
                accelerator.backward(g_loss)

                # clipper_bert_enc = QuantileClip(model.bert_encoder.parameters(), quantile=0.9, history_length=1000) # Adaptive clipping of gradient
                # clipper_bert = QuantileClip(model.bert.parameters(), quantile=0.9, history_length=1000)
                # clipper_pred = QuantileClip(model.predictor.parameters(), quantile=0.9, history_length=1000)
                # clipper_pred_enc = QuantileClip(model.predictor_encoder.parameters(), quantile=0.9, history_length=1000)

                # accelerator.clip_grad_norm_(model.bert_encoder.parameters(), max_norm=2.0)
                # accelerator.clip_grad_norm_(model.bert.parameters(), max_norm=2.0)
                # accelerator.clip_grad_norm_(model.predictor.parameters(), max_norm=2.0)
                # accelerator.clip_grad_norm_(model.predictor_encoder.parameters(), max_norm=2.0)

                # if iters % 10 == 0:  # Monitor every 10 steps
                #     components = ['bert_encoder', 'bert', 'predictor', 'predictor_encoder']
                #     if epoch >= diff_epoch:
                #         components.append('diffusion')

                #     for key in components:
                #         if key in model:
                #             grad_norm = accelerator.clip_grad_norm_(model[key].parameters(), float('inf'))
                #             accelerator.print(f"key: {key} grad norm: {grad_norm:.4f}")

                # if torch.isnan(g_loss):
                #     from IPython.core.debugger import set_trace
                #     set_trace()

                # clipper_bert_enc.step()
                # clipper_bert.step()
                # clipper_pred.step()
                # clipper_pred_enc.step()

                # Apply gradient clipping
                for key in ['bert_encoder', 'bert', 'predictor', 'predictor_encoder']:
                    if key in model:
                        accelerator.clip_grad_norm_(model[key].parameters(), max_norm=1.0)
                    else:
                        raise ValueError(f"Grad norm: key {key} not in model")

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('predictor_encoder')

                if epoch >= diff_epoch:
                    accelerator.clip_grad_norm_(model.diffusion.parameters(), max_norm=1.0)
                    optimizer.step('diffusion')

                if epoch >= joint_epoch:

                    accelerator.clip_grad_norm_(model.style_encoder.parameters(), max_norm=1.0)
                    accelerator.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)

                    optimizer.step('style_encoder')
                    optimizer.step('decoder')

                    d_loss_slm, loss_gen_lm = 0, 0

                    # # randomly pick whether to use in-distribution text
                    # if np.random.rand() < 0.5:
                    #     use_ind = True
                    # else:
                    #     use_ind = False

                    # if use_ind:
                    #     ref_lengths = input_lengths
                    #     ref_texts = texts

                    # slm_out = slmadv(i,
                    #                  y_rec_gt,
                    #                  y_rec_gt_pred,
                    #                  waves,
                    #                  mel_input_length,
                    #                  ref_texts,
                    #                  ref_lengths, use_ind, s_trg.detach(), ref if multispeaker else None)

                    # if slm_out is None:
                    #     continue
                    # # if slm_out is not None:
                    # #     d_loss_slm, loss_gen_lm, y_pred = slm_out
                    # #     optimizer.zero_grad()
                    # #     # accelerator.clip_grad_norm_(model.decoder.parameters(), 1)
                    # #     # print("here")
                    # #     accelerator.backward(loss_gen_lm)
                    # #     # print("here2")
                    # #     # SLM discriminator loss

                    # # # compute the gradient norm

                    # # total_norm = {}
                    # # for key in model.keys():
                    # #     total_norm[key] = 0
                    # #     parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                    # #     for p in parameters:
                    # #         param_norm = p.grad.detach().data.norm(2)
                    # #         total_norm[key] += param_norm.item() ** 2
                    # #     total_norm[key] = total_norm[key] ** 0.5

                    # # # gradient scaling
                    # # if total_norm['predictor'] > slmadv_params.thresh:
                    # #     for key in model.keys():
                    # #         for p in model[key].parameters():
                    # #             if p.grad is not None:
                    # #                 p.grad *= (1 / total_norm['predictor'])

                    # # for p in model.predictor.module.duration_proj.parameters():
                    # #     if p.grad is not None:
                    # #         p.grad *= slmadv_params.scale

                    # # for p in model.predictor.module.lstm.parameters():
                    # #     if p.grad is not None:
                    # #         p.grad *= slmadv_params.scale

                    # # for p in model.diffusion.module.parameters():
                    # #     if p.grad is not None:
                    # #         p.grad *= slmadv_params.scale

                    # # optimizer.step('bert_encoder')
                    # # optimizer.step('bert')
                    # # optimizer.step('predictor')
                    # # optimizer.step('diffusion')

                    # # # SLM discriminator loss
                    # # if d_loss_slm != 0:
                    # #     optimizer.zero_grad()
                    # #     # print("hey1")
                    # #     accelerator.backward(d_loss_slm, retain_graph=True)
                    # #     optimizer.step('wd')
                    # #     # print("hey2")

                else:
                    d_loss_slm, loss_gen_lm = 0, 0

                iters = iters + 1
                if (i + 1) % log_interval == 0:
                    logger.info(
                        'Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                        % (epoch + 1, epochs, log_step + 1, len(train_list) // batch_size, running_loss / log_interval, d_loss,
                           loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty, loss_diff,
                           d_loss_slm, loss_gen_lm), main_process_only=True)
                    if accelerator.is_main_process:
                        print(
                            'Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f'
                            % (epoch + 1, epochs, log_step + 1, len(train_list) // batch_size, running_loss / log_interval,
                               d_loss,
                               loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_lm, loss_gen_all, loss_sty,
                               loss_diff,
                               d_loss_slm, loss_gen_lm))
                    accelerator.log({'train/mel_loss': float(running_loss / log_interval),
                                     'train/gen_loss': float(loss_gen_all),
                                     'train/d_loss': float(d_loss),
                                     'train/ce_loss': float(loss_ce),
                                     'train/dur_loss': float(loss_dur),
                                     'train/slm_loss': float(loss_lm),
                                     'train/norm_loss': float(loss_norm_rec),
                                     'train/F0_loss': float(loss_F0_rec),
                                     'train/sty_loss': float(loss_sty),
                                     'train/diff_loss': float(loss_diff),
                                     'train/d_loss_slm': float(d_loss_slm),
                                     'train/gen_loss_slm': float(loss_gen_lm),
                                     'epoch': int(epoch) + 1}, step=iters)

                    running_loss = 0

                    accelerator.print('Time elasped:', time.time() - start_time)

        loss_test = 0
        loss_align = 0
        loss_f = 0

        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        # print("t_en", t_en.shape, t_en)
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()
                    # print("texts", texts.shape, texts)
                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s,
                                           input_lengths,
                                           s2s_attn_mono,
                                           text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start + mel_len])
                        p_en.append(p[bib, :, random_start:random_start + mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

                        y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor(texts=p_en, style=s, f0=True)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1],
                                              _text_input[1:_text_length - 1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(1), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += accelerator.gather(loss_mel).mean()
                    loss_align += accelerator.gather(loss_dur).mean()
                    loss_f += accelerator.gather(loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    accelerator.print(f"Eval errored with: \n {str(e)}")
                    continue

        accelerator.print('Epochs:', epoch + 1)
        try:
            logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (
                loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n', main_process_only=True)

            accelerator.log({'eval/mel_loss': float(loss_test / iters_test),
                             'eval/dur_loss': float(loss_test / iters_test),
                             'eval/F0_loss': float(loss_f / iters_test)},
                            step=(i + 1) * (epoch + 1))
        except ZeroDivisionError:
            accelerator.print("Eval loss was divided by zero... skipping eval cycle")

        if epoch < diff_epoch:
            # generating reconstruction examples with GT duration

            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    gt = mels[bib, :, :mel_length].unsqueeze(0)
                    en = asr[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    F0_real = F0_real.unsqueeze(0)
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    try:
                        y_rec = model.decoder(en, F0_real.squeeze(0), real_norm, s)
                    except Exception as e:
                        accelerator.print(str(e))
                        accelerator.print(F0_real.size())
                        accelerator.print(F0_real.squeeze(0).size())

                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    p_en = p[bib, :, :mel_length // 2].unsqueeze(0)
                    F0_fake, N_fake = model.predictor(texts=p_en, style=s_dur, f0=True)

                    y_pred = model.decoder(en, F0_fake, N_fake, s)

                    if accelerator.is_main_process:
                        audio_buffer = audio_to_wav_buffer(y_pred.detach().cpu().numpy().squeeze(), sr)
                        accelerator.log({f"pred/y{bib}": audio_buffer}, step=epoch)

                        # Log ground truth audio only for the first epoch
                    if epoch == 0 and accelerator.is_main_process:
                        gt_audio_buffer = audio_to_wav_buffer(waves[bib].squeeze(), sr)
                        accelerator.log({f"gt/y{bib}": gt_audio_buffer}, step=epoch)

                    if bib >= 10:
                        break
        else:

            try:
                # generating sampled speech from text directly
                with torch.no_grad():
                    # compute reference styles
                    if multispeaker and epoch >= diff_epoch:
                        ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                        ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref_s = torch.cat([ref_ss, ref_sp], dim=1)

                    for bib in range(len(d_en)):
                        if multispeaker:
                            s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                                             embedding=bert_dur[bib].unsqueeze(0),
                                             embedding_scale=1,
                                             features=ref_s[bib].unsqueeze(0),
                                             # reference from the same speaker as the embedding
                                             num_steps=5).squeeze(1)
                        else:
                            s_pred = sampler(noise=torch.ones((1, 1, 256)).to(texts.device) * 0.5,
                                             embedding=bert_dur[bib].unsqueeze(0),
                                             embedding_scale=1,
                                             num_steps=5).squeeze(1)

                        s = s_pred[:, 128:]
                        ref = s_pred[:, :128]
                        # print(model.predictor)
                        # print(d_en[bib, :, :input_lengths[bib]])
                        d = model.predictor.module.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0),
                                                                s, input_lengths[bib, ...].unsqueeze(0),
                                                                text_mask[bib, :input_lengths[bib]].unsqueeze(0))

                        x = model.predictor.module.lstm(d)
                        x_mod = model.predictor.module.prepare_projection(x)  # 640 -> 512
                        duration = model.predictor.module.duration_proj(x_mod)

                        duration = torch.sigmoid(duration).sum(axis=-1)
                        pred_dur = torch.round(duration.squeeze(0)).clamp(min=1)

                        pred_dur[-1] += 5

                        pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                        c_frame = 0
                        for i in range(pred_aln_trg.size(0)):
                            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                            c_frame += int(pred_dur[i].data)

                        # encode prosody
                        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device))
                        F0_pred, N_pred = model.predictor(texts=en, style=s, f0=True)
                        out = model.decoder(
                            (t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(
                                texts.device)),
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

                        # writer.add_audio('pred/y' + str(bib), out.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                        if accelerator.is_main_process:
                            audio_buffer = audio_to_wav_buffer(out.detach().cpu().numpy().squeeze(), sr)
                            accelerator.log({f"pred/y{bib}": audio_buffer}, step=epoch)

                        if bib >= 5:
                            break
            except Exception as e:
                accelerator.print('error ->   ', e)
                accelerator.print("some of the samples couldn't be evaluated, skipping those.")

        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            try:
                accelerator.print('Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
            except ZeroDivisionError:
                accelerator.print('No iter test, Re-Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': 0.1,  # not zero just in case
                    'epoch': epoch,
                }

            if accelerator.is_main_process:
                save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
                torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)
        if accelerator.is_main_process:
            print('Saving last pth..')
            state = {
                'net': {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, '2nd_phase_last.pth')
            torch.save(state, save_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
