INTROTXT = """# 
Repo -> [Hugging Face - 🤗](https://huggingface.co/Respair/Tsukasa_Speech/edit/main/app_tsuka.py)
This space uses Tsukasa (24khz).
**Check the Read me tabs down below.** <br>
Enjoy!
"""
import gradio as gr
import random
import importable
import torch
import os
from Utils.phonemize.mixed_phon import smart_phonemize
import numpy as np
import pickle
import re

def is_japanese(text):
    if not text:  # Handle empty string
        return False
        
    # Define ranges for Japanese characters
    japanese_ranges = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # Kanji
        (0x3000, 0x303F),  # Japanese punctuation and symbols
        (0xFF00, 0xFFEF),  # Full-width characters
    ]
    
    # Define range for Latin alphabets
    latin_alphabet_ranges = [
        (0x0041, 0x005A),  # Uppercase Latin
        (0x0061, 0x007A),  # Lowercase Latin
    ]
    
    # Define symbols to skip
    symbols_to_skip = {'\'', '*', '!', '?', ',', '.', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}', '"'}
    
    for char in text:
        if char.isspace() or char in symbols_to_skip:  # Skip spaces and specified symbols
            continue
            
        char_code = ord(char)
        
        # Check if the character is a Latin alphabet
        is_latin_char = False
        for start, end in latin_alphabet_ranges:
            if start <= char_code <= end:
                is_latin_char = True
                break
                
        if is_latin_char:
            return False  # Return False if a Latin alphabet character is found
            
        # Check if the character is a Japanese character
        is_japanese_char = False
        for start, end in japanese_ranges:
            if start <= char_code <= end:
                is_japanese_char = True
                break
                
        if not is_japanese_char:
            return False
            
    return True

  
  
voices = {}
example_texts = {}
prompts = []
inputs = []


theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

voicelist = [v for v in os.listdir("/home/ubuntu/Kanade_Project/gradio/Tsukasa_Speech/reference_sample_wavs")]



for v in voicelist:
    voices[v] = f'reference_sample_wavs/{v}'
    

with open(f'Inference/random_texts.txt', 'r') as r:
    random_texts = [line.strip() for line in r]

    example_texts = {f"{text[:30]}...": text for text in random_texts}
    
def update_text_input(preview):

    return example_texts[preview]

def get_random_text():
    return random.choice(random_texts)



with open('Inference/prompt.txt', 'r') as p:
    prompts = [line.strip() for line in p]
    
with open('Inference/input_for_prompt.txt', 'r') as i:
    inputs = [line.strip() for line in i]


last_idx = None

def get_random_prompt_pair():
    global last_idx
    max_idx = min(len(prompts), len(inputs)) - 1
    

    random_idx = random.randint(0, max_idx)
    while random_idx == last_idx:
        random_idx = random.randint(0, max_idx)
    
    last_idx = random_idx
    return inputs[random_idx], prompts[random_idx]

def Synthesize_Audio(text, voice, voice2, vcsteps, embscale, alpha, beta, ros, progress=gr.Progress()):

    
    text = smart_phonemize(text)
    
    
    if voice2 is not None:
        voice2 = {"path": voice2, "meta": {"_type": "gradio.FileData"}} 
        print(voice2)
        voice_style = importable.compute_style_through_clip(voice2['path'])
        
    else:  
        voice_style = importable.compute_style_through_clip(voices[voice])
    
    wav = importable.inference(
        text, 
        voice_style,
        alpha=alpha, 
        beta=beta, 
        diffusion_steps=vcsteps, 
        embedding_scale=embscale, 
        rate_of_speech=ros
    )

    return (24000, wav)

    
def LongformSynth_Text(text, s_prev=None, Kotodama=None, alpha=.0, beta=0, t=.8, diffusion_steps=5, embedding_scale=1, rate_of_speech=1.):

    japanese = text

    # raw_jpn = japanese[japanese.find(":") + 2:]
    # speaker = japanese[:japanese.find(":") + 2]


    if ":" in japanese[:10]:
        raw_jpn = japanese[japanese.find(":") + 2:]
        speaker = japanese[:japanese.find(":") + 2]
    else:
        raw_jpn = japanese
        speaker = ""
        
    sentences = importable.sent_tokenizer.tokenize(raw_jpn)
    sentences = importable.merging_sentences(sentences)
    
    
    # if is_japanese(raw_jpn):
    #   kotodama_prompt = kotodama_prompt

      
    # else:
    #   kotodama_prompt = speaker + importable.p2g(smart_phonemize(raw_jpn))
    #   print('kimia activated! the converted text is: ', kotodama_prompt)
      


    silence = 24000 * 0.5 # 500 ms of silence between outputs for a more natural transition
    # sentences = sent_tokenize(text)
    print(sentences)
    wavs = []
    s_prev = None
    for text in sentences:
        
        text_input = smart_phonemize(text)
        print('phonemes -> ', text_input)
        
        if is_japanese(text):
          kotodama_prompt = text

          
        else:
          kotodama_prompt = importable.p2g(smart_phonemize(text))
          kotodama_prompt = re.sub(r'\s+', ' ', kotodama_prompt).strip()
          print('kimia activated! the converted text is:\n ', kotodama_prompt)
          


        Kotodama = importable.Kotodama_Sampler(importable.model, text=speaker + kotodama_prompt, device=importable.device) 

        wav, s_prev = importable.Longform(text_input, 
                                s_prev, 
                                Kotodama, 
                                alpha = alpha, 
                                beta = beta, 
                                t = t, 
                                diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, rate_of_speech=rate_of_speech)
        wavs.append(wav)
        wavs.append(np.zeros(int(silence)))
        
    print('Synthesized: ')
    return (24000, np.concatenate(wavs))
    
    

def Inference_Synth_Prompt(text, description, Kotodama, alpha, beta, diffusion_steps, embedding_scale, rate_of_speech , progress=gr.Progress()):
    
    if is_japanese(text):
      text = text

      
    else:
      text = importable.p2g(smart_phonemize(text))
      
      print('kimia activated! the converted text is: ', text)
      
    
    prompt = f"""{description} \n text: {text}"""
    
    print('prompt ->: ', prompt)

    text = smart_phonemize(text)
    
    print('phonemes ->: ', text)

    Kotodama = importable.Kotodama_Prompter(importable.model, text=prompt, device=importable.device) 

    wav = importable.inference(text, 
                            Kotodama, 
                            alpha = alpha, 
                            beta = beta, 
                            diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, rate_of_speech=rate_of_speech)
    
    wav = importable.trim_long_silences(wav)


    print('Synthesized: ')
    return (24000, wav)

with gr.Blocks() as audio_inf:
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Text", info="Enter the text", value="きみの存在は、私の心の中で燃える小さな光のよう。きみがいない時、世界は白黒の写真みたいに寂しくて、何も輝いてない。きみの笑顔だけが、私の灰色の日々に色を塗ってくれる。離れてる時間は、めちゃくちゃ長く感じられて、きみへの想いは風船みたいにどんどん膨らんでいく。きみなしの世界なんて、想像できないよ。",  interactive=True, scale=5)
            voice = gr.Dropdown(voicelist, label="Voice", info="Select a default voice.", value=voicelist[5], interactive=True)
            voice_2 = gr.Audio(label="Upload your own Audio", interactive=True, type='filepath', max_length=300, waveform_options={'waveform_color': '#a3ffc3', 'waveform_progress_color': '#e972ab'})
            
            with gr.Accordion("Advanced Parameters", open=False):

                alpha = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1, label="Alpha", info="a Diffusion sampler parameter handling the timbre, higher means less affected by the reference | 0 = diffusion is disabled", interactive=True)
                beta = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1, label="Beta", info="a Diffusion sampler parameter, higher means less affected by the reference | 0 = diffusion is disabled", interactive=True)
                multispeakersteps = gr.Slider(minimum=3, maximum=15, value=5, step=1, label="Diffusion Steps", interactive=True)
                embscale = gr.Slider(minimum=1, maximum=5, value=1, step=0.1, label="Intensity", info="will impact the expressiveness, if you raise it too much it'll break.", interactive=True)
                rate_of_speech = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Rate of Speech", info="Higher -> Faster", interactive=True)
        
        with gr.Column(scale=1):
            btn = gr.Button("Synthesize", variant="primary")
            audio = gr.Audio(interactive=False, label="Synthesized Audio", waveform_options={'waveform_color': '#a3ffc3', 'waveform_progress_color': '#e972ab'})
            btn.click(Synthesize_Audio, inputs=[inp, voice, voice_2, multispeakersteps, embscale, alpha, beta, rate_of_speech], outputs=[audio], concurrency_limit=4)

# Kotodama Text sampler Synthesis Block
with gr.Blocks() as longform:
    with gr.Row():
        with gr.Column(scale=1):
            inp_longform = gr.Textbox(
                label="Text",
                info="Enter the text [Speaker: Text] | Also works without any name.",
                value=list(example_texts.values())[0],  
                interactive=True,
                scale=5
            )
            
            with gr.Row():
                example_dropdown = gr.Dropdown(
                    choices=list(example_texts.keys()),  
                    label="Example Texts [pick one!]",
                    value=list(example_texts.keys())[0], 
                    interactive=True
                )
                
            example_dropdown.change(
                fn=update_text_input,
                inputs=[example_dropdown],
                outputs=[inp_longform]
            )
            
            with gr.Accordion("Advanced Parameters", open=False):

                alpha_longform = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1, 
                                           label="Alpha", 
                                           info="a Diffusion parameter handling the timbre, higher means less affected by the reference | 0 = diffusion is disabled", 
                                           interactive=True)
                beta_longform = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1, 
                                          label="Beta", 
                                          info="a Diffusion parameter, higher means less affected by the reference | 0 = diffusion is disabled", 
                                          interactive=True)
                diffusion_steps_longform = gr.Slider(minimum=3, maximum=15, value=10, step=1, 
                                                     label="Diffusion Steps", 
                                                     interactive=True)
                embedding_scale_longform = gr.Slider(minimum=1, maximum=5, value=1.25, step=0.1, 
                                              label="Intensity", 
                                              info="a Diffusion parameter, it will impact the expressiveness, if you raise it too much it'll break.", 
                                              interactive=True)

                rate_of_speech_longform = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, 
                                                    label="Rate of Speech", 
                                                    info="Higher = Faster", 
                                                    interactive=True)

        with gr.Column(scale=1):
            btn_longform = gr.Button("Synthesize", variant="primary")
            audio_longform = gr.Audio(interactive=False, 
                                      label="Synthesized Audio", 
                                      waveform_options={'waveform_color': '#a3ffc3', 'waveform_progress_color': '#e972ab'})
            
            btn_longform.click(LongformSynth_Text, 
                                inputs=[inp_longform, 
                                        gr.State(None),  # s_prev 
                                        gr.State(None),  # Kotodama
                                        alpha_longform, 
                                        beta_longform, 
                                        gr.State(.8),   # t parameter 
                                        diffusion_steps_longform, 
                                        embedding_scale_longform, 
                                        rate_of_speech_longform], 
                                outputs=[audio_longform], 
                                concurrency_limit=4)

# Kotodama prompt sampler Inference Block
with gr.Blocks() as prompt_inference:
    with gr.Row():
        with gr.Column(scale=1):
            text_prompt = gr.Textbox(
                label="Text", 
                info="Enter the text to synthesize. This text will also be fed to the encoder. Make sure to see the Read Me for more details!",
                value=inputs[0],
                interactive=True,
                scale=5
            )
            description_prompt = gr.Textbox(
                label="Description",
                info="Enter a highly detailed, descriptive prompt that matches the vibe of your text to guide the synthesis.",
                value=prompts[0],
                interactive=True, 
                scale=7
            )
            
            with gr.Row():
                random_btn = gr.Button('Random Example', variant='secondary')
            
            with gr.Accordion("Advanced Parameters", open=True):
                embedding_scale_prompt = gr.Slider(minimum=1, maximum=5, value=1, step=0.25,
                                            label="Intensity",
                                            info="it will impact the expressiveness, if you raise it too much it'll break.",
                                            interactive=True)
                alpha_prompt = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1,
                                         label="Alpha",
                                         info="a Diffusion sampler parameter handling the timbre, higher means less affected by the reference | 0 = diffusion is disabled",
                                         interactive=True)
                beta_prompt = gr.Slider(minimum=0, maximum=1, value=0.0, step=0.1,
                                        label="Beta",
                                        info="a Diffusion sampler parameter, higher means less affected by the reference | 0 = diffusion is disabled",
                                        interactive=True)
                diffusion_steps_prompt = gr.Slider(minimum=3, maximum=15, value=10, step=1,
                                                   label="Diffusion Steps",
                                                   interactive=True)
                rate_of_speech_prompt = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1,
                                                  label="Rate of Speech",
                                                  info="Higher = Faster",
                                                  interactive=True)
        with gr.Column(scale=1):
            btn_prompt = gr.Button("Synthesize with Prompt", variant="primary")
            audio_prompt = gr.Audio(interactive=False,
                                    label="Prompt-based Synthesized Audio",
                                    waveform_options={'waveform_color': '#a3ffc3', 'waveform_progress_color': '#e972ab'})
           
 
            random_btn.click(
                fn=get_random_prompt_pair,
                inputs=[],
                outputs=[text_prompt, description_prompt]
            )
            
            btn_prompt.click(Inference_Synth_Prompt,
                              inputs=[text_prompt,
                                      description_prompt,
                                      gr.State(None),
                                      alpha_prompt,
                                      beta_prompt,
                                      diffusion_steps_prompt,
                                      embedding_scale_prompt,
                                      rate_of_speech_prompt],
                              outputs=[audio_prompt],
                              concurrency_limit=4)

notes = """
<h1>Notes</h1>

<p>
This work is somewhat different from your typical speech model. It offers a high degree of control<br>
over the generation process, which means it's easy to inadvertently produce unimpressive outputs.
</p>

<p>
<b>Kotodama</b> and the <b>Diffusion sampler</b> can significantly help guide the generation towards<br>
something that aligns with your input, but they aren't foolproof. turn off the diffusion sampler or <br>
set it to very low values if it doesn't sound good to you. <br>
</p>

<p>
The prompt encoder is also highly experimental and should be treated as a proof of concept. Due to the<br>
overwhelming ratio of female to male speakers and the wide variation in both speakers and their expressions,<br>
the prompt encoder may occasionally produce subpar or contradicting outputs. For example, high expressiveness alongside <br>
high pitch has been associated with females speakers simply because I had orders of magnitude more of them in the dataset.<br>
</p>

<p>
________________________________________________________ <br>
<strong>A useful note about the voice design and prompting:</strong><br>\n
The vibe of the dialogue impacts the generated voice since the Japanese dialogue  <br>
and the prompts were jointly trained. This is a peculiar feature of the Japanese lanuage.<br>
For example if you use 俺 (ore)、僕(boku) or your input is overall masculine  <br>
you may get a guy's voice, even if you describe it as female in the prompt. <br> \n
The Japanese text that is fed to the prompt doesn't necessarily have to be  <br>
the same as your input, but we can't do it in this demo <br>
to not make the page too convoluted. In a real world scenario, you can just use a <br>
prompt with a suitable Japanese text to guide the model, get the style<br>
then move on to apply it to whatever dialogue you wish your model to speak.<br>


</p>
________________________________________________________ <br>
<p>
The pitch information in my data was accurately calculated, but it only works in comparison to the other speakers <br>
so you may find a deep pitch may not be exactly too deep; although it actually is <br> 
when you compare it to others within the same data, also some of the gender labels <br>
are inaccurate since we used a model to annotate them. <br> \n
The main goal of this inference method is to demonstrate that style can be mapped to description's embeddings <br>
yielding reasonably good results.
</p>

<p>
Overall, I'm confident that with a bit of experimentation, you can achieve reasonbaly good results. <br>
The model should work well out of the box 90% of the time without the need for extensive tweaking.<br>
However, here are some tips in case you encounter issues:
</p>

<h2>Tips:</h2>

<ul>
  <li>
    Ensure that your input closely matches your reference (audio or text prompt) in terms of tone,<br>
    non-verbal cues, duration, etc.
  </li>
  
  <li>
    If your audio is too long but the input is too short, the speech rate will be slow, and vice versa.
  </li>
  
  <li>
    Experiment with the <b>alpha</b>, <b>beta</b>, and <b>Intensity</b> parameters. The Diffusion<br>
    sampler is non-deterministic, so regenerate a few times if you're not satisfied with the output.
  </li>
  
  <li>
    The speaker's share and expressive distribution in the dataset significantly impact the quality;<br>
    you won't necessarily get perfect results with all speakers.
  </li>
  
  <li>
    Punctuation is very important, for example if you add «!» mark it will raise the voice or make it more intense.
  </li>
  
  <li>
    Not all speakers are equal. Less represented speakers or out-of-distribution inputs may result<br>
    in artifacts.
  </li>
  
  <li>
    If the Diffusion sampler works but the speaker didn't have a certain expression (e.g., extreme anger)<br>
    in the dataset, try raising the diffusion sampler's parameters and let it handle everything. Though<br>
    it may result in less speaker similarity, the ideal way to handle this is to cook new vectors by<br>
    transferring an emotion from one speaker to another. But you can't do that in this space.
  </li>
  
  <li>
    For voice-based inference, you can use litagin's awesome <a href="https://huggingface.co/datasets/litagin/Moe-speech" target="_blank">Moe-speech dataset</a>,<br>
    as part of the training data includes a portion of that.
  </li>
  
  <li>
    you may also want to tweak the phonemes if you're going for something wild. <br>
    i have used cutlet in the backend, but that doesn't seem to like some of my mappings.
  </li>


</ul>
"""


notes_jp = """
<h1>メモ</h1>

<p>
この作業は、典型的なスピーチモデルとは少し異なります。生成プロセスに対して高い制御を提供するため、意図せずに<br>
比較的にクオリティーの低い出力を生成してしまうことが容易です。
</p>

<p>
<b>Kotodama</b>と<b>Diffusionサンプラー</b>は、入力に沿ったものを生成するための大きな助けとなりますが、<br>
万全というわけではありません。良いアウトプットが出ない場合は、ディフュージョンサンプラーをオフにするか、非常に低い値に設定してください。
</p>


_____________________________________________<br>\n
<strong>音声デザインとプロンプトに関する有用なメモ:</strong><br>
ダイアログの雰囲気は、日本語のダイアログとプロンプトが共同でTrainされたため、生成される音声に影響を与えます。<br>
これは日本語の特徴的な機能です。例えば、「俺」や「僕」を使用したり、全体的に男性らしい入力をすると、<br>
プロンプトで女性と記述していても、男性の声が得られる可能性があります。<br>
プロンプトに入力される日本語のテキストは、必ずしも入力内容と同じである必要はありませんが、<br>
このデモではページが複雑になりすぎないようにそれを行うことはできません。<br>
実際のシナリオでは、適切な日本語のテキストを含むプロンプトを使用してモデルを導き、<br>
スタイルを取得した後、それを希望するダイアログに適用することができます。<br>

_____________________________________________<br>\n

<p>
プロンプトエンコーダも非常に実験的であり、概念実証として扱うべきです。女性話者対男性話者の比率が圧倒的で、<br>
また話者とその表現に大きなバリエーションがあるため、エンコーダは質の低い出力を生成する可能性があります。<br>
例えば、高い表現力は、データセットに多く含まれていた女性話者と関連付けられています。<br>
それに、データのピッチ情報は正確に計算されましたが、それは他のスピーカーとの比較でしか機能しません...<br>
だから、深いピッチが必ずしも深すぎるわけではないことに気づくかもしれません。<br>
ただし、実際には、同じデータ内の他の人と比較すると、深すぎます。このインフレンスの主な目的は、<br>
スタイルベクトルを記述にマッピングし、合理的に良い結果を得ることにあります。
</p>

<p>
全体として、少しの実験でほぼ望む結果を達成できると自信を持っています。90%のケースで、大幅な調整を必要とせず、<br>
そのままでうまく動作するはずです。しかし、問題が発生した場合のためにいくつかのヒントがあります：
</p>

<h2>ヒント：</h2>

<ul>
  <li>
    入力がリファレンス（音声またはテキストプロンプト）とトーン、非言語的な手がかり、<br>
    長さなどで密接に一致していることを確認してください。
  </li>
  
  <li>
    音声が長すぎるが入力が短すぎる場合、話速が遅くなります。その逆もまた同様です。
  </li>
  
  <li>
    アルファ、ベータ、および埋め込みスケールのパラメータを試行錯誤してください。Diffusionサンプラーは<br>
    非決定的なので、満足のいく出力が得られない場合は何度か再生成してください。
  </li>
  
  <li>
    データセット内の話者の分布と表現力の分布は品質に大きく影響します。<br>
    すべての話者で必ずしも完璧な結果が得られるわけではありません。
  </li>
  
  <li>
    句読点は重要です。たとえな、「！」を使えば、スタイルのインテンシティが上がります。
  </li>
  
  <li>
    すべての話者が平等に表現されているわけではありません。少ない表現の話者や<br>
    分布外の入力はアーティファクトを生じさせる可能性があります。
  </li>
  
  <li>
    Diffusionサンプラーが機能しているが、データセット内で特定の表現（例：極度の怒り）がない場合、<br>
    Diffusionサンプラーのパラメータを引き上げ、サンプラーにすべてを任せてください。ただし、それにより<br>
    話者の類似性が低下する可能性があります。この問題を理想的に解決する方法は、ある話者から別の話者に<br>
    感情を転送し新しいベクトルを作成することですが、ここではできません。
  </li>
  
  <li>
    音声ベースのインフレンスには、トレーニングデータの一部としてMoe-speechデータセットの一部を含む<br>
    <a href="https://huggingface.co/datasets/litagin/Moe-speech" target="_blank">litaginの素晴らしいデータセット</a>を使用できます。
  </li>
  
  <li>
    たまには音素の調整が必要になる場合もあります。バックエンドではcutletを使っているのですが、<br>
    いくつかのOODマッピングがcutletと相性が良くないみたいです。
  </li>
</ul>

"""
with gr.Blocks() as read_me:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(notes)
    
with gr.Blocks() as read_me_jp:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(notes_jp)
    

custom_css = """
.tab-label {
    color: #FFD700 !important;
}
"""




with gr.Blocks(title="Tsukasa 司", css=custom_css + "footer{display:none !important}", theme="Respair/Shiki@1.2.1") as demo:
    # gr.DuplicateButton("Duplicate Space")
    gr.Markdown(INTROTXT)


    gr.TabbedInterface([longform, audio_inf, prompt_inference, read_me, read_me_jp], 
                       ['Kotodama Text Inference', 'Voice-guided Inference','Prompt-guided Inference [Highly Experimental - not optimized]', 'Read Me! [English]', 'Read Me! [日本語]'])

if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(show_api=False, share=True)
