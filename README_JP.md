---
thumbnail: https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png
license: cc-by-nc-4.0
language:
- ja
pipeline_tag: text-to-speech
tags:
- '#StyleTTS'
- '#Japanese'
- Diffusion
- Prompt
- '#TTS'
- '#TexttoSpeech'
- '#speech'
- '#StyleTTS2'
---

<div style="text-align:center;">
  <img src="https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png" alt="Logo" style="width:300px; height:auto;">
</div>


# Tsukasa 司 Speech: Engineering the Naturalness and Rich Expressiveness 

**tl;dr** : I made a very cool japanese speech generation model.

[個人プロジェクト](https://github.com/Respaired/Project-Kanade)の一部で、日本語Speech分野のさらなる発展に焦点を当てています。 

- **Tsukasa** (24kHz)のHuggingFaceスペースを使用してください: [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Tsukasa_Speech)
- **Tsumugi** (48kHz)のHuggingFaceスペース: [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Tsumugi_48khz)

- Shoukan labのDiscordサーバーに参加してください、私がよく訪れる居心地の良い場所です -> [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/JrPSzdcM)

## これは何?

*注意*: このモデルは日本語のみをサポートしていますが、Gradioデモでローマ字、またローマ字と普通の日本語をミックスしたテキストを入力することができます。

これはスピーチ生成ネットワークで、生成された音声の表現力と制御性を最大化することを目的としています。その中核にあるのは[StyleTTS 2](https://github.com/yl4579/StyleTTS2)のアーキテクチャで、以下のような変更が加えられています:

- 完全に新しいデータ前処理パイプラインの導入
- 通常のPyTorch LSTMレイヤーではなくmLSTMレイヤーを採用し、テキストおよびプロソディエンコーダーの容量を高めるためにパラメーター数を増やしている
- PL-Bert、Pitch Extractor、Text Alignerを一から再学習
- SLMにはWavLMの代わりにWhisperのエンコーダーを使用
- 48kHzの設定
- 非言語サウンド(溜息、ポーズなど)や笑い声の表現力が向上
- スタイルベクトルのサンプリングの新しい方法
- プロンプト可能な音声合成
- ローマ字の入力や日本語とローマ字の混在に対応した賢いフォネミゼーションアルゴリズム
- DDP(Distributed Data Parallel)とBF16(Bfloat16)の訓練が修正された(ほとんど!)

2つのチェックポイントが使用できます。Tsukasa and Tsumugi(仮称)です。

Tsukasaは約800時間のデータで学習されています。主にゲームやノベルからのデータで、一部はプライベートデータセットからのものです。
そのため、日本語は「アニメ日本語」(実際の日常会話とは異なる)になります。

Tsumugi(仮称)は、この データの一部約300時間を使用し、さらに手動クリーニングや注釈付けを行った制御された方法で学習されています。 

残念ながら、Tsumugiのコンテキスト長は制限されているため、イントネーションの処理はTsukasaほど良くありません。
また、Kotodamaのインファレンスの最初のモードしかサポートしていないため、ボイスデザインはできません。


Brought to you by:

- Soshyant (私)
- [Auto Meta](https://github.com/Alignment-Lab-AI)
- [Cryptowooser](https://github.com/cryptowooser)
- [Buttercream](https://github.com/korakoe)

このプロジェクトは、StyleTTSの著者であるYinghao Aaron Li氏の成果に基づいています。<br> 彼はこの分野で最も才能あるエンジニアの一人だと思います。
また、スクリプトのデバッグで協力してくれたKarestoさんとRavenさんにも感謝します。本当に素晴らしい人たちです。

## なぜ？

最近、より大規模なモデルへの傾向がありますが、私は逆の道を行き、既存のツールを活用することで限界まで性能を引き上げることを試みています。
スケールが高くなくてもいい結果が得られるかもしれないことを試しています。

日本語に関連するいくつかの事項もあります。例えば、この言語のイントネーションをどのように改善できるか、文脈によって綴りが変わる文章をどのように正確に注釈付けできるかなどです。

## 使い方

# Inference:

Gradioデモ:
```bash
python app_tsuka.py
```

または、推論ノートブックをチェックしてください。その前に、**重要な注意事項**セクションをよく読んでください。

# Training:

**第1段階**:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
**第2段階**:
```bash
accelerate launch accelerate_train_second.py --config_path ./Configs/config.yml 
```

SLMの共同TrainはマルチGPUでは機能しません。(そもそもこの段階を行うのは必要かどうか自体が疑問です、私も使用していません。)

または:

```bash
launch train_first.py --config_path ./Configs/config.yml
```

**第3段階**(Kotodama、プロンプトエンコーディングなど):
*未予定*


## 今後の改善案

いくつかの改善点が考えられます。必ずしも私が取り組むわけではありませんが、提案として捉えてください:

- [o] デコーダーの変更(具体的に[fregrad](https://github.com/kaistmm/fregrad)が面白そう。)
- [o] 別のアルゴリズムを使ってPitch Extractorを再訓練
- [o] 非音声サウンドの生成は改善されましたが、完全な非音声出力は生成できません。これは、hard alignmentの影響かもしれません。
- [o] スタイルエンコーダーを別のモダリティとしてLLMsで使用する(Style-Talkerに似たアプローチ)

## 前提条件
1. Python >= 3.11
2. このリポジトリをクローンします:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Pythonの要件をインストールします: 
```bash
pip install -r requirements.txt
```

## 訓練の詳細

- 8x A40s + 2x V100s(32GBずつ)
- 750 ~ 800時間のデータ
- Bfloat16
- 約3週間の訓練、全体で3ヶ月(データパイプラインの作業を含む)
- Google Cloudベースで概算すると、66.6 kg CO2eq.の二酸化炭素排出(Google Cloudは使用していませんが、クラスターがアメリカにあるため、非常に大まかな推定です。) 


### 重要な注意事項

[こちらへ](https://huggingface.co/Respair/Tsukasa_Speech/blob/main/%E9%87%8D%E8%A6%81%E3%81%AA%E3%83%A1%E3%83%A2.md)

ご質問があった場合は、遠慮なく教えてください。
```
saoshiant@protonmail.com
```
Discordも可能です。



## Some cool and related projects:

[Kokoro](https://huggingface.co/spaces/hexgrad/Kokoro-TTS) - a very nice and light weight TTS, based on StyleTTS. supports Japanese and English.<br>
[VoPho](https://github.com/ShoukanLabs/VoPho) - a meta phonemizer to rule them all. it will automatically handle any languages with hand-picked high quality phonemizers.<br>



## References
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

```
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}

```