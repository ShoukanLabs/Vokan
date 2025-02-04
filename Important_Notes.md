
1. **Enabling the Diffusion sampler breaks the output:**

*Unfortunately, depending on your hardware, the diffusion sampler may not work. This issue is beyond my control and seems to be related to how different hardware handles floating point operations. I can confirm that it works on A40, V100, and Google Colab's T4 GPUs, though it is not guaranteed to work even if you have the same GPUs. It doesn't matter if you use CPUs; the issue persists. This was a serious problem in the original StyleTTS2, but with the different ways of sampling i have included, you can simply turn off the diffusion sampler and use other methods for sampling the vectors, hopefully with minimal or no hit to the quality.*

2. **Cannot reproduce the quality of the provided samples or cannot consistently get good results:**

*Controllability comes at a cost, and that cost is the ease of use. This is especially true with a network composed of inherently non-deterministic modules. The system is highly sensitive to variations in your style vector. However, I'm confident that you can almost always achieve the expression you want in the most impressively natural way possible, but you should carefully tweak and play around with the inference parameters. also, not all speakers can handle all emotions consistently as some of them maybe didn't have that expression but you can cook a new one by using from a speaker that has it. I explain in more details on how to get consistently great results in the gradio space or inference notebook.*

3. **[RuntimeError: The size of tensor a (512) must match the size of tensor b (some number) at non-singleton dimension 3]:**

*your input is too long for a single inference run. use the Longform inference function. this is particularly challenging with the Tsumugi (placeholder) checkpoint as the context length of the mLSTM layer is capped at 512, meaning you cannot generate more than ~10 seconds of audio without relying on the Longform function. but this shouldn't be an issue with the other checkpoint. all in all, this should not be a serious problem. as there's no theoretical limit to the output thanks to the Longform algoirthm.*

4. **short inputs sound un-impressive:**

*everything said in 2, applies here. make sure your style vector is suitable for that. but generally it's not recommended to use a very short input.*

5. **About the Names used in kotodama inference:**
*They are all random names mapped to the ids. they have no relation to the speaker, their role in a series or anything. there are hundreds of names so I should provide a metadata later. though the model should work with any random names thrown at it.*

6. **Nans in 2nd Stage:**

*Your gradients are probably exploding. try clipping or your batch size is way too high. if that didn't work, feel free to do the first few epochs which is the pre-training stage, using the original DP script. or use DP entriely.*

7. **Supporting English (or other languages):**

*There is a wide gap between English and other languages, so I mostly focus on non-English projects. but the good folks at Shoukan labs are trying to train a multilingual model with English included. however, if i ever do myself, it'll be focused on something specific (let's say accents).*

8. **Can I use the DDP on the original StyleTTS without any of your modifications?**

*Sure! but you need to do some changes. replace the xlstm's pre projection at around line 922 with the lines of the original's script. you also have to modify prosody encoder and put the contents of F0Ntrain into the forward function of the prosody encoder itself.*
