import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath("")))

import torch
import random
import librosa
import soundfile as sf
import numpy as np
from audiodiffusion import AudioDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)

model = "C:/Users/moise/Desktop/AI/audio-diffusion/checkpoints/ukdrill_256x256"

audio_diffusion = AudioDiffusion(model_id=model)
mel = audio_diffusion.pipe.mel

num_samples = 1
num_steps = 75
width_mult = 11 # multiplicator for noise input width
noise_input = torch.randn(1, 1, audio_diffusion.pipe.unet.sample_size[0],
                        audio_diffusion.pipe.unet.sample_size[1] * width_mult).cuda() # multiples of 256 (batch size, 1, height, width)

### Run model inference to generate mel spectrogram, audios and loops
for i in range(num_samples):
    seed = generator.seed()
    print(f'Seed = {seed}')
    generator.manual_seed(seed)
    image, (sample_rate,
            audio) = audio_diffusion.generate_spectrogram_and_audio(
                generator=generator,
                steps=num_steps,
                noise=noise_input,
                eta=1)
    # Normalize audio
    audio /= np.max(np.abs(audio),axis=0)
    # Write audio to disk
    sf.write(f'out_steps{num_steps}_seed{seed}_{i}.wav', audio, sample_rate)
