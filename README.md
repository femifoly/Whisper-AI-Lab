# Whisper-AI-Lab
## Speech recognition and training using whisper AI
```
pip install git+https://github.com/openai/whisper.git
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/installwhisper.png)

```
import whisper
model = whisper.load_model("medium")
result = model.transcribe("test1.wav", language='en', fp16=False)
print(result["text"])
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Importwhisper.png)
## Below we'll look at some low level Whisper access using whisper.decode() and whisper.detect_language():
```
model = whisper.load_model('medium')

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio('test1.wav')
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
```
# View Spectogram
```
import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
%matplotlib inline
import librosa.display
from IPython.display import Audio
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import skimage.io
```
filename = 'test1.wav'
y, sr = librosa.load(filename)
*trim silent edges*
Test1, _ = librosa.effects.trim(y)
librosa.display.waveplot(Test1, sr=sr);
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Spectogram.png)
# Detect the spoken language
_, probs = model.detect_language(mel)
lang = max(probs, key=probs.get)
prob = "{0:.0%}".format(max(probs.values()))

# print language that scored the highest liklihood
print(f'Detected language (and probability): {lang}', f'({prob})')
```
![.Wav](https://user-images.githubusercontent.com/120579796/224144962-b4436762-5617-4f53-93e2-3a94b627e984.png)
![.mp3](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Detectlangmp3.png)

# Steps
## 1. Prepare Instance to Google Colab on EC2 Instance
- **Launch a GPU instance on AWS Cloud**
- **Install CUDA with the latest drivers** 
- **Install Jupyter Notebook remotely - ssh**
- --------
## 2. Data Processing and Feature Extraction
- **Prerequisites for Training a Model**

*For the best results when training a model, you will need:*
- **Short audio recordings (at least 100?) that are:**
- **In 16-bit, mono PCM WAV format.**
- **Between 1 and 30 seconds each.**
- **Have a sample rate of 22050 Hz.**
- **Have a minimum of background noise and distortion.**
- ![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/AudioSplit.png)
- ![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Noisereduction.png)
- ![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/preprocessed%20audio.png)
---------
 *Have no long pauses of silence at the beginning, throughout the middle, and at the end.*
- **A metadata.csv file that references each WAV file and indicates what text is spoken in the WAV file.**
- **A configuration file tailored to your data set and chosen vocoder (e.g. Tacotron, WavGrad, etc).**
- **A machine with a fast CPU (ideally an nVidia GPU with CUDA support and at least 12 GB of GPU RAM; you cannot effectively use CUDA if you have less than 8 GB OF GPU RAM).**
- **Lots of RAM (at least 16 GB of RAM is preferable).**

## 3. 
