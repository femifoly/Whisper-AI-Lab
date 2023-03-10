# Whisper-AI-Lab
## Speech recognition and training using whisper AI
### Steps

#### 1. Data Processing and Feature Extraction
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

#### 2. Install whisper and the required libraries

```
pip install git+https://github.com/openai/whisper.git
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/installwhisper.png)

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
#### Transcribe raw wav file(s)
```
import whisper
model = whisper.load_model("medium")
result = model.transcribe("test1.wav", language='en', fp16=False)
print(result["text"])
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Importwhisper.png)

#### 3. Below we'll look at some low level Whisper access using whisper.decode() and whisper.detect_language():

```
model = whisper.load_model('medium')

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio('test1.wav')
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
```
#### 9. Detect the spoken language

```
_, probs = model.detect_language(mel)
lang = max(probs, key=probs.get)
prob = "{0:.0%}".format(max(probs.values()))
# print language that scored the highest liklihood
print(f'Detected language and probability): {lang}', f'({prob})')
```
![.Wav](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Detectedlangwav.png)
![.mp3](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Detectlangmp3.png)
#### 4. View raw waveform (time domain) Spectogram
```
filename = 'test1.wav'
y, sr = librosa.load(filename)
*trim silent edges*
Test1, _ = librosa.effects.trim(y)
librosa.display.waveplot(Test1, sr=sr);
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/Spectogram.png)

#### 5. Find Spectogram
The spectrum of a signal is found by taking the Fourier Transform of the signal in a time domain. The approach that is normally taken in to divide the sampled signal into equal parts (as mentioned above) and take the Fourier Transform of each part individually. This is called STFT. Thefore, when we want to take the STFT of a signal, we need to specify how many samples we should consider at a time.

#### 6. Display the spectrogram

```
librosa.display.specshow(spectrogram_librosa, sr=sr, x_axis='time', y_axis='linear',hop_length=hop_length)
plt.title('Linear Frequency Power Spectrogram')
plt.colorbar()
plt.tight_layout()
plt.show()
```

```
# Size of the Fast Fourier Transform (FFT), which will also be used as the window length
n_fft=1024

# Step or stride between windows. If the step is smaller than the window length, the windows will overlap
hop_length=320

# Specify the window type for FFT/STFT
window_type ='hann'

# Calculate the spectrogram as the square of the complex magnitude of the STFT
spectrogram_librosa = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type)) ** 2

print("The shape of spectrogram_librosa is: ", spectrogram_librosa.shape)
print("The size of the spectrogram is ([(frame_size/2) + 1 x number of frames])")
print("The frame size that we have specified is the number of samples to consider for the STFT. In our case, it is equal to the n_fft",n_fft, " samples")
print("The number of frames depends on the total length of the sampled signal, the number of samples in each frame and the hop length.")
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/fft.png)

#### 7. Transform the spectrogram output to a logarithmic scale by transforming the amplitude to decibels and frequency to a mel scale
```
mel_bins = 64 # Number of mel bands
fmin = 0
fmax= None
Mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
print("The shape of mel spectrogram is: ", Mel_spectrogram.shape)
librosa.display.specshow(Mel_spectrogram, sr=sr, x_axis='time', y_axis='mel',hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/melspectogram.png)

#### 8. Log Mel Spectrogram
Move from power (mel) spectrum and apply log and move amplitude to a log scale (decibels). While doing so we will also normalize the spectrogram so that its maximum represent the 0 dB point.
```
mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
print("The shape of Log Mel spectrogram is: ", mel_spectrogram_db.shape)
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel',hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel spectrogram')
plt.tight_layout()
plt.show()
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/logmelspectogram.png)

#### 10. Save the plot in a local directory

```
fig = plt.Figure(figsize=(8,8), dpi=128, frameon=False)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel',hop_length=hop_length)
fig.savefig('./'+str(1)+'.png', bbox_inches='tight', pad_inches=0, dpi=128)
```
##### Save the data in the spectrogram, rather than a plot (image)

*save as .npy*

*Load the saved data as a confirmation*

*save as a.txt file*

```

with open('logMel.npy', 'wb') as f:
    np.save(f,mel_spectrogram_db)
   
with open('logMel.npy', 'rb') as f:
    a = np.load(f)
    print(a.shape)
    print(a)

np.savetxt('logMel.out', mel_spectrogram_db, delimiter=',') # takes more space compared to saving as .npy
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/logmeltext.png)

#### 11. Visualize the mel filter bank
```
mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel_bins, fmin=0.0, fmax=None, htk=False, norm='slaney')
print("The shape of the mel filter bank is: ", mel_filter_bank.shape)
librosa.display.specshow(mel_filter_bank, sr=sr, x_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel filter bank')
plt.tight_layout()
plt.show()
```
![](https://github.com/femifoly/Whisper-AI-Lab/blob/main/Assets/melfilterbank.png)

#### 12. Lib for training

```
%%capture
! pip install pyopenjtalk==0.3.0
! pip install pytorch-lightning==1.7.7
! pip install -qqq evaluate==0.2.2
```

```
import IPython.display
from pathlib import Path

import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm.notebook import tqdm
import pyopenjtalk
import evaluate

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
```
```
DATASET_DIR = "/content/jvs/jvs_ver1"
SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(SEED, workers=True)
```
### 13. util

```
def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform
```
#### 14. Albanian Datasets Download
[VoxLingua107](https://www.arxiv-vanity.com/papers/2011.12998/)
```
import gdown
gdown.download ('http://bark.phon.ioc.ee/voxlingua107/sr.zip', 'sr.zip', quiet=False)
!unzip sr.zip -d ./sr
```


