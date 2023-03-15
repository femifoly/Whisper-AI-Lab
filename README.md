# Whisper-AI-Lab
## Speech recognition and training using whisper AI

Introduction:
Whisper is an open-source deep learning framework used for natural language processing, and it has been used to build models for various languages. In this presentation, we will discuss how to train whisper on Albanian dataset using cloud.

Step 1: Data Preprocessing
The first step in training Whisper on Albanian dataset is to preprocess the data. This involves cleaning and formatting the data to ensure that it is in the appropriate format for training. The data should be in a text format and each line of the text file should contain a single sentence. It is also important to remove any unnecessary characters or words from the dataset.

## Step 2: Setting up the Cloud Environment
*The next step is to set up a cloud environment for training the model. There are several cloud platforms available such as Google Cloud, Amazon Web Services (AWS), and Microsoft Azure. Choose the one that suits your needs and budget.*

Today we are going to discuss how to use AWS S3, Lambda, and Transcribe to carryout speech to text function. 

Speech to text technology is becoming more and more prevalent in our daily lives. From virtual assistants like Siri and Alexa to speech recognition software used in customer service, speech to text technology is revolutionizing the way we interact with technology.

AWS S3 is a cloud storage service that allows you to store and retrieve large amounts of data. AWS Lambda is a serverless compute service that runs code in response to events and automatically manages the compute resources for you. AWS Transcribe is a speech recognition service that makes it easy to add speech-to-text capabilities to your applications.

So how does this all work together? 

First, you will need to upload your audio file to AWS S3. Once the file is uploaded, you can trigger an AWS Lambda function to start the speech recognition process using AWS Transcribe. The Lambda function will start the transcription process and store the resulting text in a text file in S3. 

Here are the basic steps to use AWS S3, Lambda, and Transcribe to do speech to text:

1. Upload the audio file to AWS S3.
2. Create an AWS Lambda function that will trigger the Transcribe service.
3. Configure the Lambda function to listen to S3 for new audio files.
4. Set up the AWS Transcribe service to transcribe the audio file and store the results in a text file in S3.
5. Retrieve the text file from S3 and use it as needed.

Using AWS S3, Lambda, and Transcribe to do speech to text has many advantages. It's cost-effective, scalable, and easy to use. Additionally, AWS Transcribe supports a wide range of audio formats and has high accuracy. 

In conclusion, AWS S3, Lambda, and Transcribe offer a powerful and easy-to-use solution for speech to text transcription. Whether you are an individual looking to transcribe a single file or a business with large volumes of audio to transcribe, AWS provides a solution that is scalable and cost-effective. 


Step 3: Installing Whisper
After setting up the cloud environment, the next step is to install Whisper. Whisper can be installed using pip or Anaconda. It is recommended to use Anaconda as it provides an environment for installing packages and managing dependencies.

Step 4: Training the Model
Once Whisper is installed, the next step is to train the model. This involves creating a model configuration file, which includes the architecture of the model, hyperparameters, and other settings. The model architecture can be defined by selecting the appropriate layers and activation functions. The hyperparameters can be adjusted to optimize the performance of the model.

Step 5: Evaluating the Model
After training the model, the next step is to evaluate its performance. This involves testing the model on a validation dataset to measure its accuracy and other performance metrics. If the model is not performing well, adjustments can be made to the hyperparameters, or the model architecture can be modified.

Conclusion:
Training Whisper on Albanian dataset using cloud is a straightforward process that involves data preprocessing, setting up the cloud environment, installing Whisper, training the model, and evaluating its performance. By following these steps, you can build a high-quality natural language processing model for the Albanian language.
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
DATASET_DIR = "/content/sr/sr"
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


