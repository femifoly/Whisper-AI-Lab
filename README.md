# Whisper-AI-Lab
## Speech recognition and training using whisper AI

# Steps
## 1. Prepare Instance to Google Colab on EC2 Instance
- **Launch a GPU instance on AWS Cloud**
- **Install CUDA with the latest drivers** 
- **Install Jupyter Notebook remotely - ssh**
## 2. Pre-process data
- **Prerequisites for Training a Model**

*For the best results when training a model, you will need:*
- **Short audio recordings (at least 100) that are:**
- **In 16-bit, mono PCM WAV format.**
- **Between 1 and 10 seconds each.**
- **Have a sample rate of 22050 Hz.**
- **Have a minimum of background noise and distortion.**

 *Have no long pauses of silence at the beginning, throughout the middle, and at the end.*
- **A metadata.csv file that references each WAV file and indicates what text is spoken in the WAV file.**
- **A configuration file tailored to your data set and chosen vocoder (e.g. Tacotron, WavGrad, etc).**
- **A machine with a fast CPU (ideally an nVidia GPU with CUDA support and at least 12 GB of GPU RAM; you cannot effectively use CUDA if you have less than 8 GB OF GPU RAM).**
- **Lots of RAM (at least 16 GB of RAM is preferable).**

## 3. 
