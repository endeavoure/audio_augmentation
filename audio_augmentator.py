import numpy as np
import ffmpegio
import random
from variables import ffmpeg_path

ffmpegio.set_path(ffmpeg_path) # ffmpeg_path - string variable which is a path to ffmpeg and ffprobe

class AudioAugmentation:
    def __init__(self, noise_coeff=0.03, amplitude_low=0.5, amplitude_high=1.5, reflection_low=0.1, reflection_high=0.6, delay_ms=100):
        self.noise_coeff = noise_coeff
        self.amplitude_low = amplitude_low
        self.amplitude_high = amplitude_high
        self.reflection_low = reflection_low
        self.reflection_high = reflection_high
        self.delay_ms = delay_ms

    def add_noise(self, audio):
        """
        Function for adding Gaussian noise multiplied by noise_coeff to the original audio;
        audio is a numpy array;
        audio.shape = (num_channels, num_samples).
        """
        noise = np.random.randn(audio.shape[0], audio.shape[1])
        augmented_audio = audio + self.noise_coeff*noise
        return augmented_audio

    def change_amplitude(self, audio):
        """
        Function for globally changing the amplitude rate of the original audio;
        audio is a numpy array;
        audio.shape = (num_channels, num_samples).
        """
        amplitude = random.uniform(self.amplitude_low, self.amplitude_high)
        augmented_audio = audio*amplitude
        return augmented_audio
    
    def reverbation(self, audio, sample_rate):
        """
        Function for creating reverbation (echo) in the original audio;
        reflection coefficient is random in [0.1, 1.0];
        delay time is equal to 50ms.
        """
        augmented_audio = np.zeros_like(audio)
        reflection = random.uniform(self.reflection_low, self.reflection_high)
        reflect_sample = int(self.delay_ms*sample_rate/1000)
        for i in range(1, audio.shape[1]):
            augmented_audio[:, i] = audio[:, i]
            if i - reflect_sample >= 0:
                augmented_audio[:, i] += reflection*augmented_audio[:, i-reflect_sample]
        return augmented_audio

    def augment(self, audio, sample_rate):
        """
        Function for applying augmentations of this processor to the original audio in a random way;
        Returns an augmented audio numpy array.
        """
        audio = self.add_noise(audio)
        audio = self.change_amplitude(audio)
        audio = self.reverbation(audio, sample_rate)
        return audio
    
def read_audio(file_path):
    """
    Function for reading an audio file into a numpy array;
    type(file_path) = str;
    Returns sample_rate (int) and audio numpy array.
    """
    sample_rate, audio = ffmpegio.audio.read(file_path)  
    return sample_rate, audio

def write_audio(file_path, audio, sample_rate):
    """
    Function for writing a numpy array into an audio file;
    type(file_path) = str.
    """
    audio = audio.T
    ffmpegio.audio.write(file_path, sample_rate, audio)