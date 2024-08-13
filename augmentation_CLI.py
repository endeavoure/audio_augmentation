import numpy as np
import ffmpegio
import random
from variables import ffmpeg_path
from audio_augmentator import AudioAugmentation, write_audio, read_audio
import click

ffmpegio.set_path(ffmpeg_path) # ffmpeg_path - string variable which is a path to ffmpeg and ffprobe

@click.command()
def main():
    """
    It is a console application created for augmentating audio records before using them as the material for ASR/STT model to learn;
    In order to use this application you must enter your original audio record path and path for the processed one.
    """
    read_file_path = str(input("Enter the need-to-be-augmented file path: "))

    sample_rate, audio = read_audio(read_file_path)
    audio = audio.T
    processor = AudioAugmentation()
    augmented_audio = processor.augment(audio, sample_rate)

    write_file_path = str(input("Enter the path for saving the augmented file: "))
    write_audio(write_file_path, augmented_audio, sample_rate)

    print(f'Done! Your augmented audio file has been placed in this directory: {write_file_path}')

if __name__ == "__main__":
    main()