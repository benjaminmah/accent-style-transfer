import argparse
import speech_recognition as sr
import jiwer
from pydub import AudioSegment
import os


# Create an arg parser for the testing process.
def testing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-original', help='Path to original audio input')
    parser.add_argument('-original_start', type=float, default=None, help='Time of the original audio to start reading at, in seconds. If unspecified, will start from the beginning.')
    parser.add_argument('-original_end', type=float, default=None, help='Time of the original audio to stop reading at, in seconds. If unspecified, will stop at the end.')
    parser.add_argument('-transformed', help='Path to transformed audio outputted by the model')
    parser.add_argument('-transformed_start', type=float, default=None, help='Time of the transformed audio to start reading at, in seconds. If unspecified, will start from the beginning.')
    parser.add_argument('-transformed_end', type=float, default=None, help='Time of the transformed audio to stop reading at, in seconds. If unspecified, will stop at the end.')
    parser.add_argument('-text', default="Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.", help='The ground truth text. If none provided, default to that used in the Speech Accent Archive.')
    return parser


def convert_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = os.path.splitext(mp3_path)[0] + '.wav'
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_audio(audio_path, start, end):
    if audio_path.lower().endswith('.mp3'):
        audio_path = convert_to_wav(audio_path)
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source, offset=start or 0, duration=(end - start) if start and end else None)
    
    try:
        transcription = recognizer.recognize_google(audio_data, show_all=True)
        if transcription and 'alternative' in transcription:
            best = max(transcription['alternative'], key=lambda x: x.get('confidence', 0))
            print(f"Transcription: {best['transcript']} Confidence: {best.get('confidence', 0)}")
            return best['transcript'], best.get('confidence', 0)
        else:
            print("No transcription result.")
            return "", 0.0
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return "", 0.0
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return "", 0.0


def calculate_wer(transcript, ground_truth):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])
    wer_score = jiwer.wer(transformation(ground_truth), transformation(transcript))
    return wer_score


# Compares the WER of the baseline model on the original audio input and the modified audio outputted by our model.
def test_model_output(args):
    original_transcript, original_conf = transcribe_audio(args.original, args.original_start, args.original_end)
    transformed_transcript, transformed_conf = transcribe_audio(args.transformed, args.transformed_start, args.transformed_end)
    original_wer = calculate_wer(original_transcript, args.text)
    transformed_wer = calculate_wer(transformed_transcript, args.text)
    return original_wer, original_conf, transformed_wer, transformed_conf


if __name__ == "__main__":
    parser = testing_parser()
    args = parser.parse_args()
    original_wer, original_conf, transformed_wer, transformed_conf = test_model_output(args)
    print("Original Audio Metrics (WER, Confidence):", original_wer, original_conf)
    print("Transformed Audio Metrics (WER, Confidence):", transformed_wer, transformed_conf)
