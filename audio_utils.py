import librosa
import soundfile as sf
import pyworld as pw
from IPython.display import Audio
import IPython


def load_wav(wav_path, sampling_rate=16000):
    signal, sr = librosa.load(wav_path, sr=sampling_rate)
    assert sr==sampling_rate, print(f'sampling rate should be {sampling_rate}, but is {sr}')
    return signal.astype('float64')


def disp_wav(signal, sampling_rate=16000):
    IPython.display.display(Audio(signal, rate=sampling_rate))


def disp_wav_file(wav_file, sampling_rate=16000, offset=None, duration=None):
    '''
    duration: how long will be read [s]. 
            when not specified, everything will be read (default). 
    '''
    assert os.path.exists(wav_file), '{} does not exist.'.format(wav_file)
    if (offset == None) and (duration == None):
        y, _ = librosa.load(wav_file, sr=sampling_rate)
    else:
        y, _ = librosa.load(wav_file, sr=sampling_rate, offset=offset, duration=duration)
    disp_wav(y, sampling_rate=sampling_rate)


def write_wav(wav_path, signal, sampling_rate=16000):
    sf.write(wav_path, signal, sampling_rate)
    
    
def extract_features(signal, sampling_rate=16000):
    # f0: pitch
    # sp: spectrogram
    # ap: aperiodicity
    f0, sp, ap = pw.wav2world(signal, sampling_rate)
    return f0, sp, ap


def synthesize(f0, sp, ap, sampling_rate=16000):
    # f0: pitch
    # sp: spectrogram
    # ap: aperiodicity
    signal = pw.synthesize(f0, sp, ap, sampling_rate)
    return signal