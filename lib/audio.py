def join_wav(wav_data1, wav_data2):
    import wave
    import io
    
    file1 = io.BytesIO(wav_data1)
    file2 = io.BytesIO(wav_data2)

    with wave.open(file1, 'rb') as wav1:
        frames1 = wav1.readframes(wav1.getnframes())
        params = wav1.getparams()
    
    with wave.open(file2, 'rb') as wav2:
        frames2 = wav2.readframes(wav2.getnframes())
    
    combined_frames = frames1 + frames2
    
    combined_file = io.BytesIO()
    with wave.open(combined_file, 'wb') as wav_out:
        wav_out.setparams(params)  # Use the parameters from the first file
        wav_out.writeframes(combined_frames)
    
    return combined_file.getvalue()

def join_wavs(wavs: list):
    import copy
    joined = copy.copy(wavs[0])
    for wav in wavs[1:]:
        joined = join_wav(joined, wav)
    return joined