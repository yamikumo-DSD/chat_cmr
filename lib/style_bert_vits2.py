#"""
#Useful documentation for this usage:
#https://note.com/f6844710/n/nf9ed9fbb8dc6
#"""

# No use of sever_fastapi
PATH = "path_to_style-bert-vits2_dir"
import sys ; sys.path.append(PATH)
import common.tts_model as tts_model
import pathlib
import os
from lib.uis import ipyplayaudio
from lib.utils import change_directory

program_working_dir = os.getcwd()

model_holder: None|tts_model.ModelHolder = None

def load_models(device='cpu'):
    global model_holder

    allowed_device = ['cpu', 'mps', 'cuda']
    if not device in allowed_device:
        raise ValueError(f'Device must be one of {allowed_device}.')

    model_holder = tts_model.ModelHolder(
        pathlib.Path(os.path.join(PATH, 'model_assets')), 
        device=device
    )
    
    model_holder.models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = tts_model.Model(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        model.load_net_g()
        model_holder.models.append(model)


def infer(model_id: int = 0, **kwargs):
    os.chdir(PATH)
    
    global model_holder
    from io import BytesIO
    from scipy.io import wavfile
    
    if not model_holder:
        raise RuntimeError('Run load_models() before running infer()')
    model = model_holder.models[model_id]
    sampling_rate, audio = model.infer(
        text=kwargs["text"],
        sid=kwargs["speaker_id"],
        reference_audio_path=kwargs["reference_audio_path"],
        sdp_ratio=kwargs["sdp_ratio"],
        noise=kwargs["noise"],
        noisew=kwargs["noisew"],
        length=kwargs["length"],
        line_split=kwargs["auto_split"],
        split_interval=kwargs["split_interval"],
        assist_text=kwargs["assist_text"],
        assist_text_weight=kwargs["assist_text_weight"],
        use_assist_text=bool(kwargs["assist_text"]),
        style=kwargs["style"],
        style_weight=kwargs["style_weight"],
        verbose=False,
    )
    with BytesIO() as wavContent:
        wavfile.write(wavContent, sampling_rate, audio)
        os.chdir(program_working_dir)
        return wavContent.getvalue()


def synth_voice_single_shot(
        text: str, 
        speaker_id: int = 0, 
        model_id: int = 0,
        speaker_name: str|None = None,
        sdp_ratio: float = 0.2,
        noise: float = 0.6,
        noisew: float = 0.8,
        length: float = 1.0,
        language: str = 'JP',
        auto_split: bool = True,
        split_interval: int = 1,
        assist_text: str|None = None,
        assist_text_weight: float = 1.0,
        style: str = 'Neutral',
        style_weight: float = 5.0,
        reference_audio_path: str|None = None,
    ):

    if len(text) > 100:
        raise ValueError('style-bert-vits2 cannot synthesize voices for text longer than 100.')
    
    import requests
    import json

    global PORT, HOST, ROOT
    
    params = {
        'text': text,
        'speaker_id': speaker_id,
        'model_id': model_id,
        'speaker_name': speaker_name,
        'sdp_ratio': sdp_ratio,
        'noise': noise,
        'noisew': noisew,
        'length': length,
        'language': language,
        'auto_split': str(auto_split).lower(),  # Ensure bool translates correctly to string
        'split_interval': split_interval,
        'assist_text': assist_text,
        'assist_text_weight': assist_text_weight,
        'style': style,
        'style_weight': style_weight,
        'reference_audio_path': reference_audio_path
    }
    return infer(**params)
    

def synth_voice(text, *args, max_workers: int = 4, **kwargs):
    """ Refer synth_voice_single_shot for arguments. """
    from lib.utils import split_text
    from lib.audio import join_wavs
    from concurrent.futures.thread import ThreadPoolExecutor
    
    chunks = split_text(text, max_length=90)
    wavs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(synth_voice_single_shot, chunk, *args, **kwargs) for chunk in chunks]
        for future in futures:
            wavs.append(future.result())

    return join_wavs(wavs)

    
def ipyspeak(text, *args, **kwargs) -> None:
    """ Refer synth_voice_single_shot for arguments. """
    from lib.uis import ipyplayaudio
    ipyplayaudio(synth_voice(text=text, *args, **kwargs))