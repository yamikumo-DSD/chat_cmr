def synth_voice(
        text: str, 
        speaker_id: int = 3, 
        pitch: float = 0.0,
        speed: float = 1.0,
    ) -> None:
    
    import requests
    import json
    from urllib.parse import urljoin

    HOST = 'localhost'
    PORT = 50021 # Default Voicevox server port.
    
    params = (('text', text),
              ('speaker', speaker_id),)

    root_url = f'http://{HOST}:{PORT}'

    try:
        query = requests.post(
            urljoin(root_url, 'audio_query'), 
            params=params
        )
    
        query_data = query.json()
        synth_payload = {'speaker': speaker_id}
        query_data['speedScale'] = speed
        query_data['pitchScale'] = pitch
    
        synthesis = requests.post(urljoin(root_url, 'synthesis'),
                                  headers={"Content-Type": "application/json"},
                                  params=synth_payload,
                                  data=json.dumps(query_data))
    except:
        raise(RuntimeError(f"No matched speaker of ID={speaker_id}"))

    return synthesis.content


def ipyspeak(
        text: str, 
        speaker_id: int = 3, 
        pitch: float = 0.0,
        speed: float = 1.0,
    ) -> None:
    from lib.uis import ipyplayaudio
    ipyplayaudio(synth_voice(text, speaker_id, pitch, speed))