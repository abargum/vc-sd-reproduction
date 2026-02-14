import os
import json
import torch
import torch.nn as nn
from audiotools import AudioSignal
from audiotools import transforms as tfm
import IPython.display as ipd
from IPython.display import display, HTML
import numpy as np
from torchfcpe import spawn_bundled_infer_model


def get_speaker_embeddings_json(targets, json_path='speaker_dicts/speaker_dict_new.json'):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Speaker embedding JSON not found at: {json_path}")
    
    with open(json_path, 'r') as f:
        speaker_dict = json.load(f)

    emb_list = nn.ParameterList()
    one_emb_list = nn.ParameterList()
    f0_mean_list = []

    for speaker in targets:
        if speaker not in speaker_dict:
            raise KeyError(f"Speaker '{speaker}' not found in speaker dictionary. "
                           f"Available speakers: {list(speaker_dict.keys())}")
        
        stats = speaker_dict[speaker]

        try:
            emb = torch.tensor(stats['avg_emb'], dtype=torch.float32).unsqueeze(0)
            emb_list.append(nn.Parameter(emb))
            one_emb = torch.tensor(stats['one_emb'], dtype=torch.float32).unsqueeze(0)
            one_emb_list.append(nn.Parameter(one_emb))

            f0_mean_list.append(stats['f0_mean'])

        except KeyError as e:
            raise KeyError(f"Missing required field in speaker '{speaker}': {e}")

    return emb_list, one_emb_list, f0_mean_list

def normalize(audio, transform):
    signal = AudioSignal(audio, 16000)
    kwargs = transform.instantiate(state=0, signal=signal)
    output_signal = transform(signal.clone().to('cpu'), **kwargs)
    transformed_signal = output_signal.audio_data 
    transformed_signal = transformed_signal.to('cpu')
    return transformed_signal


def process_in_chunks(x, model, pitch, chunk_size, use_yin=True, get_pitch=False):
    num_chunks = (x.shape[-1] + chunk_size - 1) // chunk_size
    processed_chunks = []
    for i in range(num_chunks):
        # Extract the current chunk
        start = i * chunk_size
        end = min((i + 1) * chunk_size, x.shape[-1])
        chunk = x[:, :, start:end]
        
        # If the last chunk is smaller than chunk_size, pad it
        if chunk.shape[-1] < chunk_size:
            padding = torch.zeros(1, 1, chunk_size - chunk.shape[-1])
            chunk = torch.cat([chunk, padding], dim=-1)
        
        # Process the chunk
        chunk = chunk.float()
        if get_pitch:
            y = model.get_pitch(chunk, use_yin)
        else:
            y = model((chunk, torch.tensor([pitch]).to(chunk)))
            
        processed_chunks.append(y)
    
    out = torch.cat(processed_chunks, dim=-1)
    return out


def display_audios(audio_list, sample_rate=16000):
    html_blocks = []
    
    for item in audio_list:
        if len(item) == 3:
            label, audio, sr = item
        else:
            label, audio = item
            sr = sample_rate

        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # Handle case where audio has extra dimensions
        audio = np.squeeze(audio)

        widget = ipd.Audio(audio, rate=sr)
        html_blocks.append(f"""
        <div style="margin-right:20px; text-align:center;">
            <p><b>{label}</b></p>
            {widget._repr_html_()}
        </div>
        """)

    display(HTML(f"""
    <div style="display:flex; flex-direction:row; gap:20px;">
        {''.join(html_blocks)}
    </div>
    """))