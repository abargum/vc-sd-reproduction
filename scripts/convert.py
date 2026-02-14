import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import librosa
from tqdm import tqdm
import soundfile as sf
from utils.demo_utils import *
from audiotools import transforms as tfm

transform = tfm.Compose(
    tfm.VolumeNorm(),
    tfm.RescaleAudio()
)

def convert_audio_from_audio(vc_model, source_path, target_audio, transform):
    x, sr = librosa.load(source_path, sr=16000, mono=True)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        vc_model.reset_pitch()
        vc_model.set_embedding_from_audio(target_audio)
        out = vc_model(normalize(x, transform))
    
    return out.squeeze().cpu().numpy()


def convert_audio_from_embedding(vc_model, source_path, speaker_mean, speaker_embedding, transform):
    x, sr = librosa.load(source_path, sr=16000, mono=True)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        vc_model.reset_pitch()
        vc_model.set_new_speaker_from_embedding(speaker_mean, speaker_embedding)
        out = vc_model(normalize(x, transform))
    
    return out.squeeze().cpu().numpy()


def batch_convert(source_folder, target_folder, output_folder, vc_model, transform, 
                  use_embedding=False, json_path=None):
    
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    output_folder = Path(output_folder)
    
    source_files = sorted(source_folder.glob('*.wav'))
    print(f"Found {len(source_files)} source files")
    
    target_dirs = sorted([d for d in target_folder.iterdir() if d.is_dir()])
    
    for target_dir in target_dirs:
        target_id = target_dir.name
        print(f"\nProcessing target: {target_id}")
        
        if use_embedding:
            speaker_embedding_avg, speaker_embedding_one, speaker_mean_list = \
                get_speaker_embeddings_json([target_id], json_path)
            
            speaker_mean = torch.tensor([speaker_mean_list[0]], dtype=torch.float32)
            speaker_embedding = speaker_embedding_avg[0]
            
        else:
            target_files = list(target_dir.glob('*.wav'))
            if not target_files:
                print(f"  Warning: No audio file found for {target_id}, skipping")
                continue
            
            target_path = target_files[0]
            t, sr = librosa.load(target_path, sr=16000, mono=True)
            t = torch.tensor(t[46000:], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        output_dir = output_folder / target_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for source_file in tqdm(source_files):
            output_path = output_dir / source_file.name
            if use_embedding:
                converted = convert_audio_from_embedding(vc_model, source_file, speaker_mean, speaker_embedding, transform)
            else:
                converted = convert_audio_from_audio(vc_model, source_file, t, transform)
            
            sf.write(output_path, converted, 16000)
    
    print(f"\nDone! Output saved to: {output_folder}")


if __name__ == "__main__":
    print("Loading model...")
    vc_model = torch.jit.load("pretrained/model-nc.ts")
    vc_model = vc_model.eval()
    
    source_folder = "test-clean-subset"  
    target_folder = "concat_targets"  
    output_folder = "conversions"
    
    use_embedding = True                
    json_path = "utils/speaker_dict.json" 
    
    print(f"Conversion method: {'Embedding' if use_embedding else 'Audio'}")
    
    batch_convert(
        source_folder,
        target_folder,
        output_folder,
        vc_model,
        transform,
        use_embedding=use_embedding,
        json_path=json_path
    )