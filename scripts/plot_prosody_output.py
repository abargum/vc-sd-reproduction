import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from utils.demo_utils import *
from audiotools import transforms as tfm

def plot_prosody_features(x, pitch_logits, f0, uv, periodicity, loudness, output_path, sr=16000, hop_length=256):
    
    distributions = torch.nn.functional.softmax(pitch_logits, dim=1)
    distributions = torch.log(distributions)
    distributions[torch.isinf(distributions)] = distributions[~torch.isinf(distributions)].min()
    distributions = distributions.squeeze().detach().cpu()
    
    x_np = x.squeeze().detach().cpu().numpy()
    f0_np = f0.squeeze().detach().cpu().numpy()
    periodicity_np = periodicity.squeeze().detach().cpu().numpy()
    uv_np = uv.squeeze().detach().cpu().numpy()
    loudness_np = loudness.squeeze().detach().cpu().numpy()
    
    print(f"Distributions shape: {distributions.shape}, Loudness shape: {loudness_np.shape}")
    
    f0_uv = f0_np * uv_np
    
    # Time arrays
    t_audio = np.arange(len(x_np)) / sr
    n_frames = len(f0_np)
    t_frames = np.arange(n_frames) * hop_length / sr
    
    fig, axs = plt.subplots(6, 1, figsize=(12, 7), sharex=True)
    
    axs[0].plot(t_audio, x_np, color='black')
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform")
    
    frame_dur = hop_length / sr
    extent = [t_frames[0], t_frames[-1] + frame_dur, 0, distributions.size(0)]
    axs[1].imshow(distributions, aspect='auto', origin='lower', extent=extent)
    axs[1].set_ylabel("F0 (Hz)")
    axs[1].set_title("Predicted Pitch (Posteriorgram)")
    
    axs[2].plot(t_frames, f0_uv, color='blue')
    axs[2].set_ylabel("F0 (Hz)")
    axs[2].set_title("Predicted Pitch (with UV)")
    
    axs[3].plot(t_frames, periodicity_np, color='green')
    axs[3].set_ylabel("Periodicity")
    axs[3].set_title("Periodicity Curve")
    
    axs[4].step(t_frames, uv_np, color='red', where='mid')
    axs[4].set_ylabel("UV")
    axs[4].set_title("Voiced/Unvoiced Mask")
    
    axs[5].step(t_frames, loudness_np, color='orange')
    axs[5].set_ylabel("dB")
    axs[5].set_title("Loudness")
    axs[5].set_xlabel("Time (s)")
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    print("Loading model...")
    vc_model = torch.jit.load("pretrained/model-nc.ts")
    vc_model = vc_model.eval()
    
    source_path = "audio/target_p228.wav"
    plot_output_path = "plots/prosody_analysis.png"
    
    x, sr = librosa.load(source_path, sr=16000, mono=True)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        pitch_logits, f0, uv, periodicity, loudness = vc_model.get_prosody_output(x)
    
    plot_prosody_features(
        x, 
        pitch_logits, 
        f0, 
        uv, 
        periodicity, 
        loudness, 
        plot_output_path
    )