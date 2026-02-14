import random
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

torch.set_grad_enabled(False)

device = torch.device('cpu')

vc_model = torch.jit.load("pretrained/model-nc.ts")
vc_model = vc_model.eval()
vc_model = vc_model.to(device)

sns.set_style("white")
sns.set_context("notebook")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ['DejaVu Serif'],
    "text.usetex": False,
    "font.size": 18,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})
mpl.rcParams["lines.linewidth"] = 2.0
mpl.rcParams["axes.grid"] = False

streamlined_colors = [
    "#023858",  
    "#05a692",  
    "#0570b0",  
    "#3690c0",  
    "#74a9cf", 
]


def extract_temporal_embeddings(content_repr, num_frames=None):
    B, C, T = content_repr.shape
    
    embeddings = content_repr.squeeze(0).transpose(0, 1)
    embeddings = embeddings.detach().cpu().numpy()
        
    if num_frames is not None and num_frames < T:
        indices = np.random.choice(T, size=num_frames, replace=False)
        indices = np.sort(indices) 
        embeddings = embeddings[indices]
        
    return embeddings

def load_audio(path, start_sample=16000, end_sample=48760):
    x, sr = librosa.load(path, sr=16000, mono=True)
    x = torch.tensor(x[start_sample:end_sample]).unsqueeze(0).unsqueeze(0)
    return x.to(device)

def plot_tsne(source_embeddings, converted_embeddings_list, target_speaker_ids,
              model_name="model.ts", save_path="tsne_plot.png"):
    
    all_embeddings = np.vstack([source_embeddings] + converted_embeddings_list)
    n_samples = all_embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    num_source_frames = source_embeddings.shape[0]
    ax.scatter(embeddings_2d[:num_source_frames, 0], 
               embeddings_2d[:num_source_frames, 1], 
               c='black', s=80, label='Source', marker='o', alpha=1.0, edgecolors='none')
    
    start_idx = num_source_frames
    for i in range(len(converted_embeddings_list)):
        num_frames = converted_embeddings_list[i].shape[0]
        end_idx = start_idx + num_frames
        color = streamlined_colors[i % len(streamlined_colors)]
        
        ax.scatter(embeddings_2d[start_idx:end_idx, 0], 
                   embeddings_2d[start_idx:end_idx, 1],
                   c=color, s=80, marker='x', linewidths=2.0,
                   label=f'{target_speaker_ids[i]}', alpha=0.75)
        start_idx = end_idx
    
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_title(f"{model_name}\nContent Embeddings", fontweight='bold')
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="gray", fontsize=12)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f">> Plot saved to {save_path}")

if __name__ == "__main__":
    source_path = "audio/target_p228.wav"
    target_folder = "targets"
    num_frames_per_sample = 200
    model_name = "pretrained/model-nc.ts"
        
    source_speech = load_audio(source_path)

    target_paths = glob.glob(os.path.join(target_folder, "*.wav"))
    target_paths = sorted(target_paths)[:4]
    target_speaker_ids = [os.path.basename(p).split('_')[0] for p in target_paths]
        
    with torch.no_grad():
        source_content = vc_model.get_content_z(source_speech)
    
    source_embeddings = extract_temporal_embeddings(source_content, num_frames=num_frames_per_sample)
    print(f"  Source embeddings shape: {source_embeddings.shape}")
    
    converted_embeddings_list = []
    
    for i, (target_path, speaker_id) in enumerate(zip(target_paths, target_speaker_ids)):
        print(f"  Converting to speaker {i+1}/{len(target_paths)}: {speaker_id}")
        
        t, sr = librosa.load(target_path, sr=16000, mono=True)
        t = torch.tensor(t[46000:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            vc_model.reset_pitch()
            vc_model.set_embedding_from_audio(t)
            converted_out = vc_model(source_speech)
        
        with torch.no_grad():
            converted_content = vc_model.get_content_z(converted_out)
        
        converted_emb = extract_temporal_embeddings(converted_content, num_frames=num_frames_per_sample)
        converted_embeddings_list.append(converted_emb)
        print(f"    Converted embeddings shape: {converted_emb.shape}")
    
    plot_tsne(source_embeddings, converted_embeddings_list, target_speaker_ids,
              model_name=model_name, save_path=os.path.join("plots", "tsne_plot.png"))