import torch
import librosa
import string
import jiwer
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertForCTC
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
from resemblyzer import VoiceEncoder, preprocess_wav
import torchcrepe
import soundfile as sf
import tempfile
import os
from pathlib import Path

# ============================================================================
# HARDCODED PATHS - MODIFY THESE
# ============================================================================
SOURCE_DIR = "test-clean-subset"
CONVERTED_DIR = "conversions/"
GT_TRANSCRIPT_DIR = "test-clean-subset"
TARGET_SPEAKER_DIR = "VCTK-Corpus/wav48"
MAX_SAMPLES = 3000
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Global models
dnsmos_model = None
encoder = None
asr_processor = None
asr_model = None

def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def load_asr_model():
    global asr_processor, asr_model
    print("Loading HuBERT ASR model...")
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    asr_model.eval()

def transcribe_audio(audio_path):
    audio_16k, _ = librosa.load(audio_path, sr=16000)
    asr_inputs = asr_processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        asr_logits = asr_model(**asr_inputs).logits
    
    predicted_ids = torch.argmax(asr_logits, dim=-1)
    transcription = asr_processor.decode(predicted_ids[0])
    return transcription.lower()

def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def extract_pitch_contour(audio, sr=16000, hop_length=160, fmin=50, fmax=550):
    audio_tensor = torch.tensor(audio, device=device).float()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    pitch, confidence = torchcrepe.predict(
        audio_tensor, sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
        model='tiny', device=device, return_periodicity=True
    )
    
    return pitch.squeeze().cpu().numpy(), confidence.squeeze().cpu().numpy()

def calculate_pitch_correlation(audio1_path, audio2_path, confidence_threshold=0.5):
    audio1 = load_audio(audio1_path)
    audio2 = load_audio(audio2_path)
    
    try:
        pitch1, conf1 = extract_pitch_contour(audio1)
        pitch2, conf2 = extract_pitch_contour(audio2)
        
        min_len = min(len(pitch1), len(pitch2))
        pitch1, pitch2 = pitch1[:min_len], pitch2[:min_len]
        conf1, conf2 = conf1[:min_len], conf2[:min_len]
        
        valid_mask = (conf1 > confidence_threshold) & (conf2 > confidence_threshold)
        
        if valid_mask.sum() < 10:
            return None
        
        pitch1_valid = pitch1[valid_mask]
        pitch2_valid = pitch2[valid_mask]
        
        corr, _ = pearsonr(pitch1_valid, pitch2_valid)
        return float(corr)
    except:
        return None

def calculate_dnsmos(audio_path):
    global dnsmos_model
    if dnsmos_model is None:
        dnsmos_model = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False, device=device)
    
    audio = load_audio(audio_path)
    if len(audio) == 0:
        return None
    
    audio_tensor = torch.tensor(audio, device=device).float()
    scores = dnsmos_model(audio_tensor)
    return scores[2].item()

def find_target_speaker_dir(speaker_id):
    target_base = Path(TARGET_SPEAKER_DIR)
    
    potential_path = target_base / speaker_id
    if potential_path.exists() and potential_path.is_dir():
        return potential_path
    
    for subdir in target_base.iterdir():
        if subdir.is_dir() and subdir.name.lower() == speaker_id.lower():
            return subdir
    
    for subdir in target_base.iterdir():
        if subdir.is_dir() and speaker_id.lower() in subdir.name.lower():
            return subdir
    
    return None

def calculate_similarity(audio_path, target_speaker_path):
    global encoder
    if encoder is None:
        encoder = VoiceEncoder()
    
    if not target_speaker_path or not Path(target_speaker_path).exists():
        return None
    
    audio = load_audio(audio_path)
    
    wav_files = list(Path(target_speaker_path).glob("**/*.wav"))
    if not wav_files:
        wav_files = list(Path(target_speaker_path).glob("**/*.flac"))
    
    if not wav_files:
        return None
    
    target_wavs = []
    for wav_path in wav_files[:20]:
        try:
            preprocessed_wav = preprocess_wav(str(wav_path))
            target_wavs.append(preprocessed_wav)
        except:
            continue
    
    if not target_wavs:
        return None
    
    target_embedding = encoder.embed_speaker(target_wavs)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, 16000)
        try:
            preprocessed = preprocess_wav(tmp_path)
            generated_embedding = encoder.embed_utterance(preprocessed)
            similarity = np.inner(generated_embedding, target_embedding)
            os.unlink(tmp_path)
            return float(similarity)
        except:
            os.unlink(tmp_path)
            return None

def main():
    print("="*60)
    print("VOICE CONVERSION EVALUATION")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Converted: {CONVERTED_DIR}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("="*60 + "\n")
    
    load_asr_model()
    
    source_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')])[:MAX_SAMPLES]
    speaker_dirs = [d for d in os.listdir(CONVERTED_DIR) if os.path.isdir(os.path.join(CONVERTED_DIR, d))]
    
    print(f"Found {len(speaker_dirs)} speaker IDs")
    print(f"Processing {len(source_files)} source files\n")
    
    results = {
        'wer': [], 'cer': [], 'f0_pcc': [], 'dnsmos': [], 'sim': []
    }
    
    for audio_filename in tqdm(source_files, desc="Processing"):
        source_index = os.path.splitext(audio_filename)[0]
        source_path = os.path.join(SOURCE_DIR, audio_filename)
        
        if not os.path.exists(source_path):
            continue
        
        gt_transcript = None
        gt_transcript_path = os.path.join(SOURCE_DIR, f"{source_index}.normalized.txt")
        if os.path.exists(gt_transcript_path):
            with open(gt_transcript_path, 'r', encoding='utf-8') as f:
                gt_transcript = normalize_text(f.read().strip())
        
        for speaker_id in speaker_dirs:
            speaker_dir = os.path.join(CONVERTED_DIR, speaker_id)
            
            for f in os.listdir(speaker_dir):
                if not f.endswith('.wav'):
                    continue
                
                base_name = os.path.splitext(f)[0]
                if base_name == source_index or f == audio_filename:
                    converted_path = os.path.join(speaker_dir, f)
                    
                    # WER/CER
                    if gt_transcript:
                        vc_transcription = transcribe_audio(converted_path)
                        results['wer'].append(jiwer.wer(gt_transcript, vc_transcription))
                        results['cer'].append(jiwer.cer(gt_transcript, vc_transcription))
                    
                    # F0 PCC
                    f0_pcc = calculate_pitch_correlation(source_path, converted_path)
                    if f0_pcc is not None:
                        results['f0_pcc'].append(f0_pcc)
                    
                    # DNSMOS
                    dnsmos = calculate_dnsmos(converted_path)
                    if dnsmos is not None:
                        results['dnsmos'].append(dnsmos)
                    
                    # Similarity
                    target_speaker_path = find_target_speaker_dir(speaker_id)
                    sim = calculate_similarity(converted_path, target_speaker_path)
                    if sim is not None:
                        results['sim'].append(sim)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if results['wer']:
        print(f"\nWER:     {sum(results['wer'])/len(results['wer']):.4f}")
    
    if results['cer']:
        print(f"CER:     {sum(results['cer'])/len(results['cer']):.4f}")
    
    if results['f0_pcc']:
        print(f"F0 PCC:  {sum(results['f0_pcc'])/len(results['f0_pcc']):.4f}")
    
    if results['dnsmos']:
        print(f"DNSMOS:  {sum(results['dnsmos'])/len(results['dnsmos']):.4f}")
    
    if results['sim']:
        print(f"SIM:     {sum(results['sim'])/len(results['sim']):.4f}")
    
    total_samples = len(results['wer']) if results['wer'] else len(results['f0_pcc'])
    print(f"\nTotal conversions processed: {total_samples}")
    
    print("="*60)

if __name__ == "__main__":
    main()