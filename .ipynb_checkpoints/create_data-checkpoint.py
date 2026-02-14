import os
import shutil
from pathlib import Path
from pydub import AudioSegment


def concatenate_vctk_files(vctk_base_path, output_base_path, speaker_ids):
    vctk_path = Path(vctk_base_path)
    output_path = Path(output_base_path)
    
    for speaker_id in speaker_ids:
        speaker_dir = vctk_path / speaker_id
        
        if not speaker_dir.exists():
            print(f"Warning: Speaker directory {speaker_dir} not found, skipping...")
            continue
        
        wav_files = sorted(speaker_dir.glob("*.wav"))
        
        if len(wav_files) < 3:
            print(f"Warning: Speaker {speaker_id} has fewer than 3 audio files, using all available")
        
        files_to_concat = wav_files[:3]
        
        if not files_to_concat:
            print(f"Warning: No audio files found for speaker {speaker_id}, skipping...")
            continue
        
        print(f"Processing {speaker_id}: concatenating {len(files_to_concat)} files...")
        
        combined = AudioSegment.empty()
        for audio_file in files_to_concat:
            audio = AudioSegment.from_wav(str(audio_file))
            combined += audio
        
        output_speaker_dir = output_path / speaker_id
        output_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_speaker_dir / "target_concat.wav"
        combined.export(str(output_file), format="wav")
        print(f"  -> Saved to {output_file}")


def extract_librispeech_subset(librispeech_base_path, output_path, max_files=10):
    librispeech_path = Path(librispeech_base_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not librispeech_path.exists():
        print(f"Error: LibriTTS path {librispeech_path} not found")
        return
    
    total_copied = 0
    
    for speaker_dir in sorted(librispeech_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = speaker_dir.name
        
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            wav_files = sorted([f for f in chapter_dir.glob("*.wav") 
                              if not f.name.endswith('.normalized.txt')])
            
            files_to_copy = wav_files[:max_files]
            
            for wav_file in files_to_copy:
                base_name = wav_file.stem
                dest_wav = output_dir / wav_file.name
                shutil.copy2(wav_file, dest_wav)
                
                transcript_file = wav_file.parent / f"{base_name}.normalized.txt"
                if transcript_file.exists():
                    dest_transcript = output_dir / transcript_file.name
                    shutil.copy2(transcript_file, dest_transcript)
                    total_copied += 1
                    print(f"Copied: {wav_file.name} and {transcript_file.name}")
                else:
                    print(f"Warning: Transcript not found for {wav_file.name}")
    
    print(f"\nTotal files copied: {total_copied} pairs")


def main():
    
    VCTK_BASE = "../heka-vc-training/VCTK-Corpus/wav48"
    VCTK_OUTPUT = "concat_targets"
    VCTK_SPEAKERS = ["p231", "p334", "p345", "p360", "p361", "p362"]
    
    LIBRISPEECH_BASE = "LibriTTS/test-clean"
    LIBRISPEECH_OUTPUT = "test-clean-subset"
    
    print("=" * 60)
    print("Processing VCTK-Corpus...")
    print("=" * 60)
    concatenate_vctk_files(VCTK_BASE, VCTK_OUTPUT, VCTK_SPEAKERS)
    
    print("\n" + "=" * 60)
    print("Processing LibriTTS test-clean...")
    print("=" * 60)
    extract_librispeech_subset(LIBRISPEECH_BASE, LIBRISPEECH_OUTPUT)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()