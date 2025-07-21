# generate_audiobook_kokoro.py

import os
import time
import numpy as np
import torch
import soundfile as sf
import re
import traceback
import tempfile
from kokoro import KPipeline

# --- Constants ---
DEFAULT_SAMPLE_RATE = 24000

# This function is defined in gradio_app.py but is used as a helper here
# We assume it exists in the execution context.
def get_pipeline(lang_code, device):
    # This is a placeholder for the actual get_pipeline function in gradio_app
    # In a real scenario, you'd share this via a common utility module.
    # For this script to be runnable standalone for testing, we might need a mock.
    # However, within the Gradio app, this will work as expected.
    # This function call is actually removed from this file's functions to avoid
    # circular dependencies and assume the pipeline object is passed in correctly.
    pass

# --- Helper Functions ---
def available_voices():
    """Return the hard-coded list of available Kokoro voice identifiers."""
    return [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "pf_dora", "pm_alex", "pm_santa"
    ]

# --- Core Audio Generation for a Single File (Helper) ---
def generate_audio_for_file_kokoro(
    input_path,
    pipeline,
    voice,
    output_path,
    speed=1.0,
    split_pattern=r'\n+',
    cancellation_flag=None
):
    """
    Generates audio for a single text file. This is a helper for chapter
    mode and voice tests. It does not yield progress itself.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            print(f"      Warning: Input file '{os.path.basename(input_path)}' is empty. Skipping.")
            return False
    except Exception as e:
        print(f"      Error reading file '{os.path.basename(input_path)}': {e}")
        return False

    if cancellation_flag and cancellation_flag():
        raise InterruptedError("Processing cancelled by user.")

    audio_chunks = []
    try:
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern):
            if cancellation_flag and cancellation_flag():
                raise InterruptedError("Processing cancelled by user.")
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_chunks.append(audio)
    except Exception as e:
        print(f"      Error during Kokoro pipeline processing for '{os.path.basename(input_path)}': {e}")
        traceback.print_exc()
        return False

    if not audio_chunks:
        print(f"      Warning: No audio chunks generated for '{os.path.basename(input_path)}'.")
        return False

    try:
        combined_audio = np.concatenate(audio_chunks)
        max_abs_val = np.max(np.abs(combined_audio))
        if max_abs_val > 0:
             normalized_audio = (combined_audio / max_abs_val * 32767 * 0.95).astype(np.int16)
        else:
             normalized_audio = combined_audio.astype(np.int16)
        sf.write(output_path, normalized_audio, DEFAULT_SAMPLE_RATE)
    except Exception as e:
        print(f"      Error concatenating or saving audio for '{os.path.basename(output_path)}': {e}")
        return False

    return True

# --- Main Generator for 'chapters' mode ---
def generate_audiobooks_kokoro(
    input_dir,
    lang_code,
    voice,
    device="cuda",
    output_dir=None,
    audio_format=".wav",
    speed=1.0,
    cancellation_flag=None,
    pipeline=None
):
    """
    A generator that processes a directory of chapter files and yields progress.
    """
    print(f"\n--- Starting Audiobook Generation Task (Chapters Mode) ---")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: '{input_dir}'")

    if output_dir is None:
        parent_dir = os.path.dirname(os.path.normpath(input_dir))
        book_name = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(parent_dir, f"{book_name}_audio")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.txt')])
    total_files = len(files)
    if total_files == 0:
        print("  Warning: No .txt files found in the input directory.")
        return

    if not pipeline:
        raise RuntimeError("A valid Kokoro pipeline object must be provided.")

    for i, text_file in enumerate(files):
        if cancellation_flag and cancellation_flag():
            raise InterruptedError("Processing cancelled by user.")

        yield ('progress', i + 1, total_files)
        
        print(f"\n[{i+1}/{total_files}] Processing: '{text_file}'")
        input_path = os.path.join(input_dir, text_file)
        base_name = os.path.splitext(text_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}{audio_format}")

        generate_audio_for_file_kokoro(
            input_path=input_path, pipeline=pipeline, voice=voice,
            output_path=output_path, speed=speed, cancellation_flag=cancellation_flag
        )

    yield ('result_path', os.path.abspath(output_dir), None)

# --- Main Generator for 'whole book' stream mode (MODIFIED) ---
def generate_audio_from_stream(
    text_chunks,
    pipeline,
    voice,
    output_path,
    speed=1.0,
    cancellation_flag=None
):
    """
    Generates a single audio file from a list of text chunks by writing to
    disk incrementally. This is the memory-safe method.
    It will always save as .wav for stability.
    """
    # Force output to WAV for stable incremental writing
    output_path = f"{os.path.splitext(output_path)[0]}.wav"
    print(f"  Synthesizing audio from {len(text_chunks)} text chunks to '{output_path}'")
    total_chunks = len(text_chunks)

    try:
        # Open the output file once at the beginning in write mode
        with sf.SoundFile(output_path, mode='w', samplerate=DEFAULT_SAMPLE_RATE, channels=1) as output_file:
            for i, text_chunk in enumerate(text_chunks):
                if cancellation_flag and cancellation_flag():
                    raise InterruptedError("Audio generation cancelled.")
                
                yield ('progress', i + 1, total_chunks)

                if not text_chunk.strip():
                    continue

                # Generate audio for the current chunk
                current_audio_segments = []
                for _, _, audio in pipeline(text_chunk, voice=voice, speed=speed):
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    current_audio_segments.append(audio)
                
                # If audio was generated for this chunk, combine and write to disk
                if current_audio_segments:
                    chunk_audio = np.concatenate(current_audio_segments)
                    output_file.write(chunk_audio)

        print("Audio stream successfully written to disk.")

    except Exception as e:
        print(f"Error during streaming audio generation: {e}")
        traceback.print_exc()
        raise

# --- Function for Voice Testing ---
def test_single_voice_kokoro(
    input_text,
    voice,
    output_path,
    pipeline,
    speed=1.0
):
    """
    Generates a test audio sample for a single voice from a text string.
    """
    print(f"\n--- Starting Single Voice Test for '{voice}' at speed {speed} ---")
    if not input_text.strip():
        raise ValueError("Input text is empty.")
    
    if not pipeline: 
        raise RuntimeError("Invalid pipeline object provided for voice test.")

    temp_file_path = ''
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(input_text)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"  Generating test audio...")
        success = generate_audio_for_file_kokoro(
            input_path=temp_file_path,
            pipeline=pipeline,
            voice=voice,
            output_path=output_path,
            speed=speed
        )

        if success:
            print(f"  Successfully generated test sample to '{output_path}'")
            return output_path
        else:
            print(f"  Failed to generate test sample.")
            return None
            
    except Exception as e:
        print(f"   An unexpected error occurred during single voice test: {e}")
        traceback.print_exc()
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)