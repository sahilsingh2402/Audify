# gradio_app.py

import gradio as gr
import os
import time
import traceback
import subprocess
import sys
import re
import tempfile
import torch

# Import backend functions
from extract import extract_book
from generate_audiobook_kokoro import (
    generate_audio_from_stream,
    generate_audiobooks_kokoro,
    test_single_voice_kokoro,
    available_voices,
    KPipeline
)

# --- Global State & Configuration ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(PROJECT_DIR, "extracted_books")
AUDIOBOOKS_DIR = os.path.join(PROJECT_DIR, "audiobooks")
VOICE_TESTS_DIR = os.path.join(PROJECT_DIR, "voice_tests")

os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(AUDIOBOOKS_DIR, exist_ok=True)
os.makedirs(VOICE_TESTS_DIR, exist_ok=True)

CANCELLATION_FLAG = False
PIPELINE_CACHE = {}

# --- Helper Functions ---
def get_pipeline(lang_code, device):
    """Initializes and caches the Kokoro pipeline to prevent reloading."""
    key = f"{lang_code}_{device}"
    if key not in PIPELINE_CACHE:
        print(f"Initializing Kokoro pipeline for lang='{lang_code}' on device='{device}'...")
        try:
            PIPELINE_CACHE[key] = KPipeline(lang_code=lang_code, device=device, repo_id='hexgrad/Kokoro-82M')
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"FATAL: Could not initialize Kokoro pipeline: {e}")
            traceback.print_exc()
            PIPELINE_CACHE[key] = None
    return PIPELINE_CACHE[key]

def get_safe_basename(source_option, single_file, batch_folder, existing_text_folder):
    """Determines the base name for output folders from UI inputs."""
    base_name = ""
    if source_option == "Single Book" and single_file:
        base_name = os.path.splitext(os.path.basename(single_file.name))[0]
    elif source_option == "Batch Folder" and batch_folder:
        base_name = os.path.basename(batch_folder.strip().rstrip(os.sep))
    elif source_option == "Existing Text" and existing_text_folder:
        base_name = os.path.basename(existing_text_folder.strip().rstrip(os.sep))
    if not base_name:
        base_name = f"output_{int(time.time())}"
    return re.sub(r'[^\w\s-]', '', base_name).strip().replace(' ', '_')

def format_time(seconds):
    """Formats seconds into a human-readable string Hh Mm Ss."""
    if seconds is None or seconds < 0: return "N/A"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"~{hours}h {minutes}m"
    elif minutes > 0:
        return f"~{minutes}m {seconds}s"
    return f"~{seconds}s"

def open_folder(folder_path):
    """Opens a folder in the default file explorer."""
    if not folder_path or not os.path.isdir(folder_path):
        message = f"Folder not found: {folder_path}"
        print(message)
        return message
    try:
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])
        return f"Opened {folder_path}"
    except Exception as e:
        message = f"Could not open folder: {e}"
        print(message)
        return message

# --- Main Process Logic ---
def start_main_process(
    source_option, single_file, batch_folder, existing_text_folder,
    use_toc, extract_mode,
    voice, audio_format, device, speed,
    # This special argument updates the progress bar automatically
    progress=gr.Progress()
):
    global CANCELLATION_FLAG
    CANCELLATION_FLAG = False
    start_time = time.time()
    logs = ["--- Starting Process ---"]
    
    # This generator yields updates for the 5 components in the 'outputs' list
    yield "\n".join(logs), "Starting...", "ETA: N/A", None, gr.update(visible=False)

    try:
        # ----- 1. VALIDATION -----
        logs.append("Validating inputs...")
        yield "\n".join(logs), "Validating inputs...", "ETA: N/A", None, gr.update()
        if source_option == "Single Book" and single_file is None: raise ValueError("Please upload a book file.")
        if source_option == "Batch Folder" and not batch_folder.strip(): raise ValueError("Please provide a path for the batch folder.")
        if source_option == "Existing Text" and not existing_text_folder.strip(): raise ValueError("Please provide a path for the existing text folder.")
        if not voice: raise ValueError("Please select a voice.")

        # ----- 2. EXTRACTION PHASE -----
        safe_base_name = get_safe_basename(source_option, single_file, batch_folder, existing_text_folder)
        extracted_output_dir = os.path.join(EXTRACTED_DIR, safe_base_name)
        audiobook_output_dir = os.path.join(AUDIOBOOKS_DIR, safe_base_name)
        os.makedirs(extracted_output_dir, exist_ok=True)
        os.makedirs(audiobook_output_dir, exist_ok=True)

        logs.append(f"Extracting text in '{extract_mode}' mode...")
        yield "\n".join(logs), "Extracting text...", "ETA: N/A", None, gr.update()

        source_path = single_file.name if source_option == "Single Book" else batch_folder
        extraction_generator = extract_book(file_path=source_path, use_toc=use_toc, extract_mode=extract_mode, output_dir=extracted_output_dir)
        
        text_chunks_for_audio = []
        final_output_path = ""
        
        # Loop over the generator from extract.py
        for update_type, val1, val2 in extraction_generator:
            if CANCELLATION_FLAG: raise InterruptedError("Process cancelled by user.")
            
            if update_type == 'progress':
                current_step, total_steps = val1, val2
                percentage = current_step / total_steps if total_steps > 0 else 0
                # The 'progress' object is updated directly here
                progress(percentage, desc=f"Extracting: {int(percentage * 100)}%")
                
                elapsed_time = time.time() - start_time
                eta = ((elapsed_time / percentage) - elapsed_time) if percentage > 0.01 else None
                
                # Yield updates for the other components
                yield ("\n".join(logs), 
                       f"Extracting: {int(percentage * 100)}%", 
                       f"ETA: {format_time(eta)}", 
                       None, gr.update())
            
            elif update_type == 'data':
                text_chunks_for_audio.append(val1)
            
            elif update_type == 'result_path':
                final_output_path = val1
                break

        # ----- 3. AUDIO GENERATION PHASE -----
        if text_chunks_for_audio:
            logs.append("Text extraction complete. Initializing TTS Pipeline...")
            yield "\n".join(logs), "Initializing TTS...", "ETA: ...", None, gr.update()

            lang_code, pipeline = voice[0], get_pipeline(voice[0], device)
            if pipeline is None: raise RuntimeError("Failed to initialize TTS Pipeline.")

            output_filename = f"{safe_base_name}{audio_format}"
            final_output_path = os.path.join(audiobook_output_dir, output_filename)
            
            audio_phase_start_time = time.time()
            audio_generator = generate_audio_from_stream(
                text_chunks_for_audio, pipeline, voice, final_output_path,
                float(speed), cancellation_flag=lambda: CANCELLATION_FLAG
            )

            # Loop over the generator from generate_audio_from_stream
            for audio_update_type, audio_val1, audio_val2 in audio_generator:
                if CANCELLATION_FLAG: raise InterruptedError("Process cancelled by user.")
                if audio_update_type == 'progress':
                    current_step, total_steps = audio_val1, audio_val2
                    percentage = current_step / total_steps if total_steps > 0 else 0
                    progress(percentage, desc=f"Synthesizing audio: {int(percentage * 100)}%")

                    elapsed_time = time.time() - audio_phase_start_time
                    eta = ((elapsed_time / percentage) - elapsed_time) if percentage > 0.01 else None
                    status_message = f"Synthesizing audio: {int(percentage * 100)}%"
                    
                    yield ("\n".join(logs),
                           status_message,
                           f"ETA: {format_time(eta)}",
                           None, gr.update())
                elif audio_update_type == 'status':
                    logs.append(audio_val1)
                    yield "\n".join(logs), audio_val1, gr.update(), None, gr.update()
        
        # ----- 4. FINALIZATION -----
        progress(1.0, desc="Completed")
        success_message = f"Process completed in {format_time(time.time() - start_time)}!"
        logs.append(success_message)
        yield ("\n".join(logs),
               success_message,
               "ETA: Done",
               gr.update(value=final_output_path, visible=True) if final_output_path and os.path.exists(final_output_path) else None,
               gr.update(visible=True))

    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"An error occurred: {e}"
        logs.extend(["\n\n--- DETAILED ERROR TRACEBACK ---\n", tb_str])
        print(tb_str)
        yield "\n".join(logs), error_message, "ETA: Error", None, gr.update(visible=False)

def cancel_main_process():
    global CANCELLATION_FLAG
    CANCELLATION_FLAG = True
    return "Cancellation requested..."

def run_voice_test(voice, text, device, speed):
    if not voice or not text.strip(): raise gr.Error("Please select a voice and enter sample text.")
    gr.Info("Starting voice test...")

    lang_code = voice[0]
    pipeline = get_pipeline(lang_code, device)
    if not pipeline:
        raise gr.Error("Failed to initialize the TTS pipeline. Check console for details.")

    output_path = os.path.join(VOICE_TESTS_DIR, f"test_{voice}_{int(time.time())}.wav")
    
    result_path = test_single_voice_kokoro(
        input_text=text,
        voice=voice,
        output_path=output_path,
        pipeline=pipeline,
        speed=speed
    )
    if result_path and os.path.exists(result_path):
        gr.Info("Voice test completed.")
        return gr.Audio(value=result_path, label=f"Test: {voice}")
    else:
        raise gr.Error("Failed to generate test audio. Check console for details.")

# --- Gradio UI Definition ---
with gr.Blocks(title="🎧 Audify") as demo:
    gr.Markdown("# 🎧 Audify")
    gr.Markdown("#### Welcome to Audify 👋, the simple yet powerful app that gives voice to your documents 🗣️. Audify transforms any PDF or EPUB file on your computer into a personal, high-quality audiobook 🎧.")
    
    with gr.Tabs():
        with gr.TabItem("1. Main Process"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. Source Input")
                    source_option = gr.Radio(["Single Book", "Batch Folder", "Existing Text"], label="Source Type", value="Single Book")
                    single_file_input = gr.File(label="Upload PDF/EPUB", file_types=[".pdf", ".epub"], visible=True)
                    batch_folder_input = gr.Textbox(label="Path to Folder with Books", placeholder="/path/to/your/books", visible=False)
                    existing_text_folder_input = gr.Textbox(label="Path to Folder with .txt Files", placeholder="/path/to/extracted_text", visible=False)
                    
                    gr.Markdown("### 2. Extraction Options")
                    use_toc_checkbox = gr.Checkbox(label="Use TOC/Metadata (if available)", value=True, info="For PDF/EPUB files")
                    extract_mode = gr.Radio(["chapters", "whole"], label="Extraction Mode", info="Choose 'whole' for memory-efficient streaming.", value="whole")
                    
                    gr.Markdown("### 3. Audio Settings")
                    voice_dropdown = gr.Dropdown(available_voices(), label="Voice", value="af_heart")
                    audio_format_dropdown = gr.Dropdown([".wav", ".mp3"], label="Audio Format", value=".wav")
                    gr.Markdown("⚠️ **Note:** Saving as `.mp3` may require system libraries like `ffmpeg` or `libsndfile`. (Not Working)")
                    gr.Markdown("Please use this command to convert your WAV file to MP3 and save space:\n\n"
                        "```bash\n"
                        "ffmpeg -i \"<YOUR BOOK>.wav\" -codec:a libmp3lame -q:a 2 \"book.mp3\"\n"
                        "```"
                    )
                    device_radio = gr.Radio(["cuda", "cpu"], label="Device", value="cuda" if torch.cuda.is_available() else "cpu")
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed")

                with gr.Column(scale=3):
                    gr.Markdown("### 4. Run & Monitor")
                    with gr.Row():
                        start_button = gr.Button("▶ Start Process", variant="primary", scale=3)
                        cancel_button = gr.Button("⏹ Cancel", variant="stop", scale=1)
                    
                    with gr.Row():
                        status_output = gr.Textbox(label="Current Status", interactive=False, placeholder="Ready", scale=2)
                        eta_output = gr.Textbox(label="Est. Time Remaining", interactive=False, placeholder="ETA: N/A", scale=1)
                    
                    log_output = gr.Textbox(label="Process Logs", lines=1, interactive=False, autoscroll=True)
                    
                    gr.Markdown("### 5. Output")
                    final_audio_output = gr.Audio(label="Final Audiobook", visible=False)
                    open_audio_folder_btn = gr.Button(value="Open Output Folder", visible=True)
                    open_folder_status = gr.Textbox(label="Folder Status", interactive=False, visible=True)
        
        with gr.TabItem("Voice Test"):
            gr.Markdown("## Test TTS Voices")
            with gr.Row():
                with gr.Column():
                    test_voice_dropdown = gr.Dropdown(available_voices(), label="Select Voice", value="af_heart")
                    test_device_radio = gr.Radio(["cuda", "cpu"], label="Device", value="cuda" if torch.cuda.is_available() else "cpu")
                    test_speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed")
                    test_text_input = gr.Textbox(label="Sample Text", lines=5, value="This is a sample text to test the selected voice. The quick brown fox jumps over the lazy dog.")
                    run_test_button = gr.Button("▶️ Run Test", variant="primary")
                with gr.Column():
                    test_audio_output = gr.Audio(label="Test Audio Output")

    # --- Event Handlers ---
    def update_source_inputs(choice):
        return {single_file_input: gr.update(visible=choice == "Single Book"), batch_folder_input: gr.update(visible=choice == "Batch Folder"), existing_text_folder_input: gr.update(visible=choice == "Existing Text")}
    source_option.change(fn=update_source_inputs, inputs=source_option, outputs=[single_file_input, batch_folder_input, existing_text_folder_input])

    start_event = start_button.click(
        fn=start_main_process,
        inputs=[source_option, single_file_input, batch_folder_input, existing_text_folder_input, use_toc_checkbox, extract_mode, voice_dropdown, audio_format_dropdown, device_radio, speed_slider],
        outputs=[log_output, status_output, eta_output, final_audio_output, open_audio_folder_btn]
    )
    cancel_button.click(fn=cancel_main_process, inputs=None, outputs=status_output, cancels=[start_event])
    
    run_test_button.click(
        fn=run_voice_test, 
        inputs=[test_voice_dropdown, test_text_input, test_device_radio, test_speed_slider], 
        outputs=test_audio_output
    )
    
    def get_output_path_for_open_btn(source_opt, single_f, batch_f, existing_f):
        safe_name = get_safe_basename(source_opt, single_f, batch_f, existing_f)
        return os.path.join(AUDIOBOOKS_DIR, safe_name)
    open_folder_path_holder = gr.Textbox(visible=False)
    open_audio_folder_btn.click(fn=get_output_path_for_open_btn, inputs=[source_option, single_file_input, batch_folder_input, existing_text_folder_input], outputs=open_folder_path_holder).then(fn=open_folder, inputs=open_folder_path_holder, outputs=open_folder_status)

if __name__ == "__main__":
    demo.launch(inbrowser=True)