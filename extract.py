# extract.py

import fitz  # PyMuPDF
import regex as re
import os
import sys
import zipfile
import tempfile
import time
import unicodedata  # For normalization
from bs4 import BeautifulSoup  # For improved EPUB parsing
from num2words import num2words
import traceback  # For detailed error logging if needed

# --- Configuration ---
HEADER_THRESHOLD = 50  # Pixels from top to ignore
FOOTER_THRESHOLD = 50  # Pixels from bottom to ignore
OVERLAP_CHECK_LINES = 20  # Number of lines to check for overlap between chapters

# --- Text Cleaning and Processing Functions ---
def normalize_text(text):
    """Apply Unicode normalization and fix common problematic characters."""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('—', ', ')
    text = text.replace('–', ', ')
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    return text

def expand_abbreviations_and_initials(text):
    """Expand common abbreviations and fix spaced initials."""
    abbreviations = {
        r'\bMr\.': 'Mister', r'\bMrs\.': 'Misses', r'\bMs\.': 'Miss', r'\bDr\.': 'Doctor',
        r'\bProf\.': 'Professor', r'\bJr\.': 'Junior', r'\bSr\.': 'Senior',
        r'\bvs\.': 'versus', r'\betc\.': 'etcetera', r'\bi\.e\.': 'that is',
        r'\be\.g\.': 'for example', r'\bcf\.': 'compare', r'\bSt\.': 'Saint',
        r'\bVol\.': 'Volume', r'\bNo\.': 'Number', r'\bpp\.': 'pages', r'\bp\.': 'page',
    }
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
    text = re.sub(r'([A-Z])\.(?=\s*[A-Z])', r'\1', text)
    text = re.sub(r' +', ' ', text)
    return text

def convert_numbers(text):
    """Convert integers and years to words. Leaves decimals and other numbers."""
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    def replace_match(match):
        num_str = match.group(0)
        try:
            if '.' in num_str: return num_str
            num = int(num_str)
            if 1500 <= num <= 2100: return num2words(num, to='year')
            elif match.group(1): return num2words(num, to='ordinal')
            else: return num2words(num)
        except ValueError: return num_str
    pattern = r'\b(\d+)(st|nd|rd|th)?\b'
    text = re.sub(pattern, replace_match, text)
    return text

def handle_sentence_ends_and_pauses(text):
    """Ensure sentences end cleanly and handle potential pauses."""
    text = re.sub(r'(?<=\w)([.,!?;:])', r' \1', text)
    text = re.sub(r' +', ' ', text)
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not re.search(r'[.!?;:]$', stripped_line) and not re.match(r'^[-\*\u2022•\d+\.\s]+', stripped_line) and len(stripped_line.split()) > 3:
             line += '.'
        processed_lines.append(line)
    text = '\n'.join(processed_lines)
    text = text.replace(';', ',')
    text = re.sub(r'\s+-\s+', ', ', text)
    text = re.sub(r'([.!?:])\s*', r'\1\n', text)
    return text

def remove_artifacts(text):
    """Remove common extraction artifacts like citations, excessive newlines etc."""
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[.,;:!?\-—–_]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    return text

def join_wrapped_lines(text):
    """Join lines that seem to be wrapped mid-sentence. More robust."""
    lines = text.splitlines()
    result_lines = []
    if not lines:
        return ""
    buffer = lines[0]
    for i in range(1, len(lines)):
        current_line = lines[i]
        prev_line_stripped = buffer.strip()
        if (prev_line_stripped and
            not re.search(r'[.!?:)"»’]$', prev_line_stripped) and
            not re.match(r'^[\sA-Z\d"«‘\[\*\-\u2022•]', current_line.strip()) and
            len(prev_line_stripped.split()) > 1):
             buffer += " " + current_line.strip()
        else:
             result_lines.append(buffer)
             buffer = current_line
    result_lines.append(buffer)
    return '\n'.join(filter(None, [line.strip() for line in result_lines]))

def basic_html_to_text(html_content):
    """Extract text from HTML using BeautifulSoup, removing scripts/styles."""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

def clean_pipeline(text):
    """Apply the full cleaning pipeline in order."""
    if not text: return ""
    text = normalize_text(text)
    text = join_wrapped_lines(text)
    text = expand_abbreviations_and_initials(text)
    text = convert_numbers(text)
    text = handle_sentence_ends_and_pauses(text)
    text = remove_artifacts(text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = text.strip()
    return text

# --- PDF Extraction ---
def extract_pdf_text_by_page(doc):
    """
    Extracts text page by page from PDF, filtering headers/footers.
    """
    all_pages_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_height = page.rect.height
        blocks = page.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
        filtered_lines = []
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if y1 < HEADER_THRESHOLD or y0 > page_height - FOOTER_THRESHOLD:
                continue
            cleaned_block_text = re.sub(r'\s+', ' ', text).strip()
            if cleaned_block_text:
                filtered_lines.append(cleaned_block_text)
        page_text = "\n".join(filtered_lines)
        all_pages_text.append(page_text)
    return all_pages_text

# --- TOC and Chapter Structuring ---
def get_toc(doc):
    """Extract TOC from PDF."""
    toc = doc.get_toc()
    if not toc:
        print("  No Table of Contents found in the document.")
        return []
    else:
        print(f"  Table of Contents extracted with {len(toc)} entries.")
        return toc

def deduplicate_toc(toc):
    """Removes TOC entries that point to the exact same page number."""
    seen_pages = set()
    deduplicated_toc = []
    for entry in toc:
        level, title, page_number = entry
        if page_number not in seen_pages:
            deduplicated_toc.append(entry)
            seen_pages.add(page_number)
        else:
            print(f"    Info: Duplicate TOC entry page removed: Level {level}, '{title}', Page {page_number}")
    return deduplicated_toc

def remove_overlap(prev_text, curr_text, num_lines=OVERLAP_CHECK_LINES):
    """
    Checks if the end of prev_text overlaps with the start of curr_text
    and returns prev_text with the overlap removed. Based on line comparison.
    """
    if not prev_text or not curr_text:
        return prev_text
    prev_lines = prev_text.splitlines()
    curr_lines = curr_text.splitlines()
    if not prev_lines or not curr_lines:
        return prev_text
    max_possible_overlap = min(len(prev_lines), len(curr_lines), num_lines)
    for overlap_size in range(max_possible_overlap, 0, -1):
        prev_suffix = prev_lines[-overlap_size:]
        curr_prefix = curr_lines[:overlap_size]
        if prev_suffix == curr_prefix:
            print(f"    Overlap detected ({overlap_size} lines). Removing from previous chapter end.")
            return "\n".join(prev_lines[:-overlap_size])
    return prev_text

def structure_pdf_by_toc(deduplicated_toc, all_pages_text):
    """
    Structures the PDF text into chapters based on TOC page numbers,
    applies cleaning pipeline per chapter, and removes overlap.
    """
    chapters = []
    num_pages_total = len(all_pages_text)
    print(f"  Structuring PDF text ({num_pages_total} pages) using {len(deduplicated_toc)} TOC entries...")
    last_processed_chapter = None
    for i, entry in enumerate(deduplicated_toc):
        level, title, start_page = entry
        start_page_idx = start_page - 1
        if i < len(deduplicated_toc) - 1:
            _, _, next_start_page = deduplicated_toc[i + 1]
            end_page_idx = max(start_page_idx, next_start_page - 2)
        else:
            end_page_idx = num_pages_total - 1
        if start_page_idx < 0 or start_page_idx >= num_pages_total:
            print(f"    Warning: Invalid start page index ({start_page_idx}) for TOC entry '{title}'. Skipping.")
            continue
        if end_page_idx < start_page_idx:
             print(f"    Info: Chapter '{title}' seems to have zero pages. Assigning one page.")
             end_page_idx = start_page_idx
        elif end_page_idx >= num_pages_total:
             end_page_idx = num_pages_total - 1
        chapter_pages = all_pages_text[start_page_idx : end_page_idx + 1]
        raw_chapter_text = "\n".join(chapter_pages)
        cleaned_chapter_text = clean_pipeline(raw_chapter_text)
        clean_title = title.strip()
        if last_processed_chapter:
            previous_text_no_overlap = remove_overlap(last_processed_chapter['text'], cleaned_chapter_text)
            last_processed_chapter['text'] = previous_text_no_overlap
            chapters.append(last_processed_chapter)
        last_processed_chapter = {'level': level, 'title': clean_title, 'text': cleaned_chapter_text}
    if last_processed_chapter:
        chapters.append(last_processed_chapter)
    final_chapters = [chap for chap in chapters if chap.get('text')]
    print(f"  Finished structuring. Found {len(final_chapters)} non-empty chapters.")
    return final_chapters

def split_text_into_heuristic_chapters(full_raw_text):
    """
    Attempts to split raw text into chapters based on heuristics.
    """
    if not full_raw_text or not full_raw_text.strip():
        return []
    print("    Attempting heuristic chapter splitting...")
    potential_chunks = re.split(r'\n\s*\n{2,}', full_raw_text)
    chapters = []
    chapter_count = 0
    min_chunk_length = 100
    for chunk in potential_chunks:
        trimmed_chunk = chunk.strip()
        if len(trimmed_chunk) > min_chunk_length:
            chapter_count += 1
            cleaned_chunk_text = clean_pipeline(trimmed_chunk)
            if cleaned_chunk_text:
                chapters.append({'title': f'Chapter_{chapter_count}', 'level': None, 'text': cleaned_chunk_text})
    if chapters:
        print(f"    Heuristically split into {len(chapters)} potential chapters.")
    else:
        print("    Heuristic splitting did not yield significant chapters.")
    return chapters

def get_epub_content_files(epub_path):
    """Helper to get an ordered list of content files from an EPUB."""
    with zipfile.ZipFile(epub_path, 'r') as epub_zip:
        try:
            container_xml = epub_zip.read('META-INF/container.xml').decode('utf-8')
            container_soup = BeautifulSoup(container_xml, 'xml')
            opf_path = container_soup.find('rootfile').get('full-path')
            epub_base_path = os.path.dirname(opf_path)
            opf_content = epub_zip.read(opf_path).decode('utf-8', errors='ignore')
            opf_soup = BeautifulSoup(opf_content, 'xml')
            manifest_items = {item.get('id'): {'href': item.get('href'), 'media-type': item.get('media-type')} for item in opf_soup.find('manifest').find_all('item')}
            spine = opf_soup.find('spine')
            spine_order_refs = [item.get('idref') for item in spine.find_all('itemref')] if spine else [id for id, item in manifest_items.items() if 'html' in item.get('media-type', '')]
            content_files = []
            for idref in spine_order_refs:
                href = manifest_items.get(idref, {}).get('href')
                if href:
                    full_path = os.path.normpath(os.path.join(epub_base_path, href)).replace('\\', '/')
                    content_files.append(full_path)
            return content_files
        except Exception as e:
            print(f"Warning: Could not parse EPUB manifest/spine, falling back to all HTML files. Error: {e}")
            return sorted([f for f in epub_zip.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm'))])

def save_chapters_generic(chapters, book_name, output_dir):
    """Saves chapters (list of dicts with 'title', 'text') to files."""
    if not chapters:
        print("  No chapters found or extracted to save.")
        return
    os.makedirs(output_dir, exist_ok=True)
    num_chapters = len(chapters)
    padding = len(str(num_chapters))
    print(f"  Saving {num_chapters} chapters to '{output_dir}'...")
    for idx, chapter in enumerate(chapters, 1):
        level = chapter.get('level', None)
        title = chapter.get('title', f'Chapter_{idx}')
        text = chapter.get('text', '')
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)[:60]
        if not safe_title: safe_title = f"chapter_{idx}"
        level_prefix = f"L{level}_" if level is not None else ""
        filename = f"{str(idx).zfill(padding)}_{level_prefix}{safe_title}.txt"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            print(f"    Error saving chapter '{filename}': {e}")
    print(f"  Finished saving chapters.")

def stream_book_text(file_path):
    """
    A generator that yields structured updates ('progress' or 'data')
    from a PDF or EPUB.
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            print(f"  Streaming PDF with {total_pages} pages.")
            all_pages_text = extract_pdf_text_by_page(doc)
        
        for i, page_text in enumerate(all_pages_text):
            yield ('progress', i + 1, total_pages)
            cleaned_text = clean_pipeline(page_text)
            if cleaned_text:
                yield ('data', cleaned_text, None)

    elif file_ext == '.epub':
        print("  Streaming EPUB content...")
        content_files = get_epub_content_files(file_path)
        total_files = len(content_files)
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for i, content_path in enumerate(content_files):
                yield ('progress', i + 1, total_files)
                try:
                    html_content = epub_zip.read(content_path).decode('utf-8', errors='ignore')
                    raw_text = basic_html_to_text(html_content)
                    cleaned_text = clean_pipeline(raw_text)
                    if cleaned_text:
                        yield ('data', cleaned_text, None)
                except Exception as e:
                    print(f"    Skipping EPUB section '{content_path}': {e}")
    else:
        raise ValueError(f"Unsupported file format for streaming: '{file_ext}'")

def extract_book(file_path, use_toc=True, extract_mode="chapters", output_dir="extracted_books"):
    """
    Main extraction function.
    - In 'whole' mode, it returns a generator that streams progress and text data.
    - In 'chapters' mode, it performs the full chapter extraction and saving,
      then returns a simple generator to signal completion to the UI.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: '{file_path}'")

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Extraction for: {os.path.basename(file_path)} ---")
    print(f"    Extraction Mode: {extract_mode}")

    if extract_mode == "whole":
        print("  Mode: Whole Book (Streaming). Returning data generator.")
        return stream_book_text(file_path)
    
    # --- CHAPTERS MODE (Full logic restored) ---
    else:
        def chapter_generator():
            file_ext = os.path.splitext(file_path)[1].lower()
            book_name_base = os.path.splitext(os.path.basename(file_path))[0]
            safe_book_name = re.sub(r'[^\w\s-]', '', book_name_base).strip().replace(' ', '_')
            
            yield ('progress', 0, 1) # Signal start

            chapters_to_save = []
            if file_ext == '.pdf':
                doc = fitz.open(file_path)
                all_pages_text = extract_pdf_text_by_page(doc)
                toc = get_toc(doc)
                dedup_toc = deduplicate_toc(toc) if toc else []

                if use_toc and dedup_toc:
                    print("  Attempting to structure PDF by TOC...")
                    chapters_to_save = structure_pdf_by_toc(dedup_toc, all_pages_text)
                else:
                    print("  No usable TOC or disabled. Attempting heuristic chapter splitting.")
                    full_raw_text = "\n".join(all_pages_text)
                    chapters_to_save = split_text_into_heuristic_chapters(full_raw_text)
                doc.close()

            elif file_ext == '.epub':
                print("  Extracting all EPUB sections for chapter mode...")
                # In chapter mode, we must collect all parts first before saving.
                # This is inherently more memory-intensive than 'whole' mode.
                # temp_chapters = [{'title': f'part_{i+1}', 'text': text} for i, text in enumerate(parse_epub_content(file_path))]

                temp_chapters = []
                content_files = get_epub_content_files(file_path) # Use existing helper
                with zipfile.ZipFile(file_path, 'r') as epub_zip:
                    for i, content_path in enumerate(content_files):
                        try:
                            html_content = epub_zip.read(content_path).decode('utf-8', errors='ignore')
                            raw_text = basic_html_to_text(html_content)
                            if raw_text:
                                temp_chapters.append({'title': f'part_{i+1}', 'text': raw_text})
                        except Exception as e:
                            print(f"    Skipping EPUB section '{content_path}': {e}")
                            
                chapters_to_save = temp_chapters

            if chapters_to_save:
                save_chapters_generic(chapters_to_save, safe_book_name, output_dir)
            else:
                print("  No chapters found or extracted to save.")
            
            yield ('progress', 1, 1) # Signal completion
            yield ('result_path', os.path.abspath(output_dir), None)
        
        return chapter_generator()