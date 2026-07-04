#!/usr/bin/env python3
"""
Text to Audio Converter
Converts markdown/text files to audio using OpenAI or Google TTS APIs.
"""

import os
import sys
import argparse
from pathlib import Path
import re
from typing import Optional, Set, List, Tuple
import tempfile
import glob
import signal
import shutil
import atexit
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import random
import datetime
import hashlib
import html
import unicodedata
import json

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None

try:
    from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip
    except ImportError:
        VideoFileClip = None
        AudioFileClip = None
        ImageClip = None
        CompositeVideoClip = None
        TextClip = None

try:
    import fitz  # PyMuPDF for PDF to image conversion
except ImportError:
    fitz = None

try:
    import zipfile
    import xml.etree.ElementTree as ET
except ImportError:
    zipfile = None
    ET = None

try:
    from professor_svg import ProfessorSVG
except ImportError:
    ProfessorSVG = None

try:
    from video_utils import (
        extract_code_blocks, detect_content_type, parse_course_metadata,
        split_text_for_subtitles, calculate_subtitle_timing,
        create_temp_directory, clean_temp_directory,
        analyze_audio_for_animation, estimate_mouth_opening,
        should_blink, get_gesture_for_content, get_video_dimensions,
        path_part_to_title, has_numbered_prefix, numbered_prefix_parts,
        numbered_prefix_value
    )
except ImportError:
    extract_code_blocks = None
    detect_content_type = None
    parse_course_metadata = None
    split_text_for_subtitles = None
    calculate_subtitle_timing = None
    create_temp_directory = None
    clean_temp_directory = None
    analyze_audio_for_animation = None
    estimate_mouth_opening = None
    should_blink = None
    get_gesture_for_content = None
    get_video_dimensions = None
    path_part_to_title = None
    has_numbered_prefix = None
    numbered_prefix_parts = None
    numbered_prefix_value = None

try:
    from spectrum_visualizer import SpectrumVisualizer
except ImportError:
    SpectrumVisualizer = None

try:
    from subtitle_generator import SubtitleGenerator
except ImportError:
    SubtitleGenerator = None

try:
    from hash_manager import HashManager, AudioFragmentManager
except ImportError:
    HashManager = None
    AudioFragmentManager = None

try:
    from processing_log import ProcessingLog
except ImportError:
    ProcessingLog = None

try:
    from pdf_to_images import convert_pdf_to_images, convert_odp_to_images
except ImportError:
    convert_pdf_to_images = None
    convert_odp_to_images = None


def restore_terminal():
    """Restore terminal to normal state on exit."""
    try:
        # Reset terminal settings using stty
        subprocess.run(['stty', 'sane'], capture_output=True, timeout=2)
        # Show cursor
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
        # Reset colors and attributes
        sys.stdout.write('\033[0m')
        sys.stdout.flush()
        # Enable echo
        subprocess.run(['stty', 'echo'], capture_output=True, timeout=2)
    except:
        # If terminal commands fail, try basic ANSI escape sequences
        try:
            sys.stdout.write('\033[?25h\033[0m')
            sys.stdout.flush()
        except:
            pass


def should_skip_path(file_path: Path) -> bool:
    """
    Check if a file path should be skipped (is in recursos or resources folder).
    
    Args:
        file_path: Path to check
        
    Returns:
        True if the file should be skipped, False otherwise
    """
    # Check if any parent directory is named 'recursos' or 'resources'
    for parent in file_path.parents:
        if parent.name.lower() in ['recursos', 'resources']:
            return True
    return False


class CourseTimeEstimator:
    """Estimates total course time using sampling or real audio duration calculation."""
    
    def __init__(self, api: str, api_key: Optional[str] = None, voice: str = "nova"):
        self.api = api.lower()
        self.api_key = api_key
        self.voice = voice
        self.client = None
        
        # Load .env file if available
        if load_dotenv:
            load_dotenv()
        
        if self.api == "openai":
            self._init_openai()
        elif self.api == "google":
            self._init_google()
        else:
            raise ValueError(f"Unsupported API: {api}. Use 'openai' or 'google'")
    
    def _init_openai(self):
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            self.client = OpenAI(api_key=key)
        else:
            self.client = OpenAI()
    
    def _init_google(self):
        if texttospeech is None:
            raise ImportError("Google Cloud library not installed. Run: pip install google-cloud-texttospeech")
        
        if self.api_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.api_key
        
        self.client = texttospeech.TextToSpeechClient()
    
    def clean_markdown(self, text: str) -> str:
        """Remove markdown formatting and clean text for TTS while preserving paragraphs."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove headers (keep the text, remove the #)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove bullet points but keep the text
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Preserve paragraph structure (double newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove leading/trailing whitespace from each paragraph
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        text = '\n\n'.join(paragraphs)
        
        return text
    
    def collect_all_paragraphs(self, directory: Path) -> List[Tuple[str, str, str]]:
        """
        Collect all paragraphs from markdown files in directory.
        
        Returns:
            List of tuples: (paragraph_text, file_path, section_name)
        """
        paragraphs = []
        
        # Find all markdown files recursively
        files = list(directory.rglob("*.md"))
        
        # Filter out internal files (starting with 00-) and recursos/resources folders
        files = [f for f in files if not f.name.startswith("00-") and not should_skip_path(f)]
        
        print(f"📊 Found {len(files)} lesson files to analyze")
        
        for file_path in sorted(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Clean markdown
                clean_text = self.clean_markdown(text)
                
                if not clean_text:
                    continue
                
                # Split into paragraphs
                file_paragraphs = clean_text.split('\n\n')
                
                # Get section name from parent directory or filename
                # Supports both structures: apostila/01-*.md and scripts da aula/01-*/texto-aula.md
                parent_name = file_path.parent.name
                if parent_name and parent_name[0].isdigit() and "-" in parent_name:
                    section_name = parent_name.split("-", 1)[1].replace("-", " ")
                elif file_path.stem and file_path.stem[0].isdigit() and "-" in file_path.stem:
                    section_name = file_path.stem.split("-", 1)[1].replace("-", " ")
                else:
                    section_name = parent_name
                
                for para in file_paragraphs:
                    if para.strip():
                        paragraphs.append((para.strip(), str(file_path), section_name))
                
            except Exception as e:
                print(f"  ⚠️ Error reading {file_path}: {e}")
        
        print(f"📊 Total paragraphs collected: {len(paragraphs)}")
        return paragraphs
    
    def sample_paragraphs(self, paragraphs: List[Tuple[str, str, str]], n: int = 5) -> List[Tuple[str, str, str]]:
        """Randomly sample n paragraphs from the list."""
        if len(paragraphs) <= n:
            return paragraphs
        
        return random.sample(paragraphs, n)
    
    def generate_sample_audio(self, paragraphs: List[Tuple[str, str, str]], temp_dir: Path) -> List[Path]:
        """Generate audio for sample paragraphs."""
        audio_files = []
        
        print(f"🎙️  Generating audio for {len(paragraphs)} sample paragraphs...")
        
        for i, (para, file_path, section) in enumerate(paragraphs):
            temp_path = temp_dir / f"sample_{i:03d}.mp3"
            
            try:
                if self.api == "openai":
                    self._convert_openai_sample(para, temp_path)
                elif self.api == "google":
                    self._convert_google_sample(para, temp_path)
                
                audio_files.append(temp_path)
                print(f"  ✓ Sample {i+1}/{len(paragraphs)} generated")
            except Exception as e:
                print(f"  ✗ Error generating sample {i+1}: {e}")
        
        return audio_files
    
    def _convert_openai_sample(self, text: str, output_path: Path):
        """Convert a single sample paragraph with OpenAI."""
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice=self.voice,
            input=text,
            speed=0.95
        ) as response:
            response.stream_to_file(output_path)
    
    def _convert_google_sample(self, text: str, output_path: Path):
        """Convert a single sample paragraph with Google."""
        ssml_text = f'<speak><p>{text}</p></speak>'
        ssml_with_tone = f'<speak><prosody rate="0.95" pitch="+10%">{ssml_text[7:-8]}</prosody></speak>'
        
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_with_tone)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",
            name="pt-BR-Standard-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
            pitch=2.0
        )
        
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
    
    def calculate_average_duration(self, audio_files: List[Path]) -> float:
        """Calculate average duration in seconds from audio files."""
        if AudioSegment is None:
            raise ImportError("pydub is required for duration calculation. Run: pip install pydub")
        
        durations = []
        
        for audio_file in audio_files:
            try:
                audio = AudioSegment.from_mp3(str(audio_file))
                duration = len(audio) / 1000.0  # Convert to seconds
                durations.append(duration)
            except Exception as e:
                print(f"  ⚠️ Error reading duration from {audio_file}: {e}")
        
        if not durations:
            return 0.0
        
        avg_duration = sum(durations) / len(durations)
        print(f"📊 Average duration per paragraph: {avg_duration:.2f} seconds")
        
        return avg_duration
    
    def estimate_total_time(self, total_paragraphs: int, avg_duration: float) -> float:
        """Estimate total time in seconds."""
        return total_paragraphs * avg_duration
    
    def calculate_real_duration(self, directory: Path) -> float:
        """Calculate real duration from existing audio files."""
        if AudioSegment is None:
            raise ImportError("pydub is required for duration calculation. Run: pip install pydub")
        
        # Find all audio files for the specified API
        pattern = f"*_{self.api}.mp3"
        audio_files = list(directory.rglob(pattern))
        
        # Filter out files from 00- directories and recursos/resources folders
        audio_files = [f for f in audio_files if not "00-" in str(f) and not should_skip_path(f)]
        
        print(f"📊 Found {len(audio_files)} existing audio files")
        
        total_duration = 0.0
        
        for audio_file in audio_files:
            try:
                audio = AudioSegment.from_mp3(str(audio_file))
                duration = len(audio) / 1000.0  # Convert to seconds
                total_duration += duration
                print(f"  ✓ {audio_file.name}: {duration:.2f}s")
            except Exception as e:
                print(f"  ⚠️ Error reading {audio_file}: {e}")
        
        print(f"📊 Total real duration: {total_duration:.2f} seconds")
        
        return total_duration
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def generate_report(self, data: dict, mode: str, output_path: Path = None):
        """Generate RELATORIO.md following documentation rules."""
        if output_path is None:
            output_path = Path("RELATORIO.md")
        
        # Get current date
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Build methodology section based on mode
        if mode == 'all':
            methodology = """### Modo: Cálculo Completo (--all)

Este modo calcula o tempo real somando a duração de todos os arquivos de áudio já gerados para o curso."""
            sample_info = f"- **Arquivos de Áudio Analisados**: {data['audio_files_count']}"
        else:
            sample_percentage = data.get('sample_percentage', 5.0)
            methodology = f"""### Modo: Amostragem

Este modo utiliza uma amostragem estatística para estimar o tempo total do curso:

1. Coleta todos os parágrafos dos arquivos markdown das aulas (ignorando arquivos iniciados com "00-")
2. Seleciona aleatoriamente {sample_percentage}% dos parágrafos como amostra representativa
3. Gera áudio para cada parágrafo da amostra usando a API {data['api'].upper()}
4. Calcula a duração média por parágrafo
5. Extrapola o tempo total multiplicando a média pelo número total de parágrafos"""
            sample_info = f"- **Porcentagem de Amostragem**: {sample_percentage}%"
        
        # Build report content
        report_content = f"""![visitors](https://visitor-badge.laobi.icu/badge?page_id=smart_contracts_ethereum.course_time_report)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
![Language: Portuguese](https://img.shields.io/badge/Language-Portuguese-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Educa%C3%A7%C3%A3o-brightgreen)

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1a56db,100:10b981&height=220&section=header&text=Estimativa%20de%20Tempo%20do%20Curso&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Smart%20Contracts%20na%20Ethereum&descSize=18&descAlignY=55&descColor=94a3b8" width="100%" alt="Course Time Estimation Header"/>
</p>

## Resumo

Relatório de estimativa de tempo total do curso "Smart Contracts na Ethereum" utilizando {data['api'].upper()} TTS.

## Metodologia

{methodology}

## Estatísticas

- **API Utilizada**: {data['api'].upper()}
- **Voz**: {data['voice']}
- **Total de Parágrafos**: {data['total_paragraphs']}
- **Total de Arquivos de Aula**: {data['total_files']}
{sample_info}
- **Duração Média por Parágrafo**: {data['avg_duration']:.2f} segundos
- **Tempo Total Estimado**: {data['total_time_formatted']} ({data['total_time']:.2f} segundos)
- **Tempo Total em Horas**: {data['total_time_hours']:.2f} horas

## Detalhamento por Seção

"""

        # Add section breakdown if available
        if 'section_breakdown' in data:
            for section, count in data['section_breakdown'].items():
                section_time = count * data['avg_duration']
                report_content += f"- **{section}**: {count} parágrafos (~{self.format_duration(section_time)})\n"
        
        report_content += f"""
## Conclusão

O curso "Smart Contracts na Ethereum" possui um tempo total estimado de **{data['total_time_formatted']}** ({data['total_time_hours']:.2f} horas).

Esta estimativa considera apenas o conteúdo das aulas principais, excluindo arquivos de apresentação e documentação interna (iniciados com "00-").

## Observações

- A estimativa baseia-se na velocidade de fala configurada (0.95x para delivery natural)
- Parágrafos muito curtos ou muito longos podem afetar a precisão da estimativa
- O tempo real pode variar dependendo do conteúdo técnico e complexidade dos exemplos

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:10b981,50:1a56db,100:0f172a&height=120&section=footer" width="100%" alt="Footer"/>
</p>

---
**Resumo:** Relatório de estimativa de tempo total do curso Smart Contracts na Ethereum utilizando {data['api'].upper()} TTS no modo {mode}.
**Data de Criação:** {today}
**Autor:** Carlos Delfino
**Versão:** 1.0
**Última Atualização:** {today}
**Atualizado por:** Carlos Delfino
**Histórico de Alterações:**
- {today} - Criado por Carlos Delfino - Versão 1.0
"""
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ Report generated: {output_path}")


class TextToAudioConverter:
    def __init__(self, api: str, api_key: Optional[str] = None, voice: str = "cedar", use_threads: bool = True, max_threads: int = 4, language: str = "pt-br", cover_time: float = 5.0):
        self.api = api.lower()
        self.api_key = api_key
        self.voice = voice
        self.use_threads = use_threads
        self.max_threads = max_threads
        self.language = language
        self.cover_time = cover_time
        self.client = None
        self.progress_file = Path(f".progress_{self.api}.txt")
        self.processed_files: Set[str] = set()
        
        # Generate safety identifier for OpenAI API (hashed project name)
        self.safety_identifier = self._generate_safety_identifier()
        
        # Load .env file if available
        if load_dotenv:
            load_dotenv()
        
        # Load progress file if it exists
        self._load_progress()
        
        # Initialize processing log
        if ProcessingLog:
            self.processing_log = ProcessingLog()
        else:
            self.processing_log = None
        
        if self.api == "openai":
            self._init_openai()
        elif self.api == "google":
            self._init_google()
        else:
            raise ValueError(f"Unsupported API: {api}. Use 'openai' or 'google'")
    
    def _load_progress(self):
        """Load previously processed files from progress file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.processed_files = set(line.strip() for line in f if line.strip())
    
    def _save_progress(self, file_path: str):
        """Save a successfully processed file to progress file."""
        with open(self.progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n")
        self.processed_files.add(file_path)
    
    def _clear_progress(self):
        """Clear progress file (used with --force)."""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.processed_files.clear()
    
    def _check_dependency_timestamp(self, source_file: Path, target_file: Path, force: bool = False) -> bool:
        """
        Check if target file needs to be regenerated based on timestamps.
        
        Args:
            source_file: The source file (e.g., .md)
            target_file: The target file (e.g., .mp3, .pdf, .mp4)
            force: If True, always return True (force regeneration)
            
        Returns:
            True if target needs to be regenerated, False otherwise
        """
        if force:
            return True
        
        # Check if file exists and is not empty
        if not target_file.exists() or target_file.stat().st_size == 0:
            return True
        
        # Check if source is newer than target
        source_mtime = source_file.stat().st_mtime
        target_mtime = target_file.stat().st_mtime
        
        return source_mtime > target_mtime
    
    def _generate_safety_identifier(self) -> str:
        """Generate a stable safety identifier for OpenAI API calls."""
        if hashlib is None:
            return "smart_contracts_ethereum_user"
        
        # Hash the project name to create a stable, non-personal identifier
        project_name = "Smart Contracts na Ethereum"
        hashed = hashlib.sha256(project_name.encode('utf-8')).hexdigest()
        # Use first 16 characters for a shorter identifier
        return f"sc_eth_{hashed[:16]}"
    
    def _init_openai(self):
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        # Use provided key, environment variable, or default
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            self.client = OpenAI(api_key=key)
        else:
            self.client = OpenAI()
    
    def _init_google(self):
        if texttospeech is None:
            raise ImportError("Google Cloud library not installed. Run: pip install google-cloud-texttospeech")
        
        # Use provided key or environment variable
        if self.api_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.api_key
        
        self.client = texttospeech.TextToSpeechClient()
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using OpenAI API."""
        if self.api != "openai" or OpenAI is None:
            raise ImportError("Translation requires OpenAI API. Use --api openai")
        
        # Language code mapping
        language_names = {
            "pt-br": "Portuguese (Brazilian)",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German"
        }
        
        target_lang_name = language_names.get(target_language, target_language)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following text to {target_lang_name}. Preserve markdown formatting, code blocks, and technical terms. Keep the structure and formatting exactly as in the original."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3
            )
            
            translated_text = response.choices[0].message.content
            return translated_text
        except Exception as e:
            print(f"✗ Error translating text: {e}")
            raise
    
    def get_language_suffix(self) -> str:
        """Get file suffix based on language (empty for pt-br, language code for others)."""
        return "" if self.language == "pt-br" else f"_{self.language}"
    
    def get_translated_file_path(self, original_path: Path) -> Path:
        """Get the path for translated file based on original path and language."""
        if self.language == "pt-br":
            return original_path
        
        # Add language suffix before extension
        stem = original_path.stem
        suffix = original_path.suffix
        new_stem = f"{stem}{self.get_language_suffix()}"
        return original_path.with_name(f"{new_stem}{suffix}")
    
    def translate_file(self, input_path: Path, force: bool = False) -> Path:
        """Translate a markdown file to the target language."""
        if self.language == "pt-br":
            return input_path
        
        translated_path = self.get_translated_file_path(input_path)
        
        # Check if translation already exists and is up to date
        if not force and translated_path.exists():
            if not self._check_dependency_timestamp(input_path, translated_path, force=False):
                print(f"⊘ Reusing existing translation: {translated_path}")
                return translated_path
        
        print(f"🌐 Translating {input_path.name} to {self.language}...")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            translated_text = self.translate_text(original_text, self.language)
            
            # Write translated file
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            print(f"✓ Translation saved to: {translated_path}")
            return translated_path
        except Exception as e:
            print(f"✗ Error translating file: {e}")
            raise
    
    def clean_markdown(self, text: str, input_path: Path = None) -> str:
        """Remove markdown formatting and clean text for TTS while preserving paragraphs."""
        # Extract images before removing them
        images = []
        if input_path:
            images = self._extract_images(text, input_path)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove headers (keep the text, remove the #)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Remove images (but keep track of positions for timing)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove bullet points but keep the text
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Preserve paragraph structure (double newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove leading/trailing whitespace from each paragraph
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        text = '\n\n'.join(paragraphs)
        
        return text

    def _prepare_tts_text(self, text: str) -> str:
        """Normalize text immediately before TTS generation."""
        text = html.unescape(str(text or ""))
        text = re.sub(r'[\u2010-\u2015\u2212-]+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _normalize_metadata_text(self, text: str) -> str:
        """Make section/lesson metadata readable for cover narration."""
        if path_part_to_title:
            text = path_part_to_title(str(text or ""))
        else:
            text = re.sub(r'^\d{2}(?:\.\d+)*(?:[-_\s]+|$)', '', str(text or ""))
            text = re.sub(r'[-_]+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _format_cover_title_text(self, text: str) -> str:
        """Format slug-derived title text with Portuguese articles and accents."""
        text = self._normalize_metadata_text(text)
        if not text:
            return ""

        small_words = {"a", "as", "ao", "aos", "da", "das", "de", "do", "dos", "e", "em", "na", "nas", "no", "nos", "o", "os", "para", "por", "que"}
        replacements = {
            "introducao": "Introdução",
            "sao": "são",
            "basico": "Básico",
            "basicos": "Básicos",
            "pratica": "Prática",
            "praticas": "Práticas",
            "configuracao": "Configuração",
            "integracao": "Integração",
            "funcao": "Função",
            "funcoes": "Funções",
            "heranca": "Herança",
            "seguranca": "Segurança",
        }

        words = []
        for index, word in enumerate(text.split()):
            normalized = unicodedata.normalize("NFKD", word)
            normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
            key = normalized.lower()

            if word.isupper() and any(char.isalpha() for char in word):
                formatted = word
            elif key in replacements:
                formatted = replacements[key]
            elif key in small_words and index > 0:
                formatted = key
            else:
                formatted = word[:1].upper() + word[1:].lower()
            words.append(formatted)

        title = " ".join(words)
        title = re.sub(r'\bIntrodução as\b', 'Introdução às', title)
        title = re.sub(r'\bIntrodução aos\b', 'Introdução aos', title)
        title = re.sub(r'\bIntrodução (?!a\b|ao\b|aos\b|à\b|às\b|de\b)', 'Introdução a ', title)
        title = re.sub(r'\bSintaxe Tipos Dados\b', 'Sintaxe e Tipos de Dados', title)
        title = re.sub(r'\bTipos Dados\b', 'Tipos de Dados', title)
        title = re.sub(r'\bTestnets Wallets\b', 'Testnets e Wallets', title)
        title = re.sub(r'\bEventos Herança\b', 'Eventos e Herança', title)
        title = re.sub(r'\bFunções Modificadores\b', 'Funções e Modificadores', title)
        return title

    def _ordinal_word(self, value: Optional[int], gender: str) -> str:
        """Return a Portuguese ordinal word for common module/lesson numbers."""
        if not value:
            return ""

        masculine = {
            1: "Primeiro",
            2: "Segundo",
            3: "Terceiro",
            4: "Quarto",
            5: "Quinto",
            6: "Sexto",
            7: "Sétimo",
            8: "Oitavo",
            9: "Nono",
            10: "Décimo",
            11: "Décimo Primeiro",
            12: "Décimo Segundo",
            13: "Décimo Terceiro",
            14: "Décimo Quarto",
            15: "Décimo Quinto",
            16: "Décimo Sexto",
            17: "Décimo Sétimo",
            18: "Décimo Oitavo",
            19: "Décimo Nono",
            20: "Vigésimo",
        }
        feminine = {
            1: "Primeira",
            2: "Segunda",
            3: "Terceira",
            4: "Quarta",
            5: "Quinta",
            6: "Sexta",
            7: "Sétima",
            8: "Oitava",
            9: "Nona",
            10: "Décima",
            11: "Décima Primeira",
            12: "Décima Segunda",
            13: "Décima Terceira",
            14: "Décima Quarta",
            15: "Décima Quinta",
            16: "Décima Sexta",
            17: "Décima Sétima",
            18: "Décima Oitava",
            19: "Décima Nona",
            20: "Vigésima",
        }
        words = feminine if gender == "f" else masculine
        if value in words:
            return words[value]
        suffix = "ª" if gender == "f" else "º"
        return f"{value}{suffix}"

    def _semantic_key(self, text: str) -> str:
        """Normalize text for duplicate section/lesson checks."""
        text = unicodedata.normalize("NFKD", text or "")
        text = ''.join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r'[^a-z0-9]+', '', text.lower())
        return text

    def _numbered_prefix_value(self, name: str) -> Optional[int]:
        if numbered_prefix_value:
            return numbered_prefix_value(name)
        match = re.match(r'^(\d{2})(?:\.\d+)*(?:[-_\s]+|$)', name or "")
        return int(match.group(1)) if match else None

    def _numbered_prefix_parts(self, name: str) -> Tuple[int, ...]:
        if numbered_prefix_parts:
            return numbered_prefix_parts(name)
        match = re.match(r'^(\d{2}(?:\.\d+)*)(?:[-_\s]+|$)', name or "")
        if not match:
            return ()
        return tuple(int(part) for part in match.group(1).split("."))

    def _has_numbered_prefix(self, name: str) -> bool:
        if has_numbered_prefix:
            return has_numbered_prefix(name)
        return bool(re.match(r'^\d{2}(?:\.\d+)*(?:[-_\s]+|$)', name or ""))

    def _lesson_position_in_module(self, module_path: Optional[Path], lesson_path: Optional[Path]) -> Optional[int]:
        """Return the one-based lesson position among numbered siblings in a module."""
        if not module_path or not lesson_path or not module_path.exists() or not module_path.is_dir():
            return None

        lesson_candidates = []
        for child in module_path.iterdir():
            name = child.stem if child.is_file() else child.name
            if child.name.startswith("00-") or child.name.startswith("00_"):
                continue
            if child.is_dir() or child.suffix.lower() == ".md":
                if self._has_numbered_prefix(name):
                    lesson_candidates.append(child)

        lesson_candidates.sort(key=lambda item: (self._numbered_prefix_parts(item.stem if item.is_file() else item.name), item.name.lower()))
        current = lesson_path.resolve()
        for index, candidate in enumerate(lesson_candidates, start=1):
            if candidate.resolve() == current:
                return index
        return None

    def _cover_path_context(self, input_path: Optional[Path]) -> dict:
        """Extract module and lesson numbering from the path used to build cover narration."""
        context = {
            "module_name": None,
            "module_number": None,
            "lesson_name": None,
            "lesson_number": None,
            "lesson_position": None,
        }
        if not input_path:
            return context

        input_path = Path(input_path)
        stem_is_generic = input_path.stem.lower() in {"texto-aula", "aula", "script", "roteiro"}

        module_path = None
        lesson_path = None
        lesson_name = None

        if stem_is_generic and self._has_numbered_prefix(input_path.parent.name):
            lesson_path = input_path.parent
            lesson_name = input_path.parent.name
            if self._has_numbered_prefix(input_path.parent.parent.name):
                module_path = input_path.parent.parent
        elif self._has_numbered_prefix(input_path.stem):
            lesson_path = input_path
            lesson_name = input_path.stem
            if self._has_numbered_prefix(input_path.parent.name):
                module_path = input_path.parent
        elif self._has_numbered_prefix(input_path.parent.name):
            lesson_path = input_path.parent
            lesson_name = input_path.parent.name
            if self._has_numbered_prefix(input_path.parent.parent.name):
                module_path = input_path.parent.parent

        if module_path:
            context["module_name"] = module_path.name
            context["module_number"] = self._numbered_prefix_value(module_path.name)

        if lesson_name:
            context["lesson_name"] = lesson_name
            context["lesson_number"] = self._numbered_prefix_value(lesson_name)

        if module_path and lesson_path:
            context["lesson_position"] = self._lesson_position_in_module(module_path, lesson_path)

        return context

    def _cover_narration_text(self, section: str = None, lesson: str = None, input_path: Path = None) -> str:
        """Build cover narration from module/section and lesson names."""
        path_context = self._cover_path_context(input_path)
        module_title = self._format_cover_title_text(path_context.get("module_name") or section)
        lesson_title = self._format_cover_title_text(path_context.get("lesson_name") or lesson)
        module_number = path_context.get("module_number")
        lesson_number = path_context.get("lesson_number")
        lesson_position = path_context.get("lesson_position")

        if module_number:
            module_ordinal = self._ordinal_word(module_number, "m")
            if self._semantic_key(module_title) and self._semantic_key(module_title) == self._semantic_key(lesson_title):
                result = f"{module_ordinal} Módulo, Aula de {lesson_title}"
                return result if result.endswith('.') else f"{result}."

            lesson_ordinal = self._ordinal_word(lesson_position, "f") if lesson_position else ""
            parts = [f"{module_ordinal} Módulo"]
            if module_title:
                parts.append(module_title)
            if lesson_ordinal:
                parts.append(f"{lesson_ordinal} Aula")
            elif lesson_title:
                parts.append("Aula")
            if lesson_title:
                parts.append(lesson_title)
            result = ", ".join(part for part in parts if part)
            return result if result.endswith('.') else f"{result}."

        if lesson_number:
            lesson_ordinal = self._ordinal_word(lesson_number, "f")
            if lesson_ordinal and lesson_title:
                result = f"{lesson_ordinal} Aula, {lesson_title}"
                return result if result.endswith('.') else f"{result}."

        raw_parts = [section, lesson]
        parts = []
        seen = set()
        for raw_part in raw_parts:
            part = self._format_cover_title_text(raw_part)
            if not part or part.lower() == "general":
                continue
            comparable = self._semantic_key(part)
            if comparable in seen:
                continue
            seen.add(comparable)
            parts.append(part)
        result = ". ".join(parts)
        # Add period at the end if not present for proper TTS intonation
        if result and not result.endswith('.'):
            result += "."
        return result

    def _slides_with_cover(self, slides: List[dict], include_cover: bool, section: str = None, lesson: str = None, input_path: Path = None) -> List[dict]:
        """Prepend a synthetic cover slide when the actual slide deck starts with one."""
        if not include_cover:
            return slides
        if slides and (slides[0].get("is_cover") or (not slides[0].get("content") and not slides[0].get("bullets"))):
            return slides

        section_text = self._format_cover_title_text(section)
        lesson_text = self._format_cover_title_text(lesson)
        title_parts = [part for part in [section_text, lesson_text] if part and part.lower() != "general"]
        if len(title_parts) == 2 and self._semantic_key(title_parts[0]) == self._semantic_key(title_parts[1]):
            title_parts = title_parts[:1]
        title = "\n".join(title_parts) or lesson_text or section_text or "Aula"
        narration = self._cover_narration_text(section, lesson, input_path) or title.replace("\n", ". ")

        cover_slide = {
            "title": title,
            "level": 1,
            "content": [],
            "bullets": [],
            "is_cover": True,
            "narration": narration,
        }
        return [cover_slide] + slides

    def _deck_has_cover(self, slide_generator, input_path: Path, lang_suffix: str) -> bool:
        """Detect whether the slide deck has or will receive a cover slide."""
        odp_path = input_path.parent / f"{input_path.stem}{lang_suffix}.odp"
        if odp_path.exists() and hasattr(slide_generator, "has_cover_slide"):
            if slide_generator.has_cover_slide(odp_path):
                print(f"  → Detected cover slide in ODP: {odp_path.name}")
                return True
            if not slide_generator.cover_template:
                return False
        return bool(slide_generator.cover_template)
    
    def _extract_images(self, text: str, input_path: Path) -> List[dict]:
        """Extract images from markdown and save timing information."""
        import json
        
        images = []
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.finditer(pattern, text)
        
        # Calculate character positions for timing estimation
        total_chars = len(text)
        
        for match in matches:
            alt_text = match.group(1)
            image_path = match.group(2)
            position = match.start()
            
            # Estimate relative position (0-1) for timing
            relative_position = position / total_chars if total_chars > 0 else 0
            
            images.append({
                'alt_text': alt_text,
                'path': image_path,
                'position': relative_position
            })
        
        # Save image timing file
        if images and input_path:
            timing_file = input_path.parent / f"{input_path.stem}_images.json"
            with open(timing_file, 'w', encoding='utf-8') as f:
                json.dump(images, f, indent=2, ensure_ascii=False)
            print(f"  Image timing saved to: {timing_file}")
        
        return images
    
    def text_to_ssml(self, text: str) -> str:
        """Convert text with paragraphs to SSML with pauses between paragraphs."""
        paragraphs = text.split('\n\n')
        ssml_paragraphs = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                # Escape special XML characters
                para = para.replace('&', '&amp;')
                para = para.replace('<', '&lt;')
                para = para.replace('>', '&gt;')
                para = para.replace('"', '&quot;')
                para = para.replace("'", '&apos;')
                
                ssml_paragraphs.append(f"<p>{para}</p>")
        
        # Join paragraphs with pauses
        ssml = '<speak>' + '<break time="500ms"/>'.join(ssml_paragraphs) + '</speak>'
        return ssml
    
    def _chunk_text(self, text: str, max_chars: int = 4000) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def convert_with_openai(self, text: str, output_path: str):
        """Convert text to audio using OpenAI API with enthusiastic teacher tone."""
        text = self._prepare_tts_text(text)
        # Add natural pauses between paragraphs for better rhythm
        paragraphs = text.split('\n\n')
        text_with_pauses = '... '.join([p.strip() for p in paragraphs if p.strip()])
        
        # Check if text needs chunking
        if len(text_with_pauses) > 4000:
            if AudioSegment is None:
                raise ImportError("pydub is required for long texts. Run: pip install pydub")
            
            chunks = self._chunk_text(text_with_pauses)
            print(f"  Text too long ({len(text_with_pauses)} chars), splitting into {len(chunks)} chunks")
            
            # Convert each chunk to audio
            audio_segments = []
            
            def process_chunk(i, chunk, temp_dir):
                """Process a single chunk."""
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
                temp_path = temp_dir / f"chunk_{i}.mp3"
                
                with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1-hd",
                    voice=self.voice,
                    input=chunk,
                    speed=0.95
                ) as response:
                    response.stream_to_file(temp_path)
                return AudioSegment.from_mp3(temp_path), i
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                if self.use_threads and len(chunks) > 1:
                    max_workers = min(self.max_threads, len(chunks))
                    print(f"  Processing chunks with {max_workers} threads")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(process_chunk, i, chunk, temp_dir): i
                            for i, chunk in enumerate(chunks)
                        }
                        
                        results = {}
                        for future in as_completed(futures):
                            chunk_idx = futures[future]
                            try:
                                segment, idx = future.result()
                                results[idx] = segment
                            except Exception as e:
                                print(f"  ✗ Error processing chunk {chunk_idx+1}: {e}")
                                raise
                        
                        # Sort results by chunk index
                        for i in sorted(results.keys()):
                            audio_segments.append(results[i])
                else:
                    for i, chunk in enumerate(chunks):
                        segment, _ = process_chunk(i, chunk, temp_dir)
                        audio_segments.append(segment)
                
                # Combine all chunks
                combined = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined += segment
                
                combined.export(output_path, format="mp3")
        else:
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1-hd",  # Higher quality model for better expression
                voice=self.voice,
                input=text_with_pauses,
                speed=0.95  # Slightly slower for more natural, engaging delivery
            ) as response:
                response.stream_to_file(output_path)
        
        print(f"✓ Audio saved to: {output_path}")

    def convert_slide_segments(self, slides: List[dict], output_path: Path, force: bool = False, input_path: Path = None, section: str = None, lesson: str = None, cover_pause: float = 2.0) -> List[float]:
        """Generate one audio segment per slide and concatenate them.

        Returns the real duration of each segment in seconds. Slide video generation
        uses these values so page changes happen at the end of each narrated block.
        
        Args:
            slides: List of slide dictionaries
            output_path: Path for the combined audio file
            force: Force regeneration of all segments
            input_path: Path to input file for hash tracking
            section: Section name for cover slide audio
            lesson: Lesson name for cover slide audio
            cover_pause: Minimum total duration for the cover segment, in seconds
        """
        if AudioSegment is None:
            raise ImportError("pydub is required for synchronized slide audio. Run: pip install pydub")

        timing_path = output_path.with_name(f"{output_path.stem}_slide_timings.txt")
        manifest_path = output_path.with_name(f"{output_path.stem}_slide_segments.json")

        print(f"  Preparing synchronized audio in {len(slides)} markdown/cover audio segments...")

        audio_dir = output_path.parent / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_path.stem

        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f).get("segments", {})
            except Exception as e:
                print(f"  ⚠ Could not read segment manifest: {e}")
                manifest = {}

        cover_min_ms = max(0, int(float(cover_pause or 0) * 1000))
        cover_intro_silence_ms = 500
        segment_specs = []
        for zero_index, slide in enumerate(slides):
            index = zero_index + 1
            is_cover = bool(slide.get("is_cover")) or (
                index == 1
                and not slide.get("content")
                and not slide.get("bullets")
                and bool(slide.get("title"))
            )
            if is_cover:
                segment_text = (
                    slide.get("narration")
                    or self._cover_narration_text(section, lesson, input_path)
                    or slide.get("title", "")
                )
            else:
                segment_text = self._slide_to_narration_text(slide)
                if not segment_text:
                    segment_text = slide.get("title", "")

            segment_text = self._prepare_tts_text(segment_text)
            hash_payload = {
                "api": self.api,
                "voice": self.voice,
                "text": segment_text,
                "is_cover": is_cover,
                "cover_min_ms": cover_min_ms if is_cover else 0,
                "cover_intro_silence_ms": cover_intro_silence_ms if is_cover else 0,
            }
            segment_hash = hashlib.sha256(
                json.dumps(hash_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            segment_specs.append({
                "index": index,
                "key": f"{index:03d}",
                "slide": slide,
                "text": segment_text,
                "is_cover": is_cover,
                "hash": segment_hash,
                "intro_silence_ms": cover_intro_silence_ms if is_cover else 0,
                "path": audio_dir / f"{prefix}_slide_{index:03d}.mp3",
            })

        for spec in segment_specs:
            manifest_entry = manifest.get(spec["key"], {})
            manifest_file = manifest_entry.get("file") if isinstance(manifest_entry, dict) else None
            manifest_is_cover = manifest_entry.get("is_cover") if isinstance(manifest_entry, dict) else None
            same_segment_role = manifest_is_cover is None or bool(manifest_is_cover) == spec["is_cover"]
            if manifest_file and same_segment_role:
                manifest_segment_path = audio_dir / Path(manifest_file).name
                # Only reuse from manifest if the file actually exists
                if manifest_segment_path.exists() and manifest_segment_path.stat().st_size > 0:
                    spec["path"] = manifest_segment_path

        if segment_specs and not segment_specs[0]["is_cover"] and not segment_specs[0]["path"].exists():
            legacy_paths = [
                audio_dir / f"{prefix}_slide_{spec['index']:03d}.mp3"
                for spec in segment_specs
            ]
            # Check if legacy files exist and are not empty
            if legacy_paths[0].exists() and legacy_paths[0].stat().st_size > 0:
                print("    → Detected legacy slide segment numbering starting at 001")
                for spec, legacy_path in zip(segment_specs, legacy_paths):
                    # Only use legacy path if file exists and is not empty
                    if legacy_path.exists() and legacy_path.stat().st_size > 0:
                        spec["path"] = legacy_path

        if not segment_specs:
            raise ValueError("No slide audio segments were specified")

        has_cover_segment = bool(segment_specs and segment_specs[0]["is_cover"])
        markdown_segment_count = len(segment_specs) - (1 if has_cover_segment else 0)
        print(
            f"  Audio segment plan: {markdown_segment_count} markdown segment(s)"
            f" + {1 if has_cover_segment else 0} cover segment(s)"
            f" = {len(segment_specs)} total"
        )
        if has_cover_segment:
            print(f"    Cover TTS text: {segment_specs[0]['text']}")

        # If old audio was generated without a manifest and a cover is now the
        # first segment, every existing numbered segment may be shifted by one.
        # Reusing by filename in that state can attach the wrong narration to
        # the wrong markdown paragraph.
        manifest_missing = not bool(manifest)
        regenerate_all_for_legacy_cover_shift = has_cover_segment and manifest_missing
        if regenerate_all_for_legacy_cover_shift:
            print("    → Cover segment with no manifest detected; regenerating all segments to avoid shifted audio")

        generated_segments = set()
        adjusted_segments = set()
        segments_to_generate = []

        for spec in segment_specs:
            manifest_entry = manifest.get(spec["key"], {})
            manifest_hash = manifest_entry.get("hash") if isinstance(manifest_entry, dict) else None
            segment_path = spec["path"]
            # Check if file exists and is not empty
            file_exists = segment_path.exists() and segment_path.stat().st_size > 0
            should_generate = force or not file_exists
            if regenerate_all_for_legacy_cover_shift:
                should_generate = True
            if spec["is_cover"] and not manifest_hash:
                should_generate = True
            if manifest_hash and manifest_hash != spec["hash"]:
                should_generate = True

            if should_generate:
                reason = "force" if force else "missing"
                if regenerate_all_for_legacy_cover_shift:
                    reason = "legacy cover shift"
                if spec["is_cover"] and not manifest_hash and segment_path.exists():
                    reason = "cover narration changed"
                if manifest_hash and manifest_hash != spec["hash"]:
                    reason = "text changed"
                print(f"    → Segment {spec['index']:03d} will be generated ({reason})")
                segments_to_generate.append(spec)
                continue

            try:
                existing_segment = AudioSegment.from_mp3(str(segment_path))
                if len(existing_segment) <= 0:
                    raise ValueError("empty segment")
                if spec["is_cover"] and cover_min_ms and len(existing_segment) < cover_min_ms:
                    missing_ms = cover_min_ms - len(existing_segment)
                    existing_segment += AudioSegment.silent(duration=missing_ms)
                    existing_segment.export(str(segment_path), format="mp3")
                    adjusted_segments.add(spec["key"])
                    print(f"    ✓ Extended cover segment {spec['index']:03d} to {cover_min_ms / 1000:.1f}s")
                else:
                    print(f"    ⊘ Reusing existing segment {spec['index']:03d}")
            except Exception as e:
                print(f"    ⚠ Segment {spec['index']:03d} invalid, regenerating: {e}")
                segments_to_generate.append(spec)

        def render_segment(spec: dict, temp_dir: Path):
            """Render one segment to its final audio file."""
            temp_path = temp_dir / f"slide_audio_{spec['index']:03d}.mp3"
            print(f"    Audio segment {spec['index']:03d}/{len(segment_specs)}")
            print(f"    TTS text segment {spec['index']:03d}: {spec['text']}")

            if spec["text"]:
                if self.api == "openai":
                    self._convert_openai_segment(spec["text"], temp_path)
                elif self.api == "google":
                    self.convert_with_google(spec["text"], str(temp_path))
                else:
                    raise ValueError(f"Unsupported API: {self.api}")
                segment = AudioSegment.from_file(temp_path)
            else:
                default_ms = cover_min_ms if spec["is_cover"] and cover_min_ms else 100
                segment = AudioSegment.silent(duration=default_ms)

            if spec["intro_silence_ms"]:
                segment = AudioSegment.silent(duration=spec["intro_silence_ms"]) + segment

            if spec["is_cover"]:
                if cover_min_ms and len(segment) < cover_min_ms:
                    segment += AudioSegment.silent(duration=cover_min_ms - len(segment))
            else:
                segment += AudioSegment.silent(duration=100)

            segment.export(str(spec["path"]), format="mp3")
            return spec["key"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            if self.use_threads and len(segments_to_generate) > 1:
                max_workers = min(self.max_threads, len(segments_to_generate))
                print(f"    Processing {len(segments_to_generate)} slide segments with {max_workers} threads")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(render_segment, spec, temp_dir): spec
                        for spec in segments_to_generate
                    }
                    for future in as_completed(futures):
                        spec = futures[future]
                        try:
                            generated_segments.add(future.result())
                        except Exception as e:
                            print(f"    ✗ Error processing slide {spec['index']:03d}: {e}")
                            raise
            else:
                for spec in segments_to_generate:
                    generated_segments.add(render_segment(spec, temp_dir))

        audio_segments = []
        segment_durations = []
        for spec in segment_specs:
            segment = AudioSegment.from_mp3(str(spec["path"]))
            audio_segments.append(segment)
            segment_durations.append(len(segment) / 1000.0)

        existing_timings = []
        if timing_path.exists():
            try:
                with open(timing_path, "r", encoding="utf-8") as f:
                    existing_timings = [float(line.strip()) for line in f if line.strip()]
            except Exception:
                existing_timings = []

        timings_changed = (
            len(existing_timings) != len(segment_durations)
            or any(abs(old - new) > 0.05 for old, new in zip(existing_timings, segment_durations))
        )
        # Check if output file exists and is not empty
        output_exists = output_path.exists() and output_path.stat().st_size > 0
        output_mtime = output_path.stat().st_mtime if output_exists else 0
        segment_newer_than_output = any(spec["path"].stat().st_mtime > output_mtime for spec in segment_specs)
        rebuild_final = (
            force
            or not output_exists
            or not timing_path.exists()
            or bool(generated_segments)
            or bool(adjusted_segments)
            or segment_newer_than_output
            or timings_changed
        )

        if rebuild_final:
            combined = audio_segments[0]
            for segment in audio_segments[1:]:
                combined += segment
            combined.export(output_path, format="mp3")

            with open(timing_path, "w", encoding="utf-8") as f:
                for duration in segment_durations:
                    f.write(f"{duration:.3f}\n")

            print(f"✓ Synchronized slide audio saved to: {output_path}")
            print(f"✓ Slide timings saved to: {timing_path}")
        else:
            print(f"⊘ Reusing synchronized slide audio: {output_path}")

        manifest_data = {
            "version": 1,
            "segments": {
                spec["key"]: {
                    "hash": spec["hash"],
                    "file": spec["path"].name,
                    "duration": segment_durations[i],
                    "is_cover": spec["is_cover"],
                }
                for i, spec in enumerate(segment_specs)
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Individual segments available in: {audio_dir}")
        
        # Update hash file if input_path was provided
        if input_path and HashManager:
            hash_manager = HashManager(input_path)
            slides_hash = hash_manager.calculate_slides_hash(slides)
            hash_manager.update_hashes("", slides_hash)
            print(f"✓ Hash file updated")
        
        return segment_durations

    def _convert_openai_segment(self, text: str, output_path: Path):
        """Convert a single slide narration segment with OpenAI."""
        text = self._prepare_tts_text(text)
        chunks = self._chunk_text(text, max_chars=3900)
        if len(chunks) == 1:
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1-hd",
                voice=self.voice,
                input=chunks[0],
                speed=0.95
            ) as response:
                response.stream_to_file(output_path)
            return

        if AudioSegment is None:
            raise ImportError("pydub is required for long slide segments. Run: pip install pydub")

        segment_parts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            for i, chunk in enumerate(chunks):
                chunk_path = temp_dir / f"chunk_{i:03d}.mp3"
                with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1-hd",
                    voice=self.voice,
                    input=chunk,
                    speed=0.95
                ) as response:
                    response.stream_to_file(chunk_path)
                segment_parts.append(AudioSegment.from_file(chunk_path))

            combined = segment_parts[0]
            for part in segment_parts[1:]:
                combined += part
            combined.export(output_path, format="mp3")

    def _slide_to_narration_text(self, slide: dict) -> str:
        """Build narration text from the slide body, avoiding generated titles."""
        parts = []
        parts.extend(slide.get("content", []))
        parts.extend(slide.get("bullets", []))
        return "\n\n".join(part.strip() for part in parts if part and part.strip())
    
    def convert_file_incremental(self, input_path: Path, clean_text: str, output_path: Path, force: bool = False):
        """Convert file using incremental generation based on paragraph hashes."""
        if HashManager is None or AudioFragmentManager is None:
            raise ImportError("hash_manager module not available")
        
        print(f"🔍 Checking for changes in: {input_path.name}")
        
        # Generate cover narration as a separate segment (not prepended to text)
        metadata = parse_course_metadata(input_path)
        cover_narration = self._cover_narration_text(metadata.get("section"), metadata.get("lesson"), input_path)
        if cover_narration and not cover_narration.endswith('.'):
            cover_narration += "."
        
        # Use the same prefix as convert_slide_segments: output_path.stem
        # This ensures incremental generation reuses the same files (e.g. "texto-aula_openai_slide_002.mp3")
        prefix = output_path.stem
        
        hash_manager = HashManager(input_path, prefix=prefix)
        fragment_manager = AudioFragmentManager(hash_manager.audio_dir)
        
        # Get changed paragraphs (using clean_text WITHOUT cover narration prepended)
        changed_indices, paragraphs = hash_manager.get_changed_paragraphs(clean_text)
        
        # Check if cover segment needs to be (re)generated
        cover_audio_path = hash_manager.get_cover_audio_path()
        cover_needs_generation = False
        if cover_narration:
            if force or not cover_audio_path.exists() or cover_audio_path.stat().st_size == 0:
                cover_needs_generation = True
        
        # Check if any audio fragments are missing (always check, regardless of changed_indices)
        missing_fragments = []
        for idx in range(len(paragraphs)):
            audio_path = hash_manager.get_audio_path(idx)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                missing_fragments.append(idx)
        
        if missing_fragments:
            print(f"  ⚠️ {len(missing_fragments)} missing audio fragments detected, regenerating: {missing_fragments}")
            for idx in missing_fragments:
                if idx not in changed_indices:
                    changed_indices.append(idx)
        
        # If nothing to do, just combine existing fragments
        if not changed_indices and not force and not cover_needs_generation:
            print(f"  ✓ No changes detected, combining existing fragments...")
            fragment_manager.combine_fragments(output_path, len(paragraphs), prefix=prefix, cover_path=cover_audio_path if cover_narration else None)
            hash_manager.clean_old_audio(len(paragraphs))
            return
        
        if force:
            print(f"  Force mode: regenerating all {len(paragraphs)} paragraphs")
            changed_indices = list(range(len(paragraphs)))
        else:
            print(f"  Found {len(changed_indices)} changed paragraphs: {changed_indices}")
        
        # Generate cover narration segment
        if cover_needs_generation:
            cover_min_ms = max(5000, int(self.cover_time * 1000))
            print(f"  Generating cover narration segment (min {cover_min_ms/1000:.1f}s)...")
            print(f"    Cover TTS text: {cover_narration}")
            try:
                if self.api == "openai":
                    self._convert_openai_segment(cover_narration, cover_audio_path)
                elif self.api == "google":
                    self.convert_with_google(cover_narration, str(cover_audio_path))
                
                # Ensure cover segment meets minimum duration
                if AudioSegment is not None:
                    cover_seg = AudioSegment.from_mp3(str(cover_audio_path))
                    if len(cover_seg) < cover_min_ms:
                        cover_seg += AudioSegment.silent(duration=cover_min_ms - len(cover_seg))
                        cover_seg.export(str(cover_audio_path), format="mp3")
                        print(f"    ✓ Cover segment extended to {cover_min_ms/1000:.1f}s")
                    else:
                        print(f"    ✓ Cover segment duration: {len(cover_seg)/1000:.1f}s")
            except Exception as e:
                print(f"    ✗ Error generating cover segment: {e}")
                if AudioSegment is not None:
                    # Create silent cover as fallback
                    silent = AudioSegment.silent(duration=cover_min_ms)
                    silent.export(str(cover_audio_path), format="mp3")
                    print(f"    ✓ Created silent cover segment ({cover_min_ms/1000:.1f}s)")
        
        # Generate audio for changed paragraphs
        for idx in changed_indices:
            if idx >= len(paragraphs):
                print(f"  ⚠️ Skipping invalid index {idx}")
                continue
            
            paragraph = paragraphs[idx]
            audio_path = hash_manager.get_audio_path(idx)
            
            print(f"  Generating audio for paragraph {idx+1}/{len(paragraphs)}...")
            
            try:
                if self.api == "openai":
                    self.convert_with_openai(paragraph, str(audio_path))
                elif self.api == "google":
                    self.convert_with_google(paragraph, str(audio_path))
                print(f"    ✓ Fragment saved: {audio_path.name}")
            except Exception as e:
                print(f"    ✗ Error generating fragment {idx}: {e}")
                raise
        
        # Clean up old audio files
        hash_manager.clean_old_audio(len(paragraphs))
        
        # Combine all fragments (with cover prepended)
        print(f"  Combining {len(paragraphs)} fragments into final audio...")
        fragment_manager.combine_fragments(output_path, len(paragraphs), prefix=prefix, cover_path=cover_audio_path if cover_narration else None)
        
        # Update hash file (using clean_text WITHOUT cover narration)
        hash_manager.update_hashes(clean_text)
        print(f"  ✓ Hash file updated: {hash_manager.hash_file.name}")
    
    def convert_with_google(self, text: str, output_path: str, language_code: str = "pt-BR"):
        """Convert text to audio using Google Cloud TTS API with enthusiastic teacher tone."""
        text = self._prepare_tts_text(text)
        # Convert text to SSML with paragraph tags and breaks
        ssml_text = self.text_to_ssml(text)
        
        # Wrap with prosody for enthusiastic teacher tone
        ssml_with_tone = f'<speak><prosody rate="0.95" pitch="+10%">{ssml_text[7:-8]}</prosody></speak>'
        
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_with_tone)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name="pt-BR-Standard-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,  # Slightly slower for more natural, engaging delivery
            pitch=2.0  # Higher pitch for more enthusiastic tone
        )
        
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        print(f"✓ Audio saved to: {output_path}")
    
    def convert_file(self, input_path: Path, force: bool = False, video_generator=None, slide_generator=None, spectrum_generator=None, generate_odp=False, generate_pdf=False, skip_audio=False, use_existing_slides=False):
        """Convert a single text file to audio, saving next to the original file."""
        if not input_path.exists():
            print(f"✗ File not found: {input_path}")
            return
        
        # Skip internal files (starting with 00-)
        if input_path.name.startswith("00-"):
            print(f"⊘ Skipping (internal file): {input_path.name}")
            return
        
        # Translate file if language is not pt-br
        input_path = self.translate_file(input_path, force)
        
        # Use absolute path for progress tracking
        abs_path = str(input_path.absolute())
        
        # Determine output path early so we can check if audio actually exists
        lang_suffix = self.get_language_suffix()
        output_filename = f"{input_path.stem}{lang_suffix}_{self.api}.mp3"
        output_path = input_path.parent / output_filename
        
        # Check if already processed (unless force or slide generation is requested)
        # Also verify the output MP3 exists and is not empty — if it was deleted,
        # we need to regenerate it even if the progress file says it was processed.
        # Additionally, check if any individual audio segments are missing — if so,
        # we must re-run incremental generation to fill the gaps.
        output_exists = output_path.exists() and output_path.stat().st_size > 0
        
        # Quick check for missing audio segments (only relevant for incremental mode)
        has_missing_segments = False
        if abs_path in self.processed_files and not force and not slide_generator and output_exists:
            if HashManager is not None:
                seg_prefix = output_path.stem
                seg_audio_dir = input_path.parent / "audio"
                # Check cover segment
                cover_seg = seg_audio_dir / f"{seg_prefix}_slide_001.mp3"
                if not cover_seg.exists() or cover_seg.stat().st_size == 0:
                    has_missing_segments = True
                    print(f"  ⚠️ Missing cover segment: {cover_seg.name}")
                # Check paragraph segments by counting existing files
                if not has_missing_segments and seg_audio_dir.exists():
                    existing_count = sum(
                        1 for f in seg_audio_dir.glob(f"{seg_prefix}_slide_*.mp3")
                        if f.stat().st_size > 0 and f.name != cover_seg.name
                    )
                    # Need at least 1 paragraph segment (cover + paragraphs)
                    if existing_count < 1:
                        has_missing_segments = True
                        print(f"  ⚠️ No paragraph segments found in {seg_audio_dir}")
        
        if abs_path in self.processed_files and not force and not slide_generator and output_exists and not has_missing_segments:
            # Try to get relative path, fallback to name if structure depth differs
            try:
                rel_path = input_path.relative_to(input_path.parents[-2])
            except ValueError:
                rel_path = input_path.name
            print(f"⊘ Skipping (already processed): {rel_path}")
            return
        
        # Read text content
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean markdown (extract images for video timing)
        clean_text = self.clean_markdown(text, input_path)
        
        if not clean_text:
            print(f"✗ No text content in: {input_path}")
            return
        
        # output_path was already computed above before the progress check
        metadata = parse_course_metadata(input_path)
        content_slides = None
        precomputed_slides = None
        precomputed_slide_timings = None
        deck_has_cover = False
        
        # Check if audio file needs to be regenerated based on timestamps
        needs_audio = self._check_dependency_timestamp(input_path, output_path, force)
        
        # Check if audio file already exists
        audio_generated = False
        if skip_audio:
            # Skip audio generation in slides-only mode
            print(f"⊘ Skipping audio generation (slides-only mode)")
            audio_generated = False
            # Still parse slides for ODP/PDF generation
            if slide_generator:
                content_slides = slide_generator.parse_slides_from_markdown(text)
                deck_has_cover = self._deck_has_cover(slide_generator, input_path, lang_suffix)
                precomputed_slides = self._slides_with_cover(
                    content_slides,
                    deck_has_cover,
                    metadata.get("section"),
                    metadata.get("lesson"),
                    input_path
                )
        elif slide_generator and use_existing_slides:
            # Use existing slides mode: generate audio based on existing ODP structure
            try:
                content_slides = slide_generator.parse_slides_from_markdown(text)
                deck_has_cover = self._deck_has_cover(slide_generator, input_path, lang_suffix)
                precomputed_slides = self._slides_with_cover(
                    content_slides,
                    deck_has_cover,
                    metadata.get("section"),
                    metadata.get("lesson"),
                    input_path
                )
                precomputed_slide_timings = self.convert_slide_segments(
                    precomputed_slides,
                    output_path,
                    force,
                    input_path,
                    metadata.get("section"),
                    metadata.get("lesson"),
                    cover_pause=getattr(slide_generator, "cover_time", 0)
                )
                self._save_progress(abs_path)
                audio_generated = True
                print(f"  Using existing slides for audio generation")
            except Exception as e:
                print(f"✗ Error generating audio with existing slides for {input_path}: {e}")
                print(f"  Progress saved. Run again to continue from this point.")
                return
        elif slide_generator and not use_existing_slides:
            try:
                content_slides = slide_generator.parse_slides_from_markdown(text)
                deck_has_cover = self._deck_has_cover(slide_generator, input_path, lang_suffix)
                precomputed_slides = self._slides_with_cover(
                    content_slides,
                    deck_has_cover,
                    metadata.get("section"),
                    metadata.get("lesson"),
                    input_path
                )
                precomputed_slide_timings = self.convert_slide_segments(
                    precomputed_slides,
                    output_path,
                    force,
                    input_path,
                    metadata.get("section"),
                    metadata.get("lesson"),
                    cover_pause=getattr(slide_generator, "cover_time", 0)
                )
                self._save_progress(abs_path)
                audio_generated = True
            except Exception as e:
                print(f"✗ Error generating synchronized slide audio for {input_path}: {e}")
                print(f"  Progress saved. Run again to continue from this point.")
                return
        elif HashManager and AudioFragmentManager:
            # Use incremental generation with hash tracking (default behavior)
            try:
                self.convert_file_incremental(input_path, clean_text, output_path, force)
                self._save_progress(abs_path)
                audio_generated = True
                if self.processing_log:
                    self.processing_log.record_processing(input_path, file_type="mp3")
            except Exception as e:
                print(f"✗ Error in incremental conversion for {input_path}: {e}")
                print(f"  Falling back to standard conversion...")
                # Fallback to standard conversion
                try:
                    # Generate cover as separate segment, then combine with main text
                    cover_narration = self._cover_narration_text(metadata.get("section"), metadata.get("lesson"), input_path)
                    if cover_narration and not cover_narration.endswith('.'):
                        cover_narration += "."
                    
                    if AudioSegment is not None and cover_narration:
                        cover_min_ms = max(5000, int(self.cover_time * 1000))
                        cover_tmp = output_path.parent / "audio" / f"{output_path.stem}_slide_001.mp3"
                        cover_tmp.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            if self.api == "openai":
                                self._convert_openai_segment(cover_narration, cover_tmp)
                            elif self.api == "google":
                                self.convert_with_google(cover_narration, str(cover_tmp))
                            cover_seg = AudioSegment.from_mp3(str(cover_tmp))
                            if len(cover_seg) < cover_min_ms:
                                cover_seg += AudioSegment.silent(duration=cover_min_ms - len(cover_seg))
                                cover_seg.export(str(cover_tmp), format="mp3")
                        except Exception:
                            cover_seg = AudioSegment.silent(duration=cover_min_ms)
                            cover_seg.export(str(cover_tmp), format="mp3")
                        
                        # Generate main text audio
                        main_tmp = output_path.with_suffix('.tmp.mp3')
                        if self.api == "openai":
                            self.convert_with_openai(clean_text, str(main_tmp))
                        elif self.api == "google":
                            self.convert_with_google(clean_text, str(main_tmp))
                        main_seg = AudioSegment.from_mp3(str(main_tmp))
                        combined = cover_seg + AudioSegment.silent(duration=200) + main_seg
                        combined.export(str(output_path), format="mp3")
                        main_tmp.unlink(missing_ok=True)
                    else:
                        if self.api == "openai":
                            self.convert_with_openai(clean_text, str(output_path))
                        elif self.api == "google":
                            self.convert_with_google(clean_text, str(output_path))
                    self._save_progress(abs_path)
                    audio_generated = True
                    if self.processing_log:
                        self.processing_log.record_processing(input_path, file_type="mp3")
                except Exception as fallback_error:
                    print(f"✗ Error in fallback conversion: {fallback_error}")
        elif not needs_audio:
            print(f"⊘ Skipping (audio up to date): {output_path}")
            audio_generated = True
            self._save_progress(abs_path)
        else:
            # Convert based on API
            try:
                # Generate cover as separate segment, then combine with main text
                cover_narration = self._cover_narration_text(metadata.get("section"), metadata.get("lesson"), input_path)
                if cover_narration and not cover_narration.endswith('.'):
                    cover_narration += "."
                
                if AudioSegment is not None and cover_narration:
                    cover_min_ms = max(5000, int(self.cover_time * 1000))
                    cover_tmp = output_path.parent / "audio" / f"{output_path.stem}_slide_001.mp3"
                    cover_tmp.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if self.api == "openai":
                            self._convert_openai_segment(cover_narration, cover_tmp)
                        elif self.api == "google":
                            self.convert_with_google(cover_narration, str(cover_tmp))
                        cover_seg = AudioSegment.from_mp3(str(cover_tmp))
                        if len(cover_seg) < cover_min_ms:
                            cover_seg += AudioSegment.silent(duration=cover_min_ms - len(cover_seg))
                            cover_seg.export(str(cover_tmp), format="mp3")
                    except Exception:
                        cover_seg = AudioSegment.silent(duration=cover_min_ms)
                        cover_seg.export(str(cover_tmp), format="mp3")
                    
                    # Generate main text audio
                    main_tmp = output_path.with_suffix('.tmp.mp3')
                    if self.api == "openai":
                        self.convert_with_openai(clean_text, str(main_tmp))
                    elif self.api == "google":
                        self.convert_with_google(clean_text, str(main_tmp))
                    main_seg = AudioSegment.from_mp3(str(main_tmp))
                    combined = cover_seg + AudioSegment.silent(duration=200) + main_seg
                    combined.export(str(output_path), format="mp3")
                    main_tmp.unlink(missing_ok=True)
                else:
                    if self.api == "openai":
                        self.convert_with_openai(clean_text, str(output_path))
                    elif self.api == "google":
                        self.convert_with_google(clean_text, str(output_path))
                # Save progress on success
                self._save_progress(abs_path)
                audio_generated = True
                if self.processing_log:
                    self.processing_log.record_processing(input_path, file_type="mp3")
            except Exception as e:
                print(f"✗ Error converting {input_path}: {e}")
                print(f"  Progress saved. Run again to continue from this point.")
                return
        
        # Generate video if requested
        if video_generator:
            try:
                # Parse metadata from file path
                metadata = parse_course_metadata(input_path)
                
                # Determine video output path
                if video_generator.video_output_dir:
                    video_output_dir = Path(video_generator.video_output_dir)
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_output_dir / f"{input_path.stem}.mp4"
                else:
                    video_path = input_path.parent / f"{input_path.stem}.mp4"
                
                # Check if MP3 exists and is not empty - if not, regenerate audio and segments
                if not output_path.exists() or output_path.stat().st_size == 0:
                    print(f"⚠️ MP3 not found or empty, regenerating audio and segments: {output_path}")
                    if slide_generator and precomputed_slides:
                        precomputed_slide_timings = self.convert_slide_segments(
                            precomputed_slides,
                            output_path,
                            force=True,
                            input_path=input_path,
                            section=metadata.get("section"),
                            lesson=metadata.get("lesson"),
                            cover_pause=getattr(slide_generator, "cover_time", 0)
                        )
                        audio_generated = True
                    elif HashManager and AudioFragmentManager:
                        self.convert_file_incremental(input_path, clean_text, output_path, force=True)
                        audio_generated = True
                    else:
                        if self.api == "openai":
                            self.convert_with_openai(clean_text, output_path)
                        elif self.api == "google":
                            self.convert_with_google(clean_text, output_path)
                        audio_generated = True
                
                # Check if video needs to be regenerated based on audio timestamp
                needs_video = self._check_dependency_timestamp(output_path, video_path, force)
                
                if needs_video:
                    # Generate video
                    video_generator.generate_video(
                        output_path,
                        text,
                        video_path,
                        metadata,
                        force
                    )
                    
                    # Save video progress
                    video_generator._save_video_progress(abs_path)
                    if self.processing_log:
                        self.processing_log.record_processing(input_path, file_type="mp4")
                else:
                    print(f"⊘ Skipping (video up to date): {video_path}")
                
            except Exception as e:
                print(f"✗ Error generating video for {input_path}: {e}")
                print(f"  Continuing with next file...")
        
        # Generate slide video/ODP/PDF if requested
        if slide_generator and (audio_generated or skip_audio):
            try:
                # Parse metadata from file path
                metadata = parse_course_metadata(input_path)
                
                # Determine output paths
                # For ODP and PDF, always use input_path.parent (same folder as markdown)
                # For video, use video_output_dir if specified
                output_dir = input_path.parent
                
                # In slides-only mode, always generate slides even if audio wasn't generated
                if skip_audio and not precomputed_slides:
                    content_slides = slide_generator.parse_slides_from_markdown(text)
                    deck_has_cover = self._deck_has_cover(slide_generator, input_path, lang_suffix)
                    precomputed_slides = self._slides_with_cover(
                        content_slides,
                        deck_has_cover,
                        metadata.get("section"),
                        metadata.get("lesson"),
                        input_path
                    )

                if content_slides is None:
                    if precomputed_slides:
                        content_slides = [slide for slide in precomputed_slides if not slide.get("is_cover")]
                    else:
                        content_slides = slide_generator.parse_slides_from_markdown(text)

                content_slide_timings = precomputed_slide_timings
                if deck_has_cover and precomputed_slide_timings:
                    content_slide_timings = precomputed_slide_timings[1:]
                
                # Use HashManager to check if slides changed
                slides_changed = False
                if HashManager:
                    hash_manager = HashManager(input_path)
                    slides_changed = hash_manager.get_slides_changed(precomputed_slides)
                    if slides_changed:
                        print(f"  Slides content changed, will regenerate ODP/PDF/video")
                
                # Generate ODP if requested
                if generate_odp and slide_generator.odp_template:
                    lang_suffix = self.get_language_suffix()
                    odp_path = output_dir / f"{input_path.stem}{lang_suffix}.odp"
                    # Skip ODP generation if use_existing_slides is enabled and ODP already exists
                    if use_existing_slides and odp_path.exists():
                        print(f"  ⊘ Skipping (using existing ODP): {odp_path}")
                        needs_odp = False
                        # Inject cover slide if template provided and ODP exists
                        if slide_generator.cover_template:
                            cover_template_path = Path(slide_generator.cover_template)
                            if cover_template_path.exists():
                                print(f"  → Injecting cover slide into existing ODP")
                                slide_generator._inject_cover_slide(odp_path, cover_template_path, metadata.get("section"), metadata.get("lesson"))
                            else:
                                print(f"  ⚠ Cover template not found: {cover_template_path}")
                        else:
                            print(f"  ⊘ No cover template provided, skipping cover injection")
                    else:
                        needs_odp = slides_changed or self._check_dependency_timestamp(input_path, odp_path, force)
                        if needs_odp:
                            print(f"  Generating ODP: {odp_path}")
                            slide_generator.generate_slides_odp(content_slides, odp_path, content_slide_timings, metadata.get("section"), metadata.get("lesson"))
                            if self.processing_log:
                                self.processing_log.record_processing(input_path, file_type="odp")
                        else:
                            print(f"  ⊘ Skipping (ODP up to date): {odp_path}")
                            # Even if ODP is up to date, inject cover slide if template provided
                            if slide_generator.cover_template:
                                cover_template_path = Path(slide_generator.cover_template)
                                if cover_template_path.exists():
                                    print(f"  → Injecting cover slide into up-to-date ODP")
                                    slide_generator._inject_cover_slide(odp_path, cover_template_path, metadata.get("section"), metadata.get("lesson"))
                
                # Generate PDF if requested (either from ODP or directly)
                if generate_pdf:
                    lang_suffix = self.get_language_suffix()
                    pdf_path = output_dir / f"{input_path.stem}{lang_suffix}_slides.pdf"
                    if generate_odp and slide_generator.odp_template:
                        odp_path = output_dir / f"{input_path.stem}{lang_suffix}.odp"
                        if odp_path.exists():
                            needs_pdf = slides_changed or self._check_dependency_timestamp(odp_path, pdf_path, force)
                            if needs_pdf:
                                slide_generator.convert_odp_to_pdf(odp_path, pdf_path)
                                if self.processing_log:
                                    self.processing_log.record_processing(input_path, file_type="pdf")
                            else:
                                print(f"  ⊘ Skipping (PDF up to date): {pdf_path}")
                    else:
                        needs_pdf = slides_changed or self._check_dependency_timestamp(input_path, pdf_path, force)
                        if needs_pdf:
                            slide_generator.generate_slides_pdf(content_slides, pdf_path)
                            if self.processing_log:
                                self.processing_log.record_processing(input_path, file_type="pdf")
                        else:
                            print(f"  ⊘ Skipping (PDF up to date): {pdf_path}")
                
                # Generate slide video (skip in slides-only mode)
                if not skip_audio:
                    lang_suffix = self.get_language_suffix()
                    video_path = output_dir / f"{input_path.stem}{lang_suffix}.mp4"
                    needs_slide_video = slides_changed or self._check_dependency_timestamp(output_path, video_path, force)
                    odp_dependency = output_dir / f"{input_path.stem}{lang_suffix}.odp"
                    pdf_dependency = output_dir / f"{input_path.stem}{lang_suffix}_slides.pdf"
                    if odp_dependency.exists():
                        needs_slide_video = needs_slide_video or self._check_dependency_timestamp(odp_dependency, video_path, force)
                    if pdf_dependency.exists():
                        needs_slide_video = needs_slide_video or self._check_dependency_timestamp(pdf_dependency, video_path, force)
                    if needs_slide_video:
                        slide_generator.generate_slide_video(
                            output_path,
                            text,
                            video_path,
                            metadata,
                            force,
                            slides=precomputed_slides,
                            slide_timings=precomputed_slide_timings,
                            generate_pdf=generate_pdf,
                            generate_odp=generate_odp
                        )
                        if self.processing_log:
                            self.processing_log.record_processing(input_path, file_type="mp4")
                    else:
                        print(f"  ⊘ Skipping (slide video up to date): {video_path}")
                else:
                    print(f"  ⊘ Skipping video generation (slides-only mode)")
                
                # Save slide progress
                slide_generator._save_slide_progress(abs_path)
                
            except Exception as e:
                print(f"✗ Error generating slide content for {input_path}: {e}")
                print(f"  Continuing with next file...")
        
        # Generate spectrum video if requested
        if spectrum_generator and audio_generated:
            try:
                # Determine spectrum video output path
                if spectrum_generator.video_output_dir:
                    video_output_dir = Path(spectrum_generator.video_output_dir)
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_output_dir / f"{input_path.stem}.mp4"
                else:
                    video_path = input_path.parent / f"{input_path.stem}.mp4"
                
                # Generate spectrum video
                spectrum_generator.generate_spectrum_video(
                    output_path,
                    video_path,
                    force
                )
                
            except Exception as e:
                print(f"✗ Error generating spectrum video for {input_path}: {e}")
                print(f"  Continuing with next file...")
    
    def convert_directory(self, input_dir: Path, force: bool = False, video_generator=None, slide_generator=None, spectrum_generator=None, generate_odp=False, generate_pdf=False, skip_audio=False, use_existing_slides=False):
        """Convert all markdown/text files in a directory recursively."""
        if not input_dir.exists():
            print(f"✗ Directory not found: {input_dir}")
            return
        
        # Clear progress if force is enabled
        if force:
            self._clear_progress()
            print("Force mode enabled: clearing progress and reprocessing all files")
            if video_generator:
                video_generator._clear_video_progress()
            if slide_generator:
                slide_generator._clear_slide_progress()
            if spectrum_generator:
                spectrum_generator._clear_spectrum_progress()
        
        # Find all markdown files recursively (only .md are source files)
        files = list(input_dir.rglob("*.md"))
        
        if not files:
            print(f"✗ No markdown files found in: {input_dir}")
            return
        
        # Filter out internal files (starting with 00-)
        files = [f for f in files if not f.name.startswith("00-")]
        
        # Filter out already processed files only for audio-only runs. Video modes
        # may need to revisit files whose audio was generated in a previous pass.
        # However, if the output MP3 was deleted, we need to reprocess the file.
        if not force and not any([video_generator, slide_generator, spectrum_generator]):
            unprocessed = []
            for f in files:
                abs_f = str(f.absolute())
                if abs_f not in self.processed_files:
                    unprocessed.append(f)
                else:
                    # File was processed before, but check if output MP3 still exists
                    lang_suffix = self.get_language_suffix()
                    output_mp3 = f.parent / f"{f.stem}{lang_suffix}_{self.api}.mp3"
                    if not output_mp3.exists() or output_mp3.stat().st_size == 0:
                        print(f"  ⚠️ Output missing for processed file: {f.name}")
                        unprocessed.append(f)
                    else:
                        # Check for missing audio segments (cover or paragraphs)
                        seg_prefix = f"{f.stem}{lang_suffix}_{self.api}"
                        seg_dir = f.parent / "audio"
                        has_missing = False
                        if seg_dir.exists():
                            cover_seg = seg_dir / f"{seg_prefix}_slide_001.mp3"
                            if not cover_seg.exists() or cover_seg.stat().st_size == 0:
                                has_missing = True
                            else:
                                seg_count = sum(
                                    1 for sf in seg_dir.glob(f"{seg_prefix}_slide_*.mp3")
                                    if sf.stat().st_size > 0 and sf.name != cover_seg.name
                                )
                                if seg_count < 1:
                                    has_missing = True
                        else:
                            has_missing = True
                        if has_missing:
                            print(f"  ⚠️ Missing audio segments for processed file: {f.name}")
                            unprocessed.append(f)
            files = unprocessed
        
        if not files:
            print("All files already processed. Use --force to reprocess.")
            return
        
        print(f"Found {len(files)} files to process")
        
        if self.use_threads and len(files) > 1:
            print(f"Processing with {self.max_threads} threads")
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = {
                    executor.submit(
                        self.convert_file, 
                        file_path, 
                        force, 
                        video_generator, 
                        slide_generator, 
                        spectrum_generator,
                        generate_odp,
                        generate_pdf,
                        skip_audio,
                        use_existing_slides
                    ): file_path for file_path in sorted(files)
                }
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"✗ Error processing {file_path.relative_to(input_dir)}: {e}")
        else:
            for file_path in sorted(files):
                print(f"\nProcessing: {file_path.relative_to(input_dir)}")
                self.convert_file(file_path, force, video_generator, slide_generator, spectrum_generator, generate_odp, generate_pdf, skip_audio, use_existing_slides)


class VideoGenerator:
    """Generates educational videos with animated SVG professor."""
    
    def __init__(self, resolution: str = "1080p", background_color: str = "#f0f4f8", video_output_dir: str = None, voice: str = "cedar"):
        self.resolution = resolution
        self.background_color = background_color
        self.video_output_dir = video_output_dir
        self.voice = voice
        self.width, self.height = get_video_dimensions(resolution)
        
        # Determine professor gender based on voice
        self.professor_gender = self._determine_gender_from_voice(voice)
        
        # Initialize professor SVG with gender
        if ProfessorSVG:
            self.professor = ProfessorSVG(self.width, self.height, gender=self.professor_gender)
        else:
            self.professor = None
        
        self.subtitle_generator = SubtitleGenerator()
        self.video_progress_file = Path(".progress_video.txt")
        self.processed_videos: Set[str] = set()
        self._load_video_progress()
    
    def _determine_gender_from_voice(self, voice: str) -> str:
        """Determine professor gender based on voice selection."""
        # OpenAI voices: nova, shimmer, echo, fable, alloy, ash, coral (female), onyx, sage (male)
        # Google voices: pt-BR-Standard-A (female), pt-BR-Standard-B (male)
        female_voices = ['alloy', 'echo', 'fable', 'nova', 'shimmer', 'ash', 'coral', 'pt-BR-Standard-A']
        male_voices = ['onyx', 'sage', 'pt-BR-Standard-B']
        
        if voice.lower() in female_voices:
            return 'female'
        elif voice.lower() in male_voices:
            return 'male'
        else:
            # Default to female for unknown voices
            return 'female'
    
    def _load_video_progress(self):
        """Load previously processed videos from progress file."""
        if self.video_progress_file.exists():
            with open(self.video_progress_file, 'r', encoding='utf-8') as f:
                self.processed_videos = set(line.strip() for line in f if line.strip())
    
    def _save_video_progress(self, file_path: str):
        """Save a successfully processed video to progress file."""
        with open(self.video_progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n")
        self.processed_videos.add(file_path)
    
    def _clear_video_progress(self):
        """Clear video progress file."""
        if self.video_progress_file.exists():
            self.video_progress_file.unlink()
        self.processed_videos.clear()
    
    def generate_video(
        self,
        audio_path: Path,
        text_content: str,
        output_path: Path,
        metadata: dict,
        force: bool = False
    ):
        """Generate video from audio with animated professor."""
        if not all([VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip]):
            raise ImportError("MoviePy is required for video generation. Install with: pip install moviepy")
        
        if not all([ProfessorSVG, extract_code_blocks, detect_content_type]):
            raise ImportError("Video generation modules not available")
        
        # Check if video already exists
        if output_path.exists() and not force:
            print(f"⊘ Skipping (video already exists): {output_path}")
            return
        
        print(f"Generating video: {output_path}")
        
        try:
            # Load audio
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            
            # Load image timing if available
            image_timing = self._load_image_timing(output_path)
            
            # Analyze audio for animation
            speech_segments = analyze_audio_for_animation(audio_path)
            
            # Detect content characteristics
            code_blocks = extract_code_blocks(text_content)
            has_code = len(code_blocks) > 0
            content_type = detect_content_type(text_content)
            gesture = get_gesture_for_content(text_content, has_code)
            
            # Generate subtitles
            subtitles = self.subtitle_generator.generate_subtitles(text_content, duration)
            
            # Create temporary directory for frames
            temp_dir = create_temp_directory()
            
            try:
                # Generate professor frames
                print("  Generating professor animation frames...")
                frames = self._generate_professor_frames(
                    duration,
                    speech_segments,
                    content_type,
                    gesture,
                    has_code,
                    temp_dir
                )
                
                # Create video from frames
                print("  Compositing video...")
                video_clip = self._composite_video(
                    frames,
                    audio_clip,
                    subtitles,
                    metadata,
                    code_blocks,
                    duration,
                    temp_dir,
                    image_timing
                )
                
                # Export video
                print(f"  Exporting video to: {output_path}")
                video_clip.write_videofile(
                    str(output_path),
                    fps=30,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=str(temp_dir / "temp_audio.m4a"),
                    remove_temp=True
                )
                
                # Clean up
                audio_clip.close()
                video_clip.close()
                
                print(f"✓ Video saved to: {output_path}")
                
            finally:
                clean_temp_directory(temp_dir)
                
        except Exception as e:
            import traceback
            print(f"✗ Error generating video: {e}")
            traceback.print_exc()
            raise
    
    def _load_image_timing(self, output_path: Path) -> List[dict]:
        """Load image timing file if it exists."""
        import json
        
        # Look for image timing file next to the markdown file
        image_timing_file = output_path.parent / f"{output_path.stem.replace('.mp4', '')}_images.json"
        
        if image_timing_file.exists():
            try:
                with open(image_timing_file, 'r', encoding='utf-8') as f:
                    image_timing = json.load(f)
                print(f"  Loaded image timing from: {image_timing_file}")
                return image_timing
            except Exception as e:
                print(f"  Warning: Could not load image timing file: {e}")
        
        return []
    
    def _generate_professor_frames(
        self,
        duration: float,
        speech_segments: List[Tuple[float, float]],
        content_type: str,
        gesture: str,
        has_code: bool,
        temp_dir: Path
    ) -> List[Path]:
        """Generate professor frames for animation."""
        fps = 30
        total_frames = int(duration * fps)
        frames = []
        
        for frame_num in range(total_frames):
            current_time = frame_num / fps
            
            # Determine if speaking
            is_speaking = False
            if speech_segments:
                try:
                    is_speaking = any(start <= current_time <= end for start, end in speech_segments)
                except (ValueError, TypeError):
                    # If speech_segments has wrong structure, assume always speaking
                    is_speaking = True
            
            # Calculate mouth opening
            mouth_open = 0.3 if is_speaking else 0.0
            
            # Determine if blinking
            blink = should_blink(frame_num)
            
            # Generate frame
            frame_path = temp_dir / f"frame_{frame_num:06d}.png"
            svg_content = self.professor.generate_svg(
                expression=content_type,
                gesture=gesture,
                mouth_open=mouth_open,
                blink=blink
            )
            self.professor.save_svg_as_png(svg_content, frame_path)
            frames.append(frame_path)
            
            # Progress indicator
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"    Progress: {progress:.1f}%")
        
        return frames
    
    def _composite_video(
        self,
        frames: List[Path],
        audio_clip: 'AudioFileClip',
        subtitles: List,
        metadata: dict,
        code_blocks: List[str],
        duration: float,
        temp_dir: Path,
        image_timing: List[dict] = None
    ) -> 'CompositeVideoClip':
        """Composite all video elements."""
        if image_timing is None:
            image_timing = []
        # Create background
        background = ImageClip(
            self._create_background_image(temp_dir),
            duration=duration
        )
        
        # Create professor clip from frames
        professor_clip = ImageClip(str(frames[0]), duration=duration)
        if len(frames) > 1:
            professor_clip = professor_clip.with_duration(duration)
        
        # Position professor
        professor_clip = professor_clip.with_position('center')
        
        # Add metadata overlays
        overlays = [background, professor_clip]
        
        # Add course title (top-left)
        course_text = metadata.get("course", "")
        if course_text:
            title_clip = TextClip(
                text=course_text,
                color='white',
                method='caption',
                size=(1200, 150)
            ).with_position((20, 20)).with_duration(duration)
            overlays.append(title_clip)
        
        # Add section indicator (top-right)
        section_text = metadata.get("section", "")
        if section_text:
            section_clip = TextClip(
                text=section_text,
                color='white',
                method='caption',
                size=(900, 150)
            ).with_position(('right', 20)).with_duration(duration)
            overlays.append(section_clip)
        
        # Add lesson name (bottom-left, above subtitles)
        lesson_text = metadata.get("lesson", "")
        if lesson_text:
            lesson_clip = TextClip(
                text=lesson_text,
                color='white',
                method='caption',
                size=(1500, 150)
            ).with_position((20, self.height - 360)).with_duration(duration)
            overlays.append(lesson_clip)
        
        # Add subtitles
        if subtitles:
            subtitle_clips = self._create_subtitle_clips(subtitles, duration)
            overlays.extend(subtitle_clips)
        
        # Composite all elements
        final_clip = CompositeVideoClip(overlays, size=(self.width, self.height))
        final_clip = final_clip.with_audio(audio_clip)
        
        return final_clip
    
    def _create_background_image(self, temp_dir: Path) -> Path:
        """Create background image."""
        from PIL import Image, ImageDraw
        
        bg_path = temp_dir / "background.png"
        img = Image.new('RGB', (self.width, self.height), self.background_color)
        img.save(bg_path)
        return bg_path
    
    def _create_subtitle_clips(self, subtitles: List, duration: float) -> List:
        """Create subtitle clips with timing."""
        clips = []
        
        for subtitle in subtitles:
            # Create text clip
            text_clip = TextClip(
                text=subtitle.text,
                color='white',
                stroke_color='black',
                stroke_width=6,
                method='caption',
                size=(2400, 180)
            )
            
            # Position at bottom center
            text_clip = text_clip.with_position(('center', self.height - 240))
            
            # Set timing
            text_clip = text_clip.with_start(subtitle.start_time)
            text_clip = text_clip.with_end(subtitle.end_time)
            
            clips.append(text_clip)
        
        return clips


class SlideGenerator:
    """Generates slide-based videos from markdown content with audio synchronization."""
    
    def __init__(self, resolution: str = "1080p", background_color: str = "#ffffff", 
                 video_output_dir: str = None, slide_duration: float = 5.0, 
                 odp_template: str = None, cover_template: str = None, cover_time: float = 20.0, language: str = "pt-br"):
        self.resolution = resolution
        self.background_color = background_color
        self.video_output_dir = video_output_dir
        self.slide_duration = slide_duration
        self.odp_template = odp_template
        self.cover_template = cover_template
        self.cover_time = cover_time
        self.language = language
        self.width, self.height = get_video_dimensions(resolution)
        self.slide_progress_file = Path(".progress_slide.txt")
        self.processed_slides: Set[str] = set()
        self._load_slide_progress()
        
        # Generate safety identifier for OpenAI API (hashed project name)
        self.safety_identifier = self._generate_safety_identifier()
        
        # Initialize OpenAI client for title generation
        self.openai_client = None
        if OpenAI:
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"  Warning: Could not initialize OpenAI client: {e}")
    
    def get_language_suffix(self) -> str:
        """Get file suffix based on language (empty for pt-br, language code for others)."""
        return "" if self.language == "pt-br" else f"_{self.language}"
    
    def _generate_safety_identifier(self) -> str:
        """Generate a stable safety identifier for OpenAI API calls."""
        if hashlib is None:
            return "smart_contracts_ethereum_user"
        
        # Hash the project name to create a stable, non-personal identifier
        project_name = "Smart Contracts na Ethereum"
        hashed = hashlib.sha256(project_name.encode('utf-8')).hexdigest()
        # Use first 16 characters for a shorter identifier
        return f"sc_eth_{hashed[:16]}"
    
    def _load_slide_progress(self):
        """Load previously processed slide videos from progress file."""
        if self.slide_progress_file.exists():
            with open(self.slide_progress_file, 'r', encoding='utf-8') as f:
                self.processed_slides = set(line.strip() for line in f if line.strip())
    
    def _save_slide_progress(self, file_path: str):
        """Save a successfully processed slide video to progress file."""
        with open(self.slide_progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n")
        self.processed_slides.add(file_path)
    
    def _clear_slide_progress(self):
        """Clear slide video progress file."""
        if self.slide_progress_file.exists():
            self.slide_progress_file.unlink()
        self.processed_slides.clear()
    
    def _extract_pdf_images(self, pdf_path: Path, images_dir: Path, force: bool = False, prefix: str = None, lang_suffix: str = "") -> List[Path]:
        """Extract images from PDF to images directory with caching."""
        if not convert_pdf_to_images:
            print("  Warning: pdf_to_images module not available")
            return []
        
        # Add language suffix to prefix
        if prefix and lang_suffix:
            prefix = f"{prefix}{lang_suffix}"
        
        # Check if images already exist and PDF hasn't changed
        if not force and images_dir.exists():
            # Use prefix pattern if provided, otherwise generic pattern
            if prefix:
                pattern = f"{prefix}_page_*.png"
            else:
                pattern = "page_*.png"
            existing_images = sorted(images_dir.glob(pattern))
            if existing_images:
                # Check if PDF is newer than images
                pdf_mtime = pdf_path.stat().st_mtime
                oldest_image_mtime = min(img.stat().st_mtime for img in existing_images)
                if pdf_mtime <= oldest_image_mtime:
                    print(f"  ⊘ Reusing existing PDF images: {len(existing_images)} pages")
                    return existing_images
        
        # Extract images
        print(f"  Extracting images from PDF: {pdf_path}")
        images_dir.mkdir(parents=True, exist_ok=True)
        return convert_pdf_to_images(pdf_path, images_dir, dpi=150, prefix=prefix)
    
    def _extract_odp_images(self, odp_path: Path, images_dir: Path, force: bool = False, prefix: str = None, lang_suffix: str = "") -> List[Path]:
        """Extract images from ODP to images directory with caching."""
        if not convert_odp_to_images:
            print("  Warning: pdf_to_images module not available")
            return []
        
        # Add language suffix to prefix
        if prefix and lang_suffix:
            prefix = f"{prefix}{lang_suffix}"
        
        # Check if images already exist and ODP hasn't changed
        if not force and images_dir.exists():
            # Use prefix pattern if provided, otherwise generic pattern
            if prefix:
                pattern = f"{prefix}_page_*.png"
            else:
                pattern = "page_*.png"
            existing_images = sorted(images_dir.glob(pattern))
            if existing_images:
                # Check if ODP is newer than images
                odp_mtime = odp_path.stat().st_mtime
                oldest_image_mtime = min(img.stat().st_mtime for img in existing_images)
                if odp_mtime <= oldest_image_mtime:
                    print(f"  ⊘ Reusing existing ODP images: {len(existing_images)} pages")
                    return existing_images
        
        # Extract images
        print(f"  Extracting images from ODP: {odp_path}")
        images_dir.mkdir(parents=True, exist_ok=True)
        return convert_odp_to_images(odp_path, images_dir, dpi=150, prefix=prefix)
    
    def generate_slide_title(self, content: str) -> str:
        """Generate a concise title for slide content using OpenAI."""
        if not self.openai_client:
            # Fallback: use first 50 characters
            return content[:50] + '...' if len(content) > 50 else content
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a concise, descriptive title (max 10 words) for the following slide content. Return only the title, no additional text."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=50,
                temperature=0.7,
                safety_identifier=self.safety_identifier
            )
            title = response.choices[0].message.content.strip()
            return title if title else content[:50] + '...' if len(content) > 50 else content
        except Exception as e:
            error_str = str(e)
            # Check for quota exceeded error
            if '429' in error_str or 'insufficient_quota' in error_str:
                print(f"  Warning: OpenAI quota exceeded. Disabling AI title generation for this session.")
                self.openai_client = None  # Disable further AI title generation
            else:
                print(f"  Warning: Could not generate AI title: {e}")
            return content[:50] + '...' if len(content) > 50 else content
    
    def parse_slides_from_markdown(self, text: str) -> List[dict]:
        """Parse markdown content into slides based on paragraphs (separated by blank lines)."""
        slides = []
        lines = text.split('\n')
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and markdown syntax - these mark paragraph boundaries
            if not stripped or stripped.startswith('#') or stripped.startswith('```') or stripped.startswith('!['):
                # If we have accumulated content, create one slide for the entire paragraph
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    if paragraph_text:
                        title = self.generate_slide_title(paragraph_text)
                        slides.append({
                            'title': title,
                            'level': 2,
                            'content': [paragraph_text],
                            'bullets': []
                        })
                    current_paragraph = []
                continue
            
            # Check if line starts with bullet point
            if stripped.startswith('- ') or stripped.startswith('* '):
                # Save previous paragraph if exists
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    if paragraph_text:
                        title = self.generate_slide_title(paragraph_text)
                        slides.append({
                            'title': title,
                            'level': 2,
                            'content': [paragraph_text],
                            'bullets': []
                        })
                    current_paragraph = []
                
                # Create slide for bullet point (one slide per bullet)
                bullet_text = stripped.lstrip('-*').strip()
                title = self.generate_slide_title(bullet_text)
                slides.append({
                    'title': title,
                    'level': 2,
                    'content': [bullet_text],
                    'bullets': []
                })
            else:
                # Accumulate paragraph content
                current_paragraph.append(stripped)
        
        # Add last paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            if paragraph_text:
                title = self.generate_slide_title(paragraph_text)
                slides.append({
                    'title': title,
                    'level': 2,
                    'content': [paragraph_text],
                    'bullets': []
                })
        
        # If no slides found, create single slide from all content
        if not slides:
            content_text = ' '.join([line.strip() for line in lines if line.strip() and not line.startswith('#')])
            title = self.generate_slide_title(content_text)
            slides.append({
                'title': title,
                'level': 2,
                'content': [content_text],
                'bullets': []
            })
        
        return slides
    
    def create_slide_image(self, slide: dict, output_path: Path):
        """Create an image from slide content."""
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
        
        # Create image
        img = Image.new('RGB', (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts with better Portuguese accent support
        # Reduced sizes by 40% (multiply by 0.6)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)  # 60 * 0.6
            content_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)  # 40 * 0.6
            bullet_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)  # 36 * 0.6
        except:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()
            bullet_font = ImageFont.load_default()
        
        # Draw title with wrapping - darker color for better contrast
        y_position = 100
        title = slide['title']
        # Wrap title to fit width (approximately 50 chars per line for 1920px width)
        title_lines = textwrap.wrap(title, width=50)
        for title_line in title_lines[:2]:  # Max 2 lines for title
            draw.text((50, y_position), title_line, fill='#000000', font=title_font)  # Pure black for max contrast
            y_position += 48  # 80 * 0.6
        y_position += 60  # Extra spacing after title to prevent overlap (increased from 30 to 60)
        
        # Draw content with wrapping - darker color for better contrast
        for content in slide['content'][:5]:  # Limit to 5 content items
            # Wrap content to fit width (approximately 60 chars per line)
            content_lines = textwrap.wrap(content, width=60)
            for content_line in content_lines[:8]:  # Max 8 lines per content item
                draw.text((50, y_position), content_line, fill='#1a1a1a', font=content_font)  # Dark gray
                y_position += 30  # 50 * 0.6
            y_position += 12  # 20 * 0.6
        
        # Draw bullets with wrapping - darker color for better contrast
        for bullet in slide['bullets'][:8]:  # Limit to 8 bullets
            # Wrap bullet to fit width (approximately 55 chars per line)
            bullet_lines = textwrap.wrap(bullet, width=55)
            for bullet_line in bullet_lines[:3]:  # Max 3 lines per bullet
                draw.text((80, y_position), f"• {bullet_line}", fill='#2d2d2d', font=bullet_font)  # Medium dark
                y_position += 27  # 45 * 0.6
            y_position += 9  # 15 * 0.6
        
        img.save(output_path)
    
    def calculate_slide_timing(self, slides: List[dict], audio_duration: float) -> List[float]:
        """Calculate timing for each slide based on audio duration and punctuation."""
        if not slides:
            return []
        
        # Distribute audio duration based on content length and punctuation
        # Segments ending with semicolons get more time (major pause)
        # Segments ending with commas get standard time (minor pause)
        content_weights = []
        for slide in slides:
            weight = 1.0
            content_text = ' '.join(slide['content'])
            
            # Base weight from content length
            weight += len(content_text) * 0.02
            
            # Add weight for punctuation (pauses in speech)
            if content_text.strip().endswith(';'):
                weight += 0.5  # Major pause
            elif content_text.strip().endswith(','):
                weight += 0.2  # Minor pause
            elif content_text.strip().endswith('.'):
                weight += 0.3  # Sentence end
            
            # Add weight for bullets
            weight += len(slide['bullets']) * 0.15
            
            content_weights.append(weight)
        
        total_weight = sum(content_weights)
        timings = []
        for weight in content_weights:
            slide_duration = (weight / total_weight) * audio_duration
            # Minimum 1.5 seconds for very short segments, maximum 15 seconds
            timings.append(max(min(slide_duration, 15.0), 1.5))
        
        return timings
    
    def generate_slide_video(
        self,
        audio_path: Path,
        text_content: str,
        output_path: Path,
        metadata: dict,
        force: bool = False,
        slides: List[dict] = None,
        slide_timings: List[float] = None,
        generate_pdf: bool = True,
        generate_odp: bool = False
    ):
        """Generate slide-based video from audio.
        
        Pipeline:
        1. MD file is the source of content
        2. From MD: generates MP3 audio, hash file, ODP (if requested), PDF (if requested)
        3. From ODP: generates images and PDF
        4. From images + audio + slide timings: generates video
        """
        if not all([VideoFileClip, AudioFileClip, ImageClip]):
            raise ImportError("MoviePy is required for slide video generation. Install with: pip install moviepy")
        
        # Check if video already exists
        if output_path.exists() and not force:
            print(f"⊘ Skipping (slide video already exists): {output_path}")
            return
        
        print(f"Generating slide video: {output_path}")
        
        try:
            # Load audio
            audio_clip = AudioFileClip(str(audio_path))
            audio_duration = audio_clip.duration
            
            # Parse slides from markdown (source of content)
            if slides is None:
                slides = self.parse_slides_from_markdown(text_content)
            print(f"  Parsed {len(slides)} slides from markdown content")
            
            # Calculate slide timing
            if slide_timings is None or len(slide_timings) != len(slides):
                print("  No synchronized slide timings found; using estimated timings")
                slide_timings = self.calculate_slide_timing(slides, audio_duration)
            else:
                print("  Using real audio duration for each slide")
                slide_timings = self._normalize_slide_timings(slide_timings, audio_duration)
            
            # ODP and PDF are already generated in convert_file(), just use them
            lang_suffix = self.get_language_suffix()
            odp_path = output_path.parent / f"{output_path.stem}{lang_suffix}.odp"
            pdf_path = output_path.parent / f"{output_path.stem}{lang_suffix}_slides.pdf"
            
            # Extract images from PDF to images directory (from ODP or direct PDF)
            images_dir = output_path.parent / "images"
            # Get prefix from output_path (e.g., "01-Introducao-as-Ferramentas")
            prefix = output_path.stem
            if pdf_path.exists():
                pdf_images = self._extract_pdf_images(pdf_path, images_dir, force, prefix, lang_suffix)
            elif odp_path.exists():
                pdf_images = self._extract_odp_images(odp_path, images_dir, force, prefix, lang_suffix)
            else:
                pdf_images = []
            
            # Create temporary directory for slide images (fallback if PDF extraction fails)
            temp_dir = create_temp_directory()
            
            try:
                # Use extracted PDF images or generate slide images as fallback
                print("  Preparing slide images...")
                slide_images = []
                current_time = 0
                
                if pdf_images and len(pdf_images) >= len(slides):
                    # Use PDF images (from ODP or direct PDF)
                    print(f"  Using {len(pdf_images)} images from PDF")
                    for i, (pdf_image, duration) in enumerate(zip(pdf_images, slide_timings)):
                        slide_images.append({
                            'path': pdf_image,
                            'start_time': current_time,
                            'end_time': current_time + duration
                        })
                        current_time += duration
                else:
                    # Fallback: generate slide images with text
                    print(f"  Generating slide images from text (PDF images not available or insufficient)")
                    for i, (slide, duration) in enumerate(zip(slides, slide_timings)):
                        slide_path = temp_dir / f"slide_{i:03d}.png"
                        self.create_slide_image(slide, slide_path)
                        slide_images.append({
                            'path': slide_path,
                            'start_time': current_time,
                            'end_time': current_time + duration
                        })
                        current_time += duration
                
                # Create video clips from slides
                print("  Compositing slide video...")
                slide_clips = []
                
                for i, slide_data in enumerate(slide_images):
                    slide_clip = ImageClip(str(slide_data['path']))
                    duration = slide_data['end_time'] - slide_data['start_time']
                    slide_clip = slide_clip.with_duration(duration)
                    slide_clips.append(slide_clip)
                
                # Concatenate slides sequentially
                from moviepy import concatenate_videoclips
                final_clip = concatenate_videoclips(slide_clips, method="compose")
                
                # Adjust audio duration if cover time was added
                total_video_duration = sum(clip.duration for clip in slide_clips)
                if total_video_duration > audio_duration:
                    # Extend audio to match video duration (for cover pause)
                    final_clip = final_clip.with_duration(total_video_duration)
                    final_clip = final_clip.with_audio(audio_clip)
                else:
                    final_clip = final_clip.with_duration(audio_duration)
                    final_clip = final_clip.with_audio(audio_clip)
                
                # Add metadata overlays
                overlays = [final_clip]
                
                # Add course title (top-left)
                course_text = metadata.get("course", "")
                if course_text and TextClip:
                    title_clip = TextClip(
                        text=course_text,
                        color='white',
                        font_size=24,
                        method='caption',
                        size=(900, 50)
                    ).with_position((20, 20)).with_duration(audio_duration)
                    overlays.append(title_clip)
                
                # Add section indicator (top-right)
                section_text = metadata.get("section", "")
                if section_text and TextClip:
                    section_clip = TextClip(
                        text=section_text,
                        color='white',
                        font_size=24,
                        method='caption',
                        size=(600, 50)
                    ).with_position(('right', 20)).with_duration(audio_duration)
                    overlays.append(section_clip)
                
                # Composite final video
                if len(overlays) > 1:
                    final_clip = CompositeVideoClip(overlays, size=(self.width, self.height))
                    final_clip = final_clip.with_audio(audio_clip)
                
                # Export video
                print(f"  Exporting slide video to: {output_path}")
                final_clip.write_videofile(
                    str(output_path),
                    fps=30,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=str(temp_dir / "temp_audio.m4a"),
                    remove_temp=True
                )
                
                # Clean up
                audio_clip.close()
                final_clip.close()
                
                print(f"✓ Slide video saved to: {output_path}")
                
            finally:
                clean_temp_directory(temp_dir)
                
        except Exception as e:
            import traceback
            print(f"✗ Error generating slide video: {e}")
            traceback.print_exc()
            raise

    def _normalize_slide_timings(self, timings: List[float], audio_duration: float) -> List[float]:
        """Adjust timing drift so slide durations match the final audio duration."""
        if not timings:
            return timings

        normalized = [max(0.1, float(duration)) for duration in timings]
        drift = audio_duration - sum(normalized)

        if abs(drift) <= 0.05:
            return normalized

        normalized[-1] = max(0.1, normalized[-1] + drift)

        if sum(normalized) <= audio_duration + 0.05:
            return normalized

        scale = audio_duration / sum(normalized)
        return [max(0.1, duration * scale) for duration in normalized]
    
    def generate_slides_odp(self, slides: List[dict], odp_path: Path, slide_timings: List[float] = None, section: str = None, lesson: str = None):
        """Generate ODP presentation from slides using template."""
        if not self.odp_template:
            print("  Warning: No ODP template provided, skipping ODP generation")
            return
        
        if not zipfile or not ET:
            print("  Warning: zipfile or xml.etree not available, skipping ODP generation")
            return
        
        if not Path(self.odp_template).exists():
            print(f"  Warning: ODP template not found: {self.odp_template}")
            return
        
        try:
            print(f"  Generating ODP presentation: {odp_path}")
            
            # Extract template to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                template_dir = temp_dir / "template"
                template_dir.mkdir()
                
                with zipfile.ZipFile(self.odp_template, 'r') as zip_ref:
                    zip_ref.extractall(template_dir)
                
                # Read content.xml
                content_xml_path = template_dir / "content.xml"
                with open(content_xml_path, 'r', encoding='utf-8') as f:
                    content_xml = f.read()
                
                # Parse slides to create ODP structure
                # For each slide, we need to duplicate the template slide and replace placeholders
                slides_xml = []
                
                # Extract the slide template from content.xml
                # Find the draw:page element which represents a slide
                slide_pattern = r'<draw:page(?:\s[^>]*)?>.*?</draw:page>'
                slide_template_match = re.search(slide_pattern, content_xml, re.DOTALL)
                
                if not slide_template_match:
                    print("  Warning: Could not find slide template in ODP")
                    return
                
                slide_template = slide_template_match.group(0)
                
                # Generate slides
                for i, (slide, timing) in enumerate(zip(slides, slide_timings or [0]*len(slides))):
                    slide_xml = slide_template
                    
                    # Replace placeholders
                    title = slide.get('title', '')
                    content = ' '.join(slide.get('content', []))
                    
                    # Convert markdown to ODP formatting
                    title = self._convert_markdown_to_odp(title)
                    content = self._convert_markdown_to_odp(content)
                    
                    # Replace [título]
                    slide_xml = slide_xml.replace('[título]', title)
                    
                    # Replace [paragrafo]
                    slide_xml = slide_xml.replace('[paragrafo]', content)
                    
                    # Replace [time] with timing in MM:SS format
                    time_formatted = self._format_time_mm_ss(timing)
                    slide_xml = slide_xml.replace('[time]', time_formatted)
                    
                    # Update slide name/id
                    slide_xml = re.sub(r'draw:name="page[0-9]+"', f'draw:name="page{i+1}"', slide_xml)
                    slide_xml = re.sub(r'draw:id="page[0-9]+"', f'draw:id="page{i+1}"', slide_xml)
                    
                    slides_xml.append(slide_xml)
                
                # Replace the original slide with all generated slides
                new_content_xml = re.sub(slide_pattern, '\n'.join(slides_xml), content_xml, count=1)
                
                # Write modified content.xml
                with open(content_xml_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_xml)
                
                # Update mimetype file to ensure correct MIME type
                mimetype_path = template_dir / "mimetype"
                if mimetype_path.exists():
                    with open(mimetype_path, 'w', encoding='utf-8') as f:
                        f.write('application/vnd.oasis.opendocument.presentation')
                
                # Update manifest.xml to ensure correct MIME type
                manifest_xml_path = template_dir / "META-INF" / "manifest.xml"
                if manifest_xml_path.exists():
                    with open(manifest_xml_path, 'r', encoding='utf-8') as f:
                        manifest_xml = f.read()
                    
                    # Replace template MIME type with presentation MIME type
                    manifest_xml = manifest_xml.replace(
                        'application/vnd.oasis.opendocument.presentation-template',
                        'application/vnd.oasis.opendocument.presentation'
                    )
                    
                    with open(manifest_xml_path, 'w', encoding='utf-8') as f:
                        f.write(manifest_xml)
                
                # Update meta.xml to remove template-specific metadata
                meta_xml_path = template_dir / "meta.xml"
                if meta_xml_path.exists():
                    with open(meta_xml_path, 'r', encoding='utf-8') as f:
                        meta_xml = f.read()
                    
                    # Remove template-specific metadata
                    meta_xml = re.sub(r'<meta:template[^>]*>.*?</meta:template>', '', meta_xml, flags=re.DOTALL)
                    
                    with open(meta_xml_path, 'w', encoding='utf-8') as f:
                        f.write(meta_xml)
                
                # Create new ODP file
                with zipfile.ZipFile(odp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(template_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(template_dir)
                            zipf.write(file_path, arcname)
            
            print(f"✓ ODP presentation saved to: {odp_path}")
            
            # Inject cover slide if template provided
            if self.cover_template:
                cover_template_path = Path(self.cover_template)
                if cover_template_path.exists():
                    self._inject_cover_slide(odp_path, cover_template_path, section, lesson)
            
        except Exception as e:
            import traceback
            print(f"  Warning: Could not generate ODP: {e}")
            traceback.print_exc()
    
    def _normalize_cover_slide_structure(self, slide: str) -> str:
        """Normalize cover slide XML so title/content changes do not affect matching."""
        normalized = re.sub(r'<!--.*?-->', '', slide, flags=re.DOTALL)
        normalized = re.sub(r'\s+draw:(?:name|id)="page[0-9]+"', '', normalized)
        normalized = re.sub(r'<text:p([^>]*)>.*?</text:p>', r'<text:p\1></text:p>', normalized, flags=re.DOTALL)
        normalized = re.sub(r'>\s+<', '><', normalized)
        return ' '.join(normalized.split())

    def _mark_cover_slide(self, slide: str) -> str:
        """Add an internal marker to make future cover replacement explicit."""
        if 'codex:cover-slide' in slide:
            return slide
        return re.sub(r'(<draw:page(?:\s[^>]*)?>)', r'\1<!-- codex:cover-slide -->', slide, count=1)

    def _is_cover_like_slide(self, slide: str) -> bool:
        """Identify cover slides even after editors rewrite non-text XML details."""
        has_title = bool(re.search(r'<draw:frame(?=[^>]*presentation:class="title")[^>]*>', slide, re.DOTALL))
        has_image = '<draw:image' in slide
        has_paragraph_field = (
            'presentation:class="outline"' in slide
            or '[paragrafo]' in slide
            or '[paragraph]' in slide
        )
        return has_title and has_image and not has_paragraph_field

    def _is_existing_cover_slide(self, slide: str, cover_structure: str) -> bool:
        """Return True when a slide is an existing generated cover."""
        if 'codex:cover-slide' in slide:
            return True
        if self._normalize_cover_slide_structure(slide) == cover_structure:
            return True
        return self._is_cover_like_slide(slide)

    def has_cover_slide(self, odp_path: Path) -> bool:
        """Return True when the first slide in an ODP looks like the generated cover."""
        if not odp_path or not odp_path.exists() or not zipfile:
            return False

        try:
            with zipfile.ZipFile(odp_path, 'r') as zip_ref:
                content_xml = zip_ref.read("content.xml").decode("utf-8")
        except Exception as e:
            print(f"  Warning: Could not inspect ODP cover slide: {e}")
            return False

        slide_pattern = r'<draw:page(?:\s[^>]*)?>.*?</draw:page>'
        first_slide_match = re.search(slide_pattern, content_xml, re.DOTALL)
        if not first_slide_match:
            return False

        first_slide = first_slide_match.group(0)
        if 'codex:cover-slide' in first_slide or self._is_cover_like_slide(first_slide):
            return True

        if self.cover_template and Path(self.cover_template).exists():
            try:
                with zipfile.ZipFile(self.cover_template, 'r') as zip_ref:
                    cover_xml = zip_ref.read("content.xml").decode("utf-8")
                cover_slide_match = re.search(slide_pattern, cover_xml, re.DOTALL)
                if cover_slide_match:
                    cover_structure = self._normalize_cover_slide_structure(cover_slide_match.group(0))
                    return self._normalize_cover_slide_structure(first_slide) == cover_structure
            except Exception:
                return False

        return False

    def _extract_slide_title_text(self, slide: str) -> str:
        """Extract readable title text from a slide."""
        title_frame_pattern = r'<draw:frame(?=[^>]*presentation:class="title")[^>]*>.*?</draw:frame>'
        title_parts = []
        for frame in re.findall(title_frame_pattern, slide, re.DOTALL):
            for text in re.findall(r'<text:p[^>]*>(.*?)</text:p>', frame, re.DOTALL):
                text = re.sub(r'<text:line-break\s*/>', ' ', text)
                text = re.sub(r'<[^>]+>', ' ', text)
                title_parts.append(html.unescape(' '.join(text.split())))
        return ' '.join(part for part in title_parts if part)

    def _normalize_semantic_text(self, text: str) -> str:
        """Normalize text for semantic duplicate checks."""
        text = html.unescape(text)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if not unicodedata.combining(char))
        text = text.lower()
        text = re.sub(r'[-_]+', ' ', text)
        text = re.sub(r'[^a-z0-9]+', ' ', text)
        return ' '.join(text.split())

    def _are_semantic_cover_duplicates(self, first_slide: str, second_slide: str) -> bool:
        """Detect duplicated first-page covers while preserving the older second page."""
        if not (self._is_cover_like_slide(first_slide) and self._is_cover_like_slide(second_slide)):
            return False

        first_title = self._normalize_semantic_text(self._extract_slide_title_text(first_slide))
        second_title = self._normalize_semantic_text(self._extract_slide_title_text(second_slide))
        return bool(first_title and second_title and first_title == second_title)

    def _style_names_used_by_slide(self, slide: str) -> Set[str]:
        """Return automatic style names referenced by a slide."""
        return set(re.findall(
            r'(?:draw|presentation|text):(?:style-name|text-style-name)="([^"]+)"',
            slide
        ))

    def _rename_style_names(self, xml: str, style_map: dict) -> str:
        """Rename style definitions/references according to style_map."""
        for old_name, new_name in sorted(style_map.items(), key=lambda item: len(item[0]), reverse=True):
            old_escaped = re.escape(old_name)
            xml = re.sub(
                rf'((?:style:name|draw:style-name|draw:text-style-name|presentation:style-name|text:style-name)="){old_escaped}(")',
                rf'\1{new_name}\2',
                xml
            )
        return xml

    def _automatic_style_blocks(self, content_xml: str, style_names: Set[str]) -> List[str]:
        """Extract automatic style blocks for the provided style names."""
        auto_match = re.search(r'<office:automatic-styles>(.*?)</office:automatic-styles>', content_xml, re.DOTALL)
        if not auto_match:
            return []

        automatic_styles = auto_match.group(1)
        blocks = []
        for style_name in sorted(style_names):
            name = re.escape(style_name)
            for tag_name in ("style:style", "text:list-style"):
                pattern = rf'<{tag_name}\b(?=[^>]*\bstyle:name="{name}")[^>]*(?:/>|>.*?</{tag_name}>)'
                blocks.extend(re.findall(pattern, automatic_styles, re.DOTALL))
        return blocks

    def _remove_automatic_style_blocks(self, content_xml: str, style_names: Set[str]) -> str:
        """Remove automatic style blocks by name from content.xml."""
        if not style_names:
            return content_xml

        auto_match = re.search(r'<office:automatic-styles>(.*?)</office:automatic-styles>', content_xml, re.DOTALL)
        if not auto_match:
            return content_xml

        automatic_styles = auto_match.group(1)
        for style_name in sorted(style_names):
            name = re.escape(style_name)
            for tag_name in ("style:style", "text:list-style"):
                pattern = rf'<{tag_name}\b(?=[^>]*\bstyle:name="{name}")[^>]*(?:/>|>.*?</{tag_name}>)'
                automatic_styles = re.sub(pattern, '', automatic_styles, flags=re.DOTALL)

        return (
            content_xml[:auto_match.start(1)]
            + automatic_styles
            + content_xml[auto_match.end(1):]
        )

    def _merge_cover_automatic_styles(self, content_xml: str, cover_content_xml: str, style_map: dict) -> str:
        """Copy cover template automatic styles into the destination ODP."""
        if not style_map:
            return content_xml

        renamed_style_names = set(style_map.values())
        cover_style_blocks = self._automatic_style_blocks(cover_content_xml, set(style_map.keys()))
        if not cover_style_blocks:
            return content_xml

        renamed_blocks = [
            self._rename_style_names(block, style_map)
            for block in cover_style_blocks
        ]

        content_xml = self._remove_automatic_style_blocks(content_xml, renamed_style_names)
        insert_at = content_xml.find('</office:automatic-styles>')
        if insert_at == -1:
            return content_xml

        return content_xml[:insert_at] + ''.join(renamed_blocks) + content_xml[insert_at:]

    def _media_type_for_asset(self, asset_path: str) -> str:
        """Return a basic media type for an ODP package asset."""
        suffix = Path(asset_path).suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".svg":
            return "image/svg+xml"
        return "application/octet-stream"

    def _add_manifest_entries(self, odp_dir: Path, asset_paths: List[str]):
        """Add manifest entries for copied assets when missing."""
        if not asset_paths:
            return

        manifest_path = odp_dir / "META-INF" / "manifest.xml"
        if not manifest_path.exists():
            return

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_xml = f.read()

        entries = []
        for asset_path in sorted(set(asset_paths)):
            if f'manifest:full-path="{asset_path}"' in manifest_xml:
                continue
            media_type = self._media_type_for_asset(asset_path)
            entries.append(
                f' <manifest:file-entry manifest:full-path="{asset_path}" '
                f'manifest:media-type="{media_type}"/>\n'
            )

        if not entries:
            return

        insert_at = manifest_xml.rfind('</manifest:manifest>')
        if insert_at == -1:
            return

        manifest_xml = manifest_xml[:insert_at] + ''.join(entries) + manifest_xml[insert_at:]
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(manifest_xml)

    def _copy_cover_assets(self, cover_slide: str, cover_dir: Path, existing_dir: Path) -> str:
        """Copy image assets referenced by the cover slide into the destination ODP."""
        copied_assets = []
        for href in sorted(set(re.findall(r'xlink:href="([^"]+)"', cover_slide))):
            if ':' in href or href.startswith('/') or '..' in Path(href).parts:
                continue

            source_path = cover_dir / href
            if not source_path.exists() or not source_path.is_file():
                continue

            target_href = href
            target_path = existing_dir / target_href
            if target_path.exists() and target_path.read_bytes() != source_path.read_bytes():
                source = Path(href)
                base_name = f"cover_{source.name}"
                candidate = source.with_name(base_name)
                counter = 2
                while (existing_dir / candidate.as_posix()).exists():
                    candidate = source.with_name(f"cover_{counter}_{source.name}")
                    counter += 1
                target_href = candidate.as_posix()
                target_path = existing_dir / target_href
                cover_slide = cover_slide.replace(
                    f'xlink:href="{href}"',
                    f'xlink:href="{target_href}"'
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
            copied_assets.append(target_href)

        self._add_manifest_entries(existing_dir, copied_assets)
        return cover_slide

    def _inject_cover_slide(self, odp_path: Path, cover_template: Path, section: str = None, lesson: str = None):
        """Insert or replace the single cover slide in an existing ODP file."""
        if not cover_template or not cover_template.exists():
            print(f"  Cover template not found: {cover_template}")
            return False
        
        if not odp_path.exists():
            print(f"  ODP file not found: {odp_path}")
            return False
        
        try:
            print(f"  Injecting cover slide from template: {cover_template}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Extract existing ODP
                existing_dir = temp_dir / "existing"
                existing_dir.mkdir()
                with zipfile.ZipFile(odp_path, 'r') as zip_ref:
                    zip_ref.extractall(existing_dir)
                
                # Extract cover template
                cover_dir = temp_dir / "cover"
                cover_dir.mkdir()
                with zipfile.ZipFile(cover_template, 'r') as zip_ref:
                    zip_ref.extractall(cover_dir)
                
                # Read content.xml from existing ODP
                content_xml_path = existing_dir / "content.xml"
                with open(content_xml_path, 'r', encoding='utf-8') as f:
                    content_xml = f.read()
                
                # Read content.xml from cover template
                cover_content_xml_path = cover_dir / "content.xml"
                with open(cover_content_xml_path, 'r', encoding='utf-8') as f:
                    cover_content_xml = f.read()
                
                # Extract cover slide from template
                slide_pattern = r'<draw:page(?:\s[^>]*)?>.*?</draw:page>'
                cover_slide_match = re.search(slide_pattern, cover_content_xml, re.DOTALL)
                
                if not cover_slide_match:
                    print("  Warning: Could not find slide in cover template")
                    return False
                
                cover_slide = cover_slide_match.group(0)
                cover_style_names = self._style_names_used_by_slide(cover_slide)
                cover_style_map = {
                    style_name: f"cover_{style_name}"
                    for style_name in cover_style_names
                }
                content_xml = self._merge_cover_automatic_styles(
                    content_xml,
                    cover_content_xml,
                    cover_style_map
                )
                cover_slide = self._rename_style_names(cover_slide, cover_style_map)
                
                # Replace title placeholder when section and/or lesson are provided.
                if section or lesson:
                    formatted_section = path_part_to_title(section) if path_part_to_title else str(section or "").replace('-', ' ')
                    formatted_lesson = path_part_to_title(lesson) if path_part_to_title else str(lesson or "").replace('-', ' ')
                    formatted_section = re.sub(r'\s+', ' ', formatted_section).strip()
                    formatted_lesson = re.sub(r'\s+', ' ', formatted_lesson).strip()

                    # Format: "section\nlesson"
                    title_parts = [part for part in [formatted_section, formatted_lesson] if part]
                    if len(title_parts) == 2 and title_parts[0].lower() == title_parts[1].lower():
                        title_parts = title_parts[:1]
                    title_text = "\n".join(title_parts)
                    # Convert markdown line breaks to ODP format
                    title_text_odp = title_text.replace('\n', '<text:line-break/>')
                    # Replace [título] placeholder
                    cover_slide = cover_slide.replace('[título]', title_text_odp)
                    print(f"  Cover title set to: {' / '.join(title_parts)}")

                cover_slide = self._copy_cover_assets(cover_slide, cover_dir, existing_dir)
                
                # Update cover slide name/id to be page1
                cover_slide = re.sub(r'draw:name="page[0-9]+"', 'draw:name="page1"', cover_slide)
                cover_slide = re.sub(r'draw:id="page[0-9]+"', 'draw:id="page1"', cover_slide)
                cover_structure = self._normalize_cover_slide_structure(cover_slide)
                cover_slide = self._mark_cover_slide(cover_slide)
                
                # Find all existing slides
                slide_matches = list(re.finditer(slide_pattern, content_xml, re.DOTALL))
                if not slide_matches:
                    print("  Warning: Could not find slides in existing ODP")
                    return False

                cover_flags = [
                    self._is_existing_cover_slide(match.group(0), cover_structure)
                    for match in slide_matches
                ]
                existing_cover_count = sum(1 for is_cover in cover_flags if is_cover)
                if existing_cover_count:
                    print(f"  Replacing {existing_cover_count} existing cover slide(s)")
                else:
                    print("  No existing cover slide found; inserting one")

                # Preserve the surrounding office:presentation XML and replace only
                # draw:page blocks. This keeps the ODP valid and prevents duplicate covers.
                new_content_parts = []
                last_end = 0
                next_page_number = 2
                cover_inserted = False
                for match, is_cover in zip(slide_matches, cover_flags):
                    if not cover_inserted:
                        new_content_parts.append(content_xml[last_end:match.start()])
                        new_content_parts.append(cover_slide)
                        new_content_parts.append('\n')
                        cover_inserted = True
                    else:
                        new_content_parts.append(content_xml[last_end:match.start()])

                    if not is_cover:
                        slide = match.group(0)
                        slide = re.sub(r'draw:name="page[0-9]+"', f'draw:name="page{next_page_number}"', slide)
                        slide = re.sub(r'draw:id="page[0-9]+"', f'draw:id="page{next_page_number}"', slide)
                        new_content_parts.append(slide)
                        next_page_number += 1

                    last_end = match.end()

                new_content_parts.append(content_xml[last_end:])
                new_content_xml = ''.join(new_content_parts)
                
                # Write modified content.xml
                with open(content_xml_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_xml)
                
                # Create new ODP file. Keep mimetype first and uncompressed.
                with zipfile.ZipFile(odp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    mimetype_path = existing_dir / "mimetype"
                    if mimetype_path.exists():
                        zipf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
                    for root, dirs, files in os.walk(existing_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(existing_dir)
                            if arcname.as_posix() == "mimetype":
                                continue
                            zipf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)
                
                print(f"✓ Cover slide injected successfully")
                return True
                
        except Exception as e:
            import traceback
            print(f"  Warning: Could not inject cover slide: {e}")
            traceback.print_exc()
            return False

    def _format_odp_title_line(self, text: str) -> str:
        """Format slide title text from slug/lowercase style to display style."""
        text = html.unescape(text).strip()
        if not text:
            return text

        def capitalize_words(value: str) -> str:
            words = re.sub(r'[-_]+', ' ', value).split()
            formatted = []
            for word in words:
                if word.isupper() and len(word) > 1:
                    formatted.append(word)
                elif word:
                    formatted.append(word[:1].upper() + word[1:].lower())
            return ' '.join(formatted)

        lesson_match = re.match(r'^(\d+(?:\.\d+)*)(?:\s*[-_]\s*|\s+)(.+)$', text)
        if lesson_match:
            lesson_number = lesson_match.group(1)
            lesson_title = capitalize_words(lesson_match.group(2))
            return f"{lesson_number} - {lesson_title}"

        return capitalize_words(text)

    def _format_odp_title_fragment(self, fragment: str) -> str:
        """Format a text:p fragment while preserving ODP line breaks."""
        line_break_pattern = r'<text:line-break\s*/>'
        parts = re.split(f'({line_break_pattern})', fragment)
        formatted_parts = []

        for part in parts:
            if not part:
                continue
            if re.fullmatch(line_break_pattern, part):
                formatted_parts.append(part)
                continue

            plain_text = re.sub(r'<[^>]+>', '', part)
            formatted_text = html.escape(self._format_odp_title_line(plain_text))
            if '<' not in part:
                formatted_parts.append(formatted_text)
                continue

            tokens = re.split(r'(<[^>]+>)', part)
            text_inserted = False
            preserved_tokens = []
            for token in tokens:
                if not token:
                    continue
                if token.startswith('<') and token.endswith('>'):
                    preserved_tokens.append(token)
                elif not text_inserted and token.strip():
                    preserved_tokens.append(formatted_text)
                    text_inserted = True
                elif token.strip():
                    continue
                else:
                    preserved_tokens.append(token)

            if not text_inserted:
                preserved_tokens.append(formatted_text)
            formatted_parts.append(''.join(preserved_tokens))

        return ''.join(formatted_parts)

    def review_odp_titles(self, odp_path: Path):
        """Review ODP slide titles and normalize capitalization/lesson slugs."""
        if not odp_path.exists():
            print(f"  ODP file not found: {odp_path}")
            return False

        try:
            print(f"  Reviewing ODP slide titles: {odp_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                existing_dir = temp_dir / "existing"
                existing_dir.mkdir()
                with zipfile.ZipFile(odp_path, 'r') as zip_ref:
                    zip_ref.extractall(existing_dir)

                content_xml_path = existing_dir / "content.xml"
                with open(content_xml_path, 'r', encoding='utf-8') as f:
                    content_xml = f.read()

                slide_pattern = r'<draw:page(?:\s[^>]*)?>.*?</draw:page>'
                title_frame_pattern = r'<draw:frame(?=[^>]*presentation:class="title")[^>]*>.*?</draw:frame>'
                text_p_pattern = r'(<text:p[^>]*>)(.*?)(</text:p>)'

                changes = []

                def extract_plain_title(frame: str) -> str:
                    title_parts = []
                    for _, inner, _ in re.findall(text_p_pattern, frame, re.DOTALL):
                        plain = re.sub(r'<text:line-break\s*/>', ' ', inner)
                        plain = re.sub(r'<[^>]+>', '', plain)
                        title_parts.append(html.unescape(' '.join(plain.split())))
                    return ' | '.join(part for part in title_parts if part)

                def format_text_p(match):
                    opening, inner, closing = match.groups()
                    return f"{opening}{self._format_odp_title_fragment(inner)}{closing}"

                slide_index = 0

                def review_slide(match):
                    nonlocal slide_index
                    slide_index += 1
                    slide = match.group(0)
                    slide_number = slide_index

                    def review_title_frame(frame_match):
                        frame = frame_match.group(0)
                        before = extract_plain_title(frame)
                        updated_frame = re.sub(text_p_pattern, format_text_p, frame, flags=re.DOTALL)
                        after = extract_plain_title(updated_frame)
                        if updated_frame != frame:
                            changes.append((slide_number, before, after))
                        return updated_frame

                    return re.sub(title_frame_pattern, review_title_frame, slide, flags=re.DOTALL)

                new_content_xml = re.sub(slide_pattern, review_slide, content_xml, flags=re.DOTALL)

                if not changes:
                    print("  No slide titles needed changes")
                    return True

                with open(content_xml_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_xml)

                backup_path = odp_path.with_suffix('.odp.title-backup')
                shutil.copy2(odp_path, backup_path)
                print(f"  Backup created: {backup_path}")

                with zipfile.ZipFile(odp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    mimetype_path = existing_dir / "mimetype"
                    if mimetype_path.exists():
                        zipf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
                    for root, dirs, files in os.walk(existing_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(existing_dir)
                            if arcname.as_posix() == "mimetype":
                                continue
                            zipf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)

                print(f"  Updated {len(changes)} slide title(s)")
                for slide_number, before, after in changes[:10]:
                    print(f"    Slide {slide_number}: {before} -> {after}")
                if len(changes) > 10:
                    print(f"    ... {len(changes) - 10} more title changes")
                print(f"✓ ODP slide titles reviewed successfully: {odp_path}")
                return True

        except Exception as e:
            import traceback
            print(f"  Warning: Could not review ODP slide titles: {e}")
            traceback.print_exc()
            return False
    
    def recover_odp_duplicates(self, odp_path: Path):
        """Recover ODP file by removing duplicate slides."""
        if not odp_path.exists():
            print(f"  ODP file not found: {odp_path}")
            return False
        
        try:
            print(f"  Recovering ODP file: {odp_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Extract ODP
                existing_dir = temp_dir / "existing"
                existing_dir.mkdir()
                with zipfile.ZipFile(odp_path, 'r') as zip_ref:
                    zip_ref.extractall(existing_dir)
                
                # Read content.xml
                content_xml_path = existing_dir / "content.xml"
                with open(content_xml_path, 'r', encoding='utf-8') as f:
                    content_xml = f.read()
                
                # Extract all slides
                slide_pattern = r'<draw:page(?:\s[^>]*)?>.*?</draw:page>'
                slide_matches = list(re.finditer(slide_pattern, content_xml, re.DOTALL))
                all_slides = [match.group(0) for match in slide_matches]
                
                if not all_slides:
                    print("  No slides found in ODP")
                    return False
                
                print(f"  Found {len(all_slides)} slides")
                
                # Detect duplicates by comparing normalized slide XML. Text-only
                # comparison can mark every image/shape-only slide as duplicate
                # because the extracted text is empty. Before the generic pass,
                # handle the common broken state where a new cover was inserted
                # before an older equivalent cover; in recovery, keep the older
                # second page and remove the added first page.
                seen_slides = {}
                slides_to_keep = [True] * len(all_slides)
                duplicates_count = 0

                if len(all_slides) >= 2 and self._are_semantic_cover_duplicates(all_slides[0], all_slides[1]):
                    slides_to_keep[0] = False
                    duplicates_count += 1
                    first_title = self._extract_slide_title_text(all_slides[0])
                    second_title = self._extract_slide_title_text(all_slides[1])
                    print("  Semantic duplicate covers found at positions 1 and 2")
                    print(f"  Removing added cover at position 1; keeping previous cover at position 2")
                    print(f"    Cover 1 title: {first_title}")
                    print(f"    Cover 2 title: {second_title}")

                for i, slide in enumerate(all_slides):
                    if not slides_to_keep[i]:
                        continue

                    normalized_slide = re.sub(r'\s+draw:(?:name|id)="page[0-9]+"', '', slide)
                    normalized_slide = re.sub(r'>\s+<', '><', normalized_slide)
                    normalized_slide = ' '.join(normalized_slide.split())
                    slide_hash = hashlib.sha256(normalized_slide.encode('utf-8')).hexdigest()
                    
                    if slide_hash in seen_slides:
                        duplicates_count += 1
                        print(f"  Duplicate slide found at position {i+1} (same as position {seen_slides[slide_hash]+1})")
                        slides_to_keep[i] = False
                    else:
                        seen_slides[slide_hash] = i
                        slides_to_keep[i] = True
                
                if duplicates_count == 0:
                    print("  No duplicate slides found")
                    return True
                
                print(f"  Removed {duplicates_count} duplicate slides")
                print(f"  Remaining slides: {len(all_slides) - duplicates_count}")
                
                # Rebuild content.xml by replacing/removing only draw:page blocks.
                # The previous implementation replaced the whole office:body and
                # dropped the required office:presentation wrapper.
                new_content_parts = []
                last_end = 0
                next_page_number = 1
                for match, keep_slide in zip(slide_matches, slides_to_keep):
                    new_content_parts.append(content_xml[last_end:match.start()])
                    if keep_slide:
                        slide = match.group(0)
                        slide = re.sub(r'draw:name="page[0-9]+"', f'draw:name="page{next_page_number}"', slide)
                        slide = re.sub(r'draw:id="page[0-9]+"', f'draw:id="page{next_page_number}"', slide)
                        new_content_parts.append(slide)
                        next_page_number += 1
                    last_end = match.end()
                new_content_parts.append(content_xml[last_end:])
                new_content_xml = ''.join(new_content_parts)
                
                # Write modified content.xml
                with open(content_xml_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_xml)
                
                # Create backup of original ODP
                backup_path = odp_path.with_suffix('.odp.backup')
                shutil.copy2(odp_path, backup_path)
                print(f"  Backup created: {backup_path}")
                
                # Create new ODP file. ODF readers expect mimetype first and
                # uncompressed, so preserve that convention when present.
                with zipfile.ZipFile(odp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    mimetype_path = existing_dir / "mimetype"
                    if mimetype_path.exists():
                        zipf.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
                    for root, dirs, files in os.walk(existing_dir):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(existing_dir)
                            if arcname.as_posix() == "mimetype":
                                continue
                            zipf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)
                
                print(f"✓ ODP recovered successfully: {odp_path}")
                return True
                
        except Exception as e:
            import traceback
            print(f"  Warning: Could not recover ODP: {e}")
            traceback.print_exc()
            return False
    
    def convert_odp_to_pdf(self, odp_path: Path, pdf_path: Path):
        """Convert ODP to PDF using LibreOffice."""
        try:
            import subprocess
            
            print(f"  Converting ODP to PDF: {odp_path} -> {pdf_path}")
            
            # Try to find LibreOffice
            libreoffice_paths = [
                'libreoffice',
                'soffice',
                '/usr/bin/libreoffice',
                '/usr/bin/soffice',
                '/Applications/LibreOffice.app/Contents/MacOS/soffice'
            ]
            
            libreoffice_cmd = None
            for path in libreoffice_paths:
                try:
                    result = subprocess.run(['which', path], capture_output=True, text=True)
                    if result.returncode == 0:
                        libreoffice_cmd = path
                        break
                except:
                    continue
            
            if not libreoffice_cmd:
                print("  Warning: LibreOffice not found, skipping PDF conversion")
                print("  Install LibreOffice to enable ODP to PDF conversion")
                return
            
            # Convert using headless mode
            cmd = [
                libreoffice_cmd,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', str(pdf_path.parent),
                str(odp_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ PDF converted successfully: {pdf_path}")
            else:
                print(f"  Warning: PDF conversion failed: {result.stderr}")
                
        except Exception as e:
            print(f"  Warning: Could not convert ODP to PDF: {e}")
    
    def generate_slides_pdf(self, slides: List[dict], pdf_path: Path):
        """Generate PDF from slides with markdown formatting support."""
        try:
            from reportlab.lib.pagesizes import letter, landscape
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Frame
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib import colors
            import textwrap
            import re
            
            # Try to register fonts
            try:
                pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
                pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
                pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'))
                pdfmetrics.registerFont(TTFont('DejaVuSans-BoldOblique', '/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf'))
                font_name = 'DejaVuSans'
                font_bold = 'DejaVuSans-Bold'
                font_italic = 'DejaVuSans-Oblique'
                font_bold_italic = 'DejaVuSans-BoldOblique'
            except:
                font_name = 'Helvetica'
                font_bold = 'Helvetica-Bold'
                font_italic = 'Helvetica-Oblique'
                font_bold_italic = 'Helvetica-BoldOblique'
            
            # Use landscape orientation
            pagesize = landscape(letter)
            width, height = pagesize
            
            # Define margins
            left_margin = 50
            right_margin = 50
            top_margin = 80
            bottom_margin = 50
            
            # Calculate printable width
            printable_width = width - left_margin - right_margin
            
            c = canvas.Canvas(str(pdf_path), pagesize=pagesize)
            
            for slide in slides:
                c.setFont(font_bold, 28)
                # Draw title with proper margin
                title_y = height - top_margin
                # Convert markdown in title
                title_formatted = self._convert_markdown_to_reportlab(slide['title'])
                title_paragraph = Paragraph(title_formatted, ParagraphStyle(
                    'Title',
                    fontName=font_bold,
                    fontSize=28,
                    textColor=colors.black,
                    spaceAfter=40,  # Increased from 20 to 40 (2 lines of spacing)
                    leading=34
                ))
                title_paragraph.wrapOn(c, printable_width, height)
                title_paragraph.drawOn(c, left_margin, title_y)
                
                y_position = title_y - 100  # Increased from 80 to 100 for more spacing
                c.setFont(font_name, 18)
                
                # Draw content with markdown formatting
                for content in slide['content'][:10]:  # Increased from 5 to 10
                    # Convert markdown to reportlab format
                    content_formatted = self._convert_markdown_to_reportlab(content)
                    content_paragraph = Paragraph(content_formatted, ParagraphStyle(
                        'Content',
                        fontName=font_name,
                        fontSize=18,
                        textColor=colors.black,
                        spaceAfter=10,
                        leading=22
                    ))
                    content_paragraph.wrapOn(c, printable_width, height)
                    content_paragraph.drawOn(c, left_margin, y_position)
                    y_position -= content_paragraph.height + 15
                    if y_position < bottom_margin + 30:
                        break  # Stop if we're at the bottom of the page
                
                # Draw bullets with markdown formatting
                for bullet in slide['bullets'][:15]:  # Increased from 8 to 15
                    if y_position < bottom_margin + 30:
                        break  # Stop if we're at the bottom of the page
                    # Convert markdown to reportlab format
                    bullet_formatted = self._convert_markdown_to_reportlab(f"• {bullet}")
                    bullet_paragraph = Paragraph(bullet_formatted, ParagraphStyle(
                        'Bullet',
                        fontName=font_name,
                        fontSize=18,
                        textColor=colors.black,
                        spaceAfter=8,
                        leading=22,
                        leftIndent=20
                    ))
                    bullet_paragraph.wrapOn(c, printable_width - 20, height)
                    bullet_paragraph.drawOn(c, left_margin, y_position)
                    y_position -= bullet_paragraph.height + 10
                
                c.showPage()
            
            c.save()
            print(f"✓ Slides PDF saved to: {pdf_path}")
            
        except ImportError:
            print("  Warning: reportlab not installed, skipping PDF generation")
            print("  Install with: pip install reportlab")
        except Exception as e:
            print(f"  Warning: Could not generate PDF: {e}")
    
    def _convert_markdown_to_reportlab(self, text: str) -> str:
        """Convert markdown formatting to reportlab HTML-like markup."""
        # Escape HTML special characters first
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Convert bold: **text** or __text__ to <b>text</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # Convert italic: *text* or _text_ to <i>text</i>
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
        
        # Convert code: `text` to <font face="Courier">text</font>
        text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
        
        # Convert strikethrough: ~~text~~ to <strike>text</strike>
        text = re.sub(r'~~(.+?)~~', r'<strike>\1</strike>', text)
        
        # Convert subscript: <sub>text</sub> (already HTML-like)
        # Convert superscript: <sup>text</sup> (already HTML-like)
        # Convert underline: <u>text</u> (already HTML-like)
        
        return text
    
    def _convert_markdown_to_odp(self, text: str) -> str:
        """Convert markdown formatting to ODP XML markup."""
        # Escape XML special characters first
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        
        # Convert bold: **text** or __text__ to <text:span text:style-name="T2">text</text:span>
        text = re.sub(r'\*\*(.+?)\*\*', r'<text:span text:style-name="T2">\1</text:span>', text)
        text = re.sub(r'__(.+?)__', r'<text:span text:style-name="T2">\1</text:span>', text)
        
        # Convert italic: *text* or _text_ to <text:span text:style-name="T1">text</text:span>
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<text:span text:style-name="T1">\1</text:span>', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<text:span text:style-name="T1">\1</text:span>', text)
        
        # Convert code: `text` to <text:span text:style-name="Code">text</text:span>
        text = re.sub(r'`(.+?)`', r'<text:span text:style-name="Code">\1</text:span>', text)
        
        # Convert strikethrough: ~~text~~ to <text:span text:style-name="Strike">text</text:span>
        text = re.sub(r'~~(.+?)~~', r'<text:span text:style-name="Strike">\1</text:span>', text)
        
        # Convert subscript: <sub>text</sub> to <text:span text:style-name="Sub">text</text:span>
        text = re.sub(r'<sub>(.+?)</sub>', r'<text:span text:style-name="Sub">\1</text:span>', text)
        
        # Convert superscript: <sup>text</sup> to <text:span text:style-name="Sup">text</text:span>
        text = re.sub(r'<sup>(.+?)</sup>', r'<text:span text:style-name="Sup">\1</text:span>', text)
        
        # Convert underline: <u>text</u> to <text:span text:style-name="Underline">text</text:span>
        text = re.sub(r'<u>(.+?)</u>', r'<text:span text:style-name="Underline">\1</text:span>', text)
        
        return text
    
    def _format_time_mm_ss(self, seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


class SpectrumGenerator:
    """Generates radial FFT audio spectrum visualizations."""
    
    def __init__(self, resolution: str = "1080p", video_output_dir: str = None):
        self.resolution = resolution
        self.video_output_dir = video_output_dir
        self.width, self.height = get_video_dimensions(resolution)
        self.fps = 30
        self.spectrum_progress_file = Path(".progress_spectrum.txt")
        self.processed_spectrums: Set[str] = set()
        self._load_spectrum_progress()
    
    def _load_spectrum_progress(self):
        """Load previously processed spectrum videos from progress file."""
        if self.spectrum_progress_file.exists():
            with open(self.spectrum_progress_file, 'r', encoding='utf-8') as f:
                self.processed_spectrums = set(line.strip() for line in f if line.strip())
    
    def _save_spectrum_progress(self, file_path: str):
        """Save a successfully processed spectrum video to progress file."""
        with open(self.spectrum_progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n")
        self.processed_spectrums.add(file_path)
    
    def _clear_spectrum_progress(self):
        """Clear spectrum video progress file."""
        if self.spectrum_progress_file.exists():
            self.spectrum_progress_file.unlink()
        self.processed_spectrums.clear()
    
    def generate_spectrum_video(
        self,
        audio_path: Path,
        output_path: Path,
        force: bool = False
    ):
        """Generate spectrum video from audio."""
        if SpectrumVisualizer is None:
            print("✗ Spectrum visualizer not available")
            print("  Install with: pip install numpy")
            return
        
        # Check if video already exists
        if output_path.exists() and not force:
            print(f"⊘ Skipping (spectrum video already exists): {output_path}")
            return
        
        print(f"Generating spectrum video: {output_path}")
        
        try:
            # Initialize visualizer
            visualizer = SpectrumVisualizer(
                width=self.width,
                height=self.height,
                fps=self.fps
            )
            
            # Analyze audio
            print("  Analyzing audio with FFT...")
            spectrum_data, duration = visualizer.analyze_audio_fft(audio_path)
            
            # Generate video
            success = visualizer.generate_video(
                audio_path,
                output_path,
                spectrum_data,
                duration
            )
            
            if success:
                print(f"✓ Spectrum video saved to: {output_path}")
                self._save_spectrum_progress(str(output_path))
            else:
                print(f"✗ Error generating spectrum video")
                
        except Exception as e:
            print(f"✗ Error generating spectrum video: {e}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Interrupted by user. Exiting gracefully...")
    restore_terminal()
    sys.exit(0)

def main():
    # Register terminal cleanup on exit
    atexit.register(restore_terminal)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Convert text/markdown files to audio using OpenAI or Google TTS APIs"
    )
    parser.add_argument(
        "api",
        choices=["openai", "google"],
        help="API to use for text-to-speech conversion"
    )
    parser.add_argument(
        "path",
        help="Path to file or directory to convert"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for audio files (default: same directory as input file)"
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="API key (for OpenAI) or credentials path (for Google Cloud)"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-process files even if audio already exists"
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="nova",
        help="Voice for OpenAI TTS (default: nova, options: alloy, echo, fable, onyx, nova, shimmer, ash, sage, coral)"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate video from audio with animated SVG professor"
    )
    parser.add_argument(
        "--video-output-dir",
        default=None,
        help="Output directory for video files (default: same directory as input file)"
    )
    parser.add_argument(
        "--resolution",
        default="1080p",
        choices=["720p", "1080p", "4k"],
        help="Video resolution (default: 1080p)"
    )
    parser.add_argument(
        "--background-color",
        default="#f0f4f8",
        help="Video background color (default: #f0f4f8)"
    )
    parser.add_argument(
        "--slide",
        action="store_true",
        help="Generate slide-based video from audio (overrides --video)"
    )
    parser.add_argument(
        "--slide-duration",
        type=float,
        default=5.0,
        help="Default duration per slide in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--odp-template",
        default=None,
        help="Path to ODP template file for slide generation (from ODP_TEMPLATE env var, requires --odp)"
    )
    parser.add_argument(
        "--cover-template",
        default=None,
        help="Path to cover slide template file (.otp) to inject as first slide (default: from COVER_TEMPLATE env var, requires --odp)"
    )
    parser.add_argument(
        "--cover-time",
        type=float,
        default=20.0,
        help="Duration in seconds for cover slide pause in video (default: 20.0 from COVER_TIME env var)"
    )
    parser.add_argument(
        "--odp-recover",
        action="store_true",
        help="Recover ODP files by removing duplicate slides"
    )
    parser.add_argument(
        "--odp-review-titles",
        "--odp-title-review",
        dest="odp_review_titles",
        action="store_true",
        help="Review ODP slide titles, normalizing capitalization and lesson slugs"
    )
    parser.add_argument(
        "--odp",
        action="store_true",
        help="Generate ODP presentation from slides (requires --odp-template)"
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate PDF from slides (if --odp is used, converts ODP to PDF)"
    )
    parser.add_argument(
        "--slides-only",
        action="store_true",
        help="Generate only slides (ODP/PDF) without audio or video. Use this to create/edit slides first, then run again with --use-existing-slides"
    )
    parser.add_argument(
        "--use-existing-slides",
        action="store_true",
        help="Use existing ODP/PDF slides to generate synchronized audio and video. Requires --slide flag"
    )
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Generate video with radial FFT audio spectrum visualizer (overrides --video and --slide)"
    )
    parser.add_argument(
        "--no-threads",
        action="store_true",
        help="Disable multi-threading (process files sequentially)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use (default: 4)"
    )
    parser.add_argument(
        "--log-summary",
        action="store_true",
        help="Show summary of processed files and exit"
    )
    parser.add_argument(
        "--log-pending",
        action="store_true",
        help="Show list of files pending upload to site and exit"
    )
    parser.add_argument(
        "--log-remove",
        help="Remove a file from processing log (path to file)"
    )
    parser.add_argument(
        "--log-remove-type",
        default="mp3",
        help="File type to remove (mp3, pdf, odp, mp4)"
    )
    parser.add_argument(
        "--log-remove-section",
        help="Remove an entire section from processing log"
    )
    parser.add_argument(
        "--time-course",
        type=float,
        default=None,
        help="Percentage of paragraphs to sample for time estimation (default: 5.0%%)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Calculate real duration from existing audio files"
    )
    parser.add_argument(
        "--env",
        "-e",
        default=None,
        help="Path to custom environment file (.env) to load variables from"
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        choices=["pt-br", "en", "es", "fr", "de"],
        help="Course language (default: from COURSE_LANGUAGE env var or pt-br). When not pt-br, creates translated files with language suffix"
    )
    
    args = parser.parse_args()
    
    # Load custom environment file if provided
    if args.env and load_dotenv:
        env_path = Path(args.env)
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment variables from: {env_path}")
        else:
            print(f"⚠️  Environment file not found: {env_path}")
    elif load_dotenv:
        # Load default .env file if no custom file specified
        load_dotenv()
    
    # Determine language from argument, environment, or default
    language = args.language or os.environ.get("COURSE_LANGUAGE", "pt-br")
    print(f"🌐 Course language: {language}")
    
    # Load ODP-related defaults from environment when --odp or --use-existing-slides is used
    if args.odp or args.use_existing_slides:
        if args.odp_template is None:
            args.odp_template = os.environ.get("ODP_TEMPLATE", "template/template-1-azul-quadriculado.otp")
        if args.cover_template is None:
            args.cover_template = os.environ.get("COVER_TEMPLATE", None)
        if args.cover_time == 20.0:  # Check if still at default
            args.cover_time = float(os.environ.get("COVER_TIME", "20.0"))
    
    # Always load cover_time from env if not explicitly set (for audio-only mode)
    if args.cover_time == 20.0 and not args.odp and not args.use_existing_slides:
        args.cover_time = float(os.environ.get("COVER_TIME", "5"))
    
    # Handle --time-course estimation (must be before other validations)
    if args.time_course or args.all:
        print("🎯 Course time estimation mode enabled")
        
        try:
            # Initialize estimator
            estimator = CourseTimeEstimator(
                args.api,
                args.api_key,
                args.voice
            )
            
            input_path = Path(args.path)
            
            if not input_path.exists():
                print(f"✗ Path not found: {input_path}")
                sys.exit(1)
            
            if not input_path.is_dir():
                print(f"✗ --time-course requires a directory path, not a file")
                sys.exit(1)
            
            # Prepare data dictionary for report
            data = {
                'api': args.api,
                'voice': args.voice,
                'total_paragraphs': 0,
                'total_files': 0,
                'avg_duration': 0.0,
                'total_time': 0.0,
                'total_time_formatted': '',
                'total_time_hours': 0.0,
                'sample_percentage': args.time_course
            }
            
            if args.all:
                # Mode: Calculate real duration from existing audio files
                print("📊 Mode: Real duration calculation (--all)")
                
                total_duration = estimator.calculate_real_duration(input_path)
                
                # Count total files
                files = list(input_path.rglob("*.md"))
                files = [f for f in files if not f.name.startswith("00-")]
                data['total_files'] = len(files)
                audio_files = list(input_path.rglob(f"*_{args.api}.mp3"))
                audio_files = [f for f in audio_files if not should_skip_path(f)]
                data['audio_files_count'] = len(audio_files)
                
                # Calculate average duration per file
                if data['total_files'] > 0:
                    data['avg_duration'] = total_duration / data['total_files']
                else:
                    data['avg_duration'] = 0.0
                
                data['total_time'] = total_duration
                data['total_time_formatted'] = estimator.format_duration(total_duration)
                data['total_time_hours'] = total_duration / 3600.0
                
                # Generate report
                estimator.generate_report(data, mode='all')
                
            else:
                # Mode: Sampling estimation
                print("📊 Mode: Sampling estimation")
                
                # Collect all paragraphs
                paragraphs = estimator.collect_all_paragraphs(input_path)
                
                if not paragraphs:
                    print("✗ No paragraphs found in the course files")
                    sys.exit(1)
                
                data['total_paragraphs'] = len(paragraphs)
                
                # Count total files
                files = list(input_path.rglob("*.md"))
                files = [f for f in files if not f.name.startswith("00-") and not should_skip_path(f)]
                data['total_files'] = len(files)
                
                # Build section breakdown
                section_breakdown = {}
                for _, _, section in paragraphs:
                    section_breakdown[section] = section_breakdown.get(section, 0) + 1
                data['section_breakdown'] = section_breakdown
                
                # Calculate sample size based on percentage
                sample_size = max(1, int(len(paragraphs) * (args.time_course / 100.0)))
                print(f"📊 Sampling {args.time_course}% of paragraphs ({sample_size} out of {len(paragraphs)})")
                
                # Sample paragraphs
                sample = estimator.sample_paragraphs(paragraphs, n=sample_size)
                print(f"📊 Selected {len(sample)} paragraphs for sampling")
                
                # Generate sample audio
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    audio_files = estimator.generate_sample_audio(sample, temp_dir)
                    
                    if not audio_files:
                        print("✗ No audio files generated for sampling")
                        sys.exit(1)
                    
                    # Calculate average duration
                    avg_duration = estimator.calculate_average_duration(audio_files)
                    data['avg_duration'] = avg_duration
                    
                    # Estimate total time
                    total_time = estimator.estimate_total_time(len(paragraphs), avg_duration)
                    data['total_time'] = total_time
                    data['total_time_formatted'] = estimator.format_duration(total_time)
                    data['total_time_hours'] = total_time / 3600.0
                    
                    # Generate report
                    estimator.generate_report(data, mode='sample')
            
            print("✓ Course time estimation completed")
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ Error in course time estimation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    odp_maintenance_mode = args.odp_recover or args.odp_review_titles

    # Validate: --odp-template requires --odp (unless using existing slides)
    if args.odp_template and not args.odp and not odp_maintenance_mode and not args.use_existing_slides:
        print("✗ Error: --odp-template requires --odp flag")
        print("  Use --odp to generate ODP presentation with the template")
        sys.exit(1)
    
    # Validate: --cover-template requires --odp (unless using existing slides)
    if args.cover_template and not args.odp and not odp_maintenance_mode and not args.use_existing_slides:
        print("✗ Error: --cover-template requires --odp flag")
        print("  Use --odp to generate ODP presentation with the cover template")
        sys.exit(1)
    
    # Handle log management commands
    if ProcessingLog and (args.log_summary or args.log_pending or args.log_remove or args.log_remove_section):
        log = ProcessingLog()
        if args.log_summary:
            log.print_summary()
        elif args.log_pending:
            pending = log.get_pending_upload_list()
            print(f"\nFiles Pending Upload ({len(pending)}):")
            print(f"{'='*60}")
            for entry in pending:
                print(f"  {entry}")
            print(f"\n{'='*60}\n")
        elif args.log_remove:
            log.remove_lesson(Path(args.log_remove), args.log_remove_type)
        elif args.log_remove_section:
            log.remove_section(args.log_remove_section)
        sys.exit(0)
    
    # Initialize converter
    try:
        use_threads = not args.no_threads
        
        # Validate thread count against system limits
        if use_threads:
            max_system_threads = multiprocessing.cpu_count()
            if args.threads > max_system_threads:
                print(f"✗ Error: Requested {args.threads} threads, but system only has {max_system_threads} CPU cores")
                print(f"  Use --threads {max_system_threads} or less")
                sys.exit(1)
        
        converter = TextToAudioConverter(
            args.api, 
            args.api_key, 
            args.voice,
            use_threads=use_threads,
            max_threads=args.threads,
            language=language,
            cover_time=getattr(args, 'cover_time', None) or float(os.environ.get("COVER_TIME", "5"))
        )
        if use_threads:
            print(f"Multi-threading enabled (max {args.threads} threads)")
        else:
            print("Multi-threading disabled (sequential processing)")
    except Exception as e:
        print(f"✗ Error initializing converter: {e}")
        sys.exit(1)
    
    # --spectrum overrides --video and --slide
    if args.spectrum:
        args.video = False
        args.slide = False
        args.odp = False
        args.pdf = False
        print("Spectrum mode enabled (overrides video, slide, odp and pdf modes)")
    
    # --slide overrides --video
    if args.slide:
        args.video = False
        print("Slide mode enabled (overrides video mode)")
    
    # --odp or --pdf without --slide enables slide mode
    if (args.odp or args.pdf) and not args.slide:
        args.slide = True
        print("ODP/PDF mode enabled (enables slide mode)")
    
    # Validate --use-existing-slides requires --slide
    if args.use_existing_slides and not args.slide:
        print("✗ Error: --use-existing-slides requires --slide flag")
        sys.exit(1)
    
    # --slides-only mode: disable audio and video generation
    if args.slides_only:
        if not (args.odp or args.pdf):
            print("✗ Error: --slides-only requires --odp or --pdf flag")
            sys.exit(1)
        print("Slides-only mode enabled (generating ODP/PDF without audio or video)")
        # Disable audio generation by setting a flag that convert_file will check
        args.skip_audio = True
    
    # Initialize video generator if requested
    video_generator = None
    if args.video:
        if ProfessorSVG is None or extract_code_blocks is None or detect_content_type is None or get_video_dimensions is None:
            print("✗ Video generation dependencies not available")
            print("  Install with: pip install moviepy pillow svglib reportlab")
            print("  Continuing with audio-only mode...")
        else:
            try:
                video_generator = VideoGenerator(
                    resolution=args.resolution,
                    background_color=args.background_color,
                    video_output_dir=args.video_output_dir,
                    voice=args.voice
                )
                print(f"Video generation enabled (professor gender: {video_generator.professor_gender})")
            except Exception as e:
                print(f"✗ Error initializing video generator: {e}")
                print("  Continuing with audio-only mode...")
    
    # Initialize slide generator if requested. ODP maintenance modes use the
    # ODP helpers only, so they do not need MoviePy slide-video dependencies.
    slide_generator = None
    if args.slide or odp_maintenance_mode:
        if args.slide and not odp_maintenance_mode and (VideoFileClip is None or AudioFileClip is None or ImageClip is None):
            print("✗ Slide generation dependencies not available")
            print("  Install with: pip install moviepy pillow")
            print("  Continuing with audio-only mode...")
        else:
            try:
                slide_generator = SlideGenerator(
                    resolution=args.resolution,
                    background_color=args.background_color,
                    video_output_dir=args.video_output_dir,
                    slide_duration=args.slide_duration,
                    odp_template=args.odp_template,
                    cover_template=args.cover_template,
                    cover_time=args.cover_time,
                    language=language
                )
                if odp_maintenance_mode:
                    print("ODP maintenance helpers enabled")
                elif args.odp_template:
                    print(f"Slide generation enabled with ODP template: {args.odp_template}")
                else:
                    print(f"Slide generation enabled")
            except Exception as e:
                print(f"✗ Error initializing slide generator: {e}")
                print("  Continuing with audio-only mode...")
    
    # Initialize spectrum generator if requested
    spectrum_generator = None
    if args.spectrum:
        if SpectrumVisualizer is None or get_video_dimensions is None or AudioFileClip is None:
            print("✗ Spectrum generation dependencies not available")
            print("  Install with: pip install numpy moviepy")
            print("  Continuing with audio-only mode...")
        else:
            try:
                spectrum_generator = SpectrumGenerator(
                    resolution=args.resolution,
                    video_output_dir=args.video_output_dir
                )
                print(f"Spectrum generation enabled")
            except Exception as e:
                print(f"✗ Error initializing spectrum generator: {e}")
                print("  Continuing with audio-only mode...")
    
    # Handle ODP recovery mode
    if args.odp_recover:
        print("ODP recovery mode enabled")
        if not slide_generator:
            print("✗ Error: --odp-recover requires slide generator (--slide or --odp)")
            sys.exit(1)
        
        path_str = args.path
        has_wildcards = '*' in path_str or '?' in path_str
        
        if has_wildcards:
            matched_files = glob.glob(path_str, recursive=True)
            matched_files = [f for f in matched_files if f.endswith('.odp')]
            if not matched_files:
                print(f"✗ No ODP files found matching pattern: {path_str}")
                sys.exit(1)
            
            print(f"Found {len(matched_files)} ODP files to recover")
            for odp_file in sorted(matched_files):
                odp_path = Path(odp_file)
                print(f"\nRecovering: {odp_path}")
                slide_generator.recover_odp_duplicates(odp_path)
        else:
            input_path = Path(args.path)
            if input_path.is_file() and input_path.suffix == '.odp':
                print(f"Recovering single ODP file: {input_path}")
                slide_generator.recover_odp_duplicates(input_path)
            elif input_path.is_dir():
                # Find all ODP files in directory
                odp_files = list(input_path.glob('**/*.odp'))
                if not odp_files:
                    print(f"✗ No ODP files found in directory: {input_path}")
                    sys.exit(1)
                
                print(f"Found {len(odp_files)} ODP files to recover")
                for odp_file in sorted(odp_files):
                    print(f"\nRecovering: {odp_file}")
                    slide_generator.recover_odp_duplicates(odp_file)
            else:
                print(f"✗ Invalid path for ODP recovery: {input_path}")
                sys.exit(1)
        
        print("\nODP recovery completed")
        sys.exit(0)

    # Handle ODP title review mode
    if args.odp_review_titles:
        print("ODP title review mode enabled")
        if not slide_generator:
            print("✗ Error: --odp-review-titles requires ODP helpers")
            sys.exit(1)

        path_str = args.path
        has_wildcards = '*' in path_str or '?' in path_str

        if has_wildcards:
            matched_files = glob.glob(path_str, recursive=True)
            matched_files = [f for f in matched_files if f.endswith('.odp')]
            if not matched_files:
                print(f"✗ No ODP files found matching pattern: {path_str}")
                sys.exit(1)

            print(f"Found {len(matched_files)} ODP files to review titles")
            for odp_file in sorted(matched_files):
                odp_path = Path(odp_file)
                print(f"\nReviewing titles: {odp_path}")
                slide_generator.review_odp_titles(odp_path)
        else:
            input_path = Path(args.path)
            if input_path.is_file() and input_path.suffix == '.odp':
                print(f"Reviewing titles in single ODP file: {input_path}")
                slide_generator.review_odp_titles(input_path)
            elif input_path.is_dir():
                odp_files = list(input_path.glob('**/*.odp'))
                if not odp_files:
                    print(f"✗ No ODP files found in directory: {input_path}")
                    sys.exit(1)

                print(f"Found {len(odp_files)} ODP files to review titles")
                for odp_file in sorted(odp_files):
                    print(f"\nReviewing titles: {odp_file}")
                    slide_generator.review_odp_titles(odp_file)
            else:
                print(f"✗ Invalid path for ODP title review: {input_path}")
                sys.exit(1)

        print("\nODP title review completed")
        sys.exit(0)
    
    # Determine input path and handle wildcards
    path_str = args.path
    has_wildcards = '*' in path_str or '?' in path_str
    
    if has_wildcards:
        # Expand glob pattern
        matched_files = glob.glob(path_str, recursive=True)
        # Filter out files in recursos/resources folders
        matched_files = [f for f in matched_files if not should_skip_path(Path(f))]
        if not matched_files:
            print(f"✗ No files found matching pattern: {path_str}")
            sys.exit(1)
        
        print(f"Found {len(matched_files)} files matching pattern")
        for file_path in sorted(matched_files):
            file_path = Path(file_path)
            if file_path.is_file():
                print(f"\nProcessing: {file_path}")
                converter.convert_file(file_path, args.force, video_generator, slide_generator, spectrum_generator, args.odp, args.pdf, getattr(args, 'skip_audio', False), getattr(args, 'use_existing_slides', False))
    else:
        input_path = Path(args.path)
        
        # Process file or directory
        if input_path.is_file():
            print(f"Processing single file: {input_path}")
            converter.convert_file(input_path, args.force, video_generator, slide_generator, spectrum_generator, args.odp, args.pdf, getattr(args, 'skip_audio', False), getattr(args, 'use_existing_slides', False))
        elif input_path.is_dir():
            print(f"Processing directory: {input_path}")
            converter.convert_directory(input_path, args.force, video_generator, slide_generator, spectrum_generator, args.odp, args.pdf, getattr(args, 'skip_audio', False), getattr(args, 'use_existing_slides', False))
        else:
            print(f"✗ Path not found: {input_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
