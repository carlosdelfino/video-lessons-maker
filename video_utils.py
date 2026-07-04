#!/usr/bin/env python3
"""
Video Utilities
Helper functions for video composition and editing.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile

NUMBERED_PREFIX_RE = re.compile(r'^\d{2}(?:\.\d+)*(?:[-_\s]+|$)')
NUMBERED_PREFIX_PARTS_RE = re.compile(r'^(\d{2}(?:\.\d+)*)(?:[-_\s]+|$)')
GENERIC_LESSON_STEMS = {"texto-aula", "aula", "script", "roteiro"}


def has_numbered_prefix(name: str) -> bool:
    """Return True for course path parts like 01-Intro or 06.3.1-Topic."""
    return bool(NUMBERED_PREFIX_RE.match(name or ""))


def numbered_prefix_parts(name: str) -> Tuple[int, ...]:
    """Return numeric prefix parts for sorting, such as 06.3.1 -> (6, 3, 1)."""
    match = NUMBERED_PREFIX_PARTS_RE.match(name or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def numbered_prefix_value(name: str) -> Optional[int]:
    """Return the first two-digit prefix value, or None when absent."""
    parts = numbered_prefix_parts(name)
    return parts[0] if parts else None


def strip_numbered_prefix(name: str) -> str:
    """Remove the leading numeric course prefix from a path part."""
    return NUMBERED_PREFIX_RE.sub("", name or "", count=1)


def path_part_to_title(name: str) -> str:
    """Convert a numbered slug/path part to readable text."""
    title = strip_numbered_prefix(name)
    title = re.sub(r'[-_]+', ' ', title)
    title = re.sub(r'\s+', ' ', title)
    if title and title == title.lower():
        title = ' '.join(word.capitalize() for word in title.split())
    return title.strip()


def infer_course_name(file_path: Path) -> str:
    """Infer the course directory from common source folders."""
    source_dirs = {"apostila", "scripts da aula", "script das aulas"}
    parts = list(file_path.parts)
    for i, part in enumerate(parts):
        if part.lower() in source_dirs and i > 0:
            return parts[i - 1]
    if len(parts) >= 3:
        return parts[-3]
    return file_path.parent.name or "Course"


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown text."""
    pattern = r'```[\s\S]*?```'
    return re.findall(pattern, text)


def detect_content_type(text: str) -> str:
    """Detect content type for expression selection."""
    text_lower = text.lower()
    
    # Check for practice/exercise
    if any(keyword in text_lower for keyword in ['pratique', 'exercício', 'exercise', 'practice']):
        return "questioning"
    
    # Check for important/emphasis
    if any(keyword in text_lower for keyword in ['importante', 'lembre-se', 'atenção', 'atenção', 'note', 'remember']):
        return "enthusiastic"
    
    # Check for code blocks
    if '```' in text:
        return "thoughtful"
    
    return "neutral"


def parse_course_metadata(file_path: Path) -> dict:
    """Extract course metadata from file path structure.
    
    Supports both structures:
    - apostila/01-introducao-solidity.md (old)
    - scripts da aula/01-introducao-solidity/texto-aula.md (new)
    - script das aulas/01-modulo/02-aula.md (grouped)
    """
    numbered_dirs = [
        (index, part)
        for index, part in enumerate(file_path.parent.parts)
        if has_numbered_prefix(part)
    ]
    file_is_numbered = has_numbered_prefix(file_path.stem)
    stem_is_generic = file_path.stem.lower() in GENERIC_LESSON_STEMS

    metadata = {
        "course": infer_course_name(file_path),
        "section": "",
        "lesson": path_part_to_title(file_path.stem)
    }

    if stem_is_generic and numbered_dirs:
        # The parent directory is the lesson folder. Its numbered parent, if any,
        # is the module/section.
        metadata["lesson"] = path_part_to_title(numbered_dirs[-1][1])
        if len(numbered_dirs) >= 2:
            metadata["section"] = path_part_to_title(numbered_dirs[-2][1])
    elif file_is_numbered:
        # The file itself is the lesson. A numbered parent means grouped lessons.
        metadata["lesson"] = path_part_to_title(file_path.stem)
        if numbered_dirs:
            metadata["section"] = path_part_to_title(numbered_dirs[-1][1])
    elif numbered_dirs:
        # Non-generic file inside a numbered lesson folder.
        metadata["lesson"] = path_part_to_title(numbered_dirs[-1][1])
        if len(numbered_dirs) >= 2:
            metadata["section"] = path_part_to_title(numbered_dirs[-2][1])
    
    return metadata


def split_text_for_subtitles(text: str, max_chars: int = 80) -> List[str]:
    """Split text into subtitle segments."""
    sentences = re.split(r'[.!?]+', text)
    segments = []
    
    current_segment = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_segment) + len(sentence) + 1 <= max_chars:
            current_segment += (" " if current_segment else "") + sentence
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = sentence
    
    if current_segment:
        segments.append(current_segment)
    
    return segments


def calculate_subtitle_timing(segments: List[str], audio_duration: float) -> List[Tuple[str, float, float]]:
    """Calculate timing for subtitle segments."""
    if not segments:
        return []
    
    total_chars = sum(len(seg) for seg in segments)
    timings = []
    
    current_time = 0.0
    for segment in segments:
        # Proportional timing based on character count
        duration = (len(segment) / total_chars) * audio_duration
        end_time = current_time + duration
        timings.append((segment, current_time, end_time))
        current_time = end_time
    
    return timings


def create_temp_directory() -> Path:
    """Create a temporary directory for video generation."""
    return Path(tempfile.mkdtemp(prefix="video_gen_"))


def clean_temp_directory(temp_dir: Path):
    """Clean up temporary directory."""
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)


def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def get_video_dimensions(resolution: str = "1080p") -> Tuple[int, int]:
    """Get video dimensions based on resolution."""
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    return resolutions.get(resolution, (1920, 1080))


def analyze_audio_for_animation(audio_path: Path) -> List[Tuple[float, float]]:
    """
    Analyze audio to detect speech segments for animation.
    Returns list of (start_time, end_time) for speech segments.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except ImportError:
        return []
    
    try:
        audio = AudioSegment.from_mp3(audio_path)
        
        # Detect non-silent segments
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=500,  # 500ms silence
            silence_thresh=audio.dBFS - 16,
            seek_step=100
        )
        
        # Convert to seconds
        return [(start / 1000.0, end / 1000.0) for start, end in nonsilent_ranges]
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []


def estimate_mouth_opening(audio_amplitude: float) -> float:
    """Estimate mouth opening based on audio amplitude."""
    # Normalize amplitude to 0-1 range
    normalized = min(max(audio_amplitude, 0), 1)
    return normalized


def should_blink(frame_num: int, blink_interval: int = 90) -> bool:
    """Determine if professor should blink at this frame."""
    return frame_num % blink_interval == 0


def get_gesture_for_content(text: str, has_code: bool) -> str:
    """Determine appropriate gesture based on content."""
    text_lower = text.lower()
    
    if has_code:
        return "pointing"
    
    if any(keyword in text_lower for keyword in ['veja', 'olhe', 'observe', 'note', 'look']):
        return "pointing"
    
    if any(keyword in text_lower for keyword in ['explicar', 'explica', 'explain', 'vamos ver']):
        return "explaining"
    
    return "none"
