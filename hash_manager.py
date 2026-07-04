#!/usr/bin/env python3
"""
Hash Manager for Incremental Audio Generation
Manages file and paragraph hashes to enable incremental audio generation.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


class HashManager:
    """Manages hashes for files and paragraphs to enable incremental updates."""
    
    def __init__(self, file_path: Path, prefix: str = None):
        self.file_path = Path(file_path)
        self.hash_file = self.file_path.parent / f"{self.file_path.stem}.hash"
        self.audio_dir = self.file_path.parent / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        # Use provided prefix (e.g. "texto-aula_openai") or fall back to file stem
        self._prefix = prefix or self.file_path.stem
        
        # Load existing hashes if available
        self.file_hash = None
        self.paragraph_hashes = []
        self.slides_hash = None
        self._load_hashes()
    
    def _load_hashes(self):
        """Load existing hashes from .hash file."""
        if self.hash_file.exists():
            with open(self.hash_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    self.file_hash = lines[0].strip()
                    # Lines 1 to end are paragraph hashes, last line is slides hash if marked
                    if lines[-1].startswith("SLIDES:"):
                        self.paragraph_hashes = [line.strip() for line in lines[1:-1]]
                        self.slides_hash = lines[-1].strip().replace("SLIDES:", "")
                    else:
                        self.paragraph_hashes = [line.strip() for line in lines[1:]]
                        self.slides_hash = None
    
    def _save_hashes(self, file_hash: str, paragraph_hashes: List[str], slides_hash: str = None):
        """Save hashes to .hash file."""
        with open(self.hash_file, 'w', encoding='utf-8') as f:
            f.write(f"{file_hash}\n")
            for phash in paragraph_hashes:
                f.write(f"{phash}\n")
            if slides_hash:
                f.write(f"SLIDES:{slides_hash}\n")
        
        self.file_hash = file_hash
        self.paragraph_hashes = paragraph_hashes
        self.slides_hash = slides_hash
    
    def calculate_file_hash(self, content: str) -> str:
        """Calculate SHA256 hash of entire file content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def calculate_paragraph_hash(self, paragraph: str) -> str:
        """Calculate SHA256 hash of a single paragraph."""
        return hashlib.sha256(paragraph.strip().encode('utf-8')).hexdigest()
    
    def split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs (double newline separated)."""
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def get_changed_paragraphs(self, content: str) -> Tuple[List[int], List[str]]:
        """
        Compare current content with stored hashes and identify changed paragraphs.
        
        Returns:
            Tuple of (changed_indices, all_paragraphs)
            - changed_indices: List of paragraph indices that changed
            - all_paragraphs: List of all paragraphs in current content
        """
        current_file_hash = self.calculate_file_hash(content)
        current_paragraphs = self.split_into_paragraphs(content)
        current_paragraph_hashes = [self.calculate_paragraph_hash(p) for p in current_paragraphs]
        
        # If file hash is the same, no changes
        if self.file_hash == current_file_hash:
            return [], current_paragraphs
        
        # If this is a new file or file hash changed completely
        if not self.file_hash or self.file_hash != current_file_hash:
            # Check which paragraphs changed
            changed_indices = []
            for i, (current_hash, current_para) in enumerate(zip(current_paragraph_hashes, current_paragraphs)):
                if i >= len(self.paragraph_hashes):
                    # New paragraph
                    changed_indices.append(i)
                elif self.paragraph_hashes[i] != current_hash:
                    # Changed paragraph
                    changed_indices.append(i)
            
            return changed_indices, current_paragraphs
        
        return [], current_paragraphs
    
    def get_audio_path(self, paragraph_index: int) -> Path:
        """Get the audio file path for a specific paragraph.
        Cover is slide_001, paragraphs start at slide_002."""
        return self.audio_dir / f"{self._prefix}_slide_{paragraph_index + 2:03d}.mp3"
    
    def get_cover_audio_path(self) -> Path:
        """Get the audio file path for the cover narration segment (slide_001)."""
        return self.audio_dir / f"{self._prefix}_slide_001.mp3"
    
    def update_hashes(self, content: str, slides_hash: str = None):
        """Update stored hashes with current content."""
        current_file_hash = self.calculate_file_hash(content)
        current_paragraphs = self.split_into_paragraphs(content)
        current_paragraph_hashes = [self.calculate_paragraph_hash(p) for p in current_paragraphs]
        
        self._save_hashes(current_file_hash, current_paragraph_hashes, slides_hash)
    
    def calculate_slides_hash(self, slides: List[dict]) -> str:
        """Calculate hash of slide content for incremental slide generation."""
        import json
        slide_texts = []
        for slide in slides:
            slide_dict = {
                'title': slide.get('title', ''),
                'content': slide.get('content', []),
                'bullets': slide.get('bullets', [])
            }
            slide_texts.append(json.dumps(slide_dict, sort_keys=True))
        combined = '\n'.join(slide_texts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def get_slides_changed(self, slides: List[dict]) -> bool:
        """Check if slides have changed based on hash."""
        current_slides_hash = self.calculate_slides_hash(slides)
        return self.slides_hash != current_slides_hash
    
    def clean_old_audio(self, current_paragraph_count: int):
        """Remove audio files for paragraphs that no longer exist."""
        prefix = self._prefix
        pattern = f"{prefix}_slide_*.mp3"
        for audio_file in self.audio_dir.glob(pattern):
            # Extract index from filename (e.g., prefix_slide_001.mp3)
            match = re.match(rf'{re.escape(prefix)}_slide_(\d+)\.mp3', audio_file.name)
            if match:
                index = int(match.group(1))
                # index 1 is cover, 2..N+1 are paragraphs; keep if index <= paragraph_count + 1
                if index > current_paragraph_count + 1:
                    audio_file.unlink()
                    print(f"  Removed old audio: {audio_file.name}")


class AudioFragmentManager:
    """Manages audio fragments and combines them into complete audio."""
    
    def __init__(self, audio_dir: Path):
        self.audio_dir = Path(audio_dir)
        
        try:
            from pydub import AudioSegment
            self.AudioSegment = AudioSegment
        except ImportError:
            self.AudioSegment = None
            print("⚠️ pydub not installed. Audio combination will not work.")
    
    def combine_fragments(self, output_path: Path, paragraph_count: int, use_silence: bool = True, prefix: str = None, cover_path: Path = None):
        """
        Combine all paragraph audio fragments into a single audio file.
        
        Args:
            output_path: Path for the combined audio file
            paragraph_count: Number of paragraphs to combine
            use_silence: Whether to add silence between paragraphs
            prefix: Prefix for audio filenames (e.g., "01-Introducao-as-Ferramentas")
            cover_path: Optional path to cover narration audio segment to prepend
        """
        if self.AudioSegment is None:
            raise ImportError("pydub is required for audio combination. Run: pip install pydub")
        
        audio_segments = []
        
        # Add cover segment first if it exists and is not empty
        if cover_path and cover_path.exists() and cover_path.stat().st_size > 0:
            cover_segment = self.AudioSegment.from_mp3(str(cover_path))
            audio_segments.append(cover_segment)
            print(f"  ✓ Cover segment included: {cover_path.name}")
        
        for i in range(paragraph_count):
            if prefix:
                audio_path = self.audio_dir / f"{prefix}_slide_{i + 2:03d}.mp3"
            else:
                audio_path = self.audio_dir / f"slide_{i + 2:03d}.mp3"
            # Check if file exists and is not empty
            if audio_path.exists() and audio_path.stat().st_size > 0:
                segment = self.AudioSegment.from_mp3(str(audio_path))
                audio_segments.append(segment)
            else:
                print(f"⚠️ Missing fragment: {audio_path.name}")
        
        if not audio_segments:
            raise ValueError("No audio fragments found to combine")
        
        # Combine segments
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            if use_silence:
                # Reduced silence between paragraphs to minimize excessive pauses
                combined += self.AudioSegment.silent(duration=200)
            combined += segment
        
        combined.export(str(output_path), format="mp3")
        print(f"✓ Combined audio saved to: {output_path}")
    
    def get_fragment_duration(self, paragraph_index: int, prefix: str = None) -> Optional[float]:
        """Get duration of a specific audio fragment in seconds."""
        if self.AudioSegment is None:
            return None
        
        if prefix:
            audio_path = self.audio_dir / f"{prefix}_slide_{paragraph_index + 2:03d}.mp3"
        else:
            audio_path = self.audio_dir / f"slide_{paragraph_index + 2:03d}.mp3"
        # Check if file exists and is not empty
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return None
        
        segment = self.AudioSegment.from_mp3(str(audio_path))
        return len(segment) / 1000.0  # Convert to seconds
    
    def get_total_duration(self, paragraph_count: int, prefix: str = None) -> float:
        """Get total duration of all fragments in seconds."""
        total = 0.0
        for i in range(paragraph_count):
            duration = self.get_fragment_duration(i, prefix)
            if duration:
                total += duration
        return total


def generate_hash_file(file_path: Path):
    """
    Generate or update hash file for a markdown file.
    
    Args:
        file_path: Path to the markdown file
    """
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return
    
    manager = HashManager(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    manager.update_hashes(content)
    print(f"✓ Hash file generated: {manager.hash_file}")


def check_changes(file_path: Path) -> Dict:
    """
    Check what changed in a file compared to stored hashes.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with change information
    """
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return {}
    
    manager = HashManager(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changed_indices, paragraphs = manager.get_changed_paragraphs(content)
    
    return {
        'file_changed': manager.file_hash != manager.calculate_file_hash(content),
        'changed_indices': changed_indices,
        'total_paragraphs': len(paragraphs),
        'changed_count': len(changed_indices),
        'paragraphs': paragraphs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hash Manager for Incremental Audio Generation")
    parser.add_argument("file", help="Path to markdown file")
    parser.add_argument("--check", action="store_true", help="Check for changes")
    parser.add_argument("--generate", action="store_true", help="Generate hash file")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    if args.check:
        changes = check_changes(file_path)
        print(f"\nFile: {file_path}")
        print(f"File changed: {changes['file_changed']}")
        print(f"Total paragraphs: {changes['total_paragraphs']}")
        print(f"Changed paragraphs: {changes['changed_count']}")
        if changes['changed_indices']:
            print(f"Changed indices: {changes['changed_indices']}")
    elif args.generate:
        generate_hash_file(file_path)
    else:
        parser.print_help()
