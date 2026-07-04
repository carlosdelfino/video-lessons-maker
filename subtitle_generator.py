#!/usr/bin/env python3
"""
Subtitle Generator
Generates synchronized subtitles for educational videos.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Subtitle:
    """Represents a single subtitle with timing."""
    text: str
    start_time: float
    end_time: float
    
    def to_srt_format(self, index: int) -> str:
        """Convert to SRT format."""
        start = self._format_srt_time(self.start_time)
        end = self._format_srt_time(self.end_time)
        return f"{index}\n{start} --> {end}\n{self.text}\n"
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time in SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SubtitleGenerator:
    """Generates subtitles from text content synchronized with audio."""
    
    def __init__(self, words_per_minute: float = 150):
        self.words_per_minute = words_per_minute
    
    def generate_subtitles(
        self,
        text: str,
        audio_duration: float,
        max_chars_per_line: int = 80,
        max_lines_per_subtitle: int = 2
    ) -> List[Subtitle]:
        """Generate subtitles from text with timing based on audio duration."""
        # Clean text
        clean_text = self._clean_text(text)
        
        # Split into segments
        segments = self._split_into_segments(
            clean_text,
            max_chars_per_line,
            max_lines_per_subtitle
        )
        
        if not segments:
            return []
        
        # Calculate timing
        total_chars = sum(len(seg) for seg in segments)
        subtitles = []
        
        current_time = 0.0
        for segment in segments:
            # Proportional timing based on character count
            duration = (len(segment) / total_chars) * audio_duration
            end_time = current_time + duration
            
            # Ensure minimum duration for readability
            if duration < 1.5:
                end_time = current_time + 1.5
            
            # Ensure we don't exceed audio duration
            if end_time > audio_duration:
                end_time = audio_duration
            
            subtitles.append(Subtitle(
                text=segment,
                start_time=current_time,
                end_time=end_time
            ))
            
            current_time = end_time
        
        return subtitles
    
    def _clean_text(self, text: str) -> str:
        """Clean text for subtitles."""
        # Remove markdown formatting
        text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks
        text = re.sub(r'`[^`]+`', '', text)  # Inline code
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Images
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_into_segments(
        self,
        text: str,
        max_chars_per_line: int,
        max_lines_per_subtitle: int
    ) -> List[str]:
        """Split text into subtitle segments."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = []
        current_subtitle = ""
        current_lines = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed limits
            potential_subtitle = current_subtitle + (" " if current_subtitle else "") + sentence
            potential_lines = self._count_lines(potential_subtitle, max_chars_per_line)
            
            if potential_lines > max_lines_per_subtitle or len(potential_subtitle) > max_chars_per_line * max_lines_per_subtitle:
                # Save current subtitle if it exists
                if current_subtitle:
                    segments.append(current_subtitle.strip())
                    current_subtitle = ""
                    current_lines = 0
                
                # If sentence is too long, split it
                if len(sentence) > max_chars_per_line * max_lines_per_subtitle:
                    words = sentence.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= max_chars_per_line:
                            current_line += (" " if current_line else "") + word
                        else:
                            if current_line:
                                current_subtitle += (" " if current_subtitle else "") + current_line
                                current_lines += 1
                                if current_lines >= max_lines_per_subtitle:
                                    segments.append(current_subtitle.strip())
                                    current_subtitle = ""
                                    current_lines = 0
                            current_line = word
                    if current_line:
                        current_subtitle += (" " if current_subtitle else "") + current_line
                        current_lines += 1
                else:
                    current_subtitle = sentence
                    current_lines = self._count_lines(current_subtitle, max_chars_per_line)
            else:
                current_subtitle = potential_subtitle
                current_lines = potential_lines
        
        # Add remaining subtitle
        if current_subtitle:
            segments.append(current_subtitle.strip())
        
        return segments
    
    def _count_lines(self, text: str, max_chars_per_line: int) -> int:
        """Count how many lines a text would need."""
        if not text:
            return 0
        
        words = text.split()
        lines = 0
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += (" " if current_line else "") + word
            else:
                lines += 1
                current_line = word
        
        if current_line:
            lines += 1
        
        return lines
    
    def save_srt(self, subtitles: List[Subtitle], output_path: Path):
        """Save subtitles in SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate(subtitles, start=1):
                f.write(subtitle.to_srt_format(i))
    
    def load_srt(self, srt_path: Path) -> List[Subtitle]:
        """Load subtitles from SRT file."""
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        blocks = re.split(r'\n\n+', content.strip())
        subtitles = []
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                # Parse time line
                time_line = lines[1]
                match = re.match(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', time_line)
                if match:
                    start = self._parse_srt_time(match.group(1))
                    end = self._parse_srt_time(match.group(2))
                    text = '\n'.join(lines[2:])
                    subtitles.append(Subtitle(text=text, start_time=start, end_time=end))
        
        return subtitles
    
    @staticmethod
    def _parse_srt_time(time_str: str) -> float:
        """Parse SRT time format to seconds."""
        parts = time_str.replace(',', '.').split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def generate_subtitles_from_text(
    text: str,
    audio_duration: float,
    output_path: Optional[Path] = None
) -> List[Subtitle]:
    """Convenience function to generate subtitles from text."""
    generator = SubtitleGenerator()
    subtitles = generator.generate_subtitles(text, audio_duration)
    
    if output_path:
        generator.save_srt(subtitles, output_path)
    
    return subtitles
