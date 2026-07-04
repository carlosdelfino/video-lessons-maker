#!/usr/bin/env python3
"""
Processing Log Manager
Tracks which files have been processed, organized by section and lesson.
Simple TXT format: one file per line with file type.
"""

from pathlib import Path
from typing import List, Set, Optional


class ProcessingLog:
    """Manages processing log with simple TXT format."""
    
    def __init__(self, log_file: Path = Path(".processing_log.txt")):
        self.log_file = Path(log_file)
        self.entries: Set[str] = set()
        self._load_log()
    
    def _load_log(self):
        """Load existing log file if it exists."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.entries = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"⚠️ Error loading log file: {e}")
                print("  Starting with empty log")
    
    def _save_log(self):
        """Save log data to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            for entry in sorted(self.entries):
                f.write(f"{entry}\n")
    
    def _extract_section_name(self, file_path: Path) -> str:
        """Extract section name from file path."""
        parent = file_path.parent
        return parent.name
    
    def _extract_lesson_name(self, file_path: Path) -> str:
        """Extract lesson name from file path."""
        return file_path.stem
    
    def record_processing(self, file_path: Path, file_type: str = "mp3"):
        """
        Record that a file has been processed.
        
        Args:
            file_path: Path to the processed file
            file_type: Type of file generated (mp3, pdf, odp, mp4)
        """
        section_name = self._extract_section_name(file_path)
        lesson_name = self._extract_lesson_name(file_path)
        
        # Format: section/lesson_name.file_type
        entry = f"{section_name}/{lesson_name}.{file_type}"
        self.entries.add(entry)
        self._save_log()
    
    def remove_lesson(self, file_path: Path, file_type: str = "mp3"):
        """
        Remove a lesson from the log (manually marked as uploaded to site).
        
        Args:
            file_path: Path to the file to remove from log
            file_type: Type of file to remove
        """
        section_name = self._extract_section_name(file_path)
        lesson_name = self._extract_lesson_name(file_path)
        
        entry = f"{section_name}/{lesson_name}.{file_type}"
        if entry in self.entries:
            self.entries.remove(entry)
            self._save_log()
            print(f"  ✓ Removed from log: {entry}")
        else:
            print(f"  ⚠️ Entry not in log: {entry}")
    
    def remove_section(self, section_name: str):
        """
        Remove an entire section from the log.
        
        Args:
            section_name: Name of the section to remove
        """
        to_remove = [entry for entry in self.entries if entry.startswith(f"{section_name}/")]
        for entry in to_remove:
            self.entries.remove(entry)
        
        if to_remove:
            self._save_log()
            print(f"  ✓ Removed {len(to_remove)} entries from section: {section_name}")
        else:
            print(f"  ⚠️ Section not in log: {section_name}")
    
    def get_pending_upload_list(self) -> List[str]:
        """
        Get a list of files pending upload to the site.
        
        Returns:
            List of entry strings
        """
        return sorted(self.entries)
    
    def print_summary(self):
        """Print a summary of processed files."""
        print(f"\n{'='*60}")
        print(f"PROCESSING LOG SUMMARY")
        print(f"{'='*60}")
        print(f"Total Files Pending Upload: {len(self.entries)}")
        
        if self.entries:
            print(f"\nFiles:")
            for entry in sorted(self.entries):
                print(f"  {entry}")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Processing Log Manager")
    parser.add_argument("--summary", action="store_true", help="Print summary of processed files")
    parser.add_argument("--pending", action="store_true", help="List files pending upload")
    parser.add_argument("--remove", help="Remove a lesson from log (path to file)")
    parser.add_argument("--remove-type", default="mp3", help="File type to remove (mp3, pdf, odp, mp4)")
    parser.add_argument("--remove-section", help="Remove an entire section from log")
    parser.add_argument("--log-file", default=".processing_log.txt", help="Path to log file")
    
    args = parser.parse_args()
    
    log = ProcessingLog(Path(args.log_file))
    
    if args.summary:
        log.print_summary()
    elif args.pending:
        pending = log.get_pending_upload_list()
        print(f"\nFiles Pending Upload ({len(pending)}):")
        print(f"{'='*60}")
        for entry in pending:
            print(f"  {entry}")
        print(f"\n{'='*60}\n")
    elif args.remove:
        log.remove_lesson(Path(args.remove), args.remove_type)
    elif args.remove_section:
        log.remove_section(args.remove_section)
    else:
        parser.print_help()
