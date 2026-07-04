#!/usr/bin/env python3
"""
Radial FFT Audio Spectrum Visualizer
Generates radial/circular audio spectrum visualizations for video.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
from PIL import Image, ImageDraw

try:
    from moviepy.editor import AudioFileClip, ImageSequenceClip
except ImportError:
    try:
        from moviepy import AudioFileClip, ImageSequenceClip
    except ImportError:
        AudioFileClip = None
        ImageSequenceClip = None


class SpectrumVisualizer:
    """Generates radial FFT audio spectrum visualizations."""
    
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.center_x = width // 2
        self.center_y = height // 2
        self.max_radius = min(width, height) // 2 - 50
        
        # Color palette (gradient from center to edge)
        self.colors = self._generate_color_palette()
        
        # Smoothing factor for temporal smoothing
        self.smoothing_factor = 0.3
        self.previous_spectrum = None
    
    def _generate_color_palette(self) -> List[Tuple[int, int, int]]:
        """Generate color palette for spectrum visualization."""
        colors = []
        for i in range(256):
            # Rainbow gradient based on frequency
            hue = (i / 256) * 360
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def analyze_audio_fft(self, audio_path: Path) -> Tuple[np.ndarray, float]:
        """
        Analyze audio using FFT to get frequency spectrum over time.
        Returns (spectrum_data, duration) where spectrum_data is shape (n_frames, n_freqs).
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub is required for audio analysis. Install with: pip install pydub")
        
        try:
            audio = AudioSegment.from_mp3(audio_path)
            duration = len(audio) / 1000.0  # Convert milliseconds to seconds
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            sample_rate = audio.frame_rate
            
            # Calculate number of frames
            n_frames = int(duration * self.fps)
            samples_per_frame = len(samples) // n_frames
            
            # FFT parameters
            fft_size = 512
            spectrum_data = []
            
            for i in range(n_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                
                if end > len(samples):
                    break
                
                frame_samples = samples[start:end]
                
                # Apply window function
                window = np.hanning(len(frame_samples))
                windowed_samples = frame_samples * window
                
                # FFT
                fft_result = np.fft.fft(windowed_samples, n=fft_size)
                fft_magnitude = np.abs(fft_result[:fft_size // 2])
                
                # Log scale for better visualization
                fft_magnitude = np.log1p(fft_magnitude * 1000)
                
                spectrum_data.append(fft_magnitude)
            
            return np.array(spectrum_data), duration
            
        except Exception as e:
            raise RuntimeError(f"Error analyzing audio: {e}")
    
    def generate_frame(self, spectrum: np.ndarray, frame_num: int) -> Image.Image:
        """
        Generate a single frame with radial spectrum visualization.
        
        Args:
            spectrum: FFT magnitude data for this frame
            frame_num: Current frame number (for animation effects)
        
        Returns:
            PIL Image with the rendered frame
        """
        # Create dark background
        img = Image.new('RGB', (self.width, self.height), (10, 10, 20))
        draw = ImageDraw.Draw(img)
        
        # Apply temporal smoothing
        if self.previous_spectrum is not None:
            spectrum = (self.smoothing_factor * spectrum + 
                       (1 - self.smoothing_factor) * self.previous_spectrum)
        self.previous_spectrum = spectrum.copy()
        
        # Normalize spectrum
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        
        # Draw radial spectrum
        n_freqs = len(spectrum)
        n_rings = 50
        
        for ring in range(n_rings):
            radius = (ring + 1) * (self.max_radius / n_rings)
            
            # Get frequency range for this ring
            freq_start = int((ring / n_rings) * n_freqs)
            freq_end = int(((ring + 1) / n_rings) * n_freqs)
            
            if freq_end > n_freqs:
                freq_end = n_freqs
            
            # Average magnitude for this ring
            ring_magnitude = np.mean(spectrum[freq_start:freq_end])
            
            # Calculate color based on frequency and magnitude
            color_idx = int((ring / n_rings) * 255)
            base_color = self.colors[color_idx]
            
            # Brighten based on magnitude
            brightness = 0.3 + ring_magnitude * 0.7
            color = tuple(int(c * brightness) for c in base_color)
            
            # Draw ring with varying thickness based on magnitude
            thickness = max(1, int(ring_magnitude * 5))
            
            # Draw as arc segments for tunnel effect
            n_segments = 64
            for seg in range(n_segments):
                angle_start = (seg / n_segments) * 360
                angle_end = ((seg + 1) / n_segments) * 360
                
                # Modulate radius based on angle for wave effect
                wave_offset = np.sin((angle_start / 180) * np.pi * 4 + frame_num * 0.1) * ring_magnitude * 20
                adjusted_radius = max(5, radius + wave_offset)  # Ensure minimum radius of 5
                
                # Convert to bounding box for arc
                bbox = [
                    self.center_x - adjusted_radius,
                    self.center_y - adjusted_radius,
                    self.center_x + adjusted_radius,
                    self.center_y + adjusted_radius
                ]
                
                draw.arc(bbox, angle_start, angle_end, fill=color, width=thickness)
        
        # Draw radial lines from center
        n_lines = 32
        for line in range(n_lines):
            angle = (line / n_lines) * 2 * np.pi
            
            # Get frequency for this line
            freq_idx = int((line / n_lines) * n_freqs)
            if freq_idx >= n_freqs:
                freq_idx = n_freqs - 1
            
            magnitude = spectrum[freq_idx]
            
            # Calculate line length
            line_length = magnitude * self.max_radius * 0.8
            
            # End point
            end_x = self.center_x + line_length * np.cos(angle)
            end_y = self.center_y + line_length * np.sin(angle)
            
            # Color based on angle
            color_idx = int((line / n_lines) * 255)
            color = self.colors[color_idx]
            
            # Draw line
            draw.line(
                [self.center_x, self.center_y, end_x, end_y],
                fill=color,
                width=max(1, int(magnitude * 3))
            )
        
        # Draw center glow
        glow_radius = int(50 + np.mean(spectrum) * 50)
        glow_color = self.colors[int(frame_num % 256)]
        
        for r in range(glow_radius, 0, -5):
            alpha = int(255 * (1 - r / glow_radius) * 0.3)
            draw.ellipse(
                [self.center_x - r, self.center_y - r, 
                 self.center_x + r, self.center_y + r],
                outline=glow_color,
                width=2
            )
        
        return img
    
    def generate_video(
        self,
        audio_path: Path,
        output_path: Path,
        spectrum_data: np.ndarray,
        duration: float
    ) -> bool:
        """
        Generate video with radial spectrum visualization.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output video file
            spectrum_data: Pre-computed FFT spectrum data
            duration: Audio duration in seconds
        
        Returns:
            True if successful, False otherwise
        """
        if AudioFileClip is None or ImageSequenceClip is None:
            raise ImportError("moviepy is required for video generation. Install with: pip install moviepy")
        
        try:
            # Create temporary directory for frames
            temp_dir = Path(tempfile.mkdtemp(prefix="spectrum_"))
            
            # Generate frames
            n_frames = len(spectrum_data)
            print(f"  Generating {n_frames} spectrum frames...")
            
            frame_paths = []
            for frame_num in range(n_frames):
                if frame_num % 10 == 0:
                    progress = (frame_num / n_frames) * 100
                    print(f"    Progress: {progress:.1f}%")
                
                frame = self.generate_frame(spectrum_data[frame_num], frame_num)
                frame_path = temp_dir / f"frame_{frame_num:06d}.png"
                frame.save(frame_path)
                frame_paths.append(frame_path)
            
            print("  Creating video from frames...")
            
            # Create video from frames
            video_clip = ImageSequenceClip(
                [str(p) for p in frame_paths],
                fps=self.fps
            )
            
            # Add audio
            audio_clip = AudioFileClip(str(audio_path))
            video_clip = video_clip.with_audio(audio_clip)
            
            # Write video
            video_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=self.fps,
                preset='medium',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            return True
            
        except Exception as e:
            print(f"  Error generating spectrum video: {e}")
            return False
