#!/usr/bin/env python3
"""
Run Script - Environment Setup and Execution
Checks and prepares the Python environment, then executes text_to_audio.py.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil
from dotenv import load_dotenv


def log_event(level: str, message: str, context: str = ""):
    """Log structured events with emojis."""
    emojis = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
        "debug": "🔍",
        "start": "🚀",
        "finish": "🏁"
    }
    emoji = emojis.get(level.lower(), "ℹ️")
    print(f"{emoji} {message} {context}".strip())


def check_python_version():
    """Check if Python version is compatible."""
    log_event("info", "Checking Python version...")
    if sys.version_info < (3, 8):
        log_event("error", f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    log_event("success", f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")


def get_project_root() -> Path:
    """Get the project root directory (parent of scripts/)."""
    return Path(__file__).parent.parent


def get_venv_path() -> Path:
    """Get the virtual environment path."""
    return get_project_root() / ".venv"


def venv_exists() -> bool:
    """Check if virtual environment exists."""
    venv_path = get_venv_path()
    return venv_path.exists() and (venv_path / "bin" / "python").exists()


def create_venv():
    """Create virtual environment."""
    log_event("start", "Creating virtual environment...")
    venv_path = get_venv_path()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True
        )
        log_event("success", f"Virtual environment created at {venv_path}")
    except subprocess.CalledProcessError as e:
        log_event("error", f"Failed to create virtual environment: {e}")
        sys.exit(1)


def get_pip_path() -> Path:
    """Get the pip executable path in the venv."""
    venv_path = get_venv_path()
    if os.name == "nt":  # Windows
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def get_python_path() -> Path:
    """Get the python executable path in the venv."""
    venv_path = get_venv_path()
    if os.name == "nt":  # Windows
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_dependencies():
    """Install dependencies from requirements.txt."""
    log_event("start", "Installing dependencies...")
    
    # Look for requirements.txt in project root or scripts directory
    project_root = get_project_root()
    scripts_dir = Path(__file__).parent
    
    requirements_paths = [
        project_root / "requirements.txt",
        scripts_dir / "requirements.txt"
    ]
    
    requirements_path = None
    for path in requirements_paths:
        if path.exists():
            requirements_path = path
            break
    
    if not requirements_path:
        log_event("warning", "No requirements.txt found, creating one...")
        requirements_path = scripts_dir / "requirements.txt"
        requirements_content = """openai>=1.0.0
google-cloud-texttospeech>=2.0.0
python-dotenv>=1.0.0
pydub>=0.25.0
moviepy>=1.0.3
pillow>=10.0.0
cairosvg>=2.0.0
numpy>=1.24.0
reportlab>=4.0.0
"""
        requirements_path.write_text(requirements_content)
        log_event("success", f"Created requirements.txt at {requirements_path}")
    
    pip_path = get_pip_path()
    
    try:
        subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_path)],
            check=True
        )
        log_event("success", "Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        log_event("error", f"Failed to install dependencies: {e}")
        sys.exit(1)


def check_dependencies_installed():
    """Check if key dependencies are installed."""
    python_path = get_python_path()
    
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import openai; import pydub; import moviepy"],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def setup_environment():
    """Setup the Python environment if needed."""
    log_event("start", "Setting up environment...")
    
    check_python_version()
    
    if not venv_exists():
        log_event("warning", "Virtual environment not found")
        create_venv()
        install_dependencies()
    else:
        log_event("info", "Virtual environment found")
        
        if not check_dependencies_installed():
            log_event("warning", "Dependencies not properly installed")
            install_dependencies()
        else:
            log_event("success", "Environment is ready")
    
    log_event("finish", "Environment setup complete")


def run_text_to_audio(args):
    """Execute text_to_audio.py with the provided arguments."""
    log_event("start", "Running text_to_audio.py...")
    
    scripts_dir = Path(__file__).parent
    text_to_audio_path = scripts_dir / "text_to_audio.py"
    
    if not text_to_audio_path.exists():
        log_event("error", f"text_to_audio.py not found at {text_to_audio_path}")
        sys.exit(1)
    
    python_path = get_python_path()
    
    # Build command arguments - text_to_audio.py expects positional args: api path
    cmd = [str(python_path), str(text_to_audio_path)]
    
    # Add positional arguments first
    cmd.append(args.api)
    cmd.append(args.input)
    
    # Add all optional arguments from argparse (excluding positional ones)
    for arg_name, arg_value in vars(args).items():
        if arg_name in ['api', 'input', 'skip_audio', 'remove_type', 'remove', 'estimate', 'no_setup']:
            continue
        # Skip ODP-related arguments if --odp is not set
        if arg_name in ['odp_template', 'cover_template'] and not args.odp:
            continue
        if arg_value is not None and arg_value is not False:
            if isinstance(arg_value, bool) and arg_value:
                cmd.append(f"--{arg_name.replace('_', '-')}")
            elif not isinstance(arg_value, bool):
                cmd.append(f"--{arg_name.replace('_', '-')}")
                cmd.append(str(arg_value))
    
    # Map skip-audio to slides-only
    if args.skip_audio:
        cmd.append("--slides-only")
    
    log_event("debug", f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        log_event("success", "text_to_audio.py completed successfully")
    except subprocess.CalledProcessError as e:
        log_event("error", f"text_to_audio.py failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        log_event("warning", "Interrupted by user")
        sys.exit(130)


def main():
    """Main entry point."""
    # Load .env file from project root or current directory
    project_root = get_project_root()
    env_paths = [
        project_root / ".env",
        Path.cwd() / ".env",
        Path(__file__).parent / ".env"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
    
    # Get default values from environment
    default_voice = os.environ.get("VOICE", "nova")
    # ODP-related defaults will be set only when --odp is used
    default_odp_template = None
    default_cover_template = None
    default_cover_time = None
    
    parser = argparse.ArgumentParser(
        description="Setup environment and run text_to_audio.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py --input "apostila/01-introducao.md" --api openai
  python scripts/run.py --input "scripts da aula/01-introducao/texto-aula.md" --api google --video
  python scripts/run.py --input "apostila/" --api openai --slides --force
        """
    )
    
    # Add all arguments that text_to_audio.py accepts
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--api", "-a", choices=["openai", "google"], default="openai", help="TTS API to use")
    parser.add_argument("--voice", "-v", default=default_voice, help=f"Voice for OpenAI TTS (default: {default_voice} from VOICE env var)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--video", action="store_true", help="Generate video")
    parser.add_argument("--slides", action="store_true", help="Generate slides")
    parser.add_argument("--slide", action="store_true", help="Generate slide-based video from audio (overrides --video)")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF")
    parser.add_argument("--odp", action="store_true", help="Generate ODP")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio generation (mapped to --slides-only)")
    parser.add_argument("--use-existing-slides", action="store_true", help="Use existing ODP structure for audio")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--estimate", action="store_true", help="Estimate course time without generating audio")
    parser.add_argument("--remove", action="store_true", help="Remove generated files")
    parser.add_argument("--remove-type", default="mp3", help="File type to remove (mp3, pdf, odp, mp4)")
    parser.add_argument("--no-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--env", "-e", default=None, help="Path to custom environment file (.env) to load variables from")
    parser.add_argument("--language", "-l", default=None, choices=["pt-br", "en", "es", "fr", "de"], help="Course language (default: from COURSE_LANGUAGE env var or pt-br)")
    parser.add_argument("--odp-template", default=default_odp_template, help="Path to ODP template file for slide generation (from ODP_TEMPLATE env var, requires --odp)")
    parser.add_argument("--cover-template", default=default_cover_template, help="Path to cover slide template file (.otp) to inject as first slide (from COVER_TEMPLATE env var, requires --odp)")
    parser.add_argument("--cover-time", type=float, default=default_cover_time, help="Duration in seconds for cover slide pause in video (from COVER_TIME env var, requires --odp)")
    parser.add_argument("--odp-recover", action="store_true", help="Recover ODP files by removing duplicate slides")
    parser.add_argument("--odp-review-titles", "--odp-title-review", dest="odp_review_titles", action="store_true", help="Review ODP slide titles, normalizing capitalization and lesson slugs")
    
    args = parser.parse_args()
    
    # Load ODP-related defaults from environment when --odp is used
    if args.odp or args.use_existing_slides:
        if args.odp_template is None:
            args.odp_template = os.environ.get("ODP_TEMPLATE", "template/template-1-azul-quadriculado.otp")
        if args.cover_template is None:
            args.cover_template = os.environ.get("COVER_TEMPLATE", None)
        if args.cover_time is None:
            args.cover_time = float(os.environ.get("COVER_TIME", "10"))
    
    # Always load cover_time from env for audio-only mode
    if args.cover_time is None:
        args.cover_time = float(os.environ.get("COVER_TIME", "5"))
    
    # Setup environment unless explicitly skipped
    if not args.no_setup:
        setup_environment()
    
    # Run text_to_audio.py
    run_text_to_audio(args)


if __name__ == "__main__":
    main()
