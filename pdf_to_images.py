#!/usr/bin/env python3
"""
PDF to Images Converter
Converts each page of a PDF to separate PNG images.
"""

import sys
from pathlib import Path
from typing import List, Optional


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150, prefix: str = None) -> List[Path]:
    """
    Convert each page of a PDF to a separate PNG image.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        dpi: Resolution for the images (default: 150)
        prefix: Prefix for image filenames (e.g., "01-Introducao-as-Ferramentas")
    
    Returns:
        List of paths to the generated images
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF is required. Install with: pip install pymupdf")
        sys.exit(1)
    
    # Suppress MuPDF warnings by redirecting stderr temporarily
    import io
    import contextlib
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open PDF with stderr suppressed to avoid MuPDF structure tree warnings
    with contextlib.redirect_stderr(io.StringIO()):
        doc = fitz.open(str(pdf_path))
    
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page to image with stderr suppressed
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            except Exception as e:
                print(f"  Warning: Error rendering page {page_num + 1}: {e}")
                continue
        
        # Save as PNG with prefix if provided
        if prefix:
            image_path = output_dir / f"{prefix}_page_{page_num + 1:03d}.png"
        else:
            image_path = output_dir / f"page_{page_num + 1:03d}.png"
        pix.save(str(image_path))
        
        image_paths.append(image_path)
        print(f"  Converted page {page_num + 1}/{len(doc)} -> {image_path}")
    
    doc.close()
    
    print(f"✓ Converted {len(image_paths)} pages to images in {output_dir}")
    return image_paths


def convert_odp_to_images(odp_path: Path, output_dir: Path, dpi: int = 150, prefix: str = None) -> List[Path]:
    """
    Convert ODP to PDF first, then to images.
    
    Args:
        odp_path: Path to the ODP file
        output_dir: Directory to save the images
        dpi: Resolution for the images (default: 150)
        prefix: Prefix for image filenames (e.g., "01-Introducao-as-Ferramentas")
    
    Returns:
        List of paths to the generated images
    """
    try:
        import subprocess
    except ImportError:
        print("Error: subprocess module not available")
        sys.exit(1)
    
    # Create temporary PDF path
    pdf_path = output_dir / f"{odp_path.stem}.pdf"
    
    # Convert ODP to PDF using LibreOffice
    try:
        print(f"  Converting ODP to PDF using LibreOffice...")
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", 
             "--outdir", str(output_dir), str(odp_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  Warning: LibreOffice conversion failed: {result.stderr}")
            print("  Trying alternative method using unoconv...")
            result = subprocess.run(
                ["unoconv", "-f", "pdf", "-o", str(pdf_path), str(odp_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  Error: Could not convert ODP to PDF: {result.stderr}")
                return []
        
        # Check if PDF was created
        if not pdf_path.exists():
            # LibreOffice might have saved it with a different name
            pdf_files = list(output_dir.glob("*.pdf"))
            if pdf_files:
                pdf_path = pdf_files[0]
            else:
                print(f"  Error: PDF not found after conversion")
                return []
        
        # Now convert PDF to images with prefix
        return convert_pdf_to_images(pdf_path, output_dir, dpi, prefix)
        
    except FileNotFoundError:
        print("  Error: LibreOffice or unoconv not found. Please install LibreOffice.")
        return []
    except Exception as e:
        print(f"  Error converting ODP: {e}")
        return []


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_images.py <pdf_or_odp_path> [output_dir] [dpi]")
        print("Example: python pdf_to_images.py slides.pdf images 150")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = input_path.parent / "images"
    
    # Determine DPI
    dpi = 150
    if len(sys.argv) >= 4:
        try:
            dpi = int(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid DPI '{sys.argv[3]}', using default 150")
    
    # Convert based on file type
    if input_path.suffix.lower() == '.pdf':
        convert_pdf_to_images(input_path, output_dir, dpi)
    elif input_path.suffix.lower() == '.odp':
        convert_odp_to_images(input_path, output_dir, dpi)
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")
        print("Supported types: .pdf, .odp")
        sys.exit(1)


if __name__ == "__main__":
    main()
