"""
Package the output folder for sharing.

Creates a ZIP file with the correct folder structure that
recipients can extract and browse locally.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime


def create_readme():
    """Create a README for recipients."""
    return """
CYCLIC PEPTIDE DRUG TARGET ANALYSIS
===================================

How to view this report:
1. Extract all files from this ZIP to a folder
2. Open 'index.html' in your web browser
3. Click on any target name to see detailed analysis

IMPORTANT: 
- Keep all files in the same folder structure
- Do not move individual HTML files separately
- All files must remain together for links to work

Folder structure:
  index.html              <- Main report (open this first)
  target_pivot_table.csv  <- Data export
  plots/                  <- Interactive charts
  target_pages/           <- Individual target analysis pages

Generated: {date}
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def package_output(output_dir: str, zip_path: str = None):
    """
    Package the output directory into a ZIP file for sharing.
    
    Args:
        output_dir: Path to the output directory
        zip_path: Optional custom path for the ZIP file
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return None
    
    # Default zip path
    if zip_path is None:
        zip_path = output_path.parent / f"cyclic_peptide_analysis_{datetime.now().strftime('%Y%m%d')}.zip"
    
    print(f"Creating package: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add README
        zf.writestr('README.txt', create_readme())
        
        # Add all files from output directory
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_path)
                print(f"  Adding: {arcname}")
                zf.write(file_path, arcname)
    
    print(f"\nPackage created successfully!")
    print(f"Location: {zip_path}")
    print(f"\nShare this ZIP file. Recipients should:")
    print("  1. Extract all files to a folder")
    print("  2. Open index.html in a web browser")
    
    return zip_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Package output for sharing')
    parser.add_argument('--output-dir', '-o', type=str, 
                        default=str(Path(__file__).parent / "output"),
                        help='Path to output directory')
    parser.add_argument('--zip-path', '-z', type=str, default=None,
                        help='Custom path for ZIP file')
    
    args = parser.parse_args()
    package_output(args.output_dir, args.zip_path)

