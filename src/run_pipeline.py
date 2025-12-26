#!/usr/bin/env python3
"""
Cyclic Peptide Drug Analysis Pipeline

This is the main entry point for running the complete analysis workflow:
1. Target analysis with summary tables and plots
2. Individual target page generation

Usage:
    python run_pipeline.py [--data-path PATH] [--output-dir PATH] [--min-drugs N]
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from target_analysis import run_analysis
from generate_target_pages import run_page_generation


def main():
    """Run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Cyclic Peptide Drug Target Analysis Pipeline'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--min-drugs',
        type=int,
        default=1,
        help='Minimum number of drugs per target to generate a page'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot generation (faster for testing)'
    )
    parser.add_argument(
        '--skip-pages',
        action='store_true',
        help='Skip individual target page generation'
    )

    args = parser.parse_args()

    # Set default paths
    project_root = Path(__file__).parent.parent
    data_path = args.data_path or str(project_root / "data" / "cyclic_peptide_drug_level_active.csv")
    output_dir = args.output_dir or str(project_root / "output")
    template_dir = str(project_root / "templates")

    print("=" * 60)
    print("CYCLIC PEPTIDE DRUG TARGET ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Run target analysis
    print("-" * 60)
    print("STEP 1: Target Analysis")
    print("-" * 60)

    if not args.skip_plots:
        df, analysis_results = run_analysis(data_path, output_dir)
    else:
        print("Skipping plot generation...")
        from target_analysis import load_data, analyze_targets
        df = load_data(data_path)
        analysis_results = analyze_targets(df)

    # Step 2: Generate target pages
    if not args.skip_pages:
        print()
        print("-" * 60)
        print("STEP 2: Target Page Generation")
        print("-" * 60)

        target_pages_dir = os.path.join(output_dir, "target_pages")
        generated_pages = run_page_generation(
            data_path, template_dir, target_pages_dir, min_drugs=args.min_drugs
        )
    else:
        print("\nSkipping target page generation...")

    # Summary
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  - Main report: {output_dir}/index.html")
    print(f"  - Pivot table: {output_dir}/target_pivot_table.csv")
    print(f"  - Plots: {output_dir}/plots/")
    if not args.skip_pages:
        print(f"  - Target pages: {output_dir}/target_pages/")
        print(f"  - Total pages generated: {len(generated_pages)}")


if __name__ == "__main__":
    main()
