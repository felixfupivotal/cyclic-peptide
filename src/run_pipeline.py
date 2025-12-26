#!/usr/bin/env python3
"""
Cyclic Peptide Drug Analysis Pipeline

This is the main entry point for running the complete analysis workflow:
1. Target analysis with summary tables and plots
2. Individual target page generation
3. AI-powered deep research summaries (optional)

Usage:
    python run_pipeline.py [--data-path PATH] [--output-dir PATH] [--min-drugs N]
    
    # With LLM research enabled:
    python run_pipeline.py --llm-research --llm-api-key YOUR_API_KEY
    
    # Or set environment variable:
    export OPENAI_API_KEY=your-key-here
    python run_pipeline.py --llm-research
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
    # LLM Research Options
    parser.add_argument(
        '--llm-research',
        action='store_true',
        help='Enable AI-powered deep research using OpenAI GPT models'
    )
    parser.add_argument(
        '--llm-api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-4o',
        help='LLM model to use. OpenAI: gpt-4o, gpt-4-turbo, o1, o1-mini. Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of LLM research results (regenerate all)'
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
    if args.llm_research:
        print(f"LLM Research: Enabled (model: {args.llm_model})")
        if args.no_cache:
            print("LLM Cache: Disabled")
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
        if args.llm_research:
            print("         (with AI-powered research summaries)")
        print("-" * 60)

        target_pages_dir = os.path.join(output_dir, "target_pages")
        generated_pages = run_page_generation(
            data_path=data_path,
            template_dir=template_dir,
            output_dir=target_pages_dir,
            min_drugs=args.min_drugs,
            enable_llm_research=args.llm_research,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            use_cache=not args.no_cache
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
        if args.llm_research:
            cache_dir = Path(output_dir).parent / "cache" / "llm_research"
            print(f"  - LLM research cache: {cache_dir}/")
            print(f"\nAI-powered research summaries were generated for each target.")
            print("Note: Cached results will be reused for 30 days unless --no-cache is specified.")


if __name__ == "__main__":
    main()
