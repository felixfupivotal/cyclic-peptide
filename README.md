# Cyclic Peptide Drug Target Analysis

A comprehensive analysis pipeline for cyclic peptide drug data, providing target-level analysis, visualizations, and individual target summary pages.

## Features

- **Target Analysis**: Analyze drug targets from the cyclic peptide drug dataset
  - Drug counts per target grouped by development phase
  - Company involvement analysis
  - Therapeutic area distribution

- **Visualizations**: Generate publication-ready plots
  - Development phase distribution (bar chart)
  - Top targets by drug count (horizontal bar chart)
  - Target-phase heatmap
  - Top companies in cyclic peptide development

- **Individual Target Pages**: Generate detailed HTML pages for each target
  - Target biology overview
  - Suitability for cyclic peptide therapeutics
  - Competitive landscape analysis
  - Drug pipeline table with development status
  - Companies and indications in development

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python src/run_pipeline.py
```

### Command Line Options

```bash
python src/run_pipeline.py --help

Options:
  --data-path PATH     Path to the input CSV file
  --output-dir PATH    Output directory for results
  --min-drugs N        Minimum number of drugs per target to generate a page
  --skip-plots         Skip plot generation
  --skip-pages         Skip individual target page generation
```

### Run Individual Components

**Target Analysis Only:**
```bash
python src/target_analysis.py
```

**Target Page Generation Only:**
```bash
python src/generate_target_pages.py
```

## Output Structure

```
output/
├── index.html                    # Main summary report
├── target_pivot_table.csv        # Pivot table of targets by phase
├── plots/
│   ├── phase_distribution.png    # Static phase distribution chart
│   ├── phase_distribution.html   # Interactive phase distribution
│   ├── top_targets.png           # Static top targets chart
│   ├── top_targets.html          # Interactive top targets
│   ├── target_phase_heatmap.png  # Static heatmap
│   ├── target_phase_heatmap.html # Interactive heatmap
│   ├── top_companies.png         # Static top companies chart
│   └── top_companies.html        # Interactive top companies
└── target_pages/
    ├── SSTR2.html                # Individual target pages
    ├── FAP.html
    ├── CAIX.html
    └── ...
```

## Data Requirements

The input CSV file should contain the following columns:
- `drug_name_en`: Drug name in English
- `targets`: Target(s) separated by semicolons
- `global_highest_phase_all_indication_standard`: Development phase
- `research_institution_list`: List of companies involved
- `therapeutic_area_primary_en`: Primary therapeutic area
- `indications_json`: JSON array of indication details

## Key Findings

Based on the analysis of 504 active cyclic peptide drugs:
- **204 unique targets** identified
- Top targets include FAP (26 drugs), SSTR2 (25 drugs), SSTR (21 drugs)
- Strong focus on oncology indications
- Significant radiopharmaceutical development activity

## Dependencies

- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0
- jinja2 >= 3.1.0
- numpy >= 1.24.0
