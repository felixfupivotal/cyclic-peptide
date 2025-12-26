"""
Target Analysis for Cyclic Peptide Drugs

This module analyzes targets from the cyclic peptide drug dataset,
creating summary tables and visualizations grouped by development phase.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from collections import Counter
import warnings

# Configure matplotlib to support CJK (Chinese) characters
# Try different fonts in order of preference
CJK_FONTS = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'Arial Unicode MS', 'Noto Sans CJK SC']
for font in CJK_FONTS:
    try:
        mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
        break
    except:
        continue

# Suppress font warnings for missing glyphs
warnings.filterwarnings('ignore', message='Glyph .* missing from')
mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


# Define phase order for sorting
PHASE_ORDER = {
    'pre-clinical': -1,
    'ind': 0,
    'phase 1': 1,
    'phase 2': 2,
    'phase 3': 3,
    'pre-registration': 4,
    'approved': 5,
    'launched': 6
}

PHASE_DISPLAY_NAMES = {
    'pre-clinical': 'Pre-clinical',
    'ind': 'IND',
    'phase 1': 'Phase 1',
    'phase 2': 'Phase 2',
    'phase 3': 'Phase 3',
    'pre-registration': 'Pre-registration',
    'approved': 'Approved',
    'launched': 'Launched'
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load the cyclic peptide drug data."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def parse_indications_json(json_str: str) -> list:
    """Parse the indications JSON column."""
    try:
        if pd.isna(json_str):
            return []
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def clean_company_list(company_str: str) -> list:
    """Parse company list from string representation."""
    try:
        if pd.isna(company_str):
            return []
        # Handle Python list string representation
        if company_str.startswith('['):
            import ast
            return ast.literal_eval(company_str)
        return [company_str]
    except (ValueError, SyntaxError):
        return [company_str] if company_str else []


def analyze_targets(df: pd.DataFrame) -> dict:
    """
    Analyze targets in the dataset.

    Returns a dictionary with:
    - target_summary: DataFrame with drug counts per target by phase
    - target_companies: DataFrame with companies per target
    - target_details: Detailed information for each target
    """
    results = {}

    # Get unique targets and their drugs
    target_drugs = {}
    for _, row in df.iterrows():
        targets = str(row['targets']).split(';') if pd.notna(row['targets']) else []
        for target in targets:
            target = target.strip()
            if target:
                if target not in target_drugs:
                    target_drugs[target] = []
                target_drugs[target].append(row)

    # Create target summary by phase
    summary_data = []
    for target, drugs in target_drugs.items():
        phase_counts = Counter()
        companies = set()
        therapeutic_areas = set()

        for drug in drugs:
            phase = drug['global_highest_phase_all_indication_standard']
            if pd.notna(phase):
                phase_counts[phase] += 1

            # Collect companies
            company_list = clean_company_list(drug.get('research_institution_list', ''))
            companies.update(company_list)

            # Collect therapeutic areas
            if pd.notna(drug.get('therapeutic_area_primary_en')):
                therapeutic_areas.add(drug['therapeutic_area_primary_en'])

        # Determine highest phase for this target
        highest_phase = None
        highest_phase_val = -2
        for phase, count in phase_counts.items():
            if phase in PHASE_ORDER and PHASE_ORDER[phase] > highest_phase_val:
                highest_phase = phase
                highest_phase_val = PHASE_ORDER[phase]

        summary_data.append({
            'target': target,
            'total_drugs': len(drugs),
            'highest_phase': highest_phase,
            'highest_phase_display': PHASE_DISPLAY_NAMES.get(highest_phase, highest_phase),
            'phase_counts': dict(phase_counts),
            'companies': list(companies),
            'company_count': len(companies),
            'therapeutic_areas': list(therapeutic_areas),
            'drugs': [{'name': d['drug_name_en'], 'phase': d['global_highest_phase_all_indication_standard']}
                     for d in drugs]
        })

    results['target_summary'] = pd.DataFrame(summary_data)
    results['target_summary'] = results['target_summary'].sort_values(
        by=['total_drugs'], ascending=False
    )

    return results


def get_targets_by_indication_therapeutic_area(df: pd.DataFrame, therapeutic_areas: list) -> pd.DataFrame:
    """
    Get targets that have indications in specified therapeutic areas.

    Args:
        df: DataFrame with drug data
        therapeutic_areas: List of therapeutic area names to filter by (e.g., ["Immunology", "Endocrinology & Metabolism"])

    Returns:
        DataFrame with target information for matching drugs
    """
    matching_targets = {}

    for _, row in df.iterrows():
        targets = str(row['targets']).split(';') if pd.notna(row['targets']) else []
        indications = parse_indications_json(row.get('indications_json', ''))

        # Check if any indication has a matching therapeutic area
        matching_indications = []
        for ind in indications:
            ta = ind.get('therapeutic_area_en', '')
            if ta in therapeutic_areas:
                matching_indications.append({
                    'indication': ind.get('indication_en', 'Unknown'),
                    'therapeutic_area': ta,
                    'phase': ind.get('highest_phase', 'pre-clinical')
                })

        if matching_indications:
            for target in targets:
                target = target.strip()
                if target:
                    if target not in matching_targets:
                        matching_targets[target] = {
                            'target': target,
                            'drugs': [],
                            'indications': [],
                            'therapeutic_areas': set(),
                            'highest_phase': 'pre-clinical',
                            'highest_phase_val': -1
                        }

                    matching_targets[target]['drugs'].append(row['drug_name_en'])
                    matching_targets[target]['indications'].extend(matching_indications)

                    for ind in matching_indications:
                        matching_targets[target]['therapeutic_areas'].add(ind['therapeutic_area'])
                        phase_val = PHASE_ORDER.get(ind['phase'], -1)
                        if phase_val > matching_targets[target]['highest_phase_val']:
                            matching_targets[target]['highest_phase_val'] = phase_val
                            matching_targets[target]['highest_phase'] = ind['phase']

    # Convert to list and prepare for DataFrame
    result_data = []
    for target, data in matching_targets.items():
        # Get unique indications
        unique_indications = list(set([ind['indication'] for ind in data['indications']]))
        result_data.append({
            'target': data['target'],
            'drug_count': len(set(data['drugs'])),
            'drugs': list(set(data['drugs'])),
            'indications': unique_indications[:5],  # Top 5 indications
            'therapeutic_areas': list(data['therapeutic_areas']),
            'highest_phase': data['highest_phase'],
            'highest_phase_display': PHASE_DISPLAY_NAMES.get(data['highest_phase'], data['highest_phase'])
        })

    result_df = pd.DataFrame(result_data)
    if not result_df.empty:
        result_df = result_df.sort_values('drug_count', ascending=False)

    return result_df


def create_pivot_table(df: pd.DataFrame, analysis_results: dict) -> pd.DataFrame:
    """Create a pivot table of targets by phase."""
    summary = analysis_results['target_summary']

    # Expand phase counts into columns
    phases = list(PHASE_ORDER.keys())
    pivot_data = []

    for _, row in summary.iterrows():
        row_data = {
            'Target': row['target'],
            'Total Drugs': row['total_drugs'],
            'Highest Phase': row['highest_phase_display'],
            'Companies': len(row['companies']),
        }
        for phase in phases:
            display_name = PHASE_DISPLAY_NAMES.get(phase, phase)
            row_data[display_name] = row['phase_counts'].get(phase, 0)
        pivot_data.append(row_data)

    pivot_df = pd.DataFrame(pivot_data)

    # Sort by total drugs
    pivot_df = pivot_df.sort_values('Total Drugs', ascending=False)

    return pivot_df


def create_phase_distribution_plot(analysis_results: dict, output_dir: str):
    """Create a bar chart showing drug distribution by phase."""
    summary = analysis_results['target_summary']

    # Aggregate drugs by highest phase
    phase_counts = Counter()
    for _, row in summary.iterrows():
        for phase, count in row['phase_counts'].items():
            phase_counts[phase] += count

    # Sort by phase order
    phases = sorted(phase_counts.keys(), key=lambda x: PHASE_ORDER.get(x, -2))
    counts = [phase_counts[p] for p in phases]
    display_names = [PHASE_DISPLAY_NAMES.get(p, p) for p in phases]

    # Create plot with matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(phases))
    bars = ax.bar(display_names, counts, color=colors)

    ax.set_xlabel('Development Phase', fontsize=12)
    ax.set_ylabel('Number of Drugs', fontsize=12)
    ax.set_title('Cyclic Peptide Drugs by Development Phase', fontsize=14)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_distribution.png'), dpi=150)
    plt.close()

    # Create interactive plotly version
    fig = px.bar(
        x=display_names, y=counts,
        labels={'x': 'Development Phase', 'y': 'Number of Drugs'},
        title='Cyclic Peptide Drugs by Development Phase',
        color=counts,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    fig.write_html(os.path.join(output_dir, 'phase_distribution.html'))


def create_top_targets_plot(analysis_results: dict, output_dir: str, top_n: int = 20):
    """Create a horizontal bar chart of top targets by drug count."""
    summary = analysis_results['target_summary'].head(top_n)

    # Matplotlib version
    fig, ax = plt.subplots(figsize=(12, 10))

    targets = summary['target'].tolist()[::-1]  # Reverse for horizontal bar
    counts = summary['total_drugs'].tolist()[::-1]
    phases = summary['highest_phase_display'].tolist()[::-1]

    # Color by highest phase
    phase_colors = {
        'Pre-clinical': '#1f77b4',
        'IND': '#2ca02c',
        'Phase 1': '#ff7f0e',
        'Phase 2': '#d62728',
        'Phase 3': '#9467bd',
        'Pre-registration': '#8c564b',
        'Approved': '#e377c2',
        'Launched': '#7f7f7f'
    }
    colors = [phase_colors.get(p, '#17becf') for p in phases]

    bars = ax.barh(targets, counts, color=colors)
    ax.set_xlabel('Number of Drugs', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title(f'Top {top_n} Targets by Number of Cyclic Peptide Drugs', fontsize=14)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontsize=9)

    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in phase_colors.values()]
    legend_labels = list(phase_colors.keys())
    ax.legend(legend_handles, legend_labels, loc='lower right', title='Highest Phase')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_targets.png'), dpi=150)
    plt.close()

    # Interactive plotly version
    fig = px.bar(
        summary.head(top_n),
        x='total_drugs', y='target',
        orientation='h',
        color='highest_phase_display',
        labels={'total_drugs': 'Number of Drugs', 'target': 'Target',
                'highest_phase_display': 'Highest Phase'},
        title=f'Top {top_n} Targets by Number of Cyclic Peptide Drugs',
        color_discrete_map=phase_colors
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.write_html(os.path.join(output_dir, 'top_targets.html'))


def create_target_phase_heatmap(analysis_results: dict, output_dir: str, top_n: int = 30):
    """Create a heatmap of targets vs development phases."""
    summary = analysis_results['target_summary'].head(top_n)

    phases = ['Pre-clinical', 'IND', 'Phase 1', 'Phase 2', 'Phase 3', 'Pre-registration', 'Approved', 'Launched']
    phase_key_map = {v: k for k, v in PHASE_DISPLAY_NAMES.items()}

    # Create matrix
    matrix_data = []
    for _, row in summary.iterrows():
        row_counts = []
        for phase in phases:
            phase_key = phase_key_map.get(phase, phase.lower())
            row_counts.append(row['phase_counts'].get(phase_key, 0))
        matrix_data.append(row_counts)

    matrix_df = pd.DataFrame(
        matrix_data,
        columns=phases,
        index=summary['target'].tolist()
    )

    # Matplotlib heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Number of Drugs'})
    ax.set_title(f'Top {top_n} Targets: Drug Count by Development Phase', fontsize=14)
    ax.set_xlabel('Development Phase', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_phase_heatmap.png'), dpi=150)
    plt.close()

    # Interactive plotly heatmap
    fig = px.imshow(
        matrix_df,
        labels=dict(x='Development Phase', y='Target', color='Drug Count'),
        title=f'Top {top_n} Targets: Drug Count by Development Phase',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig.update_layout(height=800)
    fig.write_html(os.path.join(output_dir, 'target_phase_heatmap.html'))


def create_company_analysis_plot(analysis_results: dict, output_dir: str, top_n: int = 20):
    """Analyze and plot top companies by drug count."""
    summary = analysis_results['target_summary']

    # Count drugs per company
    company_drugs = Counter()
    for _, row in summary.iterrows():
        for company in row['companies']:
            # Clean company name
            company_clean = company.replace('(原研)', '').replace('(Top20 MNC)', '').replace('(无权益)', '').strip()
            company_drugs[company_clean] += row['total_drugs']

    # Get top companies
    top_companies = company_drugs.most_common(top_n)
    companies, counts = zip(*top_companies) if top_companies else ([], [])

    # Matplotlib version
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette("husl", len(companies))
    bars = ax.barh(list(companies)[::-1], list(counts)[::-1], color=colors)
    ax.set_xlabel('Number of Drugs', fontsize=12)
    ax.set_ylabel('Company', fontsize=12)
    ax.set_title(f'Top {top_n} Companies in Cyclic Peptide Drug Development', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_companies.png'), dpi=150)
    plt.close()

    # Interactive plotly
    fig = px.bar(
        x=list(counts), y=list(companies),
        orientation='h',
        labels={'x': 'Number of Drugs', 'y': 'Company'},
        title=f'Top {top_n} Companies in Cyclic Peptide Drug Development'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.write_html(os.path.join(output_dir, 'top_companies.html'))


def generate_summary_html(df: pd.DataFrame, analysis_results: dict, pivot_table: pd.DataFrame, output_dir: str):
    """Generate an HTML summary report."""
    summary = analysis_results['target_summary']

    # Get filtered targets for Immunology and Endocrinology & Metabolism
    filtered_targets = get_targets_by_indication_therapeutic_area(
        df, ['Immunology', 'Endocrinology & Metabolism']
    )

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cyclic Peptide Drug Target Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .plot-container {
            margin: 30px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .interactive-link {
            display: inline-block;
            margin: 10px 0;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .interactive-link:hover {
            background-color: #2980b9;
        }
        .target-link {
            color: #3498db;
            text-decoration: none;
        }
        .target-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cyclic Peptide Drug Target Analysis</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">""" + str(len(summary)) + """</div>
                <div class="stat-label">Unique Targets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(summary['total_drugs'].sum()) + """</div>
                <div class="stat-label">Total Drug-Target Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(len(summary[summary['highest_phase'].isin(['phase 3', 'pre-registration', 'approved', 'launched'])])) + """</div>
                <div class="stat-label">Targets with Late-Stage Drugs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">""" + str(sum(summary['company_count'])) + """</div>
                <div class="stat-label">Total Company Involvements</div>
            </div>
        </div>

"""

    # Add filtered targets section for Immunology and Endocrinology & Metabolism
    if not filtered_targets.empty:
        html_content += """
        <h2>Targets with Immunology or Endocrinology & Metabolism Indications</h2>
        <p>The following targets have drugs with indications in Immunology or Endocrinology & Metabolism therapeutic areas.</p>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Drug Count</th>
                    <th>Highest Phase</th>
                    <th>Therapeutic Areas</th>
                    <th>Key Indications</th>
                </tr>
            </thead>
            <tbody>
"""
        for _, row in filtered_targets.iterrows():
            target_filename = row['target'].replace('/', '_').replace(' ', '_').replace('α', 'alpha').replace('β', 'beta')
            target_filename = ''.join(c for c in target_filename if c.isalnum() or c in '_-')
            html_content += f"""
                <tr>
                    <td><a href="target_pages/{target_filename}.html" class="target-link">{row['target']}</a></td>
                    <td>{row['drug_count']}</td>
                    <td>{row['highest_phase_display']}</td>
                    <td>{', '.join(row['therapeutic_areas'])}</td>
                    <td>{', '.join(row['indications'][:3])}</td>
                </tr>
"""
        html_content += """
            </tbody>
        </table>
        <p><em>This table shows """ + str(len(filtered_targets)) + """ targets with indications in Immunology or Endocrinology & Metabolism.</em></p>
"""

    html_content += """
        <h2>Development Phase Distribution</h2>
        <div class="plot-container">
            <img src="plots/phase_distribution.png" alt="Phase Distribution">
            <br>
            <a href="plots/phase_distribution.html" class="interactive-link">View Interactive Chart</a>
        </div>

        <h2>Top Targets by Drug Count</h2>
        <div class="plot-container">
            <img src="plots/top_targets.png" alt="Top Targets">
            <br>
            <a href="plots/top_targets.html" class="interactive-link">View Interactive Chart</a>
        </div>

        <h2>Target-Phase Heatmap</h2>
        <div class="plot-container">
            <img src="plots/target_phase_heatmap.png" alt="Target Phase Heatmap">
            <br>
            <a href="plots/target_phase_heatmap.html" class="interactive-link">View Interactive Chart</a>
        </div>

        <h2>Top Companies</h2>
        <div class="plot-container">
            <img src="plots/top_companies.png" alt="Top Companies">
            <br>
            <a href="plots/top_companies.html" class="interactive-link">View Interactive Chart</a>
        </div>

        <h2>Target Summary Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Target</th>
                    <th>Total Drugs</th>
                    <th>Highest Phase</th>
                    <th>Companies</th>
                    <th>Therapeutic Areas</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in summary.head(50).iterrows():
        target_filename = row['target'].replace('/', '_').replace(' ', '_').replace('α', 'alpha').replace('β', 'beta')
        html_content += f"""
                <tr>
                    <td><a href="target_pages/{target_filename}.html" class="target-link">{row['target']}</a></td>
                    <td>{row['total_drugs']}</td>
                    <td>{row['highest_phase_display']}</td>
                    <td>{row['company_count']}</td>
                    <td>{', '.join(row['therapeutic_areas'][:3])}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <p><em>Click on target names to view detailed analysis pages.</em></p>

        <h2>Pivot Table: Targets by Phase</h2>
        """ + pivot_table.head(30).to_html(index=False, classes='pivot-table') + """

    </div>
</body>
</html>
"""

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)


def run_analysis(data_path: str, output_dir: str):
    """Run the complete target analysis pipeline."""
    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} drugs")

    print("Analyzing targets...")
    analysis_results = analyze_targets(df)
    print(f"Found {len(analysis_results['target_summary'])} unique targets")

    print("Creating pivot table...")
    pivot_table = create_pivot_table(df, analysis_results)

    # Create output directories
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target_pages'), exist_ok=True)

    print("Generating plots...")
    create_phase_distribution_plot(analysis_results, plots_dir)
    create_top_targets_plot(analysis_results, plots_dir)
    create_target_phase_heatmap(analysis_results, plots_dir)
    create_company_analysis_plot(analysis_results, plots_dir)

    print("Generating summary HTML...")
    generate_summary_html(df, analysis_results, pivot_table, output_dir)

    # Save pivot table as CSV
    pivot_table.to_csv(os.path.join(output_dir, 'target_pivot_table.csv'), index=False)

    print(f"Analysis complete! Results saved to {output_dir}")

    return df, analysis_results


if __name__ == "__main__":
    # Default paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "cyclic_peptide_drug_level_active.csv"
    output_dir = project_root / "output"

    run_analysis(str(data_path), str(output_dir))
