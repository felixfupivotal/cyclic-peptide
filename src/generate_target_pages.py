"""
Target Page Generator for Cyclic Peptide Drugs

This module generates individual HTML pages for each drug target,
including target biology, competitive landscape, drug pipeline information,
and AI-powered deep research summaries using LLM.
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from collections import Counter
import ast
from typing import Optional

# Import LLM research module
try:
    from llm_research import (
        generate_target_research,
        format_research_for_html,
        batch_generate_research
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


# Phase display names and CSS classes
PHASE_DISPLAY = {
    'pre-clinical': ('Pre-clinical', 'pre-clinical'),
    'ind': ('IND', 'ind'),
    'phase 1': ('Phase 1', '1'),
    'phase 2': ('Phase 2', '2'),
    'phase 3': ('Phase 3', '3'),
    'pre-registration': ('Pre-registration', 'pre-registration'),
    'approved': ('Approved', 'approved'),
    'launched': ('Launched', 'launched')
}

PHASE_ORDER = {
    'pre-clinical': -1, 'ind': 0, 'phase 1': 1, 'phase 2': 2,
    'phase 3': 3, 'pre-registration': 4, 'approved': 5, 'launched': 6
}

# Target biology templates - can be extended with actual biological descriptions
TARGET_BIOLOGY_TEMPLATES = {
    'default': """
    <p><strong>{target}</strong> is a molecular target being explored for therapeutic intervention
    using cyclic peptide approaches. Based on the drug development activity in this dataset,
    {target} represents an active area of pharmaceutical research.</p>

    <div class="highlight-box">
        <strong>Key Facts:</strong>
        <ul>
            <li>Total active drug programs: {drug_count}</li>
            <li>Most advanced development stage: {highest_phase}</li>
            <li>Primary therapeutic focus: {therapeutic_areas}</li>
        </ul>
    </div>
    """,

    'SSTR': """
    <p><strong>Somatostatin Receptors (SSTR)</strong> are G protein-coupled receptors that bind somatostatin,
    a peptide hormone that inhibits the release of numerous secondary hormones. There are five SSTR subtypes
    (SSTR1-5), each with distinct tissue distribution and signaling properties.</p>

    <p>SSTR2 is particularly important in neuroendocrine tumors, where it is frequently overexpressed.
    This makes it an excellent target for both diagnostic imaging (using radiolabeled somatostatin analogs)
    and therapeutic radiopharmaceuticals.</p>

    <div class="highlight-box">
        <strong>Clinical Significance:</strong>
        <ul>
            <li>Highly expressed in neuroendocrine tumors (NETs)</li>
            <li>Established target for peptide receptor radionuclide therapy (PRRT)</li>
            <li>FDA-approved therapies include Lutathera (177Lu-DOTATATE)</li>
        </ul>
    </div>
    """,

    'SSTR2': """
    <p><strong>Somatostatin Receptor 2 (SSTR2)</strong> is a G protein-coupled receptor that mediates the
    inhibitory effects of somatostatin on hormone secretion and cell proliferation. SSTR2 is the predominant
    somatostatin receptor subtype expressed in most neuroendocrine tumors.</p>

    <p>The receptor has become a cornerstone target for molecular imaging and targeted radionuclide therapy
    in neuroendocrine malignancies, with several approved radiopharmaceuticals targeting this receptor.</p>

    <div class="highlight-box">
        <strong>Therapeutic Applications:</strong>
        <ul>
            <li>Peptide receptor radionuclide therapy (PRRT) for NETs</li>
            <li>Diagnostic imaging with 68Ga-labeled somatostatin analogs</li>
            <li>Treatment of hormone-secreting tumors</li>
        </ul>
    </div>
    """,

    'FAP': """
    <p><strong>Fibroblast Activation Protein (FAP)</strong> is a type II transmembrane serine protease
    that is selectively expressed by activated fibroblasts in tumor stroma and other pathological conditions
    characterized by tissue remodeling. FAP is minimally expressed in normal adult tissues.</p>

    <p>The tumor-selective expression pattern makes FAP an attractive target for cancer diagnostics and
    therapeutics, particularly for radiopharmaceutical approaches that can deliver cytotoxic payloads
    specifically to the tumor microenvironment.</p>

    <div class="highlight-box">
        <strong>Key Characteristics:</strong>
        <ul>
            <li>Expressed in >90% of epithelial carcinomas</li>
            <li>Minimal expression in normal tissues</li>
            <li>Emerging target for theranostic applications</li>
        </ul>
    </div>
    """,

    'GLP-1R': """
    <p><strong>Glucagon-Like Peptide-1 Receptor (GLP-1R)</strong> is a G protein-coupled receptor that mediates
    the effects of GLP-1, an incretin hormone released from intestinal L-cells in response to food intake.
    Activation of GLP-1R enhances glucose-dependent insulin secretion and has pleiotropic effects on appetite,
    gastric emptying, and cardiovascular function.</p>

    <p>GLP-1R agonists have revolutionized the treatment of type 2 diabetes and obesity, with cyclic peptide
    approaches offering potential advantages in terms of stability and bioavailability.</p>

    <div class="highlight-box">
        <strong>Therapeutic Significance:</strong>
        <ul>
            <li>Established target for diabetes and obesity treatment</li>
            <li>Cardiovascular and renal protective effects</li>
            <li>Multiple approved therapies (semaglutide, tirzepatide)</li>
        </ul>
    </div>
    """,

    'CAIX': """
    <p><strong>Carbonic Anhydrase IX (CAIX)</strong> is a transmembrane enzyme that is strongly upregulated
    under hypoxic conditions, particularly in solid tumors. CAIX plays a crucial role in pH regulation
    and is associated with aggressive tumor phenotypes and poor prognosis.</p>

    <p>The hypoxia-induced, tumor-selective expression of CAIX makes it an ideal target for
    radiopharmaceutical theranostics, allowing imaging and therapy of hypoxic tumor regions that are
    often resistant to conventional treatments.</p>

    <div class="highlight-box">
        <strong>Oncology Applications:</strong>
        <ul>
            <li>Biomarker of tumor hypoxia</li>
            <li>Target for imaging hypoxic tumor regions</li>
            <li>Therapeutic target for renal cell carcinoma</li>
        </ul>
    </div>
    """,

    'integrin': """
    <p><strong>Integrins</strong> are heterodimeric transmembrane receptors that mediate cell-cell and
    cell-extracellular matrix adhesion. The integrin family includes various alpha and beta subunit
    combinations with distinct ligand specificities and biological functions.</p>

    <p>Integrins play critical roles in tumor angiogenesis, metastasis, and immune cell trafficking,
    making them attractive targets for cancer therapy and imaging. RGD-based cyclic peptides targeting
    integrins have been extensively studied for tumor imaging and targeted drug delivery.</p>

    <div class="highlight-box">
        <strong>Therapeutic Relevance:</strong>
        <ul>
            <li>Role in tumor angiogenesis and metastasis</li>
            <li>Target for RGD-based radiopharmaceuticals</li>
            <li>Applications in tumor imaging and therapy</li>
        </ul>
    </div>
    """
}

CYCLIC_PEPTIDE_SUITABILITY_TEMPLATES = {
    'default': """
    <p>Cyclic peptides offer several advantages for targeting {target}:</p>

    <ul class="feature-list">
        <li><strong>Enhanced Stability:</strong> The cyclic structure provides resistance to proteolytic
        degradation, extending half-life compared to linear peptides.</li>
        <li><strong>Improved Binding Affinity:</strong> Conformational constraint often results in higher
        target affinity and selectivity.</li>
        <li><strong>Versatile Conjugation:</strong> Cyclic peptide scaffolds can be readily conjugated to
        various payloads including radionuclides, cytotoxic agents, and imaging probes.</li>
        <li><strong>Good Tissue Penetration:</strong> Appropriate molecular size enables efficient tumor
        penetration while maintaining target binding.</li>
    </ul>

    <div class="highlight-box">
        <strong>Development Considerations:</strong>
        <p>With {drug_count} drugs in development and the most advanced reaching {highest_phase},
        {target} represents a validated target for cyclic peptide therapeutic development.</p>
    </div>
    """,

    'radiopharmaceutical': """
    <p>Cyclic peptides are particularly well-suited for radiopharmaceutical applications targeting {target}:</p>

    <ul class="feature-list">
        <li><strong>Optimal Pharmacokinetics:</strong> Fast blood clearance and efficient tumor uptake
        provide favorable tumor-to-background ratios for imaging and therapy.</li>
        <li><strong>Chelator Compatibility:</strong> Easy incorporation of metal chelators (DOTA, NOTA)
        for radiolabeling with diagnostic (68Ga, 111In) and therapeutic (177Lu, 225Ac) radionuclides.</li>
        <li><strong>Theranostic Approach:</strong> Same peptide scaffold can be used for both diagnostic
        imaging and targeted radionuclide therapy.</li>
        <li><strong>Manufacturing Feasibility:</strong> Well-established synthesis methods enable GMP
        production of radiolabeled peptides.</li>
    </ul>

    <div class="highlight-box">
        <strong>Clinical Validation:</strong>
        <p>The success of peptide receptor radionuclide therapy (PRRT) with somatostatin analogs has
        established the clinical paradigm for cyclic peptide radiopharmaceuticals.</p>
    </div>
    """
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load the cyclic peptide drug data."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def parse_list_column(value: str) -> list:
    """Parse a string representation of a list."""
    try:
        if pd.isna(value):
            return []
        if value.startswith('['):
            return ast.literal_eval(value)
        return [value]
    except (ValueError, SyntaxError):
        return [str(value)] if value else []


def parse_indications_json(json_str: str) -> list:
    """Parse the indications JSON column."""
    try:
        if pd.isna(json_str):
            return []
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def get_target_drugs(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Get all drugs targeting a specific target."""
    mask = df['targets'].apply(lambda x: target in str(x).split(';') if pd.notna(x) else False)
    return df[mask].copy()


def get_target_biology(target: str, drug_count: int, highest_phase: str, therapeutic_areas: list) -> str:
    """Get target biology description."""
    # Check for specific target templates
    for key in TARGET_BIOLOGY_TEMPLATES:
        if key.lower() in target.lower() and key != 'default':
            template = TARGET_BIOLOGY_TEMPLATES[key]
            return template.format(
                target=target,
                drug_count=drug_count,
                highest_phase=highest_phase,
                therapeutic_areas=', '.join(therapeutic_areas[:3])
            )

    # Use default template
    return TARGET_BIOLOGY_TEMPLATES['default'].format(
        target=target,
        drug_count=drug_count,
        highest_phase=highest_phase,
        therapeutic_areas=', '.join(therapeutic_areas[:3]) if therapeutic_areas else 'Various'
    )


def get_cyclic_peptide_suitability(target: str, drug_count: int, highest_phase: str,
                                    is_radiopharmaceutical: bool) -> str:
    """Get description of target suitability for cyclic peptides."""
    if is_radiopharmaceutical:
        template = CYCLIC_PEPTIDE_SUITABILITY_TEMPLATES['radiopharmaceutical']
    else:
        template = CYCLIC_PEPTIDE_SUITABILITY_TEMPLATES['default']

    return template.format(
        target=target,
        drug_count=drug_count,
        highest_phase=highest_phase
    )


def get_competitive_landscape(drugs_df: pd.DataFrame, target: str) -> str:
    """Generate competitive landscape analysis."""
    total_drugs = len(drugs_df)

    # Count by phase
    phase_counts = drugs_df['global_highest_phase_all_indication_standard'].value_counts()

    # Get unique companies
    all_companies = []
    for companies_str in drugs_df['research_institution_list']:
        all_companies.extend(parse_list_column(companies_str))
    unique_companies = len(set(all_companies))

    # Count MNC involvement
    mnc_count = drugs_df['is_top20_mnc'].sum() if 'is_top20_mnc' in drugs_df.columns else 0

    # Determine competitive intensity
    if total_drugs >= 10:
        intensity = "highly competitive"
        intensity_desc = "significant industry interest with multiple players"
    elif total_drugs >= 5:
        intensity = "moderately competitive"
        intensity_desc = "growing interest from pharmaceutical companies"
    else:
        intensity = "emerging"
        intensity_desc = "early-stage development with opportunity for first-mover advantage"

    # Build landscape description
    html = f"""
    <p>The competitive landscape for <strong>{target}</strong>-targeting cyclic peptides is
    <strong>{intensity}</strong>, with {intensity_desc}.</p>

    <div class="competitive-landscape">
        <div class="landscape-item">
            <span>Total Active Programs</span>
            <strong>{total_drugs}</strong>
        </div>
        <div class="landscape-item">
            <span>Companies Involved</span>
            <strong>{unique_companies}</strong>
        </div>
        <div class="landscape-item">
            <span>Top 20 MNC Involvement</span>
            <strong>{mnc_count} programs</strong>
        </div>
    </div>

    <h3>Development Stage Distribution</h3>
    <div class="competitive-landscape">
    """

    for phase in ['approved', 'launched', 'pre-registration', 'phase 3', 'phase 2', 'phase 1', 'ind', 'pre-clinical']:
        count = phase_counts.get(phase, 0)
        if count > 0:
            display_name = PHASE_DISPLAY.get(phase, (phase, phase))[0]
            html += f"""
        <div class="landscape-item">
            <span>{display_name}</span>
            <strong>{count} drug(s)</strong>
        </div>
            """

    html += "</div>"

    # Add strategic insights
    if any(phase_counts.get(p, 0) > 0 for p in ['approved', 'launched']):
        html += """
        <div class="highlight-box">
            <strong>Market Insight:</strong> Approved drugs in this space demonstrate clinical
            and commercial validation of the target, though new entrants face competitive pressure.
        </div>
        """
    elif any(phase_counts.get(p, 0) > 0 for p in ['phase 3', 'pre-registration']):
        html += """
        <div class="highlight-box">
            <strong>Market Insight:</strong> Late-stage programs indicate strong clinical validation.
            Differentiation strategy will be key for new entrants.
        </div>
        """
    else:
        html += """
        <div class="highlight-box">
            <strong>Market Insight:</strong> Early-stage competitive landscape offers opportunities
            for differentiated approaches and potential first-mover advantage.
        </div>
        """

    return html


def prepare_drug_list(drugs_df: pd.DataFrame) -> list:
    """Prepare drug list for template."""
    drugs = []
    for _, row in drugs_df.iterrows():
        phase = row['global_highest_phase_all_indication_standard']
        phase_display, phase_class = PHASE_DISPLAY.get(phase, (phase, 'pre-clinical'))

        # Parse companies
        companies = parse_list_column(row.get('research_institution_list', ''))
        companies_clean = [c.replace('(原研)', '').replace('(Top20 MNC)', '').replace('(无权益)', '').strip()
                          for c in companies[:3]]

        # Parse indications
        indications = parse_indications_json(row.get('indications_json', ''))
        indication_names = list(set([ind.get('indication_en', '') for ind in indications[:5]]))

        drugs.append({
            'name': row['drug_name_en'],
            'phase': phase_display,
            'phase_class': phase_class,
            'companies': ', '.join(companies_clean) if companies_clean else 'N/A',
            'indications': ', '.join(indication_names[:3]) if indication_names else 'N/A',
            'therapeutic_area': row.get('therapeutic_area_primary_en', 'N/A')
        })

    # Sort by phase (most advanced first)
    drugs.sort(key=lambda x: -PHASE_ORDER.get(
        [k for k, v in PHASE_DISPLAY.items() if v[0] == x['phase']][0] if x['phase'] else 'pre-clinical', -1))

    return drugs


def prepare_company_list(drugs_df: pd.DataFrame) -> list:
    """Prepare company list for template."""
    company_counts = Counter()
    mnc_companies = set()

    for _, row in drugs_df.iterrows():
        companies = parse_list_column(row.get('research_institution_list', ''))
        for company in companies:
            clean_name = company.replace('(原研)', '').replace('(Top20 MNC)', '').replace('(无权益)', '').strip()
            company_counts[clean_name] += 1
            if '(Top20 MNC)' in company:
                mnc_companies.add(clean_name)

    companies = []
    for company, count in company_counts.most_common():
        companies.append({
            'name': company,
            'is_mnc': company in mnc_companies,
            'drug_count': count
        })

    return companies


def prepare_indication_list(drugs_df: pd.DataFrame) -> list:
    """Prepare indication list for template."""
    indication_data = {}

    for _, row in drugs_df.iterrows():
        indications = parse_indications_json(row.get('indications_json', ''))
        for ind in indications:
            name = ind.get('indication_en', '')
            if name:
                phase = ind.get('highest_phase', 'pre-clinical')
                phase_val = PHASE_ORDER.get(phase, -1)
                ta = ind.get('therapeutic_area_en', 'N/A')

                if name not in indication_data or phase_val > indication_data[name]['phase_val']:
                    indication_data[name] = {
                        'name': name,
                        'phase': PHASE_DISPLAY.get(phase, (phase, phase))[0],
                        'phase_val': phase_val,
                        'therapeutic_area': ta
                    }

    indications = list(indication_data.values())
    indications.sort(key=lambda x: -x['phase_val'])

    return indications[:20]  # Top 20 indications


def generate_target_page(
    target: str,
    drugs_df: pd.DataFrame,
    template,
    output_dir: str,
    enable_llm_research: bool = False,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
    use_cache: bool = True
):
    """
    Generate an HTML page for a specific target.
    
    Args:
        target: Target name.
        drugs_df: DataFrame of drugs targeting this target.
        template: Jinja2 template object.
        output_dir: Output directory for HTML files.
        enable_llm_research: Whether to generate LLM research summary.
        llm_api_key: OpenAI API key for LLM research.
        llm_model: OpenAI model to use (default: gpt-4o).
        use_cache: Whether to use cached LLM results.
    """
    # Calculate statistics
    total_drugs = len(drugs_df)

    # Get highest phase
    phase_vals = drugs_df['global_highest_phase_all_indication_standard'].apply(
        lambda x: PHASE_ORDER.get(x, -2) if pd.notna(x) else -2
    )
    max_phase_val = phase_vals.max()
    highest_phase = [k for k, v in PHASE_ORDER.items() if v == max_phase_val][0] if max_phase_val >= -1 else 'pre-clinical'
    highest_phase_display = PHASE_DISPLAY.get(highest_phase, (highest_phase, highest_phase))[0]

    # Get therapeutic areas
    therapeutic_areas = drugs_df['therapeutic_area_primary_en'].dropna().unique().tolist()

    # Check if radiopharmaceutical focused
    is_radiopharmaceutical = any(
        'radiopharmac' in str(row.get('drug_category_3', '')).lower() or
        '放射性' in str(row.get('drug_category_3', ''))
        for _, row in drugs_df.iterrows()
    )

    # Prepare template data
    drugs = prepare_drug_list(drugs_df)
    companies = prepare_company_list(drugs_df)
    indications = prepare_indication_list(drugs_df)
    
    # Generate LLM research if enabled
    llm_research_html = None
    if enable_llm_research and LLM_AVAILABLE:
        try:
            company_names = [c['name'] for c in companies]
            research_result = generate_target_research(
                target=target,
                drug_count=total_drugs,
                highest_phase=highest_phase_display,
                therapeutic_areas=therapeutic_areas,
                drugs=drugs,
                companies=company_names,
                api_key=llm_api_key,
                model=llm_model,
                use_cache=use_cache
            )
            llm_research_html = format_research_for_html(research_result)
        except Exception as e:
            print(f"Warning: LLM research generation failed for {target}: {e}")
            llm_research_html = None

    template_data = {
        'target_name': target,
        'total_drugs': total_drugs,
        'highest_phase': highest_phase_display,
        'company_count': len(companies),
        'indication_count': len(indications),
        'target_biology': get_target_biology(target, total_drugs, highest_phase_display, therapeutic_areas),
        'cyclic_peptide_suitability': get_cyclic_peptide_suitability(
            target, total_drugs, highest_phase_display, is_radiopharmaceutical
        ),
        'competitive_landscape': get_competitive_landscape(drugs_df, target),
        'drugs': drugs,
        'companies': companies,
        'indications': indications,
        'llm_research': llm_research_html,
        'generation_date': datetime.now().strftime('%Y-%m-%d')
    }

    # Render template
    html_content = template.render(**template_data)

    # Generate safe filename
    safe_filename = target.replace('/', '_').replace(' ', '_').replace('α', 'alpha').replace('β', 'beta')
    safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '_-')

    output_path = os.path.join(output_dir, f"{safe_filename}.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def generate_all_target_pages(
    df: pd.DataFrame,
    template_dir: str,
    output_dir: str,
    min_drugs: int = 1,
    enable_llm_research: bool = False,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
    use_cache: bool = True
):
    """
    Generate HTML pages for all targets.
    
    Args:
        df: DataFrame with drug data.
        template_dir: Directory containing Jinja2 templates.
        output_dir: Output directory for HTML files.
        min_drugs: Minimum number of drugs required to generate a page.
        enable_llm_research: Whether to generate LLM research summaries.
        llm_api_key: OpenAI API key for LLM research.
        llm_model: OpenAI model to use.
        use_cache: Whether to use cached LLM results.
    """
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('target_page.html')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique targets
    all_targets = set()
    for targets_str in df['targets'].dropna():
        for target in str(targets_str).split(';'):
            target = target.strip()
            if target:
                all_targets.add(target)

    print(f"Found {len(all_targets)} unique targets")
    
    if enable_llm_research:
        if not LLM_AVAILABLE:
            print("Warning: LLM research requested but llm_research module not available.")
            print("Please ensure openai package is installed: pip install openai")
            enable_llm_research = False
        else:
            # Determine provider based on model name
            is_gemini = 'gemini' in llm_model.lower()
            
            # Validate API key is available before starting
            if is_gemini:
                effective_api_key = llm_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                key_env_var = "GEMINI_API_KEY or GOOGLE_API_KEY"
            else:
                effective_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY")
                key_env_var = "OPENAI_API_KEY"
            
            if not effective_api_key:
                print(f"Warning: LLM research requested but no API key found for {'Gemini' if is_gemini else 'OpenAI'}.")
                print("Please either:")
                print(f"  - Set {key_env_var} environment variable, or")
                print("  - Pass --llm-api-key parameter")
                print("Continuing without LLM research...")
                enable_llm_research = False
            else:
                # Store the key for the session
                llm_api_key = effective_api_key
                provider = "Gemini" if is_gemini else "OpenAI"
                print(f"LLM research enabled using {provider} model: {llm_model}")
                if use_cache:
                    print("Cache enabled - previously generated research will be reused")

    generated_pages = []
    for i, target in enumerate(sorted(all_targets)):
        target_drugs = get_target_drugs(df, target)

        if len(target_drugs) >= min_drugs:
            try:
                output_path = generate_target_page(
                    target=target,
                    drugs_df=target_drugs,
                    template=template,
                    output_dir=output_dir,
                    enable_llm_research=enable_llm_research,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    use_cache=use_cache
                )
                generated_pages.append((target, output_path))
                cache_status = ""
                if enable_llm_research:
                    cache_status = " [LLM]"
                print(f"[{i+1}/{len(all_targets)}] Generated page for {target} ({len(target_drugs)} drugs){cache_status}")
            except Exception as e:
                print(f"Error generating page for {target}: {e}")

    print(f"\nGenerated {len(generated_pages)} target pages")
    return generated_pages


def run_page_generation(
    data_path: str,
    template_dir: str,
    output_dir: str,
    min_drugs: int = 1,
    enable_llm_research: bool = False,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
    use_cache: bool = True
):
    """
    Run the complete page generation workflow.
    
    Args:
        data_path: Path to the drug data CSV file.
        template_dir: Directory containing Jinja2 templates.
        output_dir: Output directory for HTML files.
        min_drugs: Minimum number of drugs required to generate a page.
        enable_llm_research: Whether to generate LLM research summaries.
        llm_api_key: OpenAI API key for LLM research.
        llm_model: OpenAI model to use (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo').
        use_cache: Whether to use cached LLM results.
    """
    print("Loading data...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} drugs")

    print("\nGenerating target pages...")
    generated = generate_all_target_pages(
        df=df,
        template_dir=template_dir,
        output_dir=output_dir,
        min_drugs=min_drugs,
        enable_llm_research=enable_llm_research,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        use_cache=use_cache
    )

    print(f"\nPage generation complete! Generated {len(generated)} pages in {output_dir}")
    return generated


if __name__ == "__main__":
    import argparse
    
    # Default paths
    project_root = Path(__file__).parent.parent
    default_data_path = project_root / "data" / "cyclic_peptide_drug_level_active.csv"
    default_template_dir = project_root / "templates"
    default_output_dir = project_root / "output" / "target_pages"
    
    parser = argparse.ArgumentParser(
        description="Generate target analysis pages for cyclic peptide drugs"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=str(default_data_path),
        help="Path to the drug data CSV file"
    )
    parser.add_argument(
        "--templates", "-t",
        type=str,
        default=str(default_template_dir),
        help="Path to the templates directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output_dir),
        help="Output directory for generated HTML pages"
    )
    parser.add_argument(
        "--min-drugs",
        type=int,
        default=1,
        help="Minimum number of drugs required to generate a target page"
    )
    parser.add_argument(
        "--llm-research",
        action="store_true",
        help="Enable AI-powered deep research using OpenAI GPT models"
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="LLM model to use. OpenAI: gpt-4o, o1, o1-mini. Gemini: gemini-1.5-pro, gemini-1.5-flash"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of LLM research results"
    )
    
    args = parser.parse_args()
    
    run_page_generation(
        data_path=args.data,
        template_dir=args.templates,
        output_dir=args.output,
        min_drugs=args.min_drugs,
        enable_llm_research=args.llm_research,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        use_cache=not args.no_cache
    )
