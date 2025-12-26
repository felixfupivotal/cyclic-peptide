"""
LLM-powered Deep Research Module for Cyclic Peptide Drug Targets

This module uses OpenAI's GPT models to generate in-depth research summaries
for drug targets, focusing on cyclic peptide modality in drug development.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Default cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "llm_research"

# Module-level client cache to avoid re-initialization issues
_cached_client: Optional["OpenAI"] = None
_cached_api_key: Optional[str] = None


def get_openai_client(api_key: Optional[str] = None) -> "OpenAI":
    """
    Initialize and return an OpenAI client.
    
    Uses a cached client if the API key matches to avoid re-initialization issues.
    
    Args:
        api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env variable.
    
    Returns:
        OpenAI client instance.
    
    Raises:
        ImportError: If openai package is not installed.
        ValueError: If no API key is available.
    """
    global _cached_client, _cached_api_key
    
    if OpenAI is None:
        raise ImportError(
            "OpenAI package not installed. Please install with: pip install openai"
        )
    
    # Determine the key to use
    key = api_key or _cached_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Return cached client if key matches
    if _cached_client is not None and _cached_api_key == key:
        return _cached_client
    
    # Create new client and cache it
    _cached_client = OpenAI(api_key=key)
    _cached_api_key = key
    
    return _cached_client


def get_cache_path(target: str, model: str) -> Path:
    """Generate cache file path for a target research."""
    # Create a hash for the target name to handle special characters
    target_hash = hashlib.md5(target.encode('utf-8')).hexdigest()[:12]
    # Only allow ASCII alphanumeric chars in filename (replace Greek letters, etc.)
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in target)[:50]
    # Remove consecutive underscores and strip
    safe_name = "_".join(filter(None, safe_name.split("_")))
    return CACHE_DIR / f"{safe_name}_{target_hash}_{model.replace('/', '_')}.json"


def load_cached_research(target: str, model: str, max_age_days: int = 30) -> Optional[dict]:
    """
    Load cached research for a target if available and not expired.
    
    Args:
        target: Target name.
        model: Model name used for generation.
        max_age_days: Maximum age of cache in days before refresh.
    
    Returns:
        Cached research dict or None if not available/expired.
    """
    cache_path = get_cache_path(target, model)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        
        # Check cache age
        cached_date = datetime.fromisoformat(cached.get("generated_at", "2000-01-01"))
        age_days = (datetime.now() - cached_date).days
        
        if age_days > max_age_days:
            return None
        
        return cached
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_research_cache(target: str, model: str, research: dict):
    """Save research to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(target, model)
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2, ensure_ascii=False)


def build_research_prompt(
    target: str,
    drug_count: int,
    highest_phase: str,
    therapeutic_areas: list,
    drugs: list,
    companies: list
) -> str:
    """
    Build the research prompt for the LLM.
    
    Args:
        target: Target name.
        drug_count: Number of drugs targeting this target.
        highest_phase: Highest development phase reached.
        therapeutic_areas: List of therapeutic areas.
        drugs: List of drug information dicts.
        companies: List of companies involved.
    
    Returns:
        Formatted prompt string.
    """
    # Format drug info
    drug_info = "\n".join([
        f"  - {d['name']}: {d['phase']} ({d.get('companies', 'N/A')})"
        for d in drugs[:10]  # Limit to top 10 drugs
    ])
    
    company_info = ", ".join(companies[:10]) if companies else "Various"
    ta_info = ", ".join(therapeutic_areas[:5]) if therapeutic_areas else "Various"
    
    prompt = f"""You are a pharmaceutical research analyst specializing in cyclic peptide drug development. 
Provide a comprehensive research summary for the following drug target, focusing specifically on its potential 
for cyclic peptide-based therapeutic development.

TARGET: {target}

CURRENT DEVELOPMENT STATUS:
- Total active drug programs: {drug_count}
- Highest development phase: {highest_phase}
- Primary therapeutic areas: {ta_info}
- Key companies involved: {company_info}

DRUGS IN DEVELOPMENT:
{drug_info}

Please provide a detailed research summary covering the following aspects:

1. **Target Biology & Mechanism**
   - Molecular function and signaling pathways
   - Role in disease pathophysiology
   - Known ligands and binding characteristics

2. **Cyclic Peptide Modality Advantages**
   - Why cyclic peptides are particularly suited for this target
   - Structural considerations (binding pocket accessibility, etc.)
   - Comparison with other modalities (small molecules, antibodies)

3. **Drug Development Potential**
   - Therapeutic opportunities and unmet medical needs
   - Key development challenges and how cyclic peptides address them
   - Regulatory pathway considerations

4. **Competitive Landscape Analysis**
   - Current market leaders and their approaches
   - Differentiation opportunities for new entrants
   - Partnership and licensing potential

5. **Future Outlook**
   - Emerging research directions
   - Potential for combination therapies
   - Expected timeline for key clinical readouts

Format the response in clear sections with headers. Be specific and evidence-based where possible.
Focus on actionable insights for drug development decision-making."""

    return prompt


def generate_target_research(
    target: str,
    drug_count: int,
    highest_phase: str,
    therapeutic_areas: list,
    drugs: list,
    companies: list,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    use_cache: bool = True,
    cache_max_age_days: int = 30,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> dict:
    """
    Generate deep research summary for a target using LLM.
    
    Args:
        target: Target name.
        drug_count: Number of drugs targeting this target.
        highest_phase: Highest development phase reached.
        therapeutic_areas: List of therapeutic areas.
        drugs: List of drug information dicts.
        companies: List of company names.
        api_key: OpenAI API key (optional, uses env var if not provided).
        model: OpenAI model to use (default: gpt-4o).
        use_cache: Whether to use cached results.
        cache_max_age_days: Maximum cache age in days.
        max_retries: Maximum number of retries on API failure.
        retry_delay: Delay between retries in seconds.
    
    Returns:
        Dict containing research summary and metadata.
    """
    # Check cache first
    if use_cache:
        cached = load_cached_research(target, model, cache_max_age_days)
        if cached:
            cached["from_cache"] = True
            return cached
    
    # Build prompt
    prompt = build_research_prompt(
        target, drug_count, highest_phase, therapeutic_areas, drugs, companies
    )
    
    # Initialize client
    client = get_openai_client(api_key)
    
    # Make API call with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert pharmaceutical research analyst with deep knowledge "
                            "of cyclic peptide drug development, target biology, and competitive "
                            "landscape analysis. Provide detailed, actionable research insights."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            research_content = response.choices[0].message.content
            
            result = {
                "target": target,
                "research_summary": research_content,
                "model": model,
                "generated_at": datetime.now().isoformat(),
                "drug_count": drug_count,
                "highest_phase": highest_phase,
                "from_cache": False,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
            
            # Cache the result
            if use_cache:
                save_research_cache(target, model, result)
            
            return result
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
    
    # Return error result if all retries failed
    return {
        "target": target,
        "research_summary": None,
        "error": str(last_error),
        "model": model,
        "generated_at": datetime.now().isoformat(),
        "from_cache": False
    }


def format_research_for_html(research: dict) -> str:
    """
    Format the research summary for HTML display.
    
    Args:
        research: Research dict from generate_target_research.
    
    Returns:
        HTML-formatted string.
    """
    if not research or not research.get("research_summary"):
        error_msg = research.get("error", "Research generation failed") if research else "No research available"
        return f"""
        <div class="research-error">
            <p><em>AI-powered research summary not available: {error_msg}</em></p>
            <p>Please ensure the OpenAI API key is configured correctly.</p>
        </div>
        """
    
    summary = research["research_summary"]
    
    # Convert markdown-style headers to HTML
    import re
    
    # Convert **bold** to <strong>
    summary = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', summary)
    
    # Convert headers (## Header) to h3/h4
    summary = re.sub(r'^### (.+)$', r'<h4>\1</h4>', summary, flags=re.MULTILINE)
    summary = re.sub(r'^## (.+)$', r'<h3>\1</h3>', summary, flags=re.MULTILINE)
    summary = re.sub(r'^# (.+)$', r'<h3>\1</h3>', summary, flags=re.MULTILINE)
    
    # Convert numbered sections (1. **Title**) to styled sections
    summary = re.sub(
        r'^(\d+)\.\s*<strong>([^<]+)</strong>',
        r'<h4 class="research-section-title">\1. \2</h4>',
        summary,
        flags=re.MULTILINE
    )
    
    # Convert bullet points to list items
    lines = summary.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                result_lines.append('<ul class="research-list">')
                in_list = True
            result_lines.append(f'<li>{stripped[2:]}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            if stripped:
                if not stripped.startswith('<'):
                    result_lines.append(f'<p>{stripped}</p>')
                else:
                    result_lines.append(stripped)
    
    if in_list:
        result_lines.append('</ul>')
    
    formatted_content = '\n'.join(result_lines)
    
    # Add metadata footer
    generated_at = research.get("generated_at", "Unknown")
    model = research.get("model", "Unknown")
    from_cache = research.get("from_cache", False)
    cache_badge = ' <span class="cache-badge">(cached)</span>' if from_cache else ''
    
    html = f"""
    <div class="llm-research-content">
        {formatted_content}
    </div>
    <div class="research-metadata">
        <span>Generated by {model}{cache_badge}</span>
        <span>Last updated: {generated_at[:10] if len(generated_at) >= 10 else generated_at}</span>
    </div>
    """
    
    return html


def batch_generate_research(
    targets: list,
    target_data: dict,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    use_cache: bool = True,
    progress_callback=None,
    delay_between_calls: float = 1.0
) -> dict:
    """
    Generate research for multiple targets with rate limiting.
    
    Args:
        targets: List of target names.
        target_data: Dict mapping target names to their data (drug_count, highest_phase, etc.)
        api_key: OpenAI API key.
        model: Model to use.
        use_cache: Whether to use cache.
        progress_callback: Optional callback(current, total, target_name) for progress.
        delay_between_calls: Delay between API calls in seconds.
    
    Returns:
        Dict mapping target names to research results.
    """
    results = {}
    total = len(targets)
    
    for i, target in enumerate(targets):
        if progress_callback:
            progress_callback(i + 1, total, target)
        
        data = target_data.get(target, {})
        
        result = generate_target_research(
            target=target,
            drug_count=data.get("drug_count", 0),
            highest_phase=data.get("highest_phase", "Unknown"),
            therapeutic_areas=data.get("therapeutic_areas", []),
            drugs=data.get("drugs", []),
            companies=data.get("companies", []),
            api_key=api_key,
            model=model,
            use_cache=use_cache
        )
        
        results[target] = result
        
        # Rate limiting (skip if result was from cache)
        if not result.get("from_cache", False) and i < total - 1:
            time.sleep(delay_between_calls)
    
    return results


def clear_cache(target: Optional[str] = None, model: Optional[str] = None):
    """
    Clear cached research results.
    
    Args:
        target: Specific target to clear (None for all).
        model: Specific model to clear (None for all).
    """
    if not CACHE_DIR.exists():
        return
    
    if target is None and model is None:
        # Clear all cache
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
    else:
        # Clear specific cache entries
        for cache_file in CACHE_DIR.glob("*.json"):
            should_delete = False
            
            if target and target.lower() in cache_file.stem.lower():
                should_delete = True
            if model and model.replace('/', '_') in cache_file.stem:
                should_delete = True
                
            if should_delete:
                cache_file.unlink()


if __name__ == "__main__":
    # Test example
    import os
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable to test.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
    else:
        # Test with a sample target
        result = generate_target_research(
            target="SSTR2",
            drug_count=15,
            highest_phase="Approved",
            therapeutic_areas=["Oncology", "Neuroendocrine tumors"],
            drugs=[
                {"name": "Lutathera", "phase": "Approved", "companies": "Novartis"},
                {"name": "DOTATATE", "phase": "Approved", "companies": "Various"}
            ],
            companies=["Novartis", "Advanced Accelerator Applications"]
        )
        
        if result.get("research_summary"):
            print("Research Summary Generated Successfully!")
            print("-" * 50)
            print(result["research_summary"][:1000] + "...")
        else:
            print(f"Error: {result.get('error')}")


