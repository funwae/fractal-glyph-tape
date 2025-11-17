#!/usr/bin/env python3
"""
Phase 5 Results Summary and Visualization

Aggregates benchmark results and creates:
- Summary tables in markdown
- Charts comparing RAW-TRUNCATE vs FGT-CONTEXT
- Analysis of where FGT helps most
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def create_summary_table(results: Dict[str, Any], output_path: Path):
    """Create markdown summary table."""
    lines = [
        "# Phase 5 Context Efficiency Results",
        "",
        f"**Dataset:** {results['dataset']}",
        f"**Episodes:** {results['num_episodes']}",
        f"**Timestamp:** {results['timestamp']}",
        f"**FGMS Available:** {results['fgms_available']}",
        "",
        "## Summary Table",
        "",
        "| Token Budget | Strategy | Success Rate | Avg Tokens | Avg Completeness | Improvement |",
        "|-------------|----------|--------------|------------|------------------|-------------|",
    ]

    for result in results["results"]:
        budget = result["token_budget"]

        # RAW-TRUNCATE row
        raw = result["raw_truncate"]
        lines.append(
            f"| {budget} | RAW-TRUNCATE | "
            f"{raw['success_rate']:.1%} ({raw['correct']}/{result['num_episodes']}) | "
            f"{raw['avg_tokens']:.0f} | "
            f"{raw['avg_completeness']:.1%} | "
            f"baseline |"
        )

        # FGT-CONTEXT/FOVEATED row
        fgt_key = 'fgt_context' if 'fgt_context' in result else 'fgt_foveated'
        if results['fgms_available'] and 'success_rate' in result.get(fgt_key, {}):
            fgt = result[fgt_key]
            improvement = (fgt['success_rate'] - raw['success_rate']) * 100
            strategy_name = "FGT-CONTEXT" if fgt_key == 'fgt_context' else "FGT-FOVEATED"
            lines.append(
                f"| {budget} | {strategy_name} | "
                f"{fgt['success_rate']:.1%} ({fgt['correct']}/{result['num_episodes']}) | "
                f"{fgt['avg_tokens']:.0f} | "
                f"{fgt['avg_completeness']:.1%} | "
                f"{improvement:+.1f}pp |"
            )

    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Calculate overall improvement
    if results['fgms_available']:
        improvements = []
        for result in results["results"]:
            fgt_key = 'fgt_context' if 'fgt_context' in result else 'fgt_foveated'
            if 'success_rate' in result.get(fgt_key, {}):
                improvement = (
                    result[fgt_key]['success_rate'] -
                    result['raw_truncate']['success_rate']
                ) * 100
                improvements.append(improvement)

        if improvements:
            avg_improvement = np.mean(improvements)
            max_improvement = np.max(improvements)
            min_improvement = np.min(improvements)

            lines.extend([
                f"- **Average improvement:** {avg_improvement:+.1f} percentage points",
                f"- **Best improvement:** {max_improvement:+.1f}pp",
                f"- **Worst improvement:** {min_improvement:+.1f}pp",
                "",
            ])

            # Analyze when FGT helps most
            if avg_improvement > 0:
                lines.append("**FGT-CONTEXT shows advantages when:**")
                lines.append("- Information needed for answering is mentioned early in conversation")
                lines.append("- Token budget is limited relative to conversation length")
                lines.append("- Relevant context is spread across non-contiguous turns")
                lines.append("")

    # Add detailed breakdown
    lines.extend([
        "## Detailed Results by Budget",
        "",
    ])

    for result in results["results"]:
        budget = result["token_budget"]
        lines.extend([
            f"### Token Budget: {budget}",
            "",
            f"**Episodes tested:** {result['num_episodes']}",
            "",
        ])

        # RAW-TRUNCATE details
        raw = result["raw_truncate"]
        lines.extend([
            "**RAW-TRUNCATE:**",
            f"- Correct answers: {raw['correct']} / {result['num_episodes']} ({raw['success_rate']:.1%})",
            f"- Average tokens used: {raw['avg_tokens']:.0f}",
            f"- Average context completeness: {raw['avg_completeness']:.1%}",
            "",
        ])

        # FGT-CONTEXT/FOVEATED details
        fgt_key = 'fgt_context' if 'fgt_context' in result else 'fgt_foveated'
        if results['fgms_available'] and 'success_rate' in result.get(fgt_key, {}):
            fgt = result[fgt_key]
            improvement = (fgt['success_rate'] - raw['success_rate']) * 100
            strategy_name = "FGT-CONTEXT" if fgt_key == 'fgt_context' else "FGT-FOVEATED"

            lines.extend([
                f"**{strategy_name}:**",
                f"- Correct answers: {fgt['correct']} / {result['num_episodes']} ({fgt['success_rate']:.1%})",
                f"- Average tokens used: {fgt['avg_tokens']:.0f}",
                f"- Average context completeness: {fgt['avg_completeness']:.1%}",
                f"- **Improvement: {improvement:+.1f} percentage points**",
                "",
            ])

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Summary table saved to {output_path}")


def plot_success_rates(results: Dict[str, Any], output_dir: Path):
    """Plot success rates by token budget."""
    budgets = []
    raw_rates = []
    fgt_rates = []

    for result in results["results"]:
        budgets.append(result["token_budget"])
        raw_rates.append(result["raw_truncate"]["success_rate"] * 100)

        if results['fgms_available'] and 'success_rate' in result['fgt_context']:
            fgt_rates.append(result["fgt_context"]["success_rate"] * 100)
        else:
            fgt_rates.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(budgets))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_rates, width, label='RAW-TRUNCATE', color='#e74c3c', alpha=0.8)
    if all(r is not None for r in fgt_rates):
        bars2 = ax.bar(x + width/2, fgt_rates, width, label='FGT-CONTEXT', color='#3498db', alpha=0.8)

    ax.set_xlabel('Token Budget', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Context Efficiency: Success Rate by Token Budget', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1] + ([bars2] if all(r is not None for r in fgt_rates) else []):
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / "success_rate_by_budget.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Success rate plot saved to {output_path}")


def plot_improvement(results: Dict[str, Any], output_dir: Path):
    """Plot improvement of FGT over RAW."""
    if not results['fgms_available']:
        print("⊘ Skipping improvement plot (FGMS not available)")
        return

    budgets = []
    improvements = []

    for result in results["results"]:
        if 'success_rate' in result['fgt_context']:
            budgets.append(result["token_budget"])
            improvement = (
                result['fgt_context']['success_rate'] -
                result['raw_truncate']['success_rate']
            ) * 100
            improvements.append(improvement)

    if not improvements:
        print("⊘ Skipping improvement plot (no FGT results)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.bar(range(len(budgets)), improvements, color=colors, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Token Budget', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('FGT-CONTEXT Improvement Over RAW-TRUNCATE', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budgets)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
               height + (1 if height > 0 else -1),
               f'{imp:+.1f}pp',
               ha='center',
               va='bottom' if height > 0 else 'top',
               fontsize=9)

    plt.tight_layout()

    output_path = output_dir / "improvement_by_budget.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Improvement plot saved to {output_path}")


def plot_token_efficiency(results: Dict[str, Any], output_dir: Path):
    """Plot average tokens used by strategy."""
    budgets = []
    raw_tokens = []
    fgt_tokens = []

    for result in results["results"]:
        budgets.append(result["token_budget"])
        raw_tokens.append(result["raw_truncate"]["avg_tokens"])

        if results['fgms_available'] and 'avg_tokens' in result['fgt_context']:
            fgt_tokens.append(result["fgt_context"]["avg_tokens"])
        else:
            fgt_tokens.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(budgets))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_tokens, width, label='RAW-TRUNCATE', color='#e74c3c', alpha=0.8)
    if all(t is not None for t in fgt_tokens):
        bars2 = ax.bar(x + width/2, fgt_tokens, width, label='FGT-CONTEXT', color='#3498db', alpha=0.8)

    # Add budget limit line
    ax.plot(x, budgets, 'k--', label='Budget Limit', linewidth=2)

    ax.set_xlabel('Token Budget', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Tokens Used', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "token_usage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Token usage plot saved to {output_path}")


def plot_episode_breakdown(results: Dict[str, Any], output_dir: Path):
    """Create a breakdown showing per-episode comparison."""
    if not results['fgms_available']:
        print("⊘ Skipping episode breakdown (FGMS not available)")
        return

    # Use the first budget for detailed breakdown
    first_result = results["results"][0]
    budget = first_result["token_budget"]

    # Count outcomes
    both_correct = 0
    only_raw_correct = 0
    only_fgt_correct = 0
    both_wrong = 0

    for episode in first_result["episodes"]:
        raw_success = episode["raw_truncate"]["success"]
        fgt_success = episode.get("fgt_context", {}).get("success", False)

        if raw_success and fgt_success:
            both_correct += 1
        elif raw_success and not fgt_success:
            only_raw_correct += 1
        elif not raw_success and fgt_success:
            only_fgt_correct += 1
        else:
            both_wrong += 1

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    sizes = [both_correct, only_fgt_correct, only_raw_correct, both_wrong]
    labels = [
        f'Both Correct\n({both_correct})',
        f'Only FGT Correct\n({only_fgt_correct})',
        f'Only RAW Correct\n({only_raw_correct})',
        f'Both Wrong\n({both_wrong})'
    ]
    colors = ['#27ae60', '#3498db', '#e67e22', '#95a5a6']
    explode = (0.05, 0.1, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title(f'Episode Outcome Breakdown (Budget: {budget} tokens)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / "episode_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Episode breakdown plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Results Summary")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/phase5/context_bench.json"),
        help="Path to benchmark results JSON"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/phase5"),
        help="Output directory for summary and plots"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 Results Summary and Visualization")
    print("=" * 70)

    # Load results
    print(f"\nLoading results from {args.results}...")
    results = load_results(args.results)

    # Create output directories
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary table
    print("\nGenerating summary table...")
    summary_path = args.out_dir / "summary_table.md"
    create_summary_table(results, summary_path)

    # Generate plots
    print("\nGenerating plots...")
    plot_success_rates(results, plots_dir)
    plot_improvement(results, plots_dir)
    plot_token_efficiency(results, plots_dir)
    plot_episode_breakdown(results, plots_dir)

    print("\n" + "=" * 70)
    print("Summary and visualization complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Summary table: {summary_path}")
    print(f"  - Plots: {plots_dir}/")


if __name__ == "__main__":
    main()
