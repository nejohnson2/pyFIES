"""Build the demo Jupyter notebook from a single source-of-truth script.

Editing the notebook directly produces noisy diffs (uuids, execution counts,
output blobs); editing this script keeps changes reviewable. Run via:

    python scripts/build_demo_notebook.py

The output is `notebooks/01_parity_demo.ipynb`. Execute it once after
generation with:

    jupyter nbconvert --execute --inplace notebooks/01_parity_demo.ipynb

so committed outputs (plots, tables) reflect the latest pyFIES + fixture
state.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        # ---------------------------------------------------------------------
        md(
            "# pyFIES end-to-end walkthrough with FAO sample data\n"
            "\n"
            "This notebook walks the full pyFIES pipeline on the four FAO sample\n"
            "country datasets that ship with [`RM.weights`](https://cran.r-project.org/package=RM.weights).\n"
            "Each step is a real call you would make in a research workflow:\n"
            "\n"
            "1. Load survey responses + sampling weights\n"
            "2. Fit the dichotomous Rasch model via weighted CML\n"
            "3. Inspect item severities, person parameters, and the raw-score distribution\n"
            "4. Equate to FAO's 2014–2016 global standard\n"
            "5. Compute the SDG 2.1.2 prevalence rates\n"
            "6. **Verify numerical parity with R's `RM.weights`** — the credibility check\n"
            "\n"
            "All R outputs were generated once by `scripts/generate_r_fixtures.R`\n"
            "and live as JSON under `tests/fixtures/r_reference/`. R is not\n"
            "required to run this notebook.\n"
        ),
        # ---------------------------------------------------------------------
        md("## Setup"),
        code(
            "from __future__ import annotations\n"
            "\n"
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "from pyfies import DEFAULT_FIES_ITEMS, FAO_2014_2016, RaschModel\n"
            "\n"
            "FIXTURE_DIR = Path('..') / 'tests' / 'fixtures' / 'r_reference'\n"
            "\n"
            "\n"
            "def load_fixture(country_id: int) -> dict:\n"
            "    \"\"\"Load the JSON fixture for FAO sample country `country_id`.\"\"\"\n"
            "    with (FIXTURE_DIR / f'country{country_id}.json').open() as f:\n"
            "        return json.load(f)\n"
            "\n"
            "\n"
            "def to_response_matrix(fixture_rows: list) -> np.ndarray:\n"
            "    \"\"\"R's jsonlite encodes NA as the literal string 'NA'. Convert to NaN.\"\"\"\n"
            "    n = len(fixture_rows)\n"
            "    k = len(fixture_rows[0])\n"
            "    out = np.empty((n, k), dtype=np.float64)\n"
            "    for i, row in enumerate(fixture_rows):\n"
            "        for j, val in enumerate(row):\n"
            "            out[i, j] = np.nan if (val is None or val == 'NA') else float(val)\n"
            "    return out\n"
            "\n"
            "\n"
            "plt.rcParams['figure.dpi'] = 110\n"
            "plt.rcParams['axes.spines.top'] = False\n"
            "plt.rcParams['axes.spines.right'] = False\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 1. The data\n"
            "\n"
            "FIES has eight questions, asked in a fixed order. Each is dichotomized\n"
            "before analysis (`Never` $\\rightarrow$ 0; `Rarely`/`Sometimes`/`Often`\n"
            "$\\rightarrow$ 1; missing $\\rightarrow$ `NaN`).\n"
            "\n"
            "We'll start with sample country 1 (n=1000)."
        ),
        code(
            "from pyfies.items import ITEM_DESCRIPTIONS\n"
            "\n"
            "fixture = load_fixture(1)\n"
            "X = to_response_matrix(fixture['data'])\n"
            "w = np.array(fixture['weights'], dtype=np.float64)\n"
            "\n"
            "print(f'Country 1: n={X.shape[0]}, items={X.shape[1]}, '\n"
            "      f'missing cells={int(np.isnan(X).sum())}')\n"
            "print(f'Sampling weights: min={w.min():.3f}  max={w.max():.3f}  '\n"
            "      f'sum={w.sum():.1f}')\n"
            "print()\n"
            "print('FIES items:')\n"
            "for name in DEFAULT_FIES_ITEMS:\n"
            "    print(f'  {name:8s}  {ITEM_DESCRIPTIONS[name]}')\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "Item-level affirmation rates (weighted) tell us how often each\n"
            "behavior was reported. Items further down the list correspond to\n"
            "more severe deprivations and are endorsed by fewer respondents:"
        ),
        code(
            "complete = ~np.isnan(X).any(axis=1)\n"
            "Xc = X[complete]\n"
            "wc = w[complete]\n"
            "wc_sum = wc.sum()\n"
            "perc_yes = (Xc.T @ wc) / wc_sum\n"
            "\n"
            "summary = pd.DataFrame(\n"
            "    {'description': [ITEM_DESCRIPTIONS[n] for n in DEFAULT_FIES_ITEMS],\n"
            "     'weighted_affirm_rate': perc_yes},\n"
            "    index=list(DEFAULT_FIES_ITEMS),\n"
            ")\n"
            "summary.style.format({'weighted_affirm_rate': '{:.1%}'})\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 2. Fit the dichotomous Rasch model\n"
            "\n"
            "`RaschModel().fit()` runs weighted Conditional Maximum Likelihood\n"
            "estimation. Item severities $\\beta_i$ are identified up to a\n"
            "constant and resolved with a sum-to-zero constraint. Rows with any\n"
            "missing item are dropped from the fit; cases at extreme raw scores\n"
            "($r=0$ or $r=k$) carry no CML information and are also excluded\n"
            "from the likelihood.\n"
        ),
        code(
            "model = RaschModel().fit(X, sample_weight=w)\n"
            "\n"
            "fit_summary = pd.DataFrame({\n"
            "    'item': DEFAULT_FIES_ITEMS,\n"
            "    'beta': model.beta,\n"
            "    'se_beta': model.se_beta,\n"
            "    'weighted_affirm_rate': perc_yes,\n"
            "})\n"
            "fit_summary['rank_severity'] = fit_summary['beta'].rank().astype(int)\n"
            "fit_summary.style.format({\n"
            "    'beta': '{:+.4f}', 'se_beta': '{:.4f}', 'weighted_affirm_rate': '{:.1%}'\n"
            "})\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "Notice that severity rank ordering aligns with affirmation rates\n"
            "(rare-to-endorse items are more severe), which is a basic Rasch\n"
            "consistency check. Now visualize the severity ladder with 95% CIs:"
        ),
        code(
            "order = np.argsort(model.beta)\n"
            "names_sorted = [DEFAULT_FIES_ITEMS[i] for i in order]\n"
            "beta_sorted = model.beta[order]\n"
            "se_sorted = model.se_beta[order]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
            "y = np.arange(len(beta_sorted))\n"
            "ax.errorbar(beta_sorted, y, xerr=1.96 * se_sorted,\n"
            "            fmt='o', color='#2e7d32', capsize=3)\n"
            "ax.axvline(0, color='gray', lw=0.6, alpha=0.5)\n"
            "ax.set_yticks(y, names_sorted)\n"
            "ax.set_xlabel(r'Item severity $\\beta$ (country metric, sum-to-zero)')\n"
            "ax.set_title('Country 1 — FIES item severity ladder (95% CI)')\n"
            "ax.invert_yaxis()\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 3. Person parameters and raw-score distribution\n"
            "\n"
            "Each raw score $r$ has a single person parameter $\\theta_r$ and a\n"
            "measurement error. The two extremes ($r=0$ and $r=k$) are anchored\n"
            "at pseudo-raw-scores 0.5 and $k-0.5$ since CML can't identify them."
        ),
        code(
            "rs = np.arange(len(model.theta))\n"
            "weighted_rs = model._fit.weighted_raw_score_counts\n"
            "rs_share = weighted_rs / weighted_rs.sum()\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))\n"
            "\n"
            "ax1.errorbar(rs, model.theta, yerr=model.se_theta,\n"
            "             fmt='o-', color='#1565c0', capsize=3)\n"
            "ax1.set_xlabel('Raw score $r$')\n"
            "ax1.set_ylabel(r'$\\theta_r$ (latent severity)')\n"
            "ax1.set_title('Person parameters per raw score')\n"
            "\n"
            "ax2.bar(rs, rs_share, color='#6a1b9a', alpha=0.85)\n"
            "ax2.set_xlabel('Raw score $r$')\n"
            "ax2.set_ylabel('Weighted share of respondents')\n"
            "ax2.set_title('Raw score distribution')\n"
            "ax2.yaxis.set_major_formatter(\n"
            "    plt.matplotlib.ticker.PercentFormatter(xmax=1.0))\n"
            "\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 4. Equate to the FAO 2014–2016 global standard\n"
            "\n"
            "Equating finds the linear transform that maps the country metric\n"
            "onto the global metric, after iteratively flagging items whose\n"
            "post-transformation discrepancy from the standard exceeds `tol`\n"
            "(default 0.35). Those items are deemed *unique* — interpreted\n"
            "differently in the country context — and dropped from the\n"
            "calibration."
        ),
        code(
            "model.equate(FAO_2014_2016)\n"
            "eq = model.equating\n"
            "\n"
            "uniqueness = ['common' if c else 'UNIQUE' for c in eq.common]\n"
            "equating_table = pd.DataFrame({\n"
            "    'item': DEFAULT_FIES_ITEMS,\n"
            "    'beta_country': model.beta,\n"
            "    'beta_equated': eq.equated_beta,\n"
            "    'beta_global_standard': FAO_2014_2016.severities,\n"
            "    'role': uniqueness,\n"
            "})\n"
            "print(f'scale = {eq.scale:.4f}, shift = {eq.shift:.4f}')\n"
            "print(f'common-items correlation = {eq.correlation:.4f}')\n"
            "equating_table.style.format({\n"
            "    'beta_country': '{:+.4f}',\n"
            "    'beta_equated': '{:+.4f}',\n"
            "    'beta_global_standard': '{:+.4f}'\n"
            "})\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 5. Compute prevalence — the SDG 2.1.2 indicator\n"
            "\n"
            "`prevalence()` defaults to the equated thresholds: items 5 and 8 of\n"
            "the FAO global standard, back-transformed onto the country metric so\n"
            "they can be applied to the country's own person parameters."
        ),
        code(
            "result = model.prevalence()\n"
            "\n"
            "print(f'Moderate or severe food insecurity: {result.moderate_or_severe:.1%}')\n"
            "print(f'Severe food insecurity:             {result.severe:.1%}')\n"
            "print()\n"
            "print(f'Thresholds on country metric: '\n"
            "      f'mod+ = {result.thresholds_country_metric[0]:+.4f}, '\n"
            "      f'severe = {result.thresholds_country_metric[1]:+.4f}')\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## 6. Numerical parity with R's `RM.weights`\n"
            "\n"
            "The same fitting + equating + prevalence pipeline run in R,\n"
            "snapshotted as a JSON fixture. We compare side by side."
        ),
        code(
            "# RM.weights does not strictly enforce sum-to-zero on β; recenter\n"
            "# its output before comparing to pyFIES (which does enforce it).\n"
            "beta_R = np.array(fixture['beta'])\n"
            "beta_R_centered = beta_R - beta_R.mean()\n"
            "\n"
            "compare = pd.DataFrame({\n"
            "    'item': DEFAULT_FIES_ITEMS,\n"
            "    'beta_pyFIES': model.beta,\n"
            "    'beta_R (centered)': beta_R_centered,\n"
            "    'abs_diff': np.abs(model.beta - beta_R_centered),\n"
            "})\n"
            "print(f'max |Δβ| = {compare[\"abs_diff\"].max():.2e}')\n"
            "compare.style.format({\n"
            "    'beta_pyFIES': '{:+.6f}',\n"
            "    'beta_R (centered)': '{:+.6f}',\n"
            "    'abs_diff': '{:.2e}',\n"
            "})\n"
        ),
        # ---------------------------------------------------------------------
        code(
            "headline = pd.DataFrame({\n"
            "    'pyFIES': [eq.scale, eq.shift, result.moderate_or_severe, result.severe],\n"
            "    'R (RM.weights)': [\n"
            "        fixture['equate_scale'],\n"
            "        fixture['equate_shift'],\n"
            "        fixture['equate_prevs'][0],\n"
            "        fixture['equate_prevs'][1],\n"
            "    ],\n"
            "}, index=['equating scale', 'equating shift',\n"
            "         'prevalence: moderate+', 'prevalence: severe'])\n"
            "headline['abs_diff'] = (headline['pyFIES'] - headline['R (RM.weights)']).abs()\n"
            "headline.style.format('{:.6f}')\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "Item severities agree with R to ~1e-5 on this country, prevalence\n"
            "rates within 0.3 percentage points. See [Parity](../parity.md) for\n"
            "the across-country tolerances and a discussion of where the\n"
            "residual sub-1e-4 noise comes from."
        ),
        # ---------------------------------------------------------------------
        md(
            "## 7. The harder case — country 4\n"
            "\n"
            "Country 4 has a very wide item severity spread (range ≈ 6 units)\n"
            "and the lowest moderate-or-severe prevalence of the four samples.\n"
            "It exercises the numerical core's behavior in the regime where\n"
            "elementary symmetric function summations come closest to\n"
            "floating-point precision limits."
        ),
        code(
            "fixture4 = load_fixture(4)\n"
            "X4 = to_response_matrix(fixture4['data'])\n"
            "w4 = np.array(fixture4['weights'], dtype=np.float64)\n"
            "\n"
            "model4 = RaschModel().fit(X4, sample_weight=w4).equate(FAO_2014_2016)\n"
            "result4 = model4.prevalence()\n"
            "\n"
            "country4_summary = pd.DataFrame({\n"
            "    'pyFIES': [\n"
            "        model4.equating.scale, model4.equating.shift,\n"
            "        result4.moderate_or_severe, result4.severe,\n"
            "    ],\n"
            "    'R (RM.weights)': [\n"
            "        fixture4['equate_scale'], fixture4['equate_shift'],\n"
            "        fixture4['equate_prevs'][0], fixture4['equate_prevs'][1],\n"
            "    ],\n"
            "}, index=['equating scale', 'equating shift',\n"
            "         'prevalence: moderate+', 'prevalence: severe'])\n"
            "country4_summary['abs_diff'] = (\n"
            "    country4_summary['pyFIES'] - country4_summary['R (RM.weights)']\n"
            ").abs()\n"
            "country4_summary.style.format('{:.6f}')\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## All four countries at a glance"
        ),
        code(
            "rows = []\n"
            "for cid in (1, 2, 3, 4):\n"
            "    fix = load_fixture(cid)\n"
            "    Xi = to_response_matrix(fix['data'])\n"
            "    wi = np.array(fix['weights'], dtype=np.float64)\n"
            "    m = RaschModel().fit(Xi, sample_weight=wi).equate(FAO_2014_2016)\n"
            "    r = m.prevalence()\n"
            "    uniques = [DEFAULT_FIES_ITEMS[i] for i, c in enumerate(m.equating.common) if not c]\n"
            "    rows.append({\n"
            "        'country': cid,\n"
            "        'n': fix['n_total'],\n"
            "        'unique_items': ', '.join(uniques) or '(none)',\n"
            "        'scale': m.equating.scale,\n"
            "        'shift': m.equating.shift,\n"
            "        'mod+_FI (pyFIES)': r.moderate_or_severe,\n"
            "        'mod+_FI (R)': fix['equate_prevs'][0],\n"
            "        'sev_FI (pyFIES)': r.severe,\n"
            "        'sev_FI (R)': fix['equate_prevs'][1],\n"
            "    })\n"
            "\n"
            "summary_all = pd.DataFrame(rows).set_index('country')\n"
            "summary_all.style.format({\n"
            "    'scale': '{:.4f}', 'shift': '{:+.4f}',\n"
            "    'mod+_FI (pyFIES)': '{:.1%}', 'mod+_FI (R)': '{:.1%}',\n"
            "    'sev_FI (pyFIES)': '{:.1%}', 'sev_FI (R)': '{:.1%}',\n"
            "})\n"
        ),
        # ---------------------------------------------------------------------
        md(
            "## Where to go next\n"
            "\n"
            "* [Quickstart](../quickstart.md) — minimal copy-paste recipe\n"
            "* [Methodology](../methodology.md) — the math behind each step\n"
            "* [Parity with RM.weights](../parity.md) — what numerical agreement to expect\n"
            "* [API reference](../api/index.md) — full signatures and options\n"
            "\n"
            "If you use pyFIES in research, please cite it via the\n"
            "[Zenodo DOI](https://doi.org/10.5281/zenodo.19838795) along with the\n"
            "underlying FAO methodology (Cafiero, Viviani, & Nord, 2018)."
        ),
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    }

    out = Path("notebooks") / "01_parity_demo.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        nbf.write(nb, f)
    print(f"Wrote {out} ({sum(1 for c in nb.cells if c.cell_type == 'code')} code cells, "
          f"{sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown cells)")
    # Sanity: confirm round-trips
    json.loads(out.read_text())


if __name__ == "__main__":
    main()
