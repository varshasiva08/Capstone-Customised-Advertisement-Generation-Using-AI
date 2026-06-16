# AdFidelity — Paper Tables

These `.tex` files contain the LaTeX source for all tables in the AdFidelity paper.
Drop them into your paper's source directory and `\input{tableN_name}` them.

---

## Files

| File | Table | Section in paper |
|------|-------|-----------------|
| `table1_comparison.tex` | Comparison of existing approaches vs AdFidelity | Related Work / Methodology |
| `table2_dfc_scores.tex` | DFC scores per demographic axis across iterations | Results |
| `table3_cdvr_pipeline.tex` | CDVR pipeline stages, inputs, outputs, tools | System Architecture |
| `table4_ablation.tex` | Ablation: prompt correction strategy comparison | Ablation Study |
| `table5_eval_setup.tex` | Evaluation dataset setup and counts | Experiments |

---

## Required LaTeX packages

Add these to your preamble:

```latex
\usepackage{booktabs}   % \toprule, \midrule, \bottomrule
\usepackage{graphicx}   % \resizebox
\usepackage{array}      % p{} column type in table3 and table1
```

---

## Notes

- **Placeholder values**: Table 2 and Table 4 contain example numeric values.
  Replace them with your actual experimental results before submission.
- **Citations**: Table 1 references `\cite{friedrich2023}` — add the FairDiffusion
  BibTeX entry to your `.bib` file.
- **`\columnwidth`**: Tables 1 and 3 use `\resizebox{\columnwidth}{!}{...}` so they
  scale to fit single-column or double-column layouts automatically.
