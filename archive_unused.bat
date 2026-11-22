@echo off
REM Moves (does NOT delete) files that neither the paper's experiments nor the app need.
REM Run from the project folder (adfidelity). Undo any time by moving things back from _archive.
if not exist _archive mkdir _archive
move rag _archive\rag
move rag_store _archive\rag_store
move quick_test.py _archive\
move requirements_before_torch.txt _archive\
move diagnosis.txt _archive\
move tests\visual_checker.py _archive\
move eval\diagnose_run.py _archive\
move eval\score_folder.py _archive\
move eval\embedding_viz.py _archive\
echo.
echo Done. Now run:  python -m pytest tests
echo and:           python -m eval.run_experiment --out results/dry_check --dry-run --max-profiles 4
echo If both work, you can delete the _archive folder before submitting.
