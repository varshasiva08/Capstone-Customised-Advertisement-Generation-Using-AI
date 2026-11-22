@echo off
REM Moves (never deletes) leftover files. Run from the project folder, with the venv active or not.
REM Everything goes to _archive\tidy\ so any file can be moved back.
if not exist _archive\tidy mkdir _archive\tidy
if not exist ..\adfidelity_backups mkdir ..\adfidelity_backups

REM 0. Safety: the fixed rating app must already be installed
findstr /C:"index=None" eval\rating_app.py >nul
if errorlevel 1 (
  echo STOP: eval\rating_app.py is not the fixed version. Copy rating_app_fixed.py over it first.
  exit /b 1
)

REM 1. Packages sent to raters and root copies of files that live in results\
for %%F in (extra1.zip extra2.zip rater2.zip extra1_rating_key.csv extra2_rating_key.csv extra1_ratings_person1.csv extra2_ratings_person1.csv main_ratings_person2.csv rating_app_fixed.py list_gender_disagreements.py new_tables.md analyze_human_eval.py second_rater_kappa.py) do (
  if exist %%F move %%F _archive\tidy\ >nul && echo archived %%F
)

REM 2. Old backups go outside the project folder
for %%D in (results_main_backup results_main_backup2) do (
  if exist %%D move %%D ..\adfidelity_backups\ >nul && echo moved %%D out of the project
)

REM 3. Dry-run and pilot leftovers
for %%D in (dry_extra pilot pilot2) do (
  if exist results\%%D move results\%%D _archive\tidy\ >nul && echo archived results\%%D
)

REM 4. Tools I added go under eval\
for %%F in (check_ratings_vs_target.py make_image_map.py make_gender_check_set.py) do (
  if exist %%F move %%F eval\ >nul && echo moved %%F to eval\
)

echo.
echo Now run:  python eval/analyze_final.py   and   python -m pytest -q
echo If either complains about a missing file, move that file back from _archive\tidy\
