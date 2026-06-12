
@echo off
cd /d "%~dp0"

echo === Compile 1/2 ===
xelatex -shell-escape -interaction=nonstopmode report.tex

echo === Compile 2/2 ===
xelatex -shell-escape -interaction=nonstopmode report.tex

echo Done.
pause
