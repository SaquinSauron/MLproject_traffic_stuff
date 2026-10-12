APRIL 2016 NYC TRAFFIC - SHARE PACKAGE

This folder has only the items needed to explain or reproduce the comparison:
- EXPLANATION_FOR_RUCHITHA.txt: why the result differs from the earlier accuracy.
- april2016_results/README.md: method, scores and presentation notes.
- april2016_results/metrics.csv and results.png: key result table and chart.
- train_april.py, requirements.txt, run_on_windows.bat and data/april2016.csv.zip: reproducible Windows run.

To reproduce: extract the whole ZIP, install Python 3.11 or newer if needed, then double-click run_on_windows.bat. First setup needs internet to install the listed packages. Outputs and detailed intermediate files are generated under april2016_results. The input traffic archive is scanned in chunks; the model comparison uses 10 links over 28 days.

The DOT archive contains speed readings rather than official congestion labels. The script derives five labels using each road's training-period speed distribution. Read EXPLANATION_FOR_RUCHITHA.txt before comparing these scores to the earlier project.

