APRIL 2016 NYC TRAFFIC 
- april2016_results/README.md: method, scores and presentation notes.
- april2016_results/metrics.csv and results.png: key result table and chart.
- train_april.py, requirements.txt, run_on_windows.bat and data/april2016.csv.zip: reproducible Windows run.

 Outputs and detailed intermediate files are generated under april2016_results. The input traffic archive is scanned in chunks; the model comparison uses 10 links over 28 days.

The DOT archive contains speed readings rather than official congestion labels. The script derives five labels using each road's training-period speed distribution. Read EXPLANATION_FOR_RUCHITHA.txt before comparing these scores to the earlier project.

