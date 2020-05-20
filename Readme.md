# Antibiotic_Resistance_Prediction

Notes: Jobs

https://proteusmaster.urcf.drexel.edu/urcfwiki/index.php/Writing_Job_Scripts

Commands: qstat, qsub, qdel, qacct, dos2unix

qsub: used to submit write a `.sh` file to store all commands. 

stdout will be returned as report file

`/lustre/sratch/tnh48`. Data are removed regularly. move to personal machine. 

Class folder: `/mnt/HA/groups/rosenclassGrp/`


- Number of lines in each file:
    - SALCIP 1759 1759 1759
    - SALFIS 1643 1643 1643
    - SALNAL 1760 1760 1758
    - SALAUG 1760 1760 1758
    - SALCOT 1759 1759 1759
    - SALSTR 931 931 929
    - SALAXO 1760 1760 1758
    - SALGEN 1760 1760 1758
    - SALTET 1759 1759 1759
    - SALCHL 1759 1759 1759
    - SALAMP 1759 1759 1759
    - SALFOX 1760 1760 1758
    - SALAZI 2565 2565 2563
    - SALTIO 1760 1760 1758
    - SALKAN 308 308 308

- Unique Values of `df["MIC"]`:
    - SALCIP Counter({'-6.643856189774724': 2389, '-6.058893689053568': 1964, '-5.058893689053568': 826, '-4.058893689053568': 38, '-1.0': 22, '-2.0': 22, '0.0': 6, '-3.0': 5, '-3.0588936890535687': 4, '1.0': 1})
    - SALFIS Counter({'9.0': 1573, '6.0': 1391, '5.0': 1352, '4.0': 520, '7.0': 78, '8.0': 15})
    - SALNAL Counter({'2.0': 3708, '1.0': 1313, '3.0': 171, '6.0': 35, '4.0': 25, '0.0': 16, '5.0': 9, '7.0': 1})
    - SALAUG Counter({'0.0': 3437, '6.0': 642, '3.0': 433, '4.0': 355, '1.0': 246, '5.0': 136, '2.0': 29})
    - SALCOT Counter({'-3.0588936890535687': 4224, '-3.0': 617, '-2.0': 339, '3.0': 55, '-1.0': 36, '2.0': 3, '0.0': 2, '1.0': 1})
    - SALSTR Counter({'7.0': 946, '6.0': 865, '3.0': 478, '4.0': 182, '2.0': 168, '5.0': 108, '1.0': 44})
    - SALAXO Counter({'-2.0': 4495, '4.0': 368, '5.0': 166, '3.0': 166, '6.0': 39, '2.0': 25, '-1.0': 12, '7.0': 5, '0.0': 1, '1.0': 1})
    - SALGEN Counter({'-1.0': 2563, '-2.0': 1599, '5.0': 456, '0.0': 372, '4.0': 177, '3.0': 68, '1.0': 32, '2.0': 11})
    - SALTET Counter({'6.0': 2708, '2.0': 2364, '5.0': 155, '3.0': 28, '4.0': 22})
    - SALCHL Counter({'3.0': 3164, '2.0': 1814, '6.0': 156, '4.0': 87, '1.0': 48, '5.0': 8})
    - SALAMP Counter({'0.0': 3122, '6.0': 1590, '1.0': 527, '2.0': 30, '3.0': 3, '5.0': 3, '4.0': 2})
    - SALFOX Counter({'1.0': 2631, '2.0': 1466, '6.0': 371, '5.0': 308, '3.0': 203, '0.0': 201, '4.0': 98})
    - SALAZI Counter({'2.0': 3458, '3.0': 3385, '1.0': 569, '6.0': 156, '4.0': 96, '5.0': 15, '0.0': 14})
    - SALTIO Counter({'0.0': 3065, '-1.0': 1285, '4.0': 612, '1.0': 157, '3.0': 141, '-2.0': 10, '2.0': 8})
    - SALKAN Counter({'3.0': 825, '7.0': 82, '4.0': 12, '5.0': 3, '6.0': 2})


Need to add:
- Precision and recall
- k-fold
- MSE

- Below is a classification for who runs tests for which antibiotics
    - Akash    - {SALCIP, SALFIS, SALNAL, SALTIO}
    - Reno     - {SALAUG, SALCOT, SALAXO}
    - Trung    - {SALGEN, SALTET, SALCHL}
    - Jonathan - {SALAMP, SALFOX, SALKAN, SALSTR, SALAZI}

From Taha:
- Run each model on each antibiotic meaning we have a different model for each antibiotic
- If it takes too long, we can run everything on 1 antibiotic (SALAMP)

run jupyuter notebook: https://stackoverflow.com/questions/35545402/how-to-run-an-ipynb-jupyter-notebook-from-terminal
