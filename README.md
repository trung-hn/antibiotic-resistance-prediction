# Antibiotic_Resistance_Prediction

Notes: Jobs

https://proteusmaster.urcf.drexel.edu/urcfwiki/index.php/Writing_Job_Scripts

Commands: qstat, qsub, qdel, qacct, dos2unix

qsub: used to submit write a `.sh` file to store all commands. 

stdout will be returned as report file

/lustre/sratch/tnh48. Data are removed regularly. move to personal machine. 

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

- Unique Values of df["MIC"]:
  - SALCIP {'-4.058893689053568', '-6.058893689053568', '-5.058893689053568', '-2.0', '-3.0', '-1.0', '0.0', '-6.643856189774724', '1.0', '-3.0588936890535687'}
  - SALFIS {'5.0', '7.0', '8.0', '4.0', '6.0', '9.0'}
  - SALNAL {'5.0', '3.0', '7.0', '4.0', '0.0', '6.0', '2.0', '1.0'}
  - SALAUG {'5.0', '3.0', '4.0', '0.0', '6.0', '2.0', '1.0'}
  - SALCOT {'3.0', '-2.0', '-3.0', '-1.0', '0.0', '2.0', '1.0', '-3.0588936890535687'}
  - SALSTR {'5.0', '3.0', '7.0', '4.0', '6.0', '2.0', '1.0'}
  - SALAXO {'5.0', '3.0', '7.0', '-2.0', '4.0', '-1.0', '0.0', '6.0', '2.0', '1.0'}
  - SALGEN {'5.0', '3.0', '-2.0', '4.0', '-1.0', '0.0', '2.0', '1.0'}
  - SALTET {'5.0', '3.0', '4.0', '6.0', '2.0'}
  - SALCHL {'5.0', '3.0', '4.0', '6.0', '2.0', '1.0'}
  - SALAMP {'5.0', '3.0', '4.0', '0.0', '6.0', '2.0', '1.0'}
  - SALFOX {'5.0', '3.0', '4.0', '0.0', '6.0', '2.0', '1.0'}
  - SALAZI {'5.0', '3.0', '4.0', '0.0', '6.0', '2.0', '1.0'}
  - SALTIO {'3.0', '-2.0', '4.0', '-1.0', '0.0', '2.0', '1.0'}
  - SALKAN {'5.0', '3.0', '7.0', '4.0', '6.0'}


From Taha:
- Run each model on each antibiotic meaning we have a different model for each antibiotic
- If it takes too long, we can run everything on 1 antibiotic (SALAMP)

run jupyuter notebook: https://stackoverflow.com/questions/35545402/how-to-run-an-ipynb-jupyter-notebook-from-terminal