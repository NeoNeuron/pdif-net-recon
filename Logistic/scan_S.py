from multiprocessing import Pool
from subprocess import run
import numpy as np

def child_proc(j):
    cml_options_list=["./a.out", 
        "--S=%.1e %.1e %.1e %.1e"%(j,j,j,j), 
        "--T_Max=1e6",
        "--P_c=0.25",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


ss = np.arange(0, 0.1, 4e-3)
p = Pool(len(ss))
results = [p.apply_async(func=child_proc, args=(j,)) for j in ss]
p.close()
p.join()
