from typing import List
import re
from pathlib import Path
from subprocess import Popen, PIPE
from tempfile import TemporaryDirectory
from typing import Dict, Any
from pysat.formula import CNF, IDPool

GANAKDIR = Path('~/Programming/Satisfiability').expanduser()

def extract_ganak(val: str) -> int:
    stats = {}
    for line in val.split('\n'):
        # if mymatch := re.match(r's\s+mc\s+([0-9]+)', line):
        #     result = int(mymatch[1])
        if mymatch := re.match(r'c\s+s exact arb int\s+([0-9]+)', line):
            stats['result'] = int(mymatch[1])
        elif mymatch := re.match(r's\s+mc\s+([0-9]+)', line):
            stats['result'] = int(mymatch[1])
        elif mymatch := re.match(r's\s+SATISFIABLE', line):
            stats['status'] = True
        elif mymatch := re.match(r's\s+UNSAT', line):
            stats['status'] = False
        elif mymatch := re.match(r'c\s+o\s+Total time \[Arjun\+GANAK\]:\s*([0-9.]+)', line):
            stats['time'] = float(mymatch[1])
        elif mymatch := re.match(r'c\s+o\s+\[appmc\+arjun\]\s+Total time:\s*([0-9.]+)', line):
            stats['time'] = float(mymatch[1])
    return stats

def ganak_count_models(cnf: CNF, pool: IDPool,
                       projected_prefixes: List[str] = [],
                       verbose: int = 0,
                       approx: bool = False,
                       options: Dict[str, Any] = {}, debug: bool = False) -> int:
    with TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / 'temp.cnf'
        with open(fpath, 'w') as fil:
            fil.write('c t pmc\n')
            if projected_prefixes != []:
                pvars = [val for key, val in pool.obj2id.items()
                         if isinstance(key, tuple) and key[0] in projected_prefixes]
            else:
                pvars = [val for key, val in pool.obj2id.items() if key is not None]
            fil.write(cnf.to_dimacs())
            if pvars:
                fil.write('\n')
                fil.write(f"c p show {' '.join(map(str,pvars))} 0\n")
                          
        # Feed it to ganak
        prog = 'approxmc/approxmc' if approx else 'ganak/ganak'
        options = [f'--{key}={val}' for key, val in options.items()]
        process = Popen([f"{GANAKDIR}/{prog}"] + options + [f"{fpath}"], stdout = PIPE)
        (output, err) = process.communicate()
        _ = process.wait()
        if debug:
            print("Output:")
            print(output.decode('utf-8'))
            if err is not None:
                print(f"Error:")
                print(err.decode('utf-8'))
        stats = extract_ganak(output.decode())
        if 'time' in stats and verbose > 0:
            print(f'Time: {stats["time"]}')
        return stats.get('result', None)
