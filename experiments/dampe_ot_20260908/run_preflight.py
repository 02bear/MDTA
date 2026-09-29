"""Run numerical tests, actual feature integration checks and five smoke runs."""
from pathlib import Path
import subprocess
import sys
from common import OUTPUT,dump

def main():
    script=Path(__file__).resolve().parent
    root=OUTPUT/'fold_1'
    result=[]
    for name in ['test_ot_alignment.py','verify_pilot.py']:
        p=subprocess.run([sys.executable,'-B',str(script/name)],capture_output=True,text=True)
        (root/(name+'.log')).write_text(p.stdout+p.stderr)
        if p.returncode:
            print(p.stdout+p.stderr);raise RuntimeError(name+' failed')
        result.append(name)
        print('PASSED',name,flush=True)
    logs=root/'smoke_logs';logs.mkdir(exist_ok=True)
    for mode in ['F0','F1','F2','F3','F4']:
        out=root/'smoke'/f'{mode}_seed42'
        if not (out/'summary.json').exists():
            with open(logs/(mode+'.log'),'x') as log:
                subprocess.run([sys.executable,'-B','-u',str(script/'train_frozen_ot.py'),
                    '--fold','1','--mode',mode,'--seed','42','--device','cpu','--smoke','--epochs','2'],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        result.append(mode)
        print('PASSED smoke',mode,flush=True)
    dump(root/'preflight_complete.json',dict(passed=True,checks=result))

if __name__=='__main__':main()
