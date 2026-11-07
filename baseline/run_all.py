"""Reproduce the baseline comparison in a new output directory."""
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import sys

BASE=Path(__file__).resolve().parent
SCRIPTS={'CVAE-EVT':'run_cvae_evt.py','EVM':'run_evm.py','W-SVM':'run_wsvm.py'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--settings',nargs='+',choices=['CIC1','CIC2','NSL1','NSL2'],default=['CIC1','CIC2','NSL1','NSL2'])
    p.add_argument('--models',nargs='+',choices=list(SCRIPTS),default=list(SCRIPTS))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--run-name',default='comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--prepare',action='store_true',help='Rebuild the pinned shared inputs before running')
    a=p.parse_args()
    if a.epochs!=100:raise ValueError('The supplied comparison CSV contract has 100 rows; use individual runners for diagnostics.')
    if a.prepare:subprocess.run([sys.executable,'-X','utf8','prepare_data.py','--settings',*a.settings],check=True,cwd=BASE)
    for setting in a.settings:
        for model in a.models:
            subprocess.run([sys.executable,'-X','utf8',SCRIPTS[model],'--setting',setting,'--epochs',str(a.epochs),'--run-name',a.run_name],check=True,cwd=BASE)
    subprocess.run([sys.executable,'-X','utf8','export_results.py','--run-name',a.run_name],check=True,cwd=BASE)

if __name__=='__main__':main()
