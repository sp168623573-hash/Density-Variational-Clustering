"""Build original W-SVM and libMR with a portable compiler."""
from pathlib import Path
import argparse
import subprocess
import json

ROOT = Path(__file__).resolve().parent
from asset_paths import ARCHIVE

ZIG = ARCHIVE / '.tools/ziglang/zig.exe'
MR = ROOT / 'upstream/libMR/libMR'
SVM = ROOT / 'upstream/W-SVM'
OUT = ROOT / 'bin'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compiler',type=Path,default=ZIG)
    compiler=parser.parse_args().compiler
    OUT.mkdir(exist_ok=True)
    common = [str(compiler), 'c++', '-O3', '-DNDEBUG', '-D_declspec=__declspec', '-fdeclspec', '-I'+str(ROOT/'windows_compat'), '-I'+str(MR), '-I'+str(SVM), '-Wno-writable-strings']
    commands = [
        common + ['-shared', str(ROOT/'native_bridge.cpp'),str(MR/'MetaRecognition.cpp'),str(MR/'weibull.c'),'-o',str(OUT/'mr_bridge.dll')],
        common + [str(SVM/'svm-train.cpp'),str(SVM/'svm.cpp'),str(MR/'MetaRecognition.cpp'),str(MR/'weibull.c'),'-o',str(OUT/'svm-train.exe')],
        common + [str(SVM/'svm-predict.cpp'),str(SVM/'svm.cpp'),str(MR/'MetaRecognition.cpp'),str(MR/'weibull.c'),'-o',str(OUT/'svm-predict.exe')],
    ]
    for command in commands:
        if Path(command[-1]).exists():
            continue
        print('Building',command[-1],flush=True)
        subprocess.run(command,check=True,cwd=ROOT)
    (OUT/'build_commands.json').write_text(json.dumps({'purpose':'Configured rebuild commands; existing output binaries are skipped.','commands':[['<zig.exe>']+c[1:] for c in commands]},indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
