from pathlib import Path
import os,sys,json,hashlib,importlib.util
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
H=Path(__file__).resolve().parent;I=H.parent;ROOT=I.parent
STATUS=dict(status='FAIL_TARGET99',published=False,final_test_executed=False)
def mod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def desc(p):
    p=Path(p).resolve(strict=True);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def put(p,v):
    def cv(a):
        if isinstance(a,np.ndarray):return a.tolist()
        if isinstance(a,np.generic):return a.item()
        if isinstance(a,Path):return str(a)
        raise TypeError(type(a).__name__)
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=cv)
    return desc(p)
def save(p,**v):
    with Path(p).open('xb') as f:np.savez_compressed(f,**v)
    return desc(p)
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def keys(x):return np.array([hashlib.sha256(np.ascontiguousarray(r,dtype=np.float64).tobytes()).hexdigest() for r in x])
B=mod(I/'target99_review_20261028/review_baselines.py','r34_baselines')
BEST=read(I/'continuation_closeout_20261104_r208/NEXT_COMPLETE_BASELINE_MANIFEST.json')['active_complete_baselines']
ACTIVE=dict(BEST)

PARENT_ROLES=read(I/'continuation_closeout_20261104_r208/CONTINUATION_BINDINGS.json')['complete_parent_roles']
