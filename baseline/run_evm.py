"""100 actual EVM refits, using the original libMR fitter and greedy set cover."""
from pathlib import Path
import argparse
import ctypes
import csv
import hashlib
import json
import time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from asset_paths import DATA, native_tool

BASE=Path(__file__).resolve().parent
THRESHOLD={'CIC1':.0586,'CIC2':.0586,'NSL1':.0016,'NSL2':.0010}

def native():
    lib=ctypes.CDLL(str(native_tool('mr_bridge.dll')))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.mr_fit_rows.argtypes=[array,ctypes.c_int,ctypes.c_int,ctypes.c_int,array]
    lib.mr_fit_rows.restype=ctypes.c_int
    lib.mr_score_rows.argtypes=[array,array,ctypes.c_int,ctypes.c_int,array]
    return lib

def probability(dist,params):
    # MetaRecognition::CDF, for FitLow and default complement_reject.
    valid=params[:,5]>0
    value=np.maximum(dist*params[:,2,None]+params[:,3,None]-params[:,4,None],0.)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        out=-np.expm1(-np.power(value/params[:,0,None],params[:,1,None]))
    out[~valid]=0.
    return np.nan_to_num(out,nan=0.,posinf=1.,neginf=0.)

def verify_probability(lib):
    v=np.ascontiguousarray(np.array([[.2,.4,.5,.6,.8],[.1,.3,.6,.7,1.]]))
    p=np.zeros((2,6),dtype='float64');assert lib.mr_fit_rows(v,2,5,1,p)==0
    d=np.ascontiguousarray(np.array([[0.,.2,.5,.8,2.],[0.,.1,.4,1.,3.]]))
    expected=np.zeros_like(d);lib.mr_score_rows(d,p,2,5,expected)
    assert np.allclose(probability(d,p),expected,atol=1e-12,rtol=1e-12)

def reduce(dist,params,y,order):
    with np.errstate(divide='ignore',invalid='ignore'):
        radii=(params[:,0]*np.power(-np.log1p(-.5),1/params[:,1])-params[:,3]+params[:,4])*params[:,2]
    radii[params[:,5]==0]=-1.
    kept=[]
    for label in np.unique(y):
        ix=order[y[order]==label]
        coverage=dist[np.ix_(ix,ix)]<=radii[ix,None]
        counts=coverage.sum(axis=1).astype('int64')
        covered=np.zeros(len(ix),dtype=bool)
        while counts.max(initial=0)>0:
            best=int(counts.argmax());kept.append(int(ix[best]))
            new=coverage[best]&~covered
            counts-=coverage[:,new].sum(axis=1)
            covered|=new
    return np.asarray(kept,dtype='int64')

def predict(x,points,params,labels,threshold):
    out=[]
    for start in range(0,len(x),512):
        d=cdist(points,x[start:start+512],'euclidean')
        p=probability(d,params)
        fused=[]
        for c in range(9):
            v=p[labels==c];n=min(4,len(v))
            fused.append(np.partition(v,-n,axis=0)[-n:].mean(0) if n else np.zeros(d.shape[1]))
        fused=np.stack(fused);label=fused.argmax(0);confidence=fused.max(0)
        coarse=np.where(label==0,0,1).astype('int8');coarse[confidence<threshold]=2
        out.append(coarse)
    return np.concatenate(out)

def run(setting,epochs,run_name):
    started=time.perf_counter();out=BASE/'results'/run_name/setting/'EVM';out.mkdir(parents=True,exist_ok=False)
    k=np.load(DATA/setting/'known.npz');q=np.load(DATA/setting/'query.npz')
    x=np.asarray(k['x_fit'],dtype='float64');y=k['y_fit'];xq=np.asarray(q['x'],dtype='float64')
    lib=native();verify_probability(lib)
    dist=cdist(x,x,'euclidean')
    tails=np.empty((len(x),50),dtype='float64')
    for c in np.unique(y):
        rows=np.flatnonzero(y==c);negative=np.flatnonzero(y!=c)
        for start in range(0,len(rows),256):
            ix=rows[start:start+256]
            tails[ix]=np.partition(dist[np.ix_(ix,negative)],49,axis=1)[:,:50]*.5
    records=[];pt=[];pq=[]
    for epoch in range(1,epochs+1):
        ep=time.perf_counter();params=np.zeros((len(x),6),dtype='float64')
        failed=lib.mr_fit_rows(tails,len(x),50,1,params)
        order=np.random.default_rng(1234+epoch).permutation(len(x))
        kept=reduce(dist,params,y,order)
        fit_seconds=time.perf_counter()-ep
        if not len(kept):raise RuntimeError('EVM set cover retained no vectors')
        pt.append(predict(x,x[kept],params[kept],y[kept],THRESHOLD[setting]))
        pq.append(predict(xq,x[kept],params[kept],y[kept],THRESHOLD[setting]))
        record=dict(Epochs=epoch,Fit_Seconds=fit_seconds,Epoch_Seconds=time.perf_counter()-ep,Elapsed_Seconds=time.perf_counter()-started,Extreme_Vectors=len(kept),Invalid_Weibull_Fits=failed,Seed=1234+epoch,
                    Model_SHA256=hashlib.sha256(params.tobytes()+kept.tobytes()).hexdigest())
        records.append(record)
        (out/'progress.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if epoch==1 or epoch%5==0:print(setting,'EVM',epoch,'vectors',len(kept),'invalid',failed,'elapsed',round(record['Elapsed_Seconds'],1),flush=True)
    with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(out/'predictions_all_epochs.npz',train=np.stack(pt),query=np.stack(pq),fit_ids=k['fit_ids'],query_ids=q['ids'])
    np.savez_compressed(out/'model_final.npz',points=x[kept],params=params[kept],fine_labels=y[kept],fit_indices=kept)
    manifest=dict(model='EVM',setting=setting,epochs=epochs,row_semantics='independent_full_refit_with_fixed_data_and_shuffled_set_cover_ties',
                  tail_size=50,margin_scale=.5,cover_threshold=.5,nfuse=4,distance='euclidean',reject_threshold=THRESHOLD[setting],
                  native_libMR=True,probability_parity_check=True,static_distance_cache=True,all_Weibull_parameters_refitted_each_round=True,
                  adaptations=['equivalent batched native libMR calls','vectorized original greedy coverage','nfuse=min(4,available vectors) for tiny classes','invalid libMR fits retain zero inclusion probability'],
                  total_seconds=time.perf_counter()-started,fit_seconds=sum(r['Fit_Seconds'] for r in records),training_complete=True,
                  query_truth_read_during_training=False,selected_epoch=epochs,selection='fixed final refit',published=False,final_test_executed=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('COMPLETE',setting,'EVM',round(manifest['total_seconds'],2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True,choices=THRESHOLD);p.add_argument('--epochs',type=int,default=100);p.add_argument('--run-name',default='comparison_20261102')
    a=p.parse_args()
    with threadpool_limits(limits=4):run(a.setting,a.epochs,a.run_name)
