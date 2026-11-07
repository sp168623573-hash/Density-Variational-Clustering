"""Run the original libsvm-openset W-SVM trainer and predictor 100 times."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import time
import numpy as np
from sklearn.datasets import dump_svmlight_file

from asset_paths import DATA, native_tool

BASE=Path(__file__).resolve().parent
THRESHOLD={'CIC1':.11,'CIC2':.007,'NSL1':.015,'NSL2':.001}

def coarse(path,n,threshold):
    with Path(path).open('r',encoding='utf-8') as f:
        lines=[line.strip() for line in f if line.strip()]
    # Native output begins with a label header in some modes.
    if lines and lines[0].startswith('labels'):lines=lines[1:]
    fine=np.array([float(line.split()[0].rstrip(':')) for line in lines])
    score=np.array([float(line.split(':',1)[1].strip().split()[0]) for line in lines])
    if not np.isfinite(score).all():raise ValueError('non-finite native W-SVM inclusion score')
    if len(fine)!=n:raise ValueError(f'native prediction row mismatch: {len(fine)} != {n}')
    if not set(np.unique(fine)).issubset(set(range(1,10))|{-99999}):raise ValueError('unexpected native W-SVM label')
    # The native W-SVM returns argmax label even below -P. Its author
    # Getresult.py applies the probability rejection as a separate step.
    out=np.where(fine==1,0,1).astype('int8');out[(fine==-99999)|(score<threshold)]=2
    return out

def run(setting,epochs,run_name):
    started=time.perf_counter();out=BASE/'results'/run_name/setting/'W-SVM';out.mkdir(parents=True,exist_ok=False)
    k=np.load(DATA/setting/'known.npz');q=np.load(DATA/setting/'query.npz')
    x=k['x_fit'];y=k['y_fit'];xq=q['x']
    train_file=out/'train.libsvm';query_file=out/'query_features_dummy_labels.libsvm'
    dump_svmlight_file(x,y+1,str(train_file),zero_based=False)
    dump_svmlight_file(xq,np.ones(len(xq)),str(query_file),zero_based=False)
    model=out/'model_final';pt=[];pq=[];records=[]
    # RBF C=1, gamma=1/d; CAP RBF, nu=.5, CAP threshold=.001.
    train_args=[str(native_tool('svm-train.exe')),'-s','8','-t','2','-c','1','-g',str(1/x.shape[1]),'-m','512','-q',str(train_file),str(model)]
    predict_args=[str(native_tool('svm-predict.exe')),'-P',str(THRESHOLD[setting]),'-C','0.001']
    (out/'commands.json').write_text(json.dumps({'train':train_args,'predict':predict_args},indent=2),encoding='utf-8')
    with (out/'native_train.log').open('w',encoding='utf-8') as log:
        log.write('NATIVE TOOL LOG ONLY. Query files contain dummy labels to isolate inference from evaluation truth.\nPrinted native query Accuracy/Precision/Recall are NOT measured results. Use experiment_B_transfer_metrics.csv and Matrix.txt.\n')
        log.flush()
        for epoch in range(1,epochs+1):
            ep=time.perf_counter();log.write(f'\nFIT_ROUND {epoch}\n');log.flush()
            subprocess.run(train_args,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=out)
            fit_seconds=time.perf_counter()-ep
            subprocess.run(predict_args+[str(train_file),str(model),str(out/'train.pred')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=out)
            subprocess.run(predict_args+[str(query_file),str(model),str(out/'query.pred')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=out)
            pt.append(coarse(out/'train.pred',len(x),THRESHOLD[setting]));pq.append(coarse(out/'query.pred',len(xq),THRESHOLD[setting]))
            record=dict(Epochs=epoch,Fit_Seconds=fit_seconds,Epoch_Seconds=time.perf_counter()-ep,Elapsed_Seconds=time.perf_counter()-started,
                        Model_SHA256=hashlib.sha256(model.read_bytes()+Path(str(model)+'_one_wsvm').read_bytes()).hexdigest())
            records.append(record);(out/'progress.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            if epoch==1 or epoch%5==0:print(setting,'W-SVM',epoch,'elapsed',round(record['Elapsed_Seconds'],1),flush=True)
    with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(out/'predictions_all_epochs.npz',train=np.stack(pt),query=np.stack(pq),fit_ids=k['fit_ids'],query_ids=q['ids'])
    manifest=dict(model='W-SVM',setting=setting,epochs=epochs,row_semantics='independent_full_native_fit_same_data_fixed_hyperparameters',
                  source='ljain2/libsvm-openset',svm_type=8,kernel='RBF',C=1,gamma=1/x.shape[1],CAP_nu=.5,CAP_threshold=.001,
                  reject_threshold=THRESHOLD[setting],published_threshold_source='Yang et al. Table IV',
                  decision_postprocessing='native CAP score gating, then reject inclusion_score < fixed threshold, matching author Getresult.py',
                  adaptations=['Windows compiler declarations only','known labels shifted +1 for native positive-class convention'],
                  total_seconds=time.perf_counter()-started,fit_seconds=sum(r['Fit_Seconds'] for r in records),training_complete=True,
                  query_truth_read_during_training=False,selected_epoch=epochs,selection='fixed final refit',published=False,final_test_executed=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('COMPLETE',setting,'W-SVM',round(manifest['total_seconds'],2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True,choices=THRESHOLD);p.add_argument('--epochs',type=int,default=100);p.add_argument('--run-name',default='comparison_20261102')
    a=p.parse_args();run(a.setting,a.epochs,a.run_name)
