"""Author CVAE1/CVAE2 networks with known-only GPD tails and per-epoch records.

The two original sequential stages each run for the requested number of epochs.
The main 100-row curve describes stage 2 with the frozen stage-1 classifier.
Evaluation labels are opened only by export_results.py after prediction saving.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import time
import warnings
import numpy as np
import torch
from torch import nn
import torch.distributions as D
from scipy.stats import genpareto
from sklearn.preprocessing import LabelBinarizer
from threadpoolctl import threadpool_limits

from asset_paths import DATA

BASE=Path(__file__).resolve().parent
SOURCES={'CIC1':'CICIDSSetting1','CIC2':'CICIDSSetting2','NSL1':'NSLKDDSetting1','NSL2':'NSLKDDSetting2'}
# Table IV of Yang et al.; fixed before any query evaluation.
EPSILON={'CIC1':.88,'CIC2':.86,'NSL1':.87,'NSL2':.87}

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def load_author_networks(setting,dim):
    name=SOURCES[setting]
    path=BASE/'upstream/CVAE-EVT'/name/(name+'.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('CVAE1','CVAE2')]
    assert len(selected)==2
    scope={'torch':torch,'nn':nn,'D':D,'LabelBinarizer':LabelBinarizer}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    c=scope['CVAE1']();r=scope['CVAE2']()
    if c.l_z_x[0].in_features!=dim:
        c.l_z_xy[0]=nn.Linear(dim+9,c.l_z_xy[0].out_features)
        c.l_z_x[0]=nn.Linear(dim,c.l_z_x[0].out_features)
        latent_dim=c.l_z_x[-1].out_features//2
        c.l_y_xz[0]=nn.Linear(dim+latent_dim,c.l_y_xz[0].out_features)
        r.l_z_xy[0]=nn.Linear(dim+9,r.l_z_xy[0].out_features)
        r.l_x_yz[-1]=nn.Linear(r.l_x_yz[-1].in_features,dim)
    for p in r.l_z_xy.parameters():p.requires_grad=False
    return c,r,path

@torch.inference_mode()
def classification(model,x):
    return torch.cat([model(a).argmax(1) for a in x.split(1024)])

@torch.inference_mode()
def errors(model,x,labels):
    result=[]
    for a,y in zip(x.split(1024),labels.split(1024)):
        one=nn.functional.one_hot(y,9).float()
        result.append((model(a,one)-a).square().sum(1).cpu().numpy())
    return np.concatenate(result).astype('float64')

def fit_tail(v,epsilon):
    """A deterministic mean-excess linearity rule, using only one FIT class."""
    v=np.asarray(v,dtype='float64')
    # The paper specifies a mean-excess linearity check but provides no code
    # or numerical rule. This explicit rule is a reproducible local adaptation.
    grid=np.unique(np.quantile(v,np.linspace(.5,.95,10)))
    candidates=[]
    for start in range(max(1,len(grid)-3)):
        us=grid[start:]
        keep=np.array([np.sum(v>u)>=max(5,int(.025*len(v))) for u in us])
        us=us[keep]
        if len(us)<4:continue
        means=np.array([np.mean(v[v>u]-u) for u in us])
        coef=np.polyfit(us,means,1)
        residual=np.sum((means-np.polyval(coef,us))**2)
        variance=np.sum((means-means.mean())**2)
        score=1-residual/max(variance,1e-30)
        candidates.append((float(us[0]),float(score)))
        if score>=.95:break
    passing=[a for a in candidates if a[1]>=.95]
    u=passing[0][0] if passing else (max(candidates,key=lambda a:a[1])[0] if candidates else 0.)
    excess=v[v>u]-u
    if len(excess)<2 or np.ptp(excess)<1e-12:
        return dict(u=u,shape=0.,scale=0.,cutoff=float(v.max()),tail_n=len(excess),method='degenerate_empirical_max_no_fabricated_tail')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shape,_,scale=genpareto.fit(excess,floc=0)
    cutoff=u+float(genpareto.ppf(epsilon,shape,loc=0,scale=scale))
    if not np.isfinite(cutoff) or scale<=0:raise ValueError('invalid GPD tail fit')
    return dict(u=u,shape=float(shape),scale=float(scale),cutoff=cutoff,tail_n=len(excess),method='GPD_MLE_mean_excess_linearity',tiny_tail=len(excess)<5)

def open_prediction(pred,err,tails):
    out=np.where(pred==0,0,1).astype('int8')
    cuts=np.array([a['cutoff'] for a in tails])
    out[err>=cuts[pred]]=2
    return out

def run(setting,epochs,run_name):
    started=time.perf_counter()
    out=BASE/'results'/run_name/setting/'CVAE-EVT';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1234);np.random.seed(1234)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(1234)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    known=np.load(DATA/setting/'known.npz');query=np.load(DATA/setting/'query.npz')
    x=torch.as_tensor(known['x_fit'],device=device);y=torch.as_tensor(known['y_fit'],device=device)
    xq=torch.as_tensor(query['x'],device=device)
    classifier,reconstructor,source=load_author_networks(setting,x.shape[1])
    classifier=classifier.to(device);reconstructor=reconstructor.to(device)
    lr=1e-4
    optimizer=torch.optim.Adam(classifier.parameters(),lr=lr)
    stage1=[];batch_size={'CIC1':500,'CIC2':100,'NSL1':50,'NSL2':50}[setting]
    for epoch in range(1,epochs+1):
        classifier.train();loss_sum=0.
        for ix in torch.randperm(len(x),device=device).split(batch_size):
            a=x[ix];one=nn.functional.one_hot(y[ix],9).float()
            q=classifier.z_xy(a,one);prior=classifier.z_x(a)
            probs=classifier.y_xz(a,q.rsample())
            loss=(D.kl.kl_divergence(q,prior).sum()+nn.functional.binary_cross_entropy(probs,one,reduction='sum'))/len(ix)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        stage1.append({'Epochs':epoch,'Stage1_Loss':loss_sum/len(x),'Elapsed_Seconds':time.perf_counter()-started})
        if epoch%10==0:print(setting,'CVAE stage1',epoch,'loss',round(loss_sum/len(x),5),flush=True)
        write_json(out/'progress.json',dict(stage=1,epoch=epoch,elapsed_seconds=time.perf_counter()-started))
    classifier.eval()
    reconstructor.l_z_xy.load_state_dict(classifier.l_z_xy.state_dict())
    for p in reconstructor.l_z_xy.parameters():p.requires_grad=False
    pf=classification(classifier,x);pq=classification(classifier,xq)
    stage2_lr=.001 if setting.startswith('CIC') else .0001
    optimizer=torch.optim.Adam([p for p in reconstructor.parameters() if p.requires_grad],lr=stage2_lr)
    records=[];train_predictions=[];query_predictions=[];all_tails=[]
    stage2_started=time.perf_counter()
    for epoch in range(1,epochs+1):
        ep_start=time.perf_counter();reconstructor.train();loss_sum=0.
        for ix in torch.randperm(len(x),device=device).split(batch_size):
            a=x[ix];one=nn.functional.one_hot(y[ix],9).float()
            z=reconstructor.z_xy(a,one).rsample()
            loss=nn.functional.mse_loss(reconstructor.x_yz(one,z),a)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        if epoch>1 and (epoch-1)%50==0:
            for group in optimizer.param_groups:group['lr']*=.5
        reconstructor.eval()
        fit_errors=errors(reconstructor,x,y)
        tails=[fit_tail(fit_errors[known['y_fit']==c],EPSILON[setting]) for c in range(9)]
        train_err=errors(reconstructor,x,pf);query_err=errors(reconstructor,xq,pq)
        train_predictions.append(open_prediction(pf.cpu().numpy(),train_err,tails))
        query_predictions.append(open_prediction(pq.cpu().numpy(),query_err,tails));all_tails.append(tails)
        record=dict(Epochs=epoch,Stage2_Loss=loss_sum/len(x),Epoch_Seconds=time.perf_counter()-ep_start,Elapsed_Seconds=time.perf_counter()-started)
        records.append(record)
        if epoch%10==0:print(setting,'CVAE stage2',epoch,'loss',round(record['Stage2_Loss'],6),'elapsed',round(record['Elapsed_Seconds'],1),flush=True)
        write_json(out/'progress.json',dict(stage=2,epoch=epoch,elapsed_seconds=time.perf_counter()-started))
    for filename,rows in [('stage1_training.csv',stage1),('training_history.csv',records)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'predictions_all_epochs.npz',train=np.stack(train_predictions),query=np.stack(query_predictions),fit_ids=known['fit_ids'],query_ids=query['ids'])
    torch.save({'classifier':classifier.state_dict(),'reconstructor':reconstructor.state_dict(),'feature_dim':x.shape[1],'setting':setting},out/'model_final.pt')
    write_json(out/'gpd_tails_all_epochs.json',all_tails)
    total=time.perf_counter()-started
    write_json(out/'run_manifest.json',dict(model='CVAE-EVT',setting=setting,epochs=epochs,stage1_epochs=epochs,stage2_epochs=epochs,curve_stage=2,
        row_semantics='stage2_training_epoch_after_complete_stage1',seed=1234,batch_size=batch_size,
        stage1_lr=lr,stage2_initial_lr=stage2_lr,epsilon=EPSILON[setting],device=str(device),
        source=str(source.relative_to(BASE)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        adaptations=['input/output widths match shared data encoding','SciPy GPD replaces paper MATLAB tail fitting','explicit FIT-only mean-excess linearity rule','no test-threshold sweep','100 epochs per sequential stage'],
        benign_clustering=False,benign_clustering_note='The paper names the version without clustering CVAE-EVT; clustering variant is CVAE-EVT*.',
        total_seconds=total,stage2_seconds=total-(stage2_started-started),training_complete=True,
        query_truth_read_during_training=False,selected_epoch=epochs,selection='fixed last epoch',published=False,final_test_executed=False))
    print('COMPLETE',setting,'CVAE-EVT',round(total,2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True,choices=SOURCES);p.add_argument('--epochs',type=int,default=100);p.add_argument('--run-name',default='comparison_20261102')
    a=p.parse_args()
    with threadpool_limits(limits=4):run(a.setting,a.epochs,a.run_name)
