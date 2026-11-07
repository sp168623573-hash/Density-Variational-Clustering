"""Post-seal exact-vector/table checks for additional fixed R108 branches."""
from io34 import *
import csv

def review(setting,outputs,role_loader,replay,check):
    sys.path.insert(0,str(I/'target99_review_20261028'))
    V=mod(I/'target99_review_20261028/finalize_round.py','r46_extra_verifier');V.BASELINES=ACTIVE
    EV=mod(I/'target99_v950_20261023/common/target99_common_v950.py','r46_extra_metrics')
    D=H/setting/os.environ['R210_KIND'];report={a:{} for a in outputs};summary={a:{} for a in outputs}
    for role in ('CAL','CP','current'):
        check();q=role_loader(role);expected=replay(role,q)
        for arm,refs in outputs.items():
            ref=refs[role]['prediction'];assert desc(ref['path'])==ref;p=load(ref['path']);assert np.array_equal(p['ids'],q['ids']) and np.array_equal(p['parent_prediction'],q['parent']) and len(set(p['ids']))==len(p['ids'])
            for k,x in expected[arm].items():assert np.array_equal(x,p[k]),k
        if role=='current':t=load(ACTIVE[setting]['truth']);ids,y=t['ids'],t['truth'];assert [int(np.sum((y==c)&(q['parent']==c))) for c in range(3)]==ACTIVE[setting]['correct']
        else:ids,y=q['control_labels']
        assert np.array_equal(ids,q['ids'])
        for arm,refs in outputs.items():
            folder=D/arm/role;p=load(refs[role]['prediction']['path']);pred=p['prediction'];base=q['parent']
            r=EV.evaluate_complete(expected_ids=ids,prediction_ids=p['ids'],prediction=pred,phase='current' if role=='current' else 'control',setting=setting,version='R210_'+arm,parent_version=ACTIVE[setting]['version'],truth_ids=ids,truth=y,parent_ids=ids,parent_prediction=base,control_name=None if role=='current' else role)
            spec=EV.prepare_three_results(r,folder/'results')
            for table in read(spec)['tables']:
                with Path(table['path']).open('x',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(table['matrix'])
            save(folder/'paired_vector.npz',ids=ids,truth=y,prediction=pred,parent_prediction=base)
            if role=='current':V.verify_candidate(setting,folder,folder/'paired_vector.npz',folder/'results/result.json',[folder/'results/tables.json'],folder/'prediction_before_truth.npz')
            V.verify_tables(folder/'results/tables.json');mat=np.zeros((3,3),int);bm=mat.copy();np.add.at(mat,(pred,y),1);np.add.at(bm,(base,y),1);assert mat.tolist()==r['confusion_matrix']
            txt=(folder/'results/Matrix.txt').read_text(encoding='utf8');assert all(c+' '+' '.join(map(str,row)) in txt.splitlines() for c,row in zip(('BEN','KNA','UNA'),mat))
            gains=[];losses=[];metrics={}
            for c,name in enumerate(('BEN','KNA','UNA')):
                gains.append(int(np.sum((y==c)&(base!=c)&(pred==c))));losses.append(int(np.sum((y==c)&(base==c)&(pred!=c))))
                n=int(bm[:,c].sum());tp,old=int(mat[c,c]),int(bm[c,c]);den,od=int(mat[c].sum()),int(bm[c].sum())
                if n:metrics[name]=dict(precision=tp*od>=old*den,recall=tp>=old,F1=tp*(n+od)>=old*(n+den))
            changed=pred!=base;coherent=True
            for group in np.unique(q['groups'][changed]):
                for before in (0,1,2):
                    at=(q['groups']==group)&(base==before)
                    if (changed&at).any() and len(np.unique(pred[at]))>1:coherent=False
            summary[arm][role]=dict(correct=np.diag(mat),support=mat.sum(0),gain=gains,loss=losses,delta=np.array(gains)-losses,no_old_correct_loss=sum(losses)==0,no_metric_regression=all(all(v.values()) for v in metrics.values()),metric_checks=metrics,group_consistent=coherent,changed_records=int(changed.sum()),changed_feature_groups=len(np.unique(q['groups'][changed])),all_model_features_calibration_and_query_DVC_replayed=True)
            report[arm][role]=dict(result=r,raw_by_truth=[int(np.sum(p['raw']&(y==c))) for c in range(3)],filtered_by_truth=[int(np.sum(p['raw']&~p['retained']&(y==c))) for c in range(3)])
            print('EXTRA_VERIFIED',setting,arm,role,np.diag(mat).tolist(),gains,losses,flush=True)
    return summary,report
