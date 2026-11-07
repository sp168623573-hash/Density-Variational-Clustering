"""Compute reports from saved predictions after fitting, never select epochs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

from asset_paths import DATA

BASE=Path(__file__).resolve().parent
MODELS=['CVAE-EVT','EVM','W-SVM']
SETTINGS=['CIC1','CIC2','NSL1','NSL2']

def metrics(y,p):
    cm=confusion_matrix(y,p,labels=[0,1,2])
    prec,rec,f1,_=precision_recall_fscore_support(y,p,labels=[0,1,2],average='weighted',zero_division=0)
    return dict(Accuracy=float(np.mean(y==p)),Precision=float(prec),Recall=float(rec),F1=float(f1),
                FPR=float(np.sum((y==0)&(p!=0))/max(np.sum(y==0),1)),
                BEN_Correct=int(cm[0,0]),BEN_Total=int(cm[0].sum()),
                KNA_Correct=int(cm[1,1]),KNA_Total=int(cm[1].sum()),
                UNA_Correct=int(cm[2,2]),UNA_Total=int(cm[2].sum()),
                Unknown_Detected=int(cm[2,2]),Unknown_Total=int(cm[2].sum()),
                Unknown_Detection_Rate=float(cm[2,2]/cm[2].sum()) if cm[2].sum() else 0.)

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(run_name):
    root=BASE/'results'/run_name;summary=[];verification=[]
    for setting in SETTINGS:
        k=np.load(DATA/setting/'known.npz');truth=np.load(DATA/setting/'truth.npz')
        ytrain=np.where(k['y_fit']==0,0,1);yquery=truth['truth']
        combined=[{'Epochs':i+1} for i in range(100)]
        available=[]
        for model in MODELS:
            out=root/setting/model
            if not (out/'run_manifest.json').exists():continue
            manifest=json.loads((out/'run_manifest.json').read_text('utf-8'))
            if model=='W-SVM':
                log_path=out/'native_train.log'
                log_text=log_path.read_text('utf-8')
                if not log_text.startswith('NATIVE TOOL LOG ONLY.'):
                    log_path.write_text('NATIVE TOOL LOG ONLY. Query files contain dummy labels to isolate inference from evaluation truth.\nPrinted native query Accuracy/Precision/Recall are NOT measured results. Use experiment_B_transfer_metrics.csv and Matrix.txt.\n'+log_text,encoding='utf-8')
            pred=np.load(out/'predictions_all_epochs.npz')
            assert manifest['training_complete'] and manifest['epochs']==100
            assert pred['train'].shape==(100,len(ytrain)) and pred['query'].shape==(100,len(yquery))
            assert np.array_equal(pred['fit_ids'],k['fit_ids']) and np.array_equal(pred['query_ids'],truth['ids'])
            assert np.isin(pred['train'],[0,1,2]).all() and np.isin(pred['query'],[0,1,2]).all()
            rows=[]
            for i in range(100):
                tr,te=metrics(ytrain,pred['train'][i]),metrics(yquery,pred['query'][i])
                row={'Epochs':i+1}
                for metric in ['Accuracy','Precision','Recall','F1','FPR']:
                    suffix='' if metric=='Accuracy' else '_'+metric
                    row[model+'_Train'+suffix]=tr[metric];row[model+'_Test'+suffix]=te[metric]
                rows.append(row);combined[i].update({a:b for a,b in row.items() if a!='Epochs'})
            write_csv(out/'experiment_B_transfer_accuracy.csv',rows)
            final=metrics(yquery,pred['query'][-1]);train_final=metrics(ytrain,pred['train'][-1])
            write_csv(out/'experiment_B_transfer_metrics.csv',[dict(Model=model,Setting=setting,Epochs=100,Scope='development',**final)])
            cm=confusion_matrix(yquery,pred['query'][-1],labels=[0,1,2]).T
            norm=cm/np.maximum(cm.sum(0),1)
            matrix='Rows = prediction; columns = truth. Order BEN, KNA, UNA.\nScope: previously observed development cohort, not independent FINAL.\n\nCounts\n'
            matrix+='\n'.join(' '.join(map(str,r)) for r in cm)
            matrix+='\n\nColumn-normalized\n'+'\n'.join(' '.join(f'{v:.8f}' for v in r) for r in norm)
            matrix+='\n\nExact correct/support\n'+'\n'.join(f'{c}: {final[c+"_Correct"]}/{final[c+"_Total"]}' for c in ['BEN','KNA','UNA'])+'\n'
            (out/'Matrix.txt').write_text(matrix,encoding='utf-8')
            write_csv(out/'predictions_final.csv',[dict(Sample_ID=str(s),True_Class=int(t),Predicted_Class=int(p)) for s,t,p in zip(truth['ids'],yquery,pred['query'][-1])])
            summary.append(dict(Setting=setting,Model=model,Epochs=100,Iteration_Type=manifest['row_semantics'],**final,
                                Train_Accuracy=train_final['Accuracy'],Fit_Seconds=manifest.get('fit_seconds',manifest['total_seconds']),Total_Seconds=manifest['total_seconds'],Scope='development'))
            verification.append(dict(setting=setting,model=model,rows=100,ids_verified=True,metrics_recomputed_from_all_saved_predictions=True,
                                     last_epoch_used=True,query_truth_read_during_training=manifest['query_truth_read_during_training'],
                                     predictions_sha256=hashlib.sha256((out/'predictions_all_epochs.npz').read_bytes()).hexdigest(),
                                     csv_sha256=hashlib.sha256((out/'experiment_B_transfer_accuracy.csv').read_bytes()).hexdigest()))
            available.append(model)
        if available:write_csv(root/setting/'experiment_B_transfer_accuracy.csv',combined)
    write_csv(root/'comparison_summary.csv',summary)
    lines=[f'三种对比模型四设置结果，已完成 {len(summary)}/12 组',
           '全部正式结果取第100行；CVAE-EVT为第二阶段epoch，EVM/W-SVM为真实完整拟合轮次。',
           '评估范围：已使用过的development队列，不是独立未见最终测试。',
           'NSL2按用户确认补入1条真实xsnoop，三个基线使用相同训练名单重跑；原运行保留。',
           '', '设置  模型  Accuracy  weighted-F1  FPR  UNA正确/总数  训练拟合秒数  含评估总分钟']
    for r in summary:
        lines.append(f"{r['Setting']:5} {r['Model']:8} {r['Accuracy']*100:8.2f}% {r['F1']*100:8.2f}% {r['FPR']*100:8.2f}% {r['UNA_Correct']}/{r['UNA_Total']}  {r['Fit_Seconds']:.2f}  {r['Total_Seconds']/60:.2f}")
    lines+=['','更完整的计数、precision/recall及运行语义见comparison_summary.csv。',
            '每个设置的experiment_B_transfer_accuracy.csv合并三个基线的真实100行记录。',
            'CVAE-EVT包含作者网络和按论文补齐的SciPy GPD适配；EVM使用原生libMR；W-SVM使用作者原生C++程序。',
            '方法和数据变动详见上级base line/README.txt；不能将本次值称为原论文数值的逐位复现。']
    (root/'结果汇总.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'verification.json').write_text(json.dumps(dict(completed_combinations=len(summary),expected_combinations=12,checks=verification),indent=2),encoding='utf-8')
    for r in summary:print(r['Setting'],r['Model'],f"Acc={r['Accuracy']:.6f}",f"UNA={r['UNA_Correct']}/{r['UNA_Total']}",f"seconds={r['Total_Seconds']:.1f}")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='comparison_20261102');main(p.parse_args().run_name)
