"""Bind the four existing development cohorts without resampling or neural features."""
from pathlib import Path
import hashlib
import os
import json
import argparse
import sys
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("PAPER_EXPERIMENT_ROOT", r"G:\Experiments\paper\software paper"))
PROCESSED = WORKSPACE / 'data/external_dev/processed'
ACTIVE = WORKSPACE / 'incremental_experiments/neighbor_evidence_20261031_r24/NEXT_COMPLETE_BASELINE_MANIFEST.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def remap(path):
    return Path(str(path).replace('F:\\shenpeng\\Experiments\\Chinese Science', str(WORKSPACE)))

def read(rel):
    return np.load(PROCESSED/rel, allow_pickle=False)

def add_authorized_xsnoop(xf,yf,fi,xc,xq):
    sys.path.insert(0,str(WORKSPACE))
    from paper_open_set.config import get_spec,NSL_FEATURE_COLUMNS
    from paper_open_set.data import _read_source,_feature_fingerprint
    source=WORKSPACE/'data/NSL-KDD/KDDTrain_Sample.txt'
    enc_path=PROCESSED/'nsl2_conditional_density_20261011_v47/known/FIT_only_full_native40_MHFD_encoding.json'
    encoding=json.loads(enc_path.read_text('utf-8'))
    frame=_read_source(get_spec('NSL-KDD Setting2'),source,'source_train')
    def encode(df):
        numeric=df[encoding['numeric_columns']].to_numpy('float64')
        pieces=[(np.log1p(numeric)-encoding['numeric_log_mean'])/encoding['numeric_log_std']]
        for name in encoding['categorical_columns']:
            values=encoding['vocabulary'][name];lookup={v:i for i,v in enumerate(values)}
            pos=np.array([lookup.get(v,len(values)) for v in df[name]])
            one=np.zeros((len(df),len(values)+1));one[np.arange(len(df)),pos]=1;pieces.append(one)
        return np.concatenate(pieces,axis=1).astype('float32')
    # Parity with the two existing physical xsnoop FIT records.
    for row in (14,8481):
        sid_prefix='nsl_kdd:source_train:'+str(row)+':'
        ix=[i for i,v in enumerate(fi) if str(v).startswith(sid_prefix)]
        assert len(ix)==1 and np.array_equal(encode(frame.iloc[[row]])[0],xf[ix[0]])
    candidate=frame.iloc[[137]]
    fp=str(_feature_fingerprint(candidate,NSL_FEATURE_COLUMNS).iloc[0])
    assert str(candidate.label.iloc[0])=='xsnoop' and fp=='0c849243e91547cd'
    value=encode(candidate)
    assert not any(np.any(np.all(a==value[0],axis=1)) for a in (xf,xc,xq))
    sid='nsl_kdd:source_train:137:'+fp
    info=dict(source=str(source.relative_to(WORKSPACE)),source_sha256=sha(source),source_row_zero_based=137,file_line_one_based=138,
              sample_id=sid,family='xsnoop',known_fine_label=8,distinct_from_FIT_CAL_query_encoded_features=True,
              authorization='User approved adding this one real source record to all three NSL2 baselines in this chat on 2026-11-02',
              reason='Original W-SVM requires at least 3 positive samples for Weibull fitting',copies_added=0,real_rows_added=1)
    return np.concatenate((xf,value)),np.append(yf,8),np.append(fi,sid),info,[source,enc_path]

def main(settings):
    active = json.loads(ACTIVE.read_text(encoding='utf-8'))['active_complete_baselines']
    configs = {
        'CIC1': ('CIC Setting1', 'cic1_fit6000_mhfd_20261009_v1/material/combined_known_roles_native78_before_fit.npz', 'cic1_mask78_conditional_mean_current_20261011_v1/current/full4056_native_inputs_before_masked_forward.npz'),
        'CIC2': ('CIC Setting2', 'cic2_shared_coverage_FIT6000_20261010_v1/shared_coverage5987_roles_native78.npz', 'cic2_live_guard_owned110_current_20261011_v1/actual3184_live_classification_geometry_before_score.npz'),
        'NSL1': ('NSL-KDD Setting1', 'nsl1_expanded_FIT6000_all41_20261009_v1/material/expanded_all41_known_inputs_before_training.npz', 'nsl1_expandedaux149_20261009_v1/current/actual_current149_geometry_before_score.npz'),
        'NSL2': ('NSL-KDD Setting2', 'nsl2_predictive_typed147_20261011_v53/train/complete_known115_inputs_before_training.npz', 'nsl2_fusion192_geometry_20261011_v67/current/actual_native115_profile77_fusion192_before_DVC.npz'),
    }
    global_manifest=BASE/'data/DATA_MANIFEST.json'
    all_info=json.loads(global_manifest.read_text('utf-8')) if global_manifest.exists() else {}
    for key,(name,kpath,qpath) in configs.items():
        if key not in settings:continue
        k,q=read(kpath),read(qpath)
        sources=[PROCESSED/kpath, PROCESSED/qpath, ACTIVE]
        if key.startswith('CIC'):
            xf,xc,xq=k['FIT_native78'],k['CAL_native78'],q['native78']
            fi,ci,qi=k['FIT_ids'],k['CAL_ids'],q['ids']
            yf,yc=k['FIT_y_fine'],k['CAL_y_fine']
            assert np.array_equal(k['FIT_y_coarse'],(yf!=0).astype('int64'))
            encoding='original 78 flow fields; shared FIT-only imputation and MinMax'
        elif key=='NSL1':
            xf,xc,xq=k['fit_features'],k['calibration_features'],q['Z149'][:,32:]
            fi,ci,qi=k['fit_ids'],k['calibration_ids'],q['ids']
            yf,yc=k['fit_known_fine'],k['calibration_known_fine']
            assert np.array_equal(k['fit_known_coarse'],(yf!=0).astype('int64'))
            old=read('nsl1_expandedaux149_20261009_v1/prepare/prepared_new_known149_before_pair_prefit.npz')
            assert np.array_equal(old['fit_Z149'][:,32:],xf)
            assert np.array_equal(old['calibration_Z149'][:,32:],xc)
            sources.append(PROCESSED/'nsl1_expandedaux149_20261009_v1/prepare/prepared_new_known149_before_pair_prefit.npz')
            encoding='existing FIT-only encoding of all 41 NSL fields into 117 coordinates; all 32 neural embedding coordinates excluded'
        else:
            xf,xc,xq=k['FIT_x115'],k['CAL_x115'],q['Z192'][:,:115]
            fi,ci,qi=k['FIT_ids'],k['CAL_ids'],q['ids']
            labels=read('nsl2_expanded6000_known_mhfd_20261009_v1/material/expanded_known5968_native40_and_features_before_training.npz')
            assert np.array_equal(fi,labels['FIT_ids']) and np.array_equal(ci,labels['CAL_ids'])
            yf,yc=labels['FIT_y_fine'],labels['CAL_y_fine']
            assert np.array_equal(labels['FIT_y_coarse'],(yf!=0).astype('int64'))
            sources.append(PROCESSED/'nsl2_expanded6000_known_mhfd_20261009_v1/material/expanded_known5968_native40_and_features_before_training.npz')
            encoding='existing FIT-only encoding of native 40 NSL fields into 115 coordinates; all 77 learned reconstruction coordinates excluded'
        truth_path=remap(active[key]['truth'])
        truth=np.load(truth_path,allow_pickle=False)
        assert np.array_equal(qi,truth['ids'])
        assert set(np.unique(yf))==set(range(9)) and set(np.unique(yc)).issubset(set(range(9)))
        # The inherited NSL2 material contains one FIT and one CAL source ID
        # also present in the existing query roster. Remove these from the
        # known roles without consulting query labels; keep every query row.
        overlaps={role:ids[np.isin(ids,qi)].tolist() for role,ids in [('FIT',fi),('CAL',ci)]}
        fm=~np.isin(fi,qi);cm=~np.isin(ci,qi)
        xf,yf,fi=xf[fm],yf[fm],fi[fm]
        xc,yc,ci=xc[cm],yc[cm],ci[cm]
        addition=None
        if key=='NSL2':
            xf,yf,fi,addition,extra_sources=add_authorized_xsnoop(xf,yf,fi,xc,xq)
            sources+=extra_sources
        assert len(set(fi))==len(fi) and not (set(fi)&set(qi) or set(ci)&set(qi) or set(fi)&set(ci))
        # Every matrix transform is fitted to FIT only, including median and range.
        arrays=[np.where(np.isfinite(x),x,np.nan) for x in (xf,xc,xq)]
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        scaler=MinMaxScaler()
        xf=scaler.fit_transform(imputer.fit_transform(arrays[0])).astype('float32')
        xc=scaler.transform(imputer.transform(arrays[1])).astype('float32')
        xq=scaler.transform(imputer.transform(arrays[2])).astype('float32')
        assert all(np.isfinite(x).all() for x in (xf,xc,xq))
        out=BASE/'data'/key
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'known.npz',x_fit=xf,y_fit=yf,fit_ids=fi,x_cal=xc,y_cal=yc,cal_ids=ci)
        np.savez_compressed(out/'query.npz',x=xq,ids=qi)
        np.savez_compressed(out/'truth.npz',truth=truth['truth'],ids=qi)
        joblib.dump({'imputer':imputer,'scaler':scaler},out/'preprocessing.joblib')
        info=dict(setting=name,setting_key=key,scope='previously observed development cohort; not an independent final test',
                  encoding=encoding,fit_rows=len(xf),cal_rows=len(xc),evaluation_rows=len(xq),features=xf.shape[1],
                  fit_fine_counts=np.bincount(yf,minlength=9).tolist(),cal_fine_counts=np.bincount(yc,minlength=9).tolist(),
                  evaluation_coarse_counts=np.bincount(truth['truth'],minlength=3).tolist(),
                  bootstrap=False,unknown_used_for_fit=False,unknown_used_for_calibration=False,known_role_id_overlap_removed=overlaps,authorized_real_known_sample_addition=addition,
                  existing_training_resampling_indices_ignored=True,evaluation_ids_match_existing_model=True,
                  upstream_sources=[{'path':str(p.relative_to(WORKSPACE)),'sha256':sha(p)} for p in sources+[truth_path]],
                  data_files={n:sha(out/n) for n in ['known.npz','query.npz','truth.npz','preprocessing.joblib']},
                  published=False,final_test_executed=False)
        (out/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        all_info[key]=info
        print(key, info['fit_rows'], info['cal_rows'],info['evaluation_rows'],info['features'],flush=True)
    (BASE/'data/DATA_MANIFEST.json').write_text(json.dumps(all_info,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--settings',nargs='+',default=['CIC1','CIC2','NSL1','NSL2']);main(p.parse_args().settings)
