"""Uniform soft raw-candidate corroboration in frozen complete32 geometry."""
import hashlib
import numpy as np

K=7;ALPHA=.9;ITERATIONS=20;FLOOR=.5
def require(v,why):
    if not v:raise ValueError(why)
def descriptor(a):
    a=np.ascontiguousarray(a);return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def recipe():
    return dict(schema_version='CIC1-uniform-V126raw-softinitial32-k7-alpha9-20-anchor0-v142',
        initial='all V126 raw candidate flags, including false positives; same raw function applied to CAL saved bankscore/eligible/stage1/rule and CP/current saved same-source cues',
        query='one canonical MHFDgroup with complete original32; allquery nodes, no trueclass/family filtering',
        anchors='all6000 genuineknownFIT original32 groups sortedlex, score0 onlyanchors clamped',
        neighbors='exact matching32 nearest7 of anchors+allquery excluding self; stable anchor-first globalindex ties',
        sigma='row distance7, Gaussian exp(-.5*(distance/sigma)^2); sigma0 exactdistance0 unitweights, no epsilon',
        updates='20 synchronous s=.9*(sum(weight*neighbor_score)/sum(weight))+.1*initial_raw; equivalent rowstochastic P, no query/raw clamp',
        scores='soft neighboring raw support indicator [0,1], not unknown probability or class-coverage guarantee',
        calibration='all trueknownCAL900 oncefinite99 + fixedengineeringfloor.5; originalstage1 cutbank',
        candidate='score strict>cut and baseline!=2; samefixedrule everyrole',
        DVC='candidate OR wholebaselinepred2 same32 hierarchical DVC, allold2 preserved',
        K=K,alpha=ALPHA,iterations=ITERATIONS,engineering_floor=FLOOR,parameter_grid=False,query_training=False,timecap_coded=False)
def raw_candidate(score,eligible,stage1,rule):
    score=np.asarray(score);eligible=np.asarray(eligible);stage1=np.asarray(stage1)
    require(score.shape==eligible.shape==stage1.shape and eligible.dtype==bool and np.isfinite(score).all() and np.isin(stage1,(0,1)).all(),'actual saved V126 candidate function completecue')
    result=np.zeros(len(score),bool)
    for c in (0,1):
        v=rule[str(c)]
        if v['enabled']:result[stage1==c]=score[stage1==c]>v['cutoff']
    return result&eligible
def collapse(z,groups,ids,raw):
    z=np.asarray(z,np.float64);groups=np.asarray(groups);ids=np.asarray(ids);raw=np.asarray(raw)
    require(z.shape==(len(ids),32) and groups.shape==raw.shape==(len(ids),) and raw.dtype==bool and np.isfinite(z).all(),'complete32 unlabelled sourcegroups/raw')
    unique,rep,inv=np.unique(groups,return_index=True,return_inverse=True)
    require(np.array_equal(z,z[rep][inv]),'samephysicalgroup identicalwhole32')
    initial=np.zeros(len(rep),bool);np.logical_or.at(initial,inv,raw)
    return dict(Z32=z[rep],groups=unique,representatives=rep,inverse=inv,initial_raw=initial,
        raw_rows=int(raw.sum()),raw_groups=int(initial.sum()),MHFD_groups=len(rep))
def diffusion(metric,anchor,query,initial_raw):
    a=np.asarray(anchor,np.float64);q=np.asarray(query,np.float64);raw=np.asarray(initial_raw)
    require(metric.dimension==32 and a.shape==(len(a),32) and q.shape==(len(q),32) and raw.shape==(len(q),) and raw.dtype==bool and len(a)>0 and len(q)>0 and np.isfinite(a).all() and np.isfinite(q).all(),'owned32 actualanchors/query/soft raw')
    before=(descriptor(a),descriptor(q),descriptor(raw));references=np.concatenate((a,q));total=len(references);require(total-1>=K,'seven nonself genuine nodes')
    neighbors=np.empty((len(q),K),np.int64);distances=np.empty((len(q),K),np.float64);calls=0
    for first in range(0,len(q),128):
        last=min(first+128,len(q));d=np.empty((last-first,total),np.float64)
        for start in range(0,total,4096):
            stop=min(start+4096,total);d[:,start:stop]=metric.pairwise_distances(q[first:last],references[start:stop]);calls+=1
        d[np.arange(last-first),len(a)+np.arange(first,last)]=np.inf
        order=np.argsort(d,axis=1,kind='stable')[:,:K];neighbors[first:last]=order;distances[first:last]=np.take_along_axis(d,order,axis=1)
    require(np.isfinite(distances).all() and np.all(neighbors!=len(a)+np.arange(len(q))[:,None]),'actualnearest7 selfexcluded')
    sigma=distances[:,-1];weights=np.empty_like(distances);positive=sigma>0
    weights[positive]=np.exp(-.5*np.square(distances[positive]/sigma[positive,None]));weights[~positive]=(distances[~positive]==0).astype(np.float64);weights/=weights.sum(1,keepdims=True)
    initial=raw.astype(np.float64);score=initial.copy();history=[]
    for step in range(ITERATIONS):
        allscore=np.concatenate((np.zeros(len(a)),score))
        expectation=np.sum(weights*allscore[neighbors],axis=1)/weights.sum(axis=1)
        score=ALPHA*expectation+(1-ALPHA)*initial
        # Deliberately no query/raw reclamp. A raw point near known anchors can
        # become .1 even when its original initial indicator was one.
        require(np.all(score>=0) and np.all(score<=1),'bounded soft corroboration');history.append(score.copy())
    require(before==(descriptor(a),descriptor(q),descriptor(raw)),'allsource coordinates and initial unchanged')
    return dict(score=score,neighbors=neighbors,distances=distances,sigma=sigma,weights=weights,initial_raw=raw,trajectory=np.stack(history),anchor_zero=np.zeros(len(a)),
        manifest=dict(recipe=recipe(),anchors=len(a),query_groups=len(q),actualdistancecalls=calls,MHFD_distance_is_plainEuclidean_not_paperEq9_10=True,knownanchors_clamped0=True,queryraw_clamped1=False,
            graph_is_unlabelled_testtime_inference=True,newfit=0,newNNforward=0,newR7=0,input_descriptors=before))
def synthetic_contract():
    class Metric:
        dimension=32
        def pairwise_distances(self,a,b):return np.sqrt(np.square(a[:,None]-b[None]).sum(2))
    rng=np.random.default_rng(133);a=rng.normal(size=(9,32));q=rng.normal(size=(4,32));raw=np.array([True,False,False,False])
    result=diffusion(Metric(),a,q,raw);require(result['trajectory'].shape==(20,4) and result['neighbors'].shape==(4,7),'actualgraphreturns')
    zero=diffusion(Metric(),a,q,np.zeros(4,bool));require(np.all(zero['score']==0),'zero raw trulycomputeszero, not roleflag')
    anchors=np.zeros((8,32));near=np.zeros((8,32));near[:,0]=1e-5*np.arange(8);allraw=np.ones(8,bool)
    suppressed=diffusion(Metric(),anchors,near,allraw);require(np.all(suppressed['score']<1) and suppressed['score'][0]<.5,'known0 anchors suppress even originalraw1, no hardseedclamp')
    far=near.copy();far[:,0]+=10;remote=diffusion(Metric(),anchors,far,allraw);require(np.all(remote['score']>.5),'mutual raw region can corroborate')
    groups=np.array(['g'+str(i) for i in range(8)]);ids=groups.copy();nodes=collapse(far,groups,ids,allraw)
    copied=collapse(np.concatenate((far,far[2:3])),np.concatenate((groups,groups[2:3])),np.concatenate((ids,np.array(['copy']))),np.r_[allraw,True])
    require(np.array_equal(nodes['Z32'],copied['Z32']) and np.array_equal(nodes['initial_raw'],copied['initial_raw']),'physicalgroup copies do not amplify graph or initial')
    rule={'0':dict(enabled=True,cutoff=.2),'1':dict(enabled=True,cutoff=.7)}
    s=np.array([.2,.3,.8,0.]);eligible=np.array([True,True,True,False]);stage=np.array([0,0,1,1])
    require(raw_candidate(s,eligible,stage,rule).tolist()==[False,True,True,False],'uniformCAL CP currentraw strict function no role or trutharg')
    large=rng.normal(size=(4100,32));small=rng.normal(size=(3,32));blocked=diffusion(Metric(),large,small,np.array([True,False,False]))
    whole=np.concatenate((large,small));direct=np.sqrt(np.square(small[:,None]-whole[None]).sum(2));direct[np.arange(3),4100+np.arange(3)]=np.inf
    require(np.array_equal(blocked['neighbors'],np.argsort(direct,axis=1,kind='stable')[:,:7]) and blocked['manifest']['actualdistancecalls']==2,
        'reference4096 blocks preserve globalexact nearest7 and stableties')
    return dict(synthetic_only=True,realtraffic=0,actualdict_return=True,softinitial_not_reclamped=True,negativeanchors_suppress_raw1=True,
        remote_mutualraw_corroborates=True,zero_actualraw_zero_inference=True,physicalcopy_no_node_mass=True,CAL_CP_current_same_candidate_function=True,
        blocked4096_globalnearest7_exact=True)
