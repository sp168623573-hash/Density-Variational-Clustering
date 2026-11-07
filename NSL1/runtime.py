from io34 import *
import pickle,csv,time
from threadpoolctl import threadpool_limits
from adapters import Adapter
from metric_cache import ReuseMetric
import relative as R
SETTING,PHASE=sys.argv[1:3];D=H/SETTING;A=Adapter(SETTING);P=I/'coherent_affinity_20261102_r34'/SETTING
ARMS=['supervised_UNA','known_correction','duration_BEN','joint']
def check():assert time.time()<float(os.environ['R34_DEADLINE'])-30
def recipe():
    r=read(H/'FROZEN_RECIPE.json')
    for v in r['code']:assert desc(v['path'])==v
    for v in r['reused_sources'][SETTING].values():assert desc(v['path'])==v
    return r
def bundle():
    s=load(P/'source.npz');units={k:np.array(v) for k,v in read(P/'units.json').items()}
    with (P/'pair_model.pkl').open('rb') as f:m=pickle.load(f)
    return s,units,m
def role(r,units):
    q=A.role(r)
    if SETTING!='NSL2':q['x']=(q['x'].astype(float)-units['mean'])/units['scale']
    return q
def geometry(r,q):
    if SETTING=='NSL2':return A.geometry(r,q)
    if SETTING=='CIC2':metric=A.A.fixed_metric()
    else:
        from paper_open_set import nsl1_complete_frequency_negative_soft149_v716c as old
        owner=mod(I/'ben_structure_20261102_r43/reencoded_metric.py','r44_active_NSL1_metric');metric=owner.PaperVariationalPairMetric.from_state(read(I/'ben_structure_20261102_r43/NSL1/reencoded/metric.json'))
    return dict(metric=ReuseMetric(metric),routes={})
def route(q,context,g):
    key=context.tobytes()
    if key in g.setdefault('routes',{}):return g['routes'][key]
    if SETTING=='NSL1':
        from paper_open_set import nsl1_complete_frequency_negative_soft149_v716c as old
        result,manifest=old.nsl.base.route_once(g['metric'],dict(ids=q['ids'],groups=q['groups'],Z149=q['density_Z'],stage1=q['stage1']),context,lambda *a:None)
        r=dict(retained=result['retained'],manifest=manifest)
    else:r=A.route(q,context,g)
    g['routes'][key]=r;return r
