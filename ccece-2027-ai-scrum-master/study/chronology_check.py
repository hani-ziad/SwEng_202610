"""Chronology check of model arm M on the 120 replayed sprints (15 Nov 2026).

Arm M pools the training part of every project, so a replayed sprint can start before a training sprint of another
project. This script counts how often, refits M at planning time (c = 0) on sprints whose outcome was known before the
quarter of each replayed sprint, and compares the calls. It was run in a scratch workspace against the journal
repository's copy of the eight-project frame (data/dev_8projects/planning_frame.csv), which reproduces the stored
m_score exactly at c = 0 (max difference 1e-16). The halfway frame there differs from Phase 3's, so only c = 0 is
reported in the paper. Edit the paths below to rerun. Output: results/chronology_check.json.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/journal/prior'); sys.path.insert(0,'/mnt/user-data/uploads/SwEng_202610/ccece-2027-ai-scrum-master')
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from phase2.data import STRICT_NUMERIC
from phase2.models import feature_columns, make_classifier
from phase3.data import CHECKPOINT_NUMERIC
SEED=42
def youden(y,s):
    best,thr=-1.0,0.5
    for t in np.unique(s):
        p=s>=t; j=(p&(y==1)).sum()/max((y==1).sum(),1)+(~p&(y==0)).sum()/max((y==0).sum(),1)-1
        if j>best: best,thr=j,t
    return float(thr)
def fit_m(train,test,numeric):
    est=make_classifier("gradient_boosting",numeric,True); cols=feature_columns(numeric,True); y=train["label_delay"].to_numpy(int)
    oof=cross_val_predict(est,train[cols],y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method="predict_proba")[:,1]
    thr=youden(y,oof); est.fit(train[cols],y); return est.predict_proba(test[cols])[:,1],thr
rp=pd.read_csv('/mnt/user-data/uploads/SwEng_202610/ccece-2027-ai-scrum-master/study/results/replay_points.csv')
D='/home/claude/journal/data/dev_8projects/'
out={}
for c,frame,numeric in ((0.0,pd.read_csv(D+'planning_frame.csv',parse_dates=['start_date']),STRICT_NUMERIC),(0.5,None,CHECKPOINT_NUMERIC)):
    if c==0.5:
        ck=pd.read_csv(D+'checkpoint_frame.csv',parse_dates=['start_date']); pf=pd.read_csv(D+'planning_frame.csv')[['sprint_id','split_full']]
        ck=ck.drop(columns=[x for x in ['split_full'] if x in ck.columns]).merge(pf,on='sprint_id',how='left'); frame=ck[(ck.checkpoint==0.5)&ck.open_at_checkpoint]
    f=frame.copy(); f['known']=f.start_date+pd.to_timedelta(f.sprint_length_days+f.late_days.clip(lower=0).fillna(0),unit='D')
    tr,te=f[f.split_full=='train'],f[f.split_full=='test']
    r=rp[rp.checkpoint==c].set_index('sprint_id'); s=te[te.sprint_id.isin(r.index)].copy()
    sc,thr=fit_m(tr,s,numeric); s['m_rep']=sc
    rep_diff=float(np.abs(s.set_index('sprint_id').m_rep-r.m_score.reindex(s.sprint_id).values).max())
    early=int(sum((tr.start_date>t).any() for t in s.start_date)); early_known=int(sum((tr.known>t).any() for t in s.start_date))
    med=float(np.median([(tr.known>t).sum() for t in s.start_date]))
    q=s.start_date.dt.to_period('Q'); parts=[]; skipped=0
    for per in sorted(q.unique()):
        g=s[q==per]; t2=tr[tr.known<per.start_time]
        if len(t2)<150 or t2.label_delay.nunique()<2: skipped+=len(g); continue
        p,th=fit_m(t2,g,numeric); parts.append(pd.DataFrame({'sprint_id':g.sprint_id.values,'tv_score':p,'tv_thr':th}))
    tv=pd.concat(parts).set_index('sprint_id'); k=r.loc[tv.index]
    tv['tv_warn']=(tv.tv_score>=tv.tv_thr).astype(int)
    def within(y,sco,proj):
        num=den=0
        for p_,g in pd.DataFrame({'y':y,'s':sco,'p':proj}).groupby('p'):
            a,b=int(g.y.sum()),int((1-g.y).sum())
            if a and b: num+=roc_auc_score(g.y,g.s)*a*b; den+=a*b
        return num/den
    out[str(c)]=dict(n=len(s),reproduction_max_abs_diff=rep_diff,start_before_some_training_start=early,start_before_some_training_outcome_known=early_known,
        median_training_sprints_not_yet_known=med,scored_time_valid=len(tv),skipped=skipped,
        auc_M=float(roc_auc_score(k.late,k.m_score)),auc_M_tv=float(roc_auc_score(k.late,tv.tv_score)),
        within_M=float(within(k.late.values,k.m_score.values,k.project.values)),within_M_tv=float(within(k.late.values,tv.tv_score.values,k.project.values)),
        warn_M=int(k.m_warn.sum()),warn_M_tv=int(tv.tv_warn.sum()),same_call=int((k.m_warn.values==tv.tv_warn.values).sum()),
        kappa=float(cohen_kappa_score(k.m_warn.values,tv.tv_warn.values)),late=int(k.late.sum()),
        hits_M=int(((k.m_warn==1)&(k.late==1)).sum()),hits_M_tv=int(((tv.tv_warn.values==1)&(k.late.values==1)).sum()))
json.dump(out,open('/home/claude/exp/asm_chrono.json','w'),indent=1); print(json.dumps(out,indent=1))
