#!/usr/bin/env python3
"""Reproduce the eight legacy time-forward cutoffs and add shift diagnostics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
import ast, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from xgboost import XGBRegressor
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.warning')
MORGAN_GENERATOR=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
RESULT_ROOT=Path(os.environ.get('CASPER_RESULTS_ROOT', ROOT/'results'))
OUT=RESULT_ROOT/'time_forward';(OUT/'predictions').mkdir(parents=True,exist_ok=True)
THREADS=int(os.environ.get('CASPER_MODEL_THREADS','8'))
WORKERS=max(1,min(int(os.environ.get('CASPER_ANALYSIS_WORKERS','8')),8))
CUTOFFS=list(range(2016,2024)); MODELS={
 'Sequence':('aa_composition',{}),'Chemistry':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_only'}),
 'Chemistry + Site':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_anchors'}),
 'Chemistry + Site + Context':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'full'}),
 'ECFP':('ecfp',{'radius':2,'nBits':2048}),'RDKit':('rdkit_full',{})}

def records():
    samples=load_samples(ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl')
    raw=pd.read_csv(ROOT/'data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv',low_memory=False).set_index('ID')
    out=[]
    for s in samples:
      rr=(s.provenance or {}).get('raw_data_sample',{}); rid=int(rr.get('ID'))
      if rid not in raw.index:continue
      r=raw.loc[rid];year=int(r.Year)
      try:mons=ast.literal_eval(r.Sequence) if isinstance(r.Sequence,str) else []
      except Exception:mons=[]
      out.append({'sample':s,'sample_id':s.sample_id,'year':year,'source':str(r.Source),'sequence':s.sequence,'family':f"{r.Source}|len={int(r.Monomer_Length)}|{r.Molecule_Shape}",'label':float(s.label),'smiles':(s.assay_metadata or {}).get('smiles',''),'monomers':set(map(str,mons)),'edits':set(str(e.edit_family) for e in s.edits)})
    return out

def split_for(rs,cut):
    tr=[r for r in rs if r['year']<cut];future=[r for r in rs if r['year']>=cut];rng=np.random.RandomState(42);rng.shuffle(future);return tr,future[len(future)//2:]

def metrics(y,p):return {'r2':r2_score(y,p),'rmse':mean_squared_error(y,p)**.5,'mae':mean_absolute_error(y,p),'spearman':spearmanr(y,p).statistic}

def run_cutoff(args):
    cut,rs=args;tr,te=split_for(rs,cut); rows=[]
    for name,(fk,kw) in MODELS.items():
      f=FEATURIZER_REGISTRY[fk](**kw);ts=[r['sample'] for r in tr];es=[r['sample'] for r in te];f.fit(ts)
      X=np.nan_to_num(f.transform(ts));Xt=np.nan_to_num(f.transform(es));y=np.array([r['label'] for r in tr]);yt=np.array([r['label'] for r in te])
      m=XGBRegressor(n_estimators=100,max_depth=5,learning_rate=.1,subsample=.8,colsample_bytree=.8,tree_method='hist',verbosity=0,n_jobs=THREADS,random_state=42);m.fit(X,y,verbose=False);pr=m.predict(Xt)
      for r,a,b in zip(te,yt,pr):rows.append({'cutoff':cut,'model':name,'sample_id':r['sample_id'],'y_true':a,'y_pred':float(b),'sequence':r['sequence'],'family':r['family'],'source':r['source']})
    return cut,rows

def fp(sm):
    m=Chem.MolFromSmiles(sm) if sm else None
    return MORGAN_GENERATOR.GetFingerprint(m) if m else None

def main():
    rs=records();allp=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
      fs={ex.submit(run_cutoff,(c,rs)):c for c in CUTOFFS}
      for f in as_completed(fs):
        c,rows=f.result();allp+=rows;pd.DataFrame(rows).to_csv(OUT/f'predictions/cutoff_{c}.csv',index=False);print(c,'models complete',flush=True)
    pred=pd.DataFrame(allp);pred.to_csv(OUT/'all_predictions.csv',index=False)
    cutoff_rows=[];simrows=[]
    for c in CUTOFFS:
      tr,te=split_for(rs,c); tl=np.array([r['label'] for r in tr]);el=np.array([r['label'] for r in te]);tseq=set(r['sequence'] for r in tr);eseq=set(r['sequence'] for r in te);tf=set(r['family'] for r in tr);ef=set(r['family'] for r in te);ted=set().union(*(r['edits'] for r in tr));eed=set().union(*(r['edits'] for r in te));tm=set().union(*(r['monomers'] for r in tr));em=set().union(*(r['monomers'] for r in te))
      def ds(a,p):return {f'{p}_mean':a.mean(),f'{p}_sd':a.std(ddof=1),f'{p}_median':np.median(a),f'{p}_iqr':np.quantile(a,.75)-np.quantile(a,.25)}
      cutoff_rows.append({'cutoff':c,'train_n':len(tr),'test_n':len(te),'train_sequences':len(tseq),'test_sequences':len(eseq),'train_families':len(tf),'test_families':len(ef),'train_edit_families':len(ted),'test_edit_families':len(eed),'unseen_test_edit_fraction':len(eed-ted)/len(eed) if eed else 0,'train_monomers':len(tm),'test_monomers':len(em),'unseen_test_monomer_fraction':len(em-tm)/len(em) if em else 0,**ds(tl,'train_label'),**ds(el,'test_label')})
      trainfp=[x for x in (fp(r['smiles']) for r in tr) if x is not None]; vals=[]
      for r in te:
        q=fp(r['smiles']); sims=DataStructs.BulkTanimotoSimilarity(q,trainfp) if q is not None else []
        s=sorted(sims,reverse=True);vals.append({'cutoff':c,'sample_id':r['sample_id'],'max_tanimoto':s[0] if s else np.nan,'mean_top5_tanimoto':np.mean(s[:5]) if s else np.nan})
      v=pd.DataFrame(vals);v.to_csv(OUT/f'similarity_cutoff_{c}.csv',index=False);a=v.max_tanimoto.dropna();simrows.append({'cutoff':c,'median_max_tanimoto':a.median(),'iqr_max_tanimoto':a.quantile(.75)-a.quantile(.25),'mean_max_tanimoto':a.mean(),'fraction_below_0_3':(a<.3).mean(),'fraction_below_0_5':(a<.5).mean(),'fraction_above_0_7':(a>.7).mean()})
    pd.DataFrame(cutoff_rows).to_csv(OUT/'cutoff_statistics.csv',index=False);pd.DataFrame(simrows).to_csv(OUT/'similarity_diagnostics.csv',index=False)
    rng=np.random.default_rng(20260717);perf=[]
    for (c,m),g in pred.groupby(['cutoff','model']):
      base=metrics(g.y_true,g.y_pred); fam=g.family.unique();boots={k:[] for k in base}
      for _ in range(2000):
        pick=rng.choice(fam,len(fam),replace=True);ix=np.concatenate([np.flatnonzero(g.family.to_numpy()==x) for x in pick]);z=metrics(g.y_true.to_numpy()[ix],g.y_pred.to_numpy()[ix])
        for k,v in z.items():boots[k].append(v)
      for k,v in base.items():perf.append({'cutoff':c,'model':m,'metric':k,'estimate':v,'ci95_low':np.nanquantile(boots[k],.025),'ci95_high':np.nanquantile(boots[k],.975),'bootstrap_unit':'peptide family','replicates':2000,'bootstrap_seed':20260717})
    pd.DataFrame(perf).to_csv(OUT/'model_performance_with_ci.csv',index=False)
    # Descriptive cutoff correlations and source composition for focal years.
    focus=[]
    for c in [2020,2021,2023]:
      tr,te=split_for(rs,c)
      for src,n in Counter(r['source'] for r in te).most_common():focus.append({'cutoff':c,'source':src,'test_n':n,'test_fraction':n/len(te)})
    pd.DataFrame(focus).to_csv(OUT/'focal_year_source_composition.csv',index=False)
    print('time-forward diagnostics complete')
if __name__=='__main__':main()
