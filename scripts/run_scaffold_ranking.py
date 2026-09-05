#!/usr/bin/env python3
"""Recompute 49-family scaffold ranking with exact pair counts/predictions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
from pathlib import Path
import os,re,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from xgboost import XGBRegressor
RESULT_ROOT=Path(os.environ.get('CASPER_RESULTS_ROOT', ROOT/'results'))
OUT=RESULT_ROOT/'scaffold_ranking';(OUT/'predictions').mkdir(parents=True,exist_ok=True)
THREADS=int(os.environ.get('CASPER_MODEL_THREADS','8'))
WORKERS=max(1,min(int(os.environ.get('CASPER_ANALYSIS_WORKERS','8')),8))
MODELS={'Sequence':('aa_composition',{}),'Chemistry':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_only'}),'Chemistry + Site':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_anchors'}),'Chemistry + Site + Context':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'full'}),'ECFP':('ecfp',{'radius':2,'nBits':2048})}
def records():
 ss=load_samples(ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl');raw=pd.read_csv(ROOT/'data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv',low_memory=False).set_index('ID');out=[]
 for s in ss:
  rid=int(s.provenance['raw_data_sample']['ID']);r=raw.loc[rid];out.append({'sample':s,'id':s.sample_id,'source':str(r.Source),'family':f"{r.Source}|len={int(r.Monomer_Length)}|{r.Molecule_Shape}",'name':str(r.Original_Name_in_Source_Literature),'label':float(s.label)})
 return out
def num(x):
 a=re.findall(r'(\d+)',x);return int(a[-1]) if a else 0
def family_job(args):
 fid,members,allr=args;members=sorted(members,key=lambda r:num(r['name']));n=max(3,int(.3*len(members)));support=members[:n];test=members[n:];ids={r['id'] for r in members};hist=[r for r in allr if r['id'] not in ids];rows=[];pred=[]
 for mn,(fk,kw) in MODELS.items():
  f=FEATURIZER_REGISTRY[fk](**kw);train=hist+support;ts=[r['sample'] for r in train];es=[r['sample'] for r in test];f.fit(ts);X=np.nan_to_num(f.transform(ts));Xt=np.nan_to_num(f.transform(es));y=np.array([r['label'] for r in train]);yt=np.array([r['label'] for r in test]);m=XGBRegressor(n_estimators=100,max_depth=5,learning_rate=.1,subsample=.8,colsample_bytree=.8,tree_method='hist',verbosity=0,n_jobs=THREADS,random_state=42);m.fit(X,y,verbose=False);pr=m.predict(Xt);correct=valid=0
  predicted_ties=label_ties=0;score=0.0
  for i in range(len(yt)):
   for j in range(i+1,len(yt)):
    if yt[i]==yt[j]:label_ties+=1;continue
    valid+=1
    if pr[i]==pr[j]:predicted_ties+=1;score+=0.5
    elif ((yt[i]>yt[j])==(pr[i]>pr[j])):correct+=1;score+=1.0
  rows.append({'family_id':fid,'source':members[0]['source'],'model':mn,'n_historical':len(hist),'n_support':len(support),'n_test':len(test),'valid_pair_count':valid,'correct_pair_count':score,'strictly_concordant_pair_count':correct,'predicted_tie_count':predicted_ties,'label_tie_count':label_ties,'pairwise_accuracy':score/valid if valid else np.nan,'within_family_spearman':spearmanr(yt,pr).statistic})
  pred += [{'family_id':fid,'model':mn,'sample_id':r['id'],'y_true':a,'y_pred':float(b)} for r,a,b in zip(test,yt,pr)]
 return rows,pred
def main():
 rs=records();fam=defaultdict(list)
 for r in rs:fam[r['family']].append(r)
 fam={k:v for k,v in fam.items() if len(v)>=8};print('families',len(fam));rows=[];pred=[]
 with ProcessPoolExecutor(max_workers=WORKERS) as ex:
  fs={ex.submit(family_job,(k,v,rs)):k for k,v in fam.items()}
  for f in as_completed(fs):a,b=f.result();rows+=a;pred+=b;print(fs[f],'complete',flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'family_level_results.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'all_predictions.csv',index=False)
 rng=np.random.default_rng(20260717);s=[]
 for m,g in d.groupby('model'):
  vals=g.pairwise_accuracy.dropna().to_numpy();bs=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(5000)];overall=g.correct_pair_count.sum()/g.valid_pair_count.sum();s.append({'model':m,'n_families':len(vals),'overall_pairwise_accuracy':overall,'mean_family_pairwise_accuracy':vals.mean(),'family_bootstrap_ci95_low':np.quantile(bs,.025),'family_bootstrap_ci95_high':np.quantile(bs,.975),'bootstrap_replicates':5000,'bootstrap_unit':'peptide family'})
 pd.DataFrame(s).to_csv(OUT/'summary_with_ci.csv',index=False)
 a=d[d.model=='Chemistry'][['family_id','pairwise_accuracy']];b=d[d.model=='Chemistry + Site'][['family_id','pairwise_accuracy']];q=b.merge(a,on='family_id',suffixes=('_AB','_A'));z=q.pairwise_accuracy_AB-q.pairwise_accuracy_A;bs=[rng.choice(z,len(z),replace=True).mean() for _ in range(5000)];pos=(z>0).sum();neg=(z<0).sum();q['difference']=z;q.to_csv(OUT/'paired_comparison_AB_vs_A_family_values.csv',index=False);pd.DataFrame([{'comparison':'Chemistry + Site minus Chemistry','mean_difference':z.mean(),'ci95_low':np.quantile(bs,.025),'ci95_high':np.quantile(bs,.975),'proportion_improved':(z>0).mean(),'proportion_tied':(z==0).mean(),'proportion_worsened':(z<0).mean(),'exact_sign_test_p':__import__('scipy').stats.binomtest(pos,pos+neg,.5).pvalue,'n_families':len(z),'bootstrap_replicates':5000}]).to_csv(OUT/'paired_comparison_AB_vs_A.csv',index=False)
 print('scaffold ranking complete')
if __name__=='__main__':main()
