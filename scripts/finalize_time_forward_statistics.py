#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd,numpy as np,warnings
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('CASPER_RESULTS_ROOT', ROOT/'results'))/'time_forward'
c=pd.read_csv(OUT/'cutoff_statistics.csv');s=pd.read_csv(OUT/'similarity_diagnostics.csv');p=pd.read_csv(OUT/'model_performance_with_ci.csv');sp=p[p.metric=='spearman'].merge(c,on='cutoff').merge(s,on='cutoff');rows=[];rng=np.random.default_rng(20260902)
for m,g in sp.groupby('model'):
 for x in ['median_max_tanimoto','test_n','unseen_test_edit_fraction','unseen_test_monomer_fraction']:
  z=spearmanr(g.estimate,g[x]);a=g.estimate.to_numpy();b=g[x].to_numpy();boot=[]
  for _ in range(5000):
   ix=rng.integers(0,len(g),len(g))
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');q=spearmanr(a[ix],b[ix]).statistic
   if np.isfinite(q):boot.append(q)
  rows.append({'model':m,'performance_metric':'Spearman','diagnostic':x,'cutoff_level_spearman':z.statistic,'p_value_descriptive':z.pvalue,'bootstrap_ci95_low':np.quantile(boot,.025) if boot else np.nan,'bootstrap_ci95_high':np.quantile(boot,.975) if boot else np.nan,'bootstrap_replicates_requested':5000,'bootstrap_replicates_valid':len(boot),'bootstrap_unit':'cutoff year','n_cutoffs':len(g)})
pd.DataFrame(rows).to_csv(OUT/'cutoff_associations.csv',index=False)
tab=sp[['cutoff','train_n','test_n','train_sequences','test_sequences','test_families','median_max_tanimoto','iqr_max_tanimoto','model','estimate','ci95_low','ci95_high']].rename(columns={'estimate':'spearman'});tab.to_csv(OUT/'supplementary_cutoff_model_table.csv',index=False)
focus=c[c.cutoff.isin([2020,2021,2023])].merge(s,on='cutoff');focus.to_csv(OUT/'focal_year_diagnostics.csv',index=False)
print('time-forward statistics finalized')
