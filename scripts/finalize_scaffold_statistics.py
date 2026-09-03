#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np,pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('CASPER_RESULTS_ROOT', ROOT/'results'))/'scaffold_ranking';d=pd.read_csv(OUT/'family_level_results.csv');rng=np.random.default_rng(20260717);rows=[]
for m,g in d.groupby('model'):
 v=g.pairwise_accuracy.dropna().to_numpy();b=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(5000)]);rows.append({'model':m,'n_families_total':g.family_id.nunique(),'n_families_with_valid_pairs':len(v),'overall_pairwise_accuracy':g.correct_pair_count.sum()/g.valid_pair_count.sum(),'mean_family_pairwise_accuracy':v.mean(),'family_bootstrap_ci95_low':np.quantile(b,.025),'family_bootstrap_ci95_high':np.quantile(b,.975),'bootstrap_replicates':5000,'bootstrap_unit':'peptide family','interval':'percentile'})
pd.DataFrame(rows).to_csv(OUT/'summary_with_ci.csv',index=False)
a=d[d.model=='Chemistry'][['family_id','pairwise_accuracy']];b=d[d.model=='Chemistry + Site'][['family_id','pairwise_accuracy']];q=b.merge(a,on='family_id',suffixes=('_AB','_A'));q['difference']=q.pairwise_accuracy_AB-q.pairwise_accuracy_A;valid=q.difference.dropna();bs=np.array([rng.choice(valid,len(valid),replace=True).mean() for _ in range(5000)]);pos=(valid>0).sum();neg=(valid<0).sum();q.to_csv(OUT/'paired_comparison_AB_vs_A_family_values.csv',index=False);pd.DataFrame([{'comparison':'Chemistry + Site minus Chemistry','mean_difference':valid.mean(),'ci95_low':np.quantile(bs,.025),'ci95_high':np.quantile(bs,.975),'proportion_improved':(valid>0).mean(),'proportion_tied':(valid==0).mean(),'proportion_worsened':(valid<0).mean(),'exact_sign_test_p':stats.binomtest(pos,pos+neg,.5).pvalue,'n_families_total':len(q),'n_families_valid':len(valid),'bootstrap_replicates':5000,'interval':'percentile'}]).to_csv(OUT/'paired_comparison_AB_vs_A.csv',index=False)
print('scaffold statistics finalized')
