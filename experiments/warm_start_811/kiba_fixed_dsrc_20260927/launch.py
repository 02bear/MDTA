"""Validate provenance and start the complete pipeline detached on GPU 1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;P=Path('/data1/ztx/MyModel-MDTA')
os.environ.pop('WARM_RUN_ROOT',None)
os.environ.update(WARM_SEED='42',WARM_GPU='1',CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import run_fixed_pipeline as pipe

def main():
    assert json.loads((H/'smoke_result.json').read_text())['passed']
    # Reconstruct the exact source-ID submatrix used by reused Davis pockets.
    old=np.load(P/'experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917/entity_similarities.npz')
    new=np.load(H/'entity_similarities.npz')
    rows=pd.read_csv(H/'data/klifs85/mapping_audit.csv')
    reused=rows[rows.method=='exact_sequence_reuse']
    lookup={x:i for i,x in enumerate(old['protein_ids'].astype(str))}
    old_ix=[lookup[Path(x).stem] for x in reused.source]
    new_ix=reused.index.to_numpy()
    discrepancy=float(np.abs(old['protein_similarity'][np.ix_(old_ix,old_ix)]-new['protein_similarity'][np.ix_(new_ix,new_ix)]).max())
    assert discrepancy<1e-7, discrepancy
    # Exercise both axes and unsupported fallback on a sparse synthetic grid.
    labels=np.full((3,3),np.nan);base=np.full((3,3),10.)
    train={'drug':np.array([0,1]),'protein':np.array([1,0]),'index':np.array([0,1])}
    query={'drug':np.array([0,2]),'protein':np.array([0,2]),'index':np.array([2,3])}
    labels[0,1]=11.;labels[1,0]=11.;labels[0,0]=99.;labels[2,2]=-99.
    cache={'labels':labels,'base':base,'split_pairs':{'train':train}}
    sim=np.array([[1.,.9,0.],[.9,1.,0.],[0.,0.,0.]])
    fixed=pipe.protocol()['fixed_dsrc'];sims={'drug_similarity':sim,'protein_similarity':sim}
    result=pipe.apply_fixed_dsrc(cache,sims,np.zeros(2),np.zeros(2),query,labels,fixed)
    assert result['drug_support'][0]>0 and result['protein_support'][0]>0
    assert result['correction'][0]>0 and result['correction'][1]==0
    labels[0,0]=np.nan;labels[2,2]=np.nan
    result2=pipe.apply_fixed_dsrc(cache,sims,np.zeros(2),np.zeros(2),query,labels,fixed)
    assert np.array_equal(result['calibrated'],result2['calibrated'])
    report={'davis_pocket_submatrix_max_error':discrepancy,'both_dsrc_axes_checked':True,'unsupported_identity_fallback':True,'synthetic_label_poisoning_invariance':True}
    pipe.dump(H/'additional_checks.json',report)
    for path,key in [(P/'data/raw/kiba/pairs.csv','pairs_sha256'),(pipe.SPLIT,'split_sha256')]:
        assert pipe.sha(path)==json.loads((H/'data_audit.json').read_text())[key]
    files={str(f.relative_to(H)):{'bytes':f.stat().st_size,'sha256':pipe.sha(f)} for f in H.rglob('*') if f.is_file() and 'smoke_output' not in f.parts and f.suffix in ['.py','.md','.npz']}
    pipe.dump(H/'final_manifest.json',{'files':files,'existing_base_features_unmodified':True,'timestamp':time.time()})
    pipe.dump(H/'protocol.json',pipe.protocol())
    root=pipe.RUN_ROOT;root.mkdir(parents=True,exist_ok=True)
    if (H/'launch.json').exists():
        previous=json.loads((H/'launch.json').read_text())
        pid=previous['pid']
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(H/'run_seed.py').encode() in cmd.read_bytes(): raise RuntimeError(f'Pipeline already active: {pid}')
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=30000,free
    with (root/'pipeline.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(H/'run_seed.py')],cwd=P,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':proc.pid,'gpu':1,'seed':42,'working_directory':str(H),'process_cwd':str(P),'results':str(root),'launched':time.time()}
    pipe.dump(H/'launch.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
