"""Prepare KIBA label-free KLIFS85/Morgan similarities using Davis conventions."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from calibrate_sequence_fallback import align_pocket

P=Path('/data1/ztx/MyModel-MDTA'); H=Path(__file__).resolve().parent
K=P/'experiments/klifs85_interaction/data/klifs85'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    df=pd.read_csv(P/'data/raw/kiba/pairs.csv',dtype={'drug_id':str,'protein_id':str})
    split_path=P/'data/splits/kiba_fixed_split_811_full.json'; split=json.loads(split_path.read_text())
    parts=[split[k+'_indices'] for k in ['train','val','test']]
    assert len(df)==118254 and sorted(sum(parts,[]))==list(range(len(df)))
    assert not df.duplicated(['drug_id','protein_id']).any()
    drugs=df[['drug_id','smiles']].drop_duplicates('drug_id'); proteins=df[['protein_id','sequence']].drop_duplicates('protein_id')
    features=[]
    for folder,ids in [('drug_1d_chemberta2',drugs.drug_id),('drug_3d',drugs.drug_id),('protein_1d_esm2',proteins.protein_id),('protein_3d_gvp',proteins.protein_id)]:
        for name in ids:
            f=P/'data/processed/kiba'/folder/(name+'.pt');x=torch.load(f,map_location='cpu',weights_only=False)
            for key,value in x.items():
                if torch.is_tensor(value) and value.is_floating_point(): assert torch.isfinite(value).all(),(f,key)
            if '1d' in folder: assert x['mean'].shape==(768 if folder.startswith('drug') else 1280,)
            elif folder=='drug_3d': assert x['x'].shape[1]==10
            else: assert x['node_s'].shape[1]==6 and len(x['residue_meta'])==len(x['node_s'])
            features.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
        print('FEATURES_VERIFIED',folder,len(ids),flush=True)
    audit={'passed':True,'dataset':'KIBA','rows':len(df),'drugs':len(drugs),'proteins':len(proteins),'split_sizes':[len(x) for x in parts],
           'pairs_sha256':sha(P/'data/raw/kiba/pairs.csv'),'split_sha256':sha(split_path),'dropped_pairs':0,'missing_features':0,'new_base_features':0,
           'rnc_mid_threshold':float(np.quantile(df.iloc[parts[0]].label,.5)), 'rnc_high_threshold':float(np.quantile(df.iloc[parts[0]].label,.75)),
           'rnc_threshold_policy':'training labels only, quantiles 0.5/0.75; raw KIBA labels retained','unobserved_pairs':'NaN, never residual references'}
    for part,idx in zip(['val','test'],parts[1:]):
        assert set(df.iloc[idx].drug_id)<=set(df.iloc[parts[0]].drug_id)
        assert set(df.iloc[idx].protein_id)<=set(df.iloc[parts[0]].protein_id)
    (H/'feature_manifest.json').write_text(json.dumps(features,indent=2))
    (H/'data_audit.json').write_text(json.dumps(audit,indent=2))
    da=pd.read_csv(K/'mapping_audit.csv',dtype={'protein_id':str,'accession':str})
    dp=pd.read_csv(P/'data/raw/davis/proteins.csv',dtype=str).merge(da,on='protein_id')
    accepted=dp[dp.status.isin(['ok','ok_sequence_fallback'])]
    records=[]; seqs=[];masks=[];out=H/'data/klifs85/by_protein';out.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']='MDTA-KIBA-academic-preprocessing/1.0'
    for i,row in enumerate(proteins.itertuples(index=False)):
        matches=accepted[accepted.sequence==row.sequence]
        status='unavailable';method='none';source=None;reference=None
        idx=np.full(85,-1,dtype=np.int64);mask=np.zeros(85,dtype=bool);actual='-'*85;identity=0.
        if len(matches):
            source=K/'by_protein'/(matches.iloc[0].protein_id+'.pt');item=torch.load(source,map_location='cpu',weights_only=False)
            idx=item['sequence_indices'].numpy();mask=item['mask'].numpy().astype(bool);actual=item['pocket_sequence']
            assert len(actual)==85
            assert all(not keep or row.sequence[int(pos)]==aa for pos,aa,keep in zip(idx,actual,mask))
            status='ok';method='exact_sequence_reuse';reference=item.get('reference_pocket_sequence',actual)
            identity=sum(a==b for a,b,m in zip(actual,reference,mask) if m)/max(1,int(mask.sum()))
        else:
            access=dp[dp.accession==row.protein_id]
            for old in access.itertuples(index=False):
                f=K/'by_protein'/(old.protein_id+'.pt')
                if f.exists():
                    candidate=torch.load(f,map_location='cpu',weights_only=False).get('reference_pocket_sequence')
                    if isinstance(candidate,str) and len(candidate)==85: reference=candidate;source=f;break
            if reference is None:
                raw=H/'data/klifs85/raw_api'/(row.protein_id+'.json');raw.parent.mkdir(parents=True,exist_ok=True)
                if raw.exists(): response=json.loads(raw.read_text())
                else:
                    for attempt in range(3):
                        try:
                            r=session.get('https://klifs.net/api/kinase_ID',params={'kinase_name':row.protein_id,'species':'HUMAN'},timeout=30)
                            if r.status_code==400 and r.json()==[400,'KLIFS error: An unknown kinase name was provided']:
                                # Explicit database no-match, not a connectivity failure.
                                raw.with_suffix('.no_match_response.json').write_text(json.dumps(r.json()))
                                response=[];break
                            r.raise_for_status();response=r.json();assert isinstance(response,list);break
                        except Exception:
                            if attempt==2: raise
                            time.sleep(2)
                    raw.write_text(json.dumps(response,indent=2))
                for kinase in response:
                    candidate=kinase.get('pocket')
                    if kinase.get('uniprot')==row.protein_id and isinstance(candidate,str) and len(candidate)==85:
                        reference=candidate;source=raw;break
            if reference is not None:
                idx=np.asarray(align_pocket(row.sequence,reference,-5.,-.02),dtype=np.int64)
                mask=(idx>=0)&(idx<len(row.sequence))
                actual=''.join(row.sequence[int(pos)] if keep else '-' for pos,keep in zip(idx,mask))
                identity=sum(a==b for a,b,m in zip(actual,reference,mask) if m)/max(1,int(mask.sum()))
                status='ok_sequence_fallback' if mask.sum()>=80 and identity>=.75 else 'low_quality_sequence_fallback'
                method='davis_calibrated_sequence_fallback'
        keep=status in ['ok','ok_sequence_fallback']
        payload={'protein_id':row.protein_id,'sequence_sha256':hashlib.sha256(row.sequence.encode()).hexdigest(),
                 'sequence_indices':torch.tensor(idx.copy()),'mask':torch.tensor(mask.copy()),'pocket_sequence':actual,
                 'reference_pocket_sequence':reference,'status':status,'method':method,'source':str(source) if source else None,
                 'source_sha256':sha(source) if source else None,'accepted':keep}
        torch.save(payload,out/(row.protein_id+'.pt'))
        seqs.append(list(actual));masks.append(mask if keep else np.zeros(85,dtype=bool))
        records.append({'protein_id':row.protein_id,'status':status,'method':method,'accepted':keep,'coverage':float(mask.mean()),'identity':float(identity),'source':payload['source']})
        print('POCKET',i,row.protein_id,status,method,flush=True)
    seqs=np.asarray(seqs,dtype='U1');masks=np.asarray(masks,dtype=bool)
    ps=np.zeros((len(proteins),len(proteins)),dtype=np.float32);overlap=np.zeros_like(ps,dtype=np.int16)
    for i in range(len(proteins)):
        valid=masks[i][None,:]&masks;overlap[i]=valid.sum(1)
        ps[i]=((seqs[i][None,:]==seqs)&valid).sum(1)/85.
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fps=[]
    for smile in drugs.smiles:
        mol=Chem.MolFromSmiles(smile);assert mol is not None;fps.append(gen.GetFingerprint(mol))
    ds=np.asarray([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps],dtype=np.float32)
    np.savez_compressed(H/'entity_similarities.npz',drug_ids=drugs.drug_id.to_numpy(dtype=str),protein_ids=proteins.protein_id.to_numpy(dtype=str),drug_similarity=ds,protein_similarity=ps,protein_overlap=overlap,protein_coverage=masks.mean(1).astype(np.float32))
    pd.DataFrame(records).to_csv(H/'data/klifs85/mapping_audit.csv',index=False)
    summary={'drug_similarity':'Morgan radius 2 2048-bit Tanimoto','protein_similarity':'KLIFS85 masked identity divided by 85, identical to Davis',
             'status_counts':pd.DataFrame(records).status.value_counts().to_dict(),'method_counts':pd.DataFrame(records).method.value_counts().to_dict(),
             'unavailable_protein_ids':[r['protein_id'] for r in records if not r['accepted']],
             'unavailable_policy':'protein DSRC branch has zero support; preserve drug branch and all pairs',
             'similarity_sha256':sha(H/'entity_similarities.npz'),'no_affinity_labels_used_for_similarities':True}
    (H/'similarity_audit.json').write_text(json.dumps(summary,indent=2))
    print('PREPARE_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
