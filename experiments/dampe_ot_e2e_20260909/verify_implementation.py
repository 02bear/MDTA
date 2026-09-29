"""Pre-launch invariants; no formal training and no test-based decisions."""
import json
from pathlib import Path
import sys
import torch

PROJECT=Path('/data1/ztx/MyModel-MDTA');sys.path.insert(0,str(PROJECT))
from models.model_p13d import MyModelMDTAP13D
from train_p13d_earlystop import set_seed
from model_periodic_ot import MyModelMDTAP13DPeriodicOT
from train_periodic_ot import load_warmstart_encoders,SPLIT_ROOT,OUTPUT_ROOT

def parameter_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters()}

def main():
    set_seed(42)
    baseline=MyModelMDTAP13D(hidden_dim=128,dropout=.1,task='regression')
    expected=parameter_dict(baseline)
    set_seed(42)
    periodic=MyModelMDTAP13DPeriodicOT(hidden_dim=128,dropout=.1,task='regression')
    current=parameter_dict(periodic)
    assert expected.keys()==current.keys()
    assert all(torch.equal(expected[k],current[k]) for k in expected)
    head_prefixes=('drug_fusion.','protein_fusion.','decoder.')
    heads={k:v.clone() for k,v in current.items() if k.startswith(head_prefixes)}
    split_path=SPLIT_ROOT/'fold_1/split.json'
    split=json.loads(split_path.read_text())
    _,base_ck=load_warmstart_encoders(periodic,1,split_path,split)
    loaded=parameter_dict(periodic)
    assert all(torch.equal(heads[k],loaded[k]) for k in heads)
    encoder_prefixes=('drug_1d_encoder.','drug_3d_encoder.',
                      'protein_1d_encoder.','protein_3d_encoder.')
    for key,value in base_ck['model_state_dict'].items():
        if key.startswith(encoder_prefixes):
            assert torch.equal(value.cpu(),loaded[key])
    assert all(p.requires_grad for p in periodic.parameters())
    checks={}
    for experiment in ['ot_warmstart_finetune_both','ot_warmup_e2e_both']:
        root=OUTPUT_ROOT/(experiment+'_smoke')/'fold_1'
        ck=torch.load(root/'best_model.pt',map_location='cpu',weights_only=False)
        required={'model_state_dict','optimizer_state_dict','epoch','early_stopping',
                  'ot_metadata','rng_state','provenance'}
        assert required<=set(ck)
        assert ck['ot_enabled'] is True
        for name in ['drug_ot_map','protein_ot_map']:
            mapping=ck['model_state_dict'][name].double()
            torch.testing.assert_close(mapping.sum(0),torch.ones(128,dtype=torch.float64),atol=2e-5,rtol=0)
            torch.testing.assert_close(mapping.sum(1),torch.ones(128,dtype=torch.float64),atol=2e-5,rtol=0)
        history=json.loads((root/'history.json').read_text())
        checks[experiment]=dict(stages=[x['stage'] for x in history],best_epoch=ck['epoch'],
            warmstart=ck['provenance']['warmstart'])
    assert checks['ot_warmstart_finetune_both']['stages']==['periodic_both_ot','periodic_both_ot']
    assert checks['ot_warmup_e2e_both']['stages']==['baseline_warmup','periodic_both_ot']
    assert checks['ot_warmstart_finetune_both']['warmstart']['loaded'] is True
    assert checks['ot_warmup_e2e_both']['warmstart']['loaded'] is False
    formal_absent={e:not (OUTPUT_ROOT/e).exists() for e in
                   ['ot_warmstart_finetune_both','ot_warmup_e2e_both']}
    assert all(formal_absent.values())
    print(json.dumps(dict(passed=True,
        periodic_random_initialization_exactly_matches_original_baseline=True,
        warmstart_only_four_encoders=True,
        warmstart_heads_unchanged_from_seed42_initialization=True,
        all_parameters_trainable=True,
        normalized_maps_have_unit_row_and_column_sums=True,
        checkpoint_resume_state_complete=True,smoke=checks,
        formal_outputs_absent=formal_absent),indent=2))

if __name__=='__main__':main()
