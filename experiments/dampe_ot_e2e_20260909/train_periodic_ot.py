"""Two requested periodic BothOT protocols on one fixed drug-cold fold."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT = Path(os.environ.get("MDTA_PROJECT", "/data1/ztx/MyModel-MDTA")).resolve()
sys.path.insert(0, str(PROJECT))
from datasets.davis_dataset_p13d import DavisDatasetP13D
from datasets.collate_p13d import mdta_collate_fn_p13d, move_batch_to_device
from models.model_p13d import MyModelMDTAP13D
from train_p13d_earlystop import set_seed, train_one_epoch, evaluate
from model_periodic_ot import MyModelMDTAP13DPeriodicOT
from ot_alignment import compute_cost_matrix, fit_alignment, SinkhornConvergenceError

SPLIT_ROOT = PROJECT/"data/splits/davis_drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final"
BASE_ROOT = PROJECT/"outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/baseline"
OUTPUT_ROOT = PROJECT/"outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final"
ENCODERS = ["drug_1d_encoder", "drug_3d_encoder", "protein_1d_encoder", "protein_3d_encoder"]
SOURCE_FILES = ["models/model_p13d.py", "models/fusion.py", "models/decoder.py",
                "models/drug_1d_encoder.py", "models/drug_3d_egnn_encoder.py",
                "models/protein_1d_encoder.py", "models/protein_3d_egnn_encoder.py",
                "datasets/davis_dataset_p13d.py", "datasets/collate_p13d.py",
                "train_p13d_earlystop.py"]


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(value,path):
    """Preserve the previous checkpoint and retry bounded transient I/O errors."""
    path=Path(path)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    for attempt in range(1,4):
        try:
            # A Python file handle exposes OS write errors more clearly than
            # the C++ filename writer; fsync precedes the atomic replacement.
            with tmp.open("wb") as handle:
                torch.save(value,handle)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
            return
        except (OSError,RuntimeError) as exc:
            if isinstance(exc,RuntimeError) and not any(
                token in str(exc) for token in
                ("PytorchStreamWriter","unexpected pos","file write failed")
            ):
                raise
            chain=[];current=exc;seen=set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                chain.append(dict(type=type(current).__name__,errno=getattr(current,"errno",None),message=str(current)))
                current=current.__cause__ or current.__context__
            try:
                fs=os.statvfs(path.parent)
                free_bytes=fs.f_bavail*fs.f_frsize
            except OSError:
                free_bytes=None
            print("CHECKPOINT_SAVE_RETRY",json.dumps(dict(path=str(path),attempt=attempt,max_attempts=3,free_bytes=free_bytes,errors=chain)),flush=True)
            if attempt==3:
                raise
            time.sleep(2*attempt)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print("CHECKPOINT_TEMP_CLEANUP_FAILED",str(tmp),repr(cleanup_error),flush=True)


def rng_state(device,train_generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device),train_generator=train_generator.get_state())


def restore_rng(state,device,train_generator):
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu());torch.cuda.set_rng_state(state["cuda"].cpu(),device)
    train_generator.set_state(state["train_generator"].cpu())


def validate_split(dataset,split):
    groups={};drugs={};audit={}
    for part in ["train","val","test"]:
        ix=split[part+"_indices"]
        if not ix or len(ix)!=len(set(ix)) or min(ix)<0 or max(ix)>=len(dataset):
            raise ValueError("Invalid "+part+" indices")
        frame=dataset.df.iloc[ix];groups[part]=set(ix);drugs[part]=set(frame.drug_id)
        audit[part]=dict(pairs=len(ix),drugs=len(drugs[part]),proteins=int(frame.protein_id.nunique()))
    for a,b in [("train","val"),("train","test"),("val","test")]:
        if groups[a]&groups[b] or drugs[a]&drugs[b]:raise ValueError(f"Leakage {a}/{b}")
    if set.union(*groups.values())!=set(range(len(dataset))):raise ValueError("Split is not a partition")
    return audit


def make_dataset():
    return DavisDatasetP13D(
        pairs_csv=PROJECT/"data/raw/davis/pairs.csv",
        drug_1d_dir=PROJECT/"data/processed/davis/drug_1d_chemberta2",
        drug_3d_dir=PROJECT/"data/processed/davis/drug_3d",use_drug_3d=True,
        protein_1d_dir=PROJECT/"data/processed/davis/protein_1d_esm2",
        protein_3d_dir=PROJECT/"data/processed/davis/protein_3d_gvp")


def make_loaders(dataset,split,batch_size,train_generator):
    loaders={}
    for part,shuffle in [("train",True),("val",False),("test",False)]:
        loaders[part]=DataLoader(Subset(dataset,split[part+"_indices"]),batch_size=batch_size,
            shuffle=shuffle,num_workers=0,collate_fn=mdta_collate_fn_p13d,pin_memory=True,
            generator=train_generator if shuffle else torch.Generator().manual_seed(42000))
    return loaders


def unique_entity_loaders(dataset,split,batch_size):
    frame=dataset.df.iloc[split["train_indices"]].copy();result={}
    for kind in ["drug","protein"]:
        col=kind+"_id"
        reps=frame.reset_index().groupby(col,sort=True)["index"].first()
        ids=[str(x) for x in reps.index.tolist()]
        loader=DataLoader(Subset(dataset,reps.tolist()),batch_size=batch_size,shuffle=False,
            num_workers=0,collate_fn=mdta_collate_fn_p13d,pin_memory=True,
            generator=torch.Generator().manual_seed(987654))
        result[kind]=(ids,loader)
    return result


def preserve_rng_begin(device):
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(device))


def preserve_rng_end(state,device):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3],device)


@torch.no_grad()
def extract_entity_embeddings(model,kind,ids,loader,device):
    h1=[];h3=[];seen=[]
    for batch in loader:
        seen.extend(map(str,batch[kind+"_id"]))
        batch=move_batch_to_device(batch,device)
        h1.append(getattr(model,kind+"_1d_encoder")(batch[kind+"_1d"]).cpu())
        h3.append(getattr(model,kind+"_3d_encoder")(batch[kind+"_3d"]).cpu())
    if seen!=ids:raise ValueError(kind+" entity order mismatch")
    return torch.cat(h3),torch.cat(h1)


def compact_ot_meta(meta,previous,mapping,h3):
    mapped=h3.double()@mapping.double()
    change=None if previous is None else float(torch.linalg.norm(mapping-previous)/torch.linalg.norm(previous).clamp_min(1e-30))
    result=dict(num_unique_train_entities=meta["num_unique_train_entities"],
        train_entity_ids_hash=meta["train_entity_ids_hash"],cost_min=meta["cost_min"],
        cost_max=meta["cost_max"],cost_mean=meta["cost_mean"],
        cost_std=meta["cost_std"],epsilon=meta["epsilon"],iterations=meta["iterations"],
        converged=meta["converged"],row_marginal_max_abs_error=meta["row_marginal_max_abs_error"],
        column_marginal_max_abs_error=meta["column_marginal_max_abs_error"],
        marginal_max_relative_error=meta["marginal_max_relative_error"],T_finite=True,
        relative_map_change=change,source_mean_l2=float(torch.linalg.norm(h3.double(),dim=1).mean()),
        mapped_mean_l2=float(torch.linalg.norm(mapped,dim=1).mean()),
        source_mean_feature_variance=float(h3.double().var(0,unbiased=False).mean()),
        mapped_mean_feature_variance=float(mapped.var(0,unbiased=False).mean()),
        solver=meta.get("solver"),tolerance_relative=meta.get("tolerance_relative"),
        mapping_source=meta.get("mapping_source","current_sinkhorn_solution"),
        used_previous_map=bool(meta.get("used_previous_map",False)))
    if meta.get("failure_message"):
        result["sinkhorn_failure"]=meta["failure_message"]
    return result


def update_ot(model,entities,device,epsilon,previous):
    state=preserve_rng_begin(device);was_training=model.training;model.eval();result={};maps={}
    try:
        for kind in ["drug","protein"]:
            ids,loader=entities[kind]
            h3,h1=extract_entity_embeddings(model,kind,ids,loader,device)
            try:
                coupling,meta=fit_alignment(h3,h1,ids,ids,ids,epsilon=epsilon)
                meta["cost_std"]=float(compute_cost_matrix(h3,h1).std())
                mapping=(coupling*coupling.shape[1]).float()
            except SinkhornConvergenceError as error:
                if previous.get(kind) is None:
                    raise
                meta=dict(error.metadata)
                meta.update(mapping_source="previous_valid_map_fallback",
                            used_previous_map=True,failure_message=str(error))
                mapping=previous[kind].detach().cpu().float().clone()
            maps[kind]=mapping
            result[kind]=compact_ot_meta(meta,previous.get(kind),mapping,h3)
        model.set_ot_maps(maps["drug"],maps["protein"],True)
    finally:
        preserve_rng_end(state,device);model.train(was_training)
    return maps,result


def build_model(device):
    return MyModelMDTAP13DPeriodicOT(hidden_dim=128,dropout=0.1,task="regression").to(device)


def load_warmstart_encoders(model,fold,split_path,split):
    path=BASE_ROOT/f"fold_{fold}"/"best_model.pt"
    if not path.exists():raise FileNotFoundError("MISSING MATCHED BASELINE CHECKPOINT "+str(path))
    ck=torch.load(path,map_location="cpu",weights_only=False)
    args=ck.get("args",{})
    expected=Path(args.get("split_json","")).resolve()
    if expected!=split_path.resolve():raise ValueError("Baseline checkpoint split path mismatch")
    saved=json.loads((path.parent/"split_indices.json").read_text())
    for part in ["train","val"]:
        if saved[part+"_indices"]!=split[part+"_indices"]:raise ValueError("Baseline split indices mismatch")
    state=ck["model_state_dict"]
    for name in ENCODERS:
        prefix=name+"."
        own={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        getattr(model,name).load_state_dict(own,strict=True)
    return path,ck


def checkpoint_payload(model,optimizer,epoch,train_metrics,val_metrics,args,early,ot_meta,train_gen,device,provenance):
    return dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch,
        train_metrics=train_metrics,val_metrics=val_metrics,args=vars(args),early_stopping=early,
        ot_enabled=bool(model.ot_enabled.item()),ot_metadata=ot_meta,rng_state=rng_state(device,train_gen),
        provenance=provenance)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--experiment",choices=["ot_warmstart_finetune_both","ot_warmup_e2e_both"],required=True)
    p.add_argument("--fold",type=int,choices=range(1,6),required=True)
    p.add_argument("--device",required=True);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--ot_warmup_epochs",type=int,default=10);p.add_argument("--epsilon",type=float,default=1e-3)
    p.add_argument("--epochs",type=int,default=500);p.add_argument("--batch_size",type=int,default=16)
    p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--weight_decay",type=float,default=1e-5)
    p.add_argument("--early_stop_patience",type=int,default=60);p.add_argument("--early_stop_min_delta",type=float,default=1e-4)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true")
    p.add_argument("--smoke_train_pairs",type=int,default=32);p.add_argument("--smoke_eval_pairs",type=int,default=32)
    args=p.parse_args()
    device=torch.device(args.device);set_seed(args.seed)
    split_path=SPLIT_ROOT/f"fold_{args.fold}"/"split.json";split=json.loads(split_path.read_text())
    dataset=make_dataset();raw=pd.read_csv(PROJECT/"data/raw/davis/pairs.csv")
    if len(raw)!=len(dataset.df) or raw.drug_id.astype(str).tolist()!=dataset.df.drug_id.tolist() or raw.protein_id.astype(str).tolist()!=dataset.df.protein_id.tolist():
        raise ValueError("Dataset filtering changes split row identity")
    audit=validate_split(dataset,split)
    if args.smoke:
        # Keep true unique train entities for OT; reduce only optimization/evaluation pair counts.
        split=dict(split)
        split["train_indices"]=split["train_indices"][:args.smoke_train_pairs]
        split["val_indices"]=split["val_indices"][:args.smoke_eval_pairs]
        split["test_indices"]=split["test_indices"][:args.smoke_eval_pairs]
    train_gen=torch.Generator().manual_seed(args.seed)
    loaders=make_loaders(dataset,split,args.batch_size,train_gen)
    # Formal entity IDs always derive from the complete original train split.
    formal_split=json.loads(split_path.read_text());entities=unique_entity_loaders(dataset,formal_split,args.batch_size)
    model=build_model(device);warmstart={"loaded":False}
    if args.experiment=="ot_warmstart_finetune_both":
        base_path,base_ck=load_warmstart_encoders(model,args.fold,split_path,formal_split)
        warmstart=dict(loaded=True,path=str(base_path),sha256=sha256(base_path),best_epoch=base_ck["epoch"],
                       baseline_val_metrics=base_ck["val_metrics"],loaded_modules=ENCODERS,
                       excluded_modules=["drug_fusion","protein_fusion","decoder"])
    out=OUTPUT_ROOT/args.experiment/f"fold_{args.fold}"
    if args.smoke:out=OUTPUT_ROOT/(args.experiment+"_smoke")/f"fold_{args.fold}"
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError("Refusing nonempty output "+str(out))
    out.mkdir(parents=True,exist_ok=True)
    # Verify filesystem can atomically replace a checkpoint before training.
    atomic_torch_save({"write_test":True},out/".checkpoint_write_test.pt");(out/".checkpoint_write_test.pt").unlink()
    sources={f:sha256(PROJECT/f) for f in SOURCE_FILES}
    sources.update({"experiment/"+f.name:sha256(f) for f in Path(__file__).parent.glob("*.py")})
    provenance=dict(experiment=args.experiment,fold=args.fold,seed=args.seed,split_path=str(split_path),
        split_sha256=sha256(split_path),split_audit=audit,pairs_sha256=sha256(PROJECT/"data/raw/davis/pairs.csv"),
        source_sha256=sources,warmstart=warmstart,normalized_ot="A=T@diag(b)^-1=128*T",
        ot_population={k:dict(count=len(v[0]),ids_hash=hashlib.sha256(json.dumps(v[0],separators=(",",":")).encode()).hexdigest()) for k,v in entities.items()},
        test_used_for_fitting_or_selection=False,smoke=args.smoke)
    write_json(out/"provenance.json",provenance)
    criterion=torch.nn.MSELoss();optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    start=1;early=dict(best_val_rmse=float("inf"),best_epoch=-1,no_improvement=0);history=[];previous={"drug":None,"protein":None}
    latest=out/"latest_model.pt"
    if args.resume:
        ck=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(ck["model_state_dict"],strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"]);start=ck["epoch"]+1;early=ck["early_stopping"]
        restore_rng(ck["rng_state"],device,train_gen);history=json.loads((out/"history.json").read_text())
        previous={"drug":model.drug_ot_map.detach().cpu().clone(),"protein":model.protein_ot_map.detach().cpu().clone()} if ck["ot_enabled"] else previous
    max_epochs=2 if args.smoke else args.epochs;started=time.monotonic()
    for epoch in range(start,max_epochs+1):
        enabled=(args.experiment=="ot_warmstart_finetune_both" or epoch>args.ot_warmup_epochs)
        if enabled:
            maps,ot_meta=update_ot(model,entities,device,args.epsilon,previous);previous=maps
        else:
            identity=torch.eye(128,device=device);model.set_ot_maps(identity,identity,False)
            ot_meta={"stage":"baseline_warmup","ot_computed":False}
        train_metrics=train_one_epoch(model,loaders["train"],criterion,optimizer,device,log_interval=200)
        val_metrics=evaluate(model,loaders["val"],criterion,device)
        if not all(np.isfinite(x) for x in list(train_metrics.values())+list(val_metrics.values())):raise RuntimeError("Nonfinite metrics")
        improved=val_metrics["rmse"]<early["best_val_rmse"]-args.early_stop_min_delta
        if improved:early=dict(best_val_rmse=val_metrics["rmse"],best_epoch=epoch,no_improvement=0)
        else:early["no_improvement"]+=1
        row=dict(epoch=epoch,stage="periodic_both_ot" if enabled else "baseline_warmup",train=train_metrics,val=val_metrics,ot=ot_meta,elapsed_seconds=time.monotonic()-started)
        history.append(row);payload=checkpoint_payload(model,optimizer,epoch,train_metrics,val_metrics,args,early,ot_meta,train_gen,device,provenance)
        atomic_torch_save(payload,latest)
        if improved:atomic_torch_save(payload,out/"best_model.pt")
        write_json(out/"history.json",history);write_json(out/"progress.json",dict(status="training",epoch=epoch,early_stopping=early,stage=row["stage"]))
        print(f'{args.experiment} fold={args.fold} epoch={epoch} stage={row["stage"]} train_mse={train_metrics["mse"]:.6f} val_mse={val_metrics["mse"]:.6f} best={early["best_epoch"]}',flush=True)
        if not args.smoke and early["no_improvement"]>=args.early_stop_patience:break
    best=torch.load(out/"best_model.pt",map_location=device,weights_only=False);model.load_state_dict(best["model_state_dict"],strict=True)
    val=evaluate(model,loaders["val"],criterion,device);test=evaluate(model,loaders["test"],criterion,device)
    replay_delta={key:float(val[key]-best["val_metrics"][key]) for key in val}
    # CUDA scatter reductions can perturb nearly tied ranks. Require strict replay
    # for continuous errors and record ranking-metric drift separately.
    for key in ["mse","rmse","mae","loss"]:
        if abs(replay_delta[key])>1e-4:raise RuntimeError("Best checkpoint VAL replay mismatch "+key)
    print("CHECKPOINT_REPLAY_DELTA",json.dumps(replay_delta),flush=True)
    metrics=dict(fold=args.fold,best_epoch=best["epoch"],val_mse=val["mse"],val_ci=val["ci"],val_rm2=val["rm2"],
        test_mse=test["mse"],test_ci=test["ci"],test_rm2=test["rm2"],split_hash=sha256(split_path),seed=args.seed,
        experiment=args.experiment,ot_enabled_at_best=best["ot_enabled"],
        selection_val_metrics=best["val_metrics"],checkpoint_replay_delta=replay_delta,
        complete=True,smoke=args.smoke)
    write_json(out/"metrics.json",metrics);write_json(out/"best_ot_metadata.json",best["ot_metadata"])
    write_json(out/"ot_metadata.json",best["ot_metadata"])
    write_json(out/"progress.json",dict(status="complete",**metrics));print("COMPLETE",json.dumps(metrics),flush=True)

if __name__=="__main__":main()
