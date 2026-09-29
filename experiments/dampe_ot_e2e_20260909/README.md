# Periodic BothOT end-to-end experiments

Two protocols requested on 2026-09-09. This directory does not modify the baseline.

- `ot_warmstart_finetune_both`: load only four encoders from the matched fold baseline checkpoint, initialize original fusion/decoder from seed42, unfreeze everything, enable periodic normalized BothOT from epoch1.
- `ot_warmup_e2e_both`: initialize the complete original architecture from seed42, load no checkpoint, run ordinary concat for epochs1-10, enable periodic normalized BothOT from epoch11.

Each OT epoch saves/restores Python, NumPy, Torch CPU and CUDA RNG states around deterministic eval/no-grad entity extraction. OT is fit only on unique train entities. Validation uses the maps fit before the same training epoch. Best and latest checkpoints include both maps, optimizer, early-stop and RNG states. Test is evaluated once after training from the validation-selected best checkpoint, without refitting OT.

The two requested protocols do not contain a matched no-OT end-to-end continuation control. Consequently, they can measure predictive outcomes but cannot by themselves isolate the causal contribution of OT from initialization/staging effects.
