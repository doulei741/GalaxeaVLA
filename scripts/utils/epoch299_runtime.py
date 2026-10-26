"""Run-specific audited checkpoint retention and fixed held-out evaluation.

Only imported by the explicitly patched 299-episode experiment. No hardware control.
"""
from pathlib import Path
import contextlib
import datetime
import json
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel()*x.element_size()
    if isinstance(x, dict):
        return sum(_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_bytes(v) for v in x)
    return 0


def _link(root, name, path):
    temp = root/(name+'.new')
    temp.unlink(missing_ok=True)
    temp.symlink_to(path.relative_to(root))
    os.replace(temp, root/name)


def _atomic_save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    need = int(_bytes(state)*1.05) + 6*2**30
    free = shutil.disk_usage(path.parent).free
    if free < need:
        raise RuntimeError(f'Insufficient checkpoint space: free={free}, required_with_reserve={need}')
    temp = path.with_suffix('.pt.partial')
    if temp.exists():
        raise RuntimeError(f'Preserve/review interrupted checkpoint before retry: {temp}')
    torch.save(state, temp)
    with temp.open('rb') as f:
        os.fsync(f.fileno())
    restored = torch.load(temp, map_location='cpu', mmap=True, weights_only=False)
    if restored['step'] != state['step'] or len(restored['model_state_dict']) != len(state['model_state_dict']):
        raise RuntimeError('Saved checkpoint reload/schema check failed; previous checkpoint retained')
    if (restored['optimizer_state_dict'] is None) != (state['optimizer_state_dict'] is None):
        raise RuntimeError('Saved optimizer presence differs')
    del restored
    os.replace(temp, path)


def save_training_checkpoint(output_dir, step, epoch, batch_idx, model=None,
                             optimizer=None, scheduler=None, ema_model=None, **extra):
    root = Path(output_dir)
    full = optimizer is not None
    state = dict(step=step, epoch=epoch, batch_idx=batch_idx,
        model_state_dict=model.state_dict() if model is not None else None,
        optimizer_state_dict=optimizer.state_dict() if full else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        ema_model_state_dict=ema_model.ema_model.state_dict() if ema_model is not None else None,
        **extra)
    name = f'step_{step}.pt' if full else f'final_step_{step}.pt'
    path = root/'checkpoints'/name
    _atomic_save(state, path)
    _link(root, 'last.pt', path)
    if full:
        _link(root, 'last_full.pt', path)
        # Commit and reload the new recovery state BEFORE pruning older recovery states.
        for old in path.parent.glob('step_*.pt'):
            if old != path and old.stem.removeprefix('step_').isdigit():
                old.unlink()
    latest = root/'fixed_eval/latest.json'
    if full and latest.exists():
        result = json.loads(latest.read_text())
        selection = root/'best_selection.json'
        old_score = json.loads(selection.read_text())['score'] if selection.exists() else math.inf
        score = result.get('normalized_physical_action_score')
        if result['step'] == step and score is not None and score < old_score:
            infer = {**state, 'optimizer_state_dict': None, 'scheduler_state_dict': None}
            _atomic_save(infer, root/'checkpoints/best_inference.pt')
            selection.write_text(json.dumps({'step': step, 'score': score,
                'criterion': 'equal mean of four fixed FM part MAEs / step0 baselines; not task success',
                'metrics': result['metrics']}, indent=2)+'\n')
    record = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'step': step, 'path': str(path), 'full_optimizer_state': full,
              'bytes': path.stat().st_size, 'free_bytes': shutil.disk_usage(root).free}
    with (root/'checkpoint_events.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    logger.info('CHECKPOINT_COMMITTED %s', json.dumps(record))
    return path


def record_metrics(root, step, metrics):
    values = {k: float(v) for k,v in metrics.items() if isinstance(v,(int,float))}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError(f'Nonfinite logged metrics at step {step}')
    row = {'step': step, 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'train_record_kind': 'last_microbatch_at_log_step; preclip_grad_norm_on_accumulated_gradient',
           **values}
    with (Path(root)/'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    logger.info('METRICS %s',json.dumps(row))


def audit_optimizer(model, optimizer, root):
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    expected = {id(p) for _,p in params}
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    if not expected.issubset(actual):
        raise RuntimeError('Trainable parameters absent from optimizer')
    if any('lora_' in n.lower() for n,_ in params):
        raise RuntimeError('LoRA parameters found in requested full training')
    groups = {}
    for category, needles in [('vision', ('vision','visual')),
                              ('action', ('action_expert','action_head','action_out')),
                              ('backbone', ('vlm','language_model'))]:
        selected = [(n,p) for n,p in params if any(s in n for s in needles) and p.grad is not None
                    and (category != 'backbone' or not any(s in n for s in ('vision','visual','action_expert')))]
        if not selected:
            raise RuntimeError(f'No backward gradients for {category}')
        samples = [(n,float(p.grad.detach().float().norm().item())) for n,p in selected[:4]]
        if not any(norm>0 and math.isfinite(norm) for _,norm in samples):
            raise RuntimeError(f'No finite nonzero gradient samples in {category}')
        groups[category] = {'tensors_with_gradient':len(selected),'gradient_norm_samples':samples}
    result = {'trainable_parameters':sum(p.numel() for _,p in params),
              'total_parameters':sum(p.numel() for p in model.parameters()),
              'all_trainable_in_optimizer':True, 'lora':False, 'backward_groups':groups,
              'optimizer_groups':[{'lr':g['lr'],'base_lr':g.get('initial_lr',g['lr']),'parameter_count':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]}
    (Path(root)/'full_parameter_backward_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    logger.info('FULL_PARAMETER_BACKWARD %s',json.dumps(result))


class FixedEvaluator:
    """Fixed 150 anchors for rollout nodes; fixed 30 midpoints for interim CE/FM."""
    def __init__(self, native):
        self.native = native
        self.output_dir = Path(native.output_dir)
        self.root = Path(os.environ['A1Z_DATASET'])
        split = json.loads(Path(os.environ['G05_FIXED_SPLIT_JSON']).read_text())
        episodes = [json.loads(s) for s in (self.root/'meta/episodes.jsonl').read_text().splitlines()]
        import pyarrow.parquet as pq
        self.anchors=[];offset=0
        for ep in split['validation_episodes']:
            n=episodes[ep]['length'];top=max(0,n-32)
            table=pq.read_table(self.root/'data/chunk-000'/f'episode_{ep:06d}.parquet')
            points=[0,n//2,top]
            for side,fallback in [('left',n//4),('right',3*n//4)]:
                grip=np.asarray(table[f'action.{side}_gripper'].to_pylist()).reshape(-1)
                transitions=np.flatnonzero(np.diff(grip>.5)!=0)+1
                points.append(max(0,min(top,int(transitions[0])-8)) if len(transitions) else min(top,fallback))
            points=list(dict.fromkeys(points))
            for fallback in [n//4,3*n//4,n//3,2*n//3]:
                if len(points)==5:break
                point=min(top,fallback)
                if point not in points:points.append(point)
            if len(points)!=5:raise RuntimeError('Expected five distinct validation anchors')
            for point in points:self.anchors.append({'episode':ep,'frame':point,'index':offset+point,'midpoint':point==n//2})
            offset+=n
        if offset!=len(native.eval_dataloader.dataset):
            raise RuntimeError('Fixed validation offsets disagree with actual dataset')
        (self.output_dir/'fixed_eval').mkdir(exist_ok=True)
        (self.output_dir/'fixed_eval/anchors.json').write_text(json.dumps(self.anchors,indent=2)+'\n')

    def evaluate(self, model, accelerator, step, eval_batch=None):
        from utils.metric import rollout_and_calculate_metrics, move_to_device
        full = step==0 or step%5000==0 or step==int(os.environ.get('G05_TOTAL_UPDATES','-1'))
        anchors=self.anchors if full else [a for a in self.anchors if a['midpoint']]
        start=time.time();rng=random.getstate();nrng=np.random.get_state();was_training=model.training
        metrics=[]; predictions=[]; truth=[]; masks=[]
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(7007);torch.cuda.manual_seed_all(7007);random.seed(7007);np.random.seed(7007)
                model.eval()
                for anchor in anchors:
                    torch.manual_seed(7007+anchor['index']);torch.cuda.manual_seed_all(7007+anchor['index'])
                    random.seed(7007+anchor['index']);np.random.seed(7007+anchor['index'])
                    batch=self.native.eval_dataloader.collate_fn([self.native.eval_dataloader.dataset[anchor['index']]])
                    if full:
                        values,preds=rollout_and_calculate_metrics(batch,model,accelerator,
                            processor=self.native.eval_processor,parts_meta=self.native.parts_meta,return_preds=True)
                        pred=preds['action_denorm'].float().numpy();gt=preds['action_gt_denorm'].float().numpy()
                        predictions.append(pred);truth.append(gt);masks.append(preds['valid_dim_mask'].numpy())
                        offset=0
                        for part,dim in self.native.parts_meta.items():
                            if part in ['left_gripper','right_gripper']:
                                valid=preds['valid_dim_mask'][...,offset:offset+int(dim)].any().item()
                                if valid:
                                    actual=gt[0,:,offset]>.5;proposed=pred[0,:,offset]>.5
                                    expected=bool(np.any(np.diff(actual)));changed=bool(np.any(np.diff(proposed)))
                                    prefix=f'gripper/{part}/'
                                    values.update({prefix+'expected_transition':float(expected),prefix+'transition_hit':float(expected and changed),
                                        prefix+'false_transition':float(not expected and changed),prefix+'final_class_correct':float(actual[-1]==proposed[-1])})
                            offset+=int(dim)
                    else:
                        with torch.no_grad(),accelerator.autocast():
                            _,losses=model(move_to_device(batch,accelerator.device))
                        values={f'teacher_forcing/{"ar_action_ce_loss" if k=="ce_loss" else k}':v for k,v in losses.items()}
                    values={k:float(v.item() if hasattr(v,'item') else v) for k,v in values.items()}
                    if not all(math.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite fixed validation')
                    metrics.append(values)
        finally:
            random.setstate(rng);np.random.set_state(nrng);model.train(was_training)
        keys=set.intersection(*(set(x) for x in metrics))
        mean={key:float(np.mean([x[key] for x in metrics])) for key in keys}
        result={'step':step,'anchors':len(anchors),'mode':'fixed_action_rollout' if full else 'fixed_teacher_forcing',
                'metrics':mean,'seconds':time.time()-start,'not_robot_success_rate':True,
                'parts_meta':{k:int(v) for k,v in self.native.parts_meta.items()}}
        if full:
            np.savez_compressed(self.output_dir/f'fixed_eval/predictions_{step}.npz',
                fm_prediction_denorm=np.concatenate(predictions),ground_truth_denorm=np.concatenate(truth),
                valid_dim_mask=np.concatenate(masks),anchors=np.array([[a['episode'],a['frame']] for a in anchors]))
            result['per_anchor_metrics']=metrics
        baseline=self.output_dir/'fixed_eval/step_0.json'
        if full and step>0 and baseline.exists():
            base=json.loads(baseline.read_text())['metrics']
            names=[f'rollout/per_key/{part}/l1/fm_action_l1' for part in ['left_control','right_control','left_gripper','right_gripper']]
            if all(k in mean and k in base and base[k]>0 for k in names):
                result['normalized_physical_action_score']=float(np.mean([mean[k]/base[k] for k in names]))
        (self.output_dir/f'fixed_eval/step_{step}.json').write_text(json.dumps(result,indent=2)+'\n')
        if full:(self.output_dir/'fixed_eval/latest.json').write_text(json.dumps(result,indent=2)+'\n')
        logger.info('FIXED_VALIDATION %s',json.dumps({k:v for k,v in result.items() if k!='per_anchor_metrics'}))
        return {**{f'eval/fixed/{k}':v for k,v in mean.items()},'_eval_time_sec':result['seconds']}
