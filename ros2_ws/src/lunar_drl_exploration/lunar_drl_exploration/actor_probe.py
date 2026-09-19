"""Frozen-Q mechanism experiment; teachers are diagnostics, never replay labels.

Uses the actual Actor and candidate-aligned Critic on new-graph observations.
A controlled score-span stress and a mid-run teacher switch test recovery. The
initial transformation and subsequent optimization gains are reported separately.
Only a small JSON report is persisted; input checkpoints are never rewritten.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path


def summarize_fit(raw_initial_objective, history):
    return dict(immediate_transform_gain=history[0]['soft_objective']-raw_initial_objective,
                learning_gain=history[-1]['soft_objective']-history[0]['soft_objective'],
                final_best_probability=history[-1]['best_probability'])


def switched_q(q):
    result=q.clone()
    result[q.argmin()]=q.max()+max(float(q.max()-q.min()), .01)
    return result


def run(checkpoint, output, *, updates=300, cases=4, bounds=(0.,10.), device='cuda'):
    import numpy as np
    import torch
    from .config import ModelConfig
    from .model import Actor, Critic
    from .batch import pack_observations, pack_privileged
    from .replay import ReplayBuffer
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    record=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model_config=replace(ModelConfig(**record['learner']['model_config']),actor_score_bound=0.)
    base=Actor(model_config).to(device).eval()
    base.load_state_dict(record['learner']['actor'])
    critics=[Critic(model_config).to(device).eval() for _ in range(2)]
    for name,model in zip(('q1','q2'),critics):model.load_state_dict(record['learner'][name])
    alpha=float(record['learner']['log_alpha'].exp())
    replay=ReplayBuffer.from_snapshot(record['replay'])
    transitions=[entry[0] for entry in replay._entries.values()]
    transitions=[t for t in transitions if len(t.observation.action_nodes)>1]
    if not transitions:raise ValueError('probe requires a nontrivial action set')
    order=np.argsort([len(t.observation.action_nodes) for t in transitions],kind='stable')
    chosen=order[np.linspace(0,len(order)-1,min(cases,len(order)),dtype=int)]
    rows=[]
    for case_index,index in enumerate(chosen):
        transition=transitions[int(index)]
        batch=pack_observations([transition.observation],device)
        truth=pack_privileged([transition.observation],[transition.privileged],replay.scenes,device)
        with torch.no_grad():q=torch.minimum(*[critic.forward_packed(batch,truth) for critic in critics]).detach()
        # Both modes see exactly the same scaled initial parameter tensors.
        for stress_span in (None,40.):
            initial=copy.deepcopy(base)
            with torch.no_grad():
                raw=initial.forward_packed(batch).logits
                scale=1. if stress_span is None else stress_span/max(float(raw.max()-raw.min()),1e-6)
                initial.head[-1].weight.mul_(scale)
                initial.head[-1].bias.mul_(scale)
                raw_policy=initial.forward_packed(batch)
                raw_initial=float((raw_policy.probs*(q-alpha*raw_policy.log_probs)).sum())
            for bound in bounds:
                actor=Actor(replace(model_config,actor_score_bound=bound)).to(device).eval()
                actor.load_state_dict(initial.state_dict())
                optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
                history=[]
                teacher=q
                switch_at=updates//2 if stress_span is not None else None
                for step in range(updates+1):
                    policy=actor.forward_packed(batch)
                    objective=(policy.probs*(teacher-alpha*policy.log_probs)).sum()
                    if step in {0,1,10,50,100,updates,switch_at}:
                        raw=policy.raw_logits
                        gradient=torch.autograd.grad(-objective,raw,retain_graph=True)[0]
                        center=raw-raw.mean()
                        saturation=float((torch.tanh(center/bound).abs()>=.95).float().mean()) if bound else 0.
                        history.append(dict(update=step,teacher='switched' if switch_at is not None and step>switch_at else 'initial',
                            soft_objective=float(objective.detach()),expected_q=float((policy.probs.detach()*teacher).sum()),
                            best_probability=float(policy.probs[teacher.argmax()].detach()),
                            expected_q_regret=float((teacher.max()-(policy.probs.detach()*teacher).sum())),
                            raw_span=float((raw.max()-raw.min()).detach()),saturation_fraction=saturation,
                            best_raw_gradient=float(gradient[teacher.argmax()].abs()),
                            max_raw_gradient=float(gradient.abs().max())))
                    if step==updates:break
                    if step==switch_at:
                        teacher=switched_q(q)
                        # Record same weights under the new teacher before optimization.
                        objective=(policy.probs*(teacher-alpha*policy.log_probs)).sum()
                        history.append(dict(update=step,teacher='switched',soft_objective=float(objective.detach()),
                            expected_q=float((policy.probs.detach()*teacher).sum()),
                            best_probability=float(policy.probs[teacher.argmax()].detach()),
                            expected_q_regret=float(teacher.max()-(policy.probs.detach()*teacher).sum())))
                    optimizer.zero_grad(set_to_none=True)
                    (-objective).backward()
                    optimizer.step()
                # Never compare objectives belonging to different Q teachers.
                initial_history=[h for h in history if h['teacher']=='initial']
                row=dict(case=case_index,action_count=len(q),score_bound=bound,stress_span=stress_span,
                         initial_weight_scale=scale,alpha=alpha,history=history,
                         initial_teacher=summarize_fit(raw_initial,initial_history))
                switched=[h for h in history if h['teacher']=='switched']
                if switched:row['switched_teacher']=summarize_fit(switched[0]['soft_objective'],switched)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    result=dict(source=str(checkpoint),source_updates=record['learner']['updates'],device=device,
                updates_per_fit=updates,learning_rate=1e-5,rows=rows,
                limits='Frozen learned Q is not ground truth. Score-span stress and Q switch are controlled diagnostics, not live-policy evidence.')
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=300)
    parser.add_argument('--cases',type=int,default=4)
    parser.add_argument('--bounds',type=float,nargs='+',default=[0.,10.])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.updates<2 or args.cases<1:parser.error('at least two updates and one case required')
    run(args.checkpoint,args.output,updates=args.updates,cases=args.cases,bounds=args.bounds,device=args.device)


if __name__=='__main__':main()
