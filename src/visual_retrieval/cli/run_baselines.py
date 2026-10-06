"""Run shared extraction/evaluation and matched comparisons across baseline models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from visual_retrieval.config import DEFAULT_BASE, DATASET_NAMES, MODEL_NAMES, dataset_root, output_directory
from visual_retrieval.common import write_json
from visual_retrieval.evaluation.comparison import compare_bundles


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',choices=DATASET_NAMES,nargs='+',default=list(DATASET_NAMES))
    parser.add_argument('--models',choices=MODEL_NAMES,nargs='+',default=list(MODEL_NAMES))
    parser.add_argument('--base-root',type=Path,default=DEFAULT_BASE)
    for name in DATASET_NAMES:
        parser.add_argument(f'--{name}-root',type=Path)
    parser.add_argument('--output-root',type=Path,help='Optional outputs under ROOT/dataset/model.')
    parser.add_argument('--stage',choices=['all','check','extract','evaluate','compare'],default='all')
    parser.add_argument('--clip4clip-checkpoint',type=Path)
    parser.add_argument('--internvideo2-checkpoint',type=Path)
    parser.add_argument('--clip-checkpoint',type=Path,help='Optional local HF CLIP directory.')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--precision',choices=['auto','bf16','fp16','fp32'],default='auto')
    parser.add_argument('--text-batch-size',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--activitynet-mode',choices=['segments','video'],default='segments')
    parser.add_argument('--activitynet-val-file',choices=['val_1.json','val_2.json'],default='val_1.json')
    parser.add_argument('--allow-missing',action='store_true',help='Permit missing files during preflight; extraction remains incomplete.')
    parser.add_argument('--summary',type=Path)
    return parser


def commands_for(args,dataset,model):
    root=getattr(args,dataset+'_root') or dataset_root(dataset,args.base_root)
    output=args.output_root/dataset/model if args.output_root else output_directory(model,root)
    extraction=[sys.executable,'-m','visual_retrieval.features.extraction','--model',model,'--dataset',dataset,
                '--data-root',str(root),'--output-dir',str(output),'--device',args.device,'--precision',args.precision,
                '--text-batch-size',str(args.text_batch_size),'--activitynet-mode',args.activitynet_mode,
                '--activitynet-val-file',args.activitynet_val_file]
    checkpoint=getattr(args,model+'_checkpoint')
    if checkpoint:
        extraction+=['--checkpoint',str(checkpoint)]
    if args.allow_missing:
        extraction+=['--allow-missing']
    evaluation=[sys.executable,'-m','visual_retrieval.evaluation.baseline_evaluation',
                '--features-dir',str(output),'--batch-size',str(args.batch_size)]
    return dict(check=extraction+['--check-data'],extract=extraction,evaluate=evaluation,output=output)


def main():
    parser=make_parser();args=parser.parse_args()
    if min(args.text_batch_size,args.batch_size)<=0:
        parser.error('Batch sizes must be positive.')
    args.models=list(dict.fromkeys(args.models));args.datasets=list(dict.fromkeys(args.datasets))
    if args.stage in ('all','extract') and 'clip4clip' in args.models and not (args.clip4clip_checkpoint and args.clip4clip_checkpoint.is_file()):
        parser.error('Supply --clip4clip-checkpoint PATH to a trained meanP/2d checkpoint, or select --models clip internvideo2.')
    source_root=str(Path(__file__).resolve().parents[2])
    env={**os.environ,'PYTHONPATH':source_root+os.pathsep+os.environ.get('PYTHONPATH','')}
    reports={}
    if args.stage in ('all','check','extract'):
        for dataset in args.datasets:
            subprocess.run(commands_for(args,dataset,args.models[0])['check'],check=True,env=env)
    if args.stage=='check':
        return
    for dataset in args.datasets:
        reports[dataset]={}
        directories=[]
        for model in args.models:
            job=commands_for(args,dataset,model);directories.append(job['output'])
            print(f'\n{dataset} / {model}',flush=True)
            if args.stage in ('all','extract'):
                subprocess.run(job['extract'],check=True,env=env)
            if args.stage in ('all','evaluate'):
                subprocess.run(job['evaluate'],check=True,env=env)
                reports[dataset][model]=json.loads((job['output']/'evaluation.json').read_text(encoding='utf-8'))
        if args.stage in ('all','compare') and len(directories)>1:
            reports[dataset]['comparison']=compare_bundles(directories,args.batch_size)
    if args.stage!='extract':
        summary=args.summary or args.base_root/'reports'/'baseline_comparison.json'
        write_json(summary,reports)
        print(f'Saved results: {summary}')


if __name__=='__main__':
    main()
