"""Prepare canonical item/query manifests without downloading model weights."""
import argparse
from pathlib import Path
from visual_retrieval.config import DEFAULT_BASE, DATASET_NAMES, dataset_root
from visual_retrieval.data.adapters import load_dataset
from visual_retrieval.common import write_json, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',choices=DATASET_NAMES,nargs='+',default=list(DATASET_NAMES))
    parser.add_argument('--base-root',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--data-root',type=Path,help='Override root for one selected dataset.')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--activitynet-mode',choices=['segments','video'],default='segments')
    args=parser.parse_args()
    if args.data_root and len(args.datasets)!=1:
        parser.error('--data-root requires exactly one dataset.')
    output=args.output_dir or args.base_root/'manifests'
    for name in args.datasets:
        root=args.data_root or dataset_root(name,args.base_root)
        data=load_dataset(name,root,activitynet_mode=args.activitynet_mode)
        items=[{**i,'path':str(i['path'].resolve())} for i in data.items]
        missing=[i['item_id'] for i in data.items if not i['path'].is_file()]
        path=output/f'{name}.json'
        write_json(path,dict(dataset=name,protocol=data.protocol,splits=data.splits,items=items,queries=data.queries,
                             subset_sha256=sha256(root/'subset.json'),
                             annotations_sha256=[sha256(p) for p in data.annotation_paths], missing_item_ids=missing))
        print(f'{name}: {len(items)} gallery items, {len(data.queries)} queries, {len(missing)} missing files; {path}')


if __name__=='__main__':
    main()
