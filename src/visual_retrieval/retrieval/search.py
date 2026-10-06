"""Search a saved gallery using its matching model and checkpoint."""
import argparse
import json
from pathlib import Path
from visual_retrieval.common import DEFAULT_ROOT, write_json
from visual_retrieval.models.registry import (checkpoint_signature, resolve_model_files,
                                              create_encoder, validate_preprocessing)
from .index import GalleryIndex


def load_saved_encoder(directory, device='cuda', checkpoint=None, tokenizer=None):
    config=json.loads((Path(directory)/'run_config.json').read_text(encoding='utf-8'))
    model=validate_preprocessing(config)
    path, tokenizer=resolve_model_files(model,checkpoint or config['checkpoint']['path'],tokenizer or config['tokenizer'])
    signature=checkpoint_signature(path)
    original=dict(config['checkpoint']);actual=dict(signature)
    original.pop('path',None);actual.pop('path',None)
    if actual!=original:
        raise ValueError('Checkpoint signature differs from extraction. Use the original weights or extract a new gallery.')
    encoder=create_encoder(model,path,device,config['precision'] if device!='cpu' else 'fp32',tokenizer)
    return encoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',required=True)
    parser.add_argument('--features-dir',type=Path,default=Path(DEFAULT_ROOT)/'features'/'internvideo2_stage2_1b_all')
    parser.add_argument('--data-root',type=Path,help='Optional relocated source-video root.')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--tokenizer')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--top-k',type=int,default=10)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.query.strip() or args.top_k<=0:
        parser.error('Query and top-k must be nonempty/positive.')
    manifest=json.loads((args.features_dir/'manifest.json').read_text(encoding='utf-8'))
    validate_preprocessing(manifest)
    gallery=GalleryIndex.from_directory(args.features_dir)
    encoder=load_saved_encoder(args.features_dir,args.device,args.checkpoint,args.tokenizer)
    query=encoder.encode_text([args.query])[0]
    items={i['item_id']:i for i in manifest.get('gallery_items',[])}
    results=gallery.search(query,args.top_k)
    for result in results:
        item=items.get(result['item_id'],dict(path=str(Path(DEFAULT_ROOT)/'raw_videos'/f"{result['item_id']}.mp4"),
                                           source_video_id=result['item_id'],start=None,end=None))
        path=args.data_root/'raw_videos'/Path(item['path']).name if args.data_root else item['path']
        result.update(video_id=result['item_id'],source_video_id=item['source_video_id'],path=str(path),
                      start=item['start'],end=item['end'])
        print(json.dumps(result,ensure_ascii=False))
    if args.output:
        write_json(args.output,dict(query=args.query,results=results))


if __name__=='__main__':
    main()
