"""Evaluate labeled multi-turn sessions against one saved model gallery."""
import argparse
import json
from pathlib import Path
import time
from visual_retrieval.common import write_json
from visual_retrieval.interaction.session import RetrievalSession
from visual_retrieval.retrieval.index import GalleryIndex
from visual_retrieval.retrieval.search import load_saved_encoder
from .ranking_metrics import metrics_from_ranks


def evaluate_sessions(sessions,gallery,encoder,method='accumulated',beta=.5,gamma=.25):
    if method not in ('latest','accumulated','feedback') or not isinstance(sessions,list) or not sessions:
        raise ValueError('Supply nonempty sessions and a supported method.')
    traces=[];by_turn={}
    for index,record in enumerate(sessions):
        target=record['target_item_id']
        if target not in gallery.lookup or not record.get('turns'):
            raise ValueError('Session target must exist in the gallery and turns must be nonempty.')
        state=RetrievalSession();turn_results=[]
        for turn,step in enumerate(record['turns'],1):
            state.add_query(step['query'])
            started=time.perf_counter()
            query=encoder.encode_text([state.query_text('latest' if method=='latest' else 'accumulated')])[0]
            if method=='feedback':
                query=state.refine(query,gallery,beta,gamma)
            results=gallery.search(query,len(gallery.item_ids))
            rank=next(r['rank'] for r in results if r['item_id']==target)
            turn_results.append(dict(turn=turn,query=state.query_text('latest' if method=='latest' else 'accumulated'),
                                     rank=rank,latency_seconds=time.perf_counter()-started))
            by_turn.setdefault(turn,[]).append(rank)
            # Feedback recorded after this turn affects the NEXT turn, never retroactively.
            for item_id in step.get('positive',[]):
                if item_id not in gallery.lookup:
                    raise ValueError(f'Feedback item absent from gallery: {item_id}')
                state.set_feedback(item_id,True)
            for item_id in step.get('negative',[]):
                if item_id not in gallery.lookup:
                    raise ValueError(f'Feedback item absent from gallery: {item_id}')
                state.set_feedback(item_id,False)
        traces.append(dict(session_id=record.get('session_id',index),target_item_id=target,turns=turn_results,
                           first_top10_turn=next((r['turn'] for r in turn_results if r['rank']<=10),None)))
    return dict(method=method,num_sessions=len(traces),by_turn={str(t):dict(num_sessions=len(ranks),
                **metrics_from_ranks(ranks)) for t,ranks in by_turn.items()},sessions=traces)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir',type=Path,required=True)
    parser.add_argument('--sessions',type=Path,required=True)
    parser.add_argument('--method',choices=['latest','accumulated','feedback'],default='accumulated')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--beta',type=float,default=.5);parser.add_argument('--gamma',type=float,default=.25)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sessions=json.loads(args.sessions.read_text(encoding='utf-8'))
    gallery=GalleryIndex.from_directory(args.features_dir)
    encoder=load_saved_encoder(args.features_dir,args.device)
    report=evaluate_sessions(sessions,gallery,encoder,args.method,args.beta,args.gamma)
    report['features_dir']=str(args.features_dir.resolve())
    write_json(args.output,report)
    print(json.dumps(report['by_turn'],indent=2))


if __name__=='__main__':
    main()
