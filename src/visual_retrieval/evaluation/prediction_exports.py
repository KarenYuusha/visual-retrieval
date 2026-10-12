"""Stream query predictions for offline label and visual error analysis."""
import csv
import json
from pathlib import Path


def save_predictions(directory, queries, video_ids, manifest, rankings):
    directory = Path(directory)
    gallery = {item['item_id']: {key: item.get(key) for key in
               ('item_id', 'source_video_id', 'path', 'start', 'end')}
               for item in manifest.get('gallery_items', [])}
    csv_path, json_path = directory / 'query_ranks.csv', directory / 'predictions.jsonl'
    csv_temp, json_temp = csv_path.with_suffix('.csv.tmp'), json_path.with_suffix('.jsonl.tmp')
    with csv_temp.open('w', newline='', encoding='utf-8') as table, json_temp.open('w', encoding='utf-8') as records:
        writer = csv.writer(table)
        writer.writerow(['query_id', 'video_id', 'caption', 'rank', 'reciprocal_rank',
                         'ground_truth_video_id', 'predicted_video_id', 'is_correct',
                         'ground_truth_score', 'predicted_score', 'score_gap',
                         'ground_truth_path', 'ground_truth_start', 'ground_truth_end',
                         'predicted_path', 'predicted_start', 'predicted_end'])
        for row, query in enumerate(queries):
            target = query['video_id']
            predicted = video_ids[int(rankings['top_indices'][row, 0])]
            rank = int(rankings['ranks'][row])
            gt_score = float(rankings['target_scores'][row])
            pred_score = float(rankings['top_scores'][row, 0])
            gap = pred_score - gt_score
            correct = predicted == target
            gt_item, pred_item = gallery.get(target), gallery.get(predicted)
            writer.writerow([query['query_id'], target, query['text'], rank, 1 / rank,
                             target, predicted, correct, gt_score, pred_score, gap,
                             *[(gt_item or {}).get(key) for key in ('path', 'start', 'end')],
                             *[(pred_item or {}).get(key) for key in ('path', 'start', 'end')]])
            candidates = [dict(rank=i + 1, video_id=video_ids[int(index)], score=float(score),
                               is_ground_truth=video_ids[int(index)] == target)
                          for i, (index, score) in enumerate(zip(rankings['top_indices'][row], rankings['top_scores'][row]))]
            record = dict(query_id=query['query_id'], caption=query['text'],
                          ground_truth_video_id=target, predicted_video_id=predicted,
                          rank=rank, reciprocal_rank=1 / rank, is_correct=correct,
                          ground_truth_score=gt_score, predicted_score=pred_score, score_gap=gap,
                          ground_truth_item=gt_item, predicted_item=pred_item, top_k=candidates)
            records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    csv_temp.replace(csv_path)
    json_temp.replace(json_path)
