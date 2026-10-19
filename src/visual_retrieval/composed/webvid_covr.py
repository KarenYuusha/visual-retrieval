"""Persist trained CoVR-2 image/edit tokens separately from frozen gallery tokens."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from PIL import Image

from visual_retrieval.common import file_signature,sha256,write_json
from visual_retrieval.features.cache import atomic_npz,ensure_cache_config
from visual_retrieval.features.webvid_covr import release_gpu
from visual_retrieval.data.webvid_covr import PROTOCOL
from .gallery import digest

QUERY_FORMAT='covr2_image364_caption64_normalized32x256_and_frozen_text_v1'


def query_path(output,query_id):
    return Path(output)/'query_cache'/(hashlib.sha256(query_id.encode()).hexdigest()+'.npz')


def query_signature(query):
    return dict(query_id=query['query_id'],reference_id=query['reference_id'],video_id=query['video_id'],
                text=query['text'],image=file_signature(query['reference_image']),image_sha256=sha256(query['reference_image']))


def validate_query(tokens,text):
    tokens,text=np.asarray(tokens,dtype=np.float32),np.asarray(text,dtype=np.float32)
    if (tokens.shape!=(32,256) or text.shape!=(256,) or not np.isfinite(tokens).all()
            or not np.isfinite(text).all() or not np.allclose(np.linalg.norm(tokens,axis=-1),1,atol=.005)
            or not np.isclose(np.linalg.norm(text),1,atol=.005)):
        raise ValueError('Invalid normalized CoVR-2 query [32,256] or scoring text [256].')
    return tokens,text


def read_query(output,query):
    try:
        with np.load(query_path(output,query['query_id']),allow_pickle=False) as stored:
            if json.loads(str(stored['signature']))!=query_signature(query):
                raise ValueError('Image/text query changed; rerun extraction.')
            return validate_query(stored['query_tokens'],stored['scoring_text'])
    except (OSError,KeyError,EOFError,BadZipFile) as error:
        raise ValueError(f'Missing/invalid CoVR-2 query {query["query_id"]}; rerun extraction.') from error


def extract_queries(data,output,base_checkpoint,query_checkpoint,device='cuda',allow_incomplete=False,factory=None):
    if device not in ('cpu','cuda'):raise ValueError('Use cpu or cuda.')
    if factory is None:
        from visual_retrieval.models.covr2_encoder import CoVR2Encoder
        factory=CoVR2Encoder
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=dict(model_name='covr2',protocol=PROTOCOL,query_format=QUERY_FORMAT,
        base_sha256=digest(base_checkpoint),query_sha256=digest(query_checkpoint),
        precision='fp32' if device=='cpu' else 'fp16',device=device)
    if not (output/'run_config.json').exists() and any((output/'query_cache').glob('*.npz')):
        raise ValueError('Orphan query caches lack model provenance; use a new output folder.')
    ensure_cache_config(output,config)
    (output/'query_cache').mkdir(exist_ok=True)
    encoder=None
    successful,failed,reused=[],[],0
    try:
        for query in data.queries:
            try:
                read_query(output,query)
                reused+=1;successful.append(query['query_id']);continue
            except (ValueError,OSError):pass
            try:
                signature=query_signature(query)
                with Image.open(query['reference_image']) as image:
                    rgb=np.asarray(image.convert('RGB'))
            except (ValueError,OSError) as error:
                failed.append(dict(query_id=query['query_id'],error=str(error)));continue
            if encoder is None:
                encoder=factory(base_checkpoint,query_checkpoint,device=device)
            tokens,text=validate_query(*encoder.encode_query(rgb,query['text']))
            atomic_npz(query_path(output,query['query_id']),query_tokens=tokens,scoring_text=text,
                       signature=json.dumps(signature,sort_keys=True))
            successful.append(query['query_id'])
            print('covr2: encoded query',query['query_id'],flush=True)
        manifest=dict(**config,subset_sha256=data.subset_sha256,split=data.split,
            requested_query_ids=[q['query_id'] for q in data.queries],successful_query_ids=successful,
            failed_queries=failed,reused_queries=reused,complete=not failed)
        write_json(output/'manifest.json',manifest)
        if failed and not allow_incomplete:
            raise ValueError('CoVR-2 query extraction incomplete; retry or use --allow-incomplete.')
        return manifest
    finally:
        encoder=None
        release_gpu()
