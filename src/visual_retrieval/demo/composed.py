"""Optional composed-search registry and mutually exclusive GPU engine routing."""
import gc
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image, UnidentifiedImageError
from visual_retrieval.composed.gallery import FORMAT, digest, rank_gallery, token_path, validate_tokens
from visual_retrieval.models.covr2_encoder import CoVR2Encoder
from .service import _item_metadata


def composer_value(value):
    """Validate a single image attachment without trusting filename extensions."""
    if isinstance(value,str): return value, None
    value = value or {}
    text, files = value.get('text','') or '', value.get('files',[]) or []
    if not isinstance(text,str) or len(text)>2000: raise ValueError('Use text up to 2000 characters.')
    if len(files)>1: raise ValueError('Attach one reference image.')
    if not files: return text, None
    file = files[0]
    path = file if isinstance(file,str) else file.get('path')
    if not path: raise ValueError('Attach a reference image.')
    try:
        with Image.open(path) as image:
            if image.format not in {'PNG','JPEG','WEBP','BMP'} or getattr(image,'n_frames',1) != 1:
                raise ValueError('Use one still PNG, JPEG, WebP or BMP image; videos are not supported.')
            if image.width*image.height>40_000_000:
                raise ValueError('Use a reference image up to 40 megapixels.')
            return text, np.asarray(image.convert('RGB'))
    except (OSError,UnidentifiedImageError) as e:
        raise ValueError('Use a still image, not a video or another file type.') from e


class ComposedSearchService:
    def __init__(self, text_service, datasets, base_checkpoint, query_checkpoint, encoder_loader=CoVR2Encoder):
        self.text = text_service
        self.device = text_service.device
        self.base, self.checkpoint = Path(base_checkpoint),Path(query_checkpoint)
        if not self.base.is_file() or not self.checkpoint.is_file():
            raise ValueError('Both CoVR-2 checkpoint files are required.')
        self.encoder_loader = encoder_loader
        self._encoder = None
        self._lock = threading.RLock()
        self.galleries = dict(text_service.galleries)
        self.issues = list(text_service.issues)
        base_hash=digest(self.base)
        for spec in datasets:
            folder=spec.root/'features'/'covr2'
            try:
                manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
                identity=manifest['identity']
                if identity['format']!=FORMAT or identity['base_sha256']!=base_hash or identity['frames']!=15:
                    raise ValueError('CoVR-2 gallery provenance differs from the frozen COCO checkpoint.')
                precision='fp32' if self.device=='cpu' else 'fp16'
                if identity['precision']!=precision:
                    raise ValueError('CoVR-2 extraction precision differs from the query device.')
                rows=manifest['gallery_items']
                if len({r['item_id'] for r in rows})!=len(rows): raise ValueError('Duplicate gallery IDs.')
                metadata=[]
                for row in rows:
                    path=token_path(folder,row['item_id'])
                    if path.is_file():
                        try:
                            validate_tokens(np.load(path,mmap_mode='r',allow_pickle=False))
                        except (ValueError,OSError,EOFError) as error:
                            self.issues.append(f"{spec.label} / covr2 / {row['item_id']}: invalid cached tokens ({error})")
                            continue
                        metadata.append(_item_metadata(spec,row['item_id'],row,row.get('start') is not None))
                if not metadata: raise ValueError('No completed CoVR-2 gallery tokens.')
                self.galleries[('covr2',spec.key)]=dict(spec=spec,folder=folder,metadata=metadata,
                    identity=identity,effective_precision=precision,omitted=len(rows)-len(metadata),runtime_errors=set())
            except (ValueError,OSError,KeyError,TypeError) as e:
                self.issues.append(f'{spec.label} / covr2: {e}')

    @property
    def available_models(self):
        return self.text.available_models

    @property
    def composed_available(self):
        return any(m=='covr2' for m,_ in self.galleries)

    def dataset_choices(self,model):
        return [('All datasets','all')]+[(g['spec'].label,d) for (m,d),g in self.galleries.items() if m==model]

    def gallery_counts(self,model):
        return {d:len(g['metadata'])-len(g.get('runtime_errors',())) for (m,d),g in self.galleries.items() if m==model}

    def omitted_counts(self,model):
        return {d:g['omitted']+len(g.get('runtime_errors',())) for (m,d),g in self.galleries.items() if m==model}

    def _release_composed(self):
        self._encoder=None
        gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def warmup(self,model):
        with self._lock:
            if model!='covr2':
                self._release_composed()
                self.text.warmup(model)

    def search(self,*args,**kwargs):
        with self._lock:
            self._release_composed()
            return self.text.search(*args,**kwargs)

    def search_composed(self,image,text,dataset='all',source=None):
        if not isinstance(text,str) or not text.strip() or len(text)>2000:
            raise ValueError('Describe how the reference image should change.')
        with self._lock:
            galleries=[g for (m,d),g in self.galleries.items() if m=='covr2' and (dataset=='all' or d==dataset)]
            if not galleries: raise ValueError('Prepare CoVR-2 gallery tokens for this dataset first.')
            self.text.close()
            if self._encoder is None:
                self._encoder=self.encoder_loader(self.base,self.checkpoint,device=self.device)
            query,scoring=self._encoder.encode_query(image,text)
            def items():
                for gallery in galleries:
                    for row in gallery['metadata']:
                        try:
                            tokens=validate_tokens(np.load(token_path(gallery['folder'],row['item_id']),allow_pickle=False))
                        except (ValueError,OSError,EOFError):
                            gallery['runtime_errors'].add(row['item_id'])
                            continue
                        gallery['runtime_errors'].discard(row['item_id'])
                        yield row,tokens
            return rank_gallery(query,scoring,items(),source=source,limit=100)

    def resolve_result(self,row):
        if row.get('model')!='covr2': return self.text.resolve_result(row)
        g=self.galleries.get(('covr2',row.get('dataset')))
        found=next((r for r in g['metadata'] if r['item_id']==row.get('item_id')),None) if g else None
        if found is None: raise ValueError('Select a result from the current search.')
        return dict(found)

    def close(self):
        with self._lock:
            self._release_composed()
            self.text.close()
