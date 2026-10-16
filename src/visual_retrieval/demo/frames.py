"""Capture the displayed video frame; keep source identity on the server."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
from PIL import Image

CAPTURE = r"""
const video = document.querySelector('#selected-preview video');
if (!video || video.readyState < 2 || !video.videoWidth) throw new Error('Wait for a video frame to load.');
video.pause();
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth; canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video,0,0);
const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/png'));
if (!blob) throw new Error('Could not capture this frame.');
const data = await new Promise(resolve => {const r=new FileReader();r.onload=()=>resolve(r.result);r.readAsDataURL(blob);});
const payload = {data,time:video.currentTime,src:video.currentSrc,token};
"""
COPY_JS = "async(token,ctx)=>{try{" + CAPTURE + r"""
try {await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);payload.copied=true;}
catch(e) {payload.copied=false;}
return [JSON.stringify(payload),ctx];
}catch(e){return [JSON.stringify({error:e.message}),ctx];}}
"""
USE_JS = "async(query,token,ctx)=>{try{" + CAPTURE + "return [query,JSON.stringify(payload),ctx];}catch(e){return [query,JSON.stringify({error:e.message}),ctx];}}"


def image_key(image):
    return hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()


def captured_frame(payload, ctx, media):
    data=json.loads(payload)
    if data.get('error'): raise ValueError(data['error'])
    item=ctx.get('frame_item')
    if not item or data.get('token')!=f"{ctx['revision']}:{ctx.get('page',0)}":
        raise ValueError('Select a video in the current results first.')
    expected=Path(media.video(item)).name
    if Path(unquote(urlsplit(data.get('src','')).path)).name!=expected:
        raise ValueError('Wait for the selected video to load before copying.')
    raw=data.get('data','')
    if not raw.startswith('data:image/png;base64,') or len(raw)>24_000_000:
        raise ValueError('Could not capture this frame.')
    try:
        image=Image.open(BytesIO(base64.b64decode(raw.split(',',1)[1],validate=True)))
        if image.format!='PNG' or image.width*image.height>4_000_000: raise ValueError('Frame is too large.')
        image=image.convert('RGB')
    except (OSError,TypeError) as e:
        raise ValueError('Invalid frame image.') from e
    time=data.get('time')
    if type(time) not in (int,float) or not np.isfinite(time) or time<0:
        raise ValueError('Invalid frame time.')
    start=item.get('start') or 0
    if item.get('end') is not None and time>item['end']-start+.1:
        raise ValueError('Frame is outside the selected segment.')
    key=image_key(image)
    ctx.setdefault('reference_sources',{})[key]=(item['dataset'],item['source_video_id'])
    # Bound session bookkeeping; recent copies remain available for paste.
    while len(ctx['reference_sources'])>20: del ctx['reference_sources'][next(iter(ctx['reference_sources']))]
    return image, key, start+time, data.get('copied',False)
