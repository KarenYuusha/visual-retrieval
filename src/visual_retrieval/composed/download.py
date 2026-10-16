"""Download the authors' released inference weights with checksum verification."""
import argparse
from pathlib import Path
import os
import urllib.request
from huggingface_hub import hf_hub_download
from .gallery import digest

BASE_URL='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_finetune_coco.pth'
BASE_SHA='053adc5b226a3d390aad3af8ceb2ecb51502cf1950775143a68a41e2163c8a9d'
QUERY_SHA='a440c917a08fb7dc265c3038aa8fc1643f053ff101e8b420d36e48df7db67a27'
REVISION='414666d2224255a157da06cf361ee6a7c90ac4a6'


def download(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    base=folder/'blip2_finetune_coco.pth'
    if not base.exists():
        temporary=base.with_suffix('.download')
        print('Downloading frozen BLIP-2 COCO weights (4.7 GB)…',flush=True)
        urllib.request.urlretrieve(BASE_URL,temporary)
        if digest(temporary)!=BASE_SHA: raise ValueError('BLIP-2 download checksum mismatch; retry the download.')
        os.replace(temporary,base)
    if digest(base)!=BASE_SHA: raise ValueError('BLIP-2 checkpoint checksum mismatch.')
    print('Downloading CoVR-2 query weights (6.0 GB; resumes through the Hub)…',flush=True)
    query=Path(hf_hub_download('lucas-ventura/CoVR2','coir+covr.ckpt',revision=REVISION,local_dir=folder))
    if digest(query)!=QUERY_SHA: raise ValueError('CoVR-2 checkpoint checksum mismatch.')
    print('Verified:',base,query,sep='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    download(parser.parse_args().output)

if __name__=='__main__':main()
