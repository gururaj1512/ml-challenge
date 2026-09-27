from __future__ import annotations
import argparse
import json
import os
import platform
from .common import Config,bind_run


def preflight(cfg):
    import importlib.metadata as im
    import psutil
    import torch
    import faiss
    from .wand import library
    print('Machine:',platform.machine(),'RAM GiB:',round(psutil.virtual_memory().total/2**30,1),'CPU threads:',os.cpu_count())
    if platform.system()!='Linux':raise RuntimeError('This execution package targets Linux EC2, not the Mac environment.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. No silent CPU fallback.')
    for gid in cfg['embedding']['gpu_ids']:
        if gid>=torch.cuda.device_count():raise RuntimeError('Configured GPU count exceeds hardware')
        print('GPU',gid,torch.cuda.get_device_name(gid))
    if not hasattr(faiss,'StandardGpuResources'):raise RuntimeError('Need conda faiss-gpu, not pip faiss-cpu')
    if psutil.virtual_memory().total<300*2**30:raise RuntimeError('Full-vocabulary run configuration requires a 384-GiB-class host (>=300 GiB visible RAM).')
    library()
    print('Native WAND loaded. Required source files must be in',cfg.processed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.json')
    p.add_argument('command',choices=['preflight','prepare','prepare-model','lexical','encode','ann','mine','fit','calibrate','audit','accept','predict','export','package'])
    p.add_argument('--split',choices=['train','test']);p.add_argument('--kind',choices=['targets','queries']);p.add_argument('--yes',action='store_true')
    a=p.parse_args();cfg=Config(a.config);bind_run(cfg)
    if a.command=='preflight':return preflight(cfg)
    if a.command=='prepare':
        from .data import prepare
        return prepare(cfg)
    if a.command=='prepare-model':
        from .embeddings import prepare_model
        return prepare_model(cfg)
    if a.command in ('lexical','encode','ann','mine'):
        if not a.split:p.error('--split required')
        if a.command=='lexical':
            from .lexical import build
            return build(cfg,a.split)
        if a.command=='encode':
            if not a.kind:p.error('--kind required')
            from .embeddings import encode
            return encode(cfg,a.split,a.kind)
        if a.command=='ann':
            from .embeddings import ann
            return ann(cfg,a.split)
        if a.command=='mine':
            if a.split=='test':
                from .output import deployment
                deployment(cfg)
            from .pairs import mine
            return mine(cfg,a.split)
    if a.command in ('fit','calibrate','audit','accept'):
        from . import training
        if a.command=='accept' and not a.yes:p.error('Review audit.json, then use accept --yes')
        return getattr(training,a.command)(cfg)
    from . import output
    return getattr(output,a.command)(cfg)


if __name__=='__main__':main()
