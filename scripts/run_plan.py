"""One final train or test plan. Parallel CPU lexical work and GPU embeddings.
No automatic rollback, alternate model, dropped route, or CPU fallback.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def cmd(*args):return [sys.executable,'-u','-m','erfast.cli',*args]


def run(*args):
    print('+',' '.join(cmd(*args)),flush=True);subprocess.run(cmd(*args),check=True)


def parallel_indexes(split):
    # Keep lexical logs separate. Terminate a still-running sibling if one stage fails;
    # completed artifacts remain in their new run folder and no alternate result is substituted.
    logs=Path('logs');logs.mkdir(exist_ok=True)
    with (logs/f'{split}-lexical.log').open('a') as log:
        lex=subprocess.Popen(cmd('lexical','--split',split),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for kind in ['targets','queries']:
                if lex.poll() not in (None,0):raise RuntimeError('Lexical indexing failed; see log')
                run('encode','--split',split,'--kind',kind)
            run('ann','--split',split)
            rc=lex.wait()
            if rc:raise RuntimeError('Lexical indexing failed; see log')
        finally:
            if lex.poll() is None:
                os.killpg(lex.pid,signal.SIGTERM);lex.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','test']);p.add_argument('--accept-audit',action='store_true');a=p.parse_args()
    run('preflight')
    if a.stage=='train':
        run('prepare');run('prepare-model');parallel_indexes('train')
        run('mine','--split','train');run('fit');run('calibrate');run('audit')
        print('Training and audit complete. Read models/audit.json before starting test.',flush=True)
    else:
        if not a.accept_audit:p.error('Use --accept-audit only after reviewing the actual audit.')
        run('accept','--yes');parallel_indexes('test');run('mine','--split','test')
        run('predict');run('export');run('package')


if __name__=='__main__':main()
