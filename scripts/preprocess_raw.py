"""Reproduction utility only. Reuse existing processed/v2 for the deadline run."""
from collections import Counter
from pathlib import Path
import argparse
import pandas as pd
import pyarrow as pa
from urllib.parse import quote
from erfast.common import atomic_parquet,write_json
from erfast.text import record_from_row

p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);a=p.parse_args()
raw=Path(a.raw).resolve();out=Path(a.out).resolve()
if out.exists() and any(out.iterdir()):raise SystemExit('Output must be an empty new processed directory')
out.mkdir(parents=True,exist_ok=True);manifest=[]
for split in ['train','test']:
    for source in (['ground_truth'] if split=='train' else [])+['source1','source2','source3']:
        path=raw/split/f'{split}_{source}.tsv';counts=Counter();n=0;parts=0
        for part,df in enumerate(pd.read_csv(path,sep='\t',dtype=str,keep_default_na=True,chunksize=50000)):
            if source=='ground_truth':
                d=df.fillna('').to_dict('records')
                for r in d:
                    r['matched_entity_id_list']=[x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()]
                    r['match_count']=len(r['matched_entity_id_list']);r['is_singleton']=not bool(r['match_count'])
                atomic_parquet(out/split/source/f'part-{part:05d}.parquet',pa.Table.from_pylist(d))
            else:
                d=pd.DataFrame([record_from_row(r) for r in df.to_dict('records')])
                d['name_missing']=d.normalized_name.eq('');d['address_missing']=d.normalized_address.eq('')
                for country,g in d.groupby('country',dropna=False):
                    counts[str(country)]+=len(g)
                    atomic_parquet(out/split/source/f'country={quote(str(country),safe="")}'/f'part-{part:05d}.parquet',pa.Table.from_pandas(g,preserve_index=False))
            n+=len(df);parts+=1
        item={'file':path.name,'split':split,'source':source,'rows':n,'parts':parts}
        if source!='ground_truth':item['countries']=dict(counts)
        manifest.append(item);print(item,flush=True)
write_json(out/'_manifest.json',{'files':manifest,'chunk_size':50000,'max_rows_per_file':0})
