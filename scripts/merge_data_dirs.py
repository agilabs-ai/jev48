#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from openjev.io import read_jsonl


def main():
    p=argparse.ArgumentParser(); p.add_argument("--inputs",nargs="+",required=True); p.add_argument("--output",required=True); args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for split in ["train","dev","calibration","test","ood"]:
        rows=[]
        for directory in map(Path,args.inputs):
            path=directory/f"{split}.jsonl"
            if path.exists(): rows.extend(read_jsonl(path))
        ids=[r.id for r in rows]
        if len(ids)!=len(set(ids)): raise ValueError(f"duplicate IDs after merge in {split}")
        (out/f"{split}.jsonl").write_text("".join(r.model_dump_json()+"\n" for r in rows),encoding="utf-8")
        print(split,len(rows))

if __name__=="__main__": main()
