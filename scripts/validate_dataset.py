#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from openjev.io import read_jsonl


def main():
    p=argparse.ArgumentParser(); p.add_argument("data_dir"); args=p.parse_args()
    d=Path(args.data_dir)
    splits={}
    for name in ["train","dev","calibration","test","ood"]:
        path=d/f"{name}.jsonl"
        if path.exists(): splits[name]=read_jsonl(path)
    ids={}; families={}; fingerprints={}; issues=[]
    for split,rows in splits.items():
        for ex in rows:
            if ex.id in ids: issues.append(f"duplicate id {ex.id}: {ids[ex.id]} and {split}")
            ids[ex.id]=split
            if ex.family_id in families and families[ex.family_id]!=split:
                issues.append(f"family leakage {ex.family_id}: {families[ex.family_id]} and {split}")
            families[ex.family_id]=split
            fp=hashlib.sha256((ex.state+'\n'+ex.question+'\n'+'\n'.join(c.text for c in ex.candidates)).encode()).hexdigest()
            if fp in fingerprints and fingerprints[fp]!=split:
                issues.append(f"exact content leakage {ex.id}: {fingerprints[fp]} and {split}")
            fingerprints[fp]=split
    if issues:
        print("FAILED")
        for x in issues[:50]: print("-",x)
        raise SystemExit(1)
    print({"ok":True,"splits":{k:len(v) for k,v in splits.items()},"unique_ids":len(ids),"unique_families":len(families)})


if __name__=="__main__": main()
