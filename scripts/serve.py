#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fastapi import FastAPI
import uvicorn

from open_system_one.schema import DecisionRequest, DecisionResponse, SystemOneRequest, SystemOneResponse
from open_system_one.serving import load_engine


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--device",default=None)
    args=p.parse_args()
    engine=load_engine(args.checkpoint,args.device)
    app=FastAPI(title="Open System One",version="0.1.0")

    @app.get("/health")
    def health(): return {"ok":True,"device":engine.device,"temperature":engine.temperature}

    @app.post("/v1/decision",response_model=DecisionResponse)
    def decision(req:DecisionRequest): return engine.predict(req)

    @app.post("/v1/systemone",response_model=SystemOneResponse)
    def systemone(req:SystemOneRequest): return engine.system_one(req)

    uvicorn.run(app,host=args.host,port=args.port)


if __name__=="__main__": main()
