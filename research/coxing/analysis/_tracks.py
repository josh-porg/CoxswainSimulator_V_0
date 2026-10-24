import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import struct, os
def parse(f,end):
    out=[]
    while f.tell()<end-8:
        st=f.tell(); h=f.read(8)
        if len(h)<8: break
        sz,t=struct.unpack(">I4s",h); t=t.decode("latin1")
        if sz==1: sz=struct.unpack(">Q",f.read(8))[0]
        if sz<8: break
        n={"type":t,"start":st,"size":sz,"kids":[]}
        if t in ("moov","trak","mdia","minf","stbl"):
            f.seek(st+8); n["kids"]=parse(f,st+sz)
        out.append(n); f.seek(st+sz)
    return out
def find(ns,t):
    for n in ns:
        if n["type"]==t: yield n
        yield from find(n["kids"],t)
def track_samples(P, ti):
    f=open(P,"rb"); top=parse(f,os.path.getsize(P))
    trak=list(find(top,"trak"))[ti]
    g=lambda t:(list(find([trak],t)) or [None])[0]
    f.seek(g("stsz")["start"]+8); _,ssz,cnt=struct.unpack(">III",f.read(12))
    sizes=[ssz]*cnt if ssz else list(struct.unpack(">%dI"%cnt,f.read(4*cnt)))
    sc=g("stsc"); f.seek(sc["start"]+8); _,ne=struct.unpack(">II",f.read(8))
    ent=[struct.unpack(">III",f.read(12)) for _ in range(ne)]
    co=g("stco") or g("co64"); f.seek(co["start"]+8); _,nc=struct.unpack(">II",f.read(8))
    offs=list(struct.unpack(">%d%s"%(nc,"I" if co["type"]=="stco" else "Q"),
                            f.read((4 if co["type"]=="stco" else 8)*nc)))
    spc=[]
    for i,(first,n,_) in enumerate(ent):
        last=ent[i+1][0]-1 if i+1<len(ent) else nc
        for ch in range(first,last+1): spc.append(n)
    out=[]; si=0
    for ci,o in enumerate(offs):
        n=spc[ci] if ci<len(spc) else spc[-1]; cur=o
        for _ in range(n):
            if si>=len(sizes): break
            f.seek(cur); out.append(f.read(sizes[si])); cur+=sizes[si]; si+=1
    f.close(); return out
