# Digitised literature data

Numbers read from published papers, committed so every result in `docs/SOURCES.md` can be
rerun. Each file's header gives the source, the figure or table, how it was read, and its
accuracy. Where a value was read off a figure rather than a table, the file says so.

| file | source | SOURCES |
|---|---|---|
| `grift2020_cd_vs_depth.csv` | Grift (2020) thesis Fig. 2.4 — plate C_D against immersion depth | §159 |
| `grift2020_entrainment_rate.csv` | Grift (2020) thesis Fig. 2.11c — wake entrainment rate against acceleration | §159 |
| `kleshnev2005_onwater_vs_machines.csv` | Kleshnev (2005) ISBS, Table 1 — on-water single against two machines | §159 |

Not here, deliberately:
- **Commercial or private data** — [BR24] (BioRow), the coxing race recordings, cox-box
  exports and transcripts — live in the gitignored `data/local/`. Only derived numbers
  enter the repository.
- **The papers themselves** (copyright) — downloaded copies sit in `data/local/literature/`.
