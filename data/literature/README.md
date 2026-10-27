# Digitised literature data

Numbers read from published papers, committed so every result in `docs/SOURCES.md` can be
rerun. Each file's header gives the source, the figure or table, how it was read, and its
accuracy. Where a value was read off a figure rather than a table, the file says so.

| file | source | SOURCES |
|---|---|---|
| `grift2020_cd_vs_depth.csv` | Grift (2020) thesis Fig. 2.4 — plate C_D against immersion depth | §159 |
| `grift2020_entrainment_rate.csv` | Grift (2020) thesis Fig. 2.11c — wake entrainment rate against acceleration | §159 |
| `cr06_fig3_measured.csv` | Cabrera, Ruina & Kleshnev (2006) Fig. 3 — one women's single's measured boat velocity, handle force, leg and back displacement and oar angle; extracted from the open preprint's vector figure by `research/cr06/extract_fig3.py` | §§45–60 |
| `kleshnev2005_onwater_vs_machines.csv` | Kleshnev (2005) ISBS, Table 1 — on-water single against two machines | §159 |
| `buckeridge2015_foot_force.csv` | Buckeridge, Bull & McGregor (2015) Scand J Med Sci Sports, Tables 3–6 — twenty elite women on an instrumented ergometer: foot force ratio at maximum handle force, average resultant, footplate angle, asymmetries | §177 |
| `vieira2017_cop.csv` | Vieira et al. (2017) PLoS ONE (CC-BY), Figs 3–4 — footplate centre of pressure (mm from the heel) through one rower's drive, and the group's mean at peak force and over the drive; digitised by `research/cr06/extract_vieira2017.py` | §177 |
| `grift2021_kinematics.csv`, `grift2021_forces.csv` | Grift, Tummers & Westerweel (2021) JFM 918 A29 (open access), Figs 4(b), 7(b), 9(a) — a 1:2 sweep blade's oar angle and tip path (an on-water M4− stroke, robot-reproduced) and the measured F_n, F_t (three realisations); extracted from the vector figures by `research/blade/extract_grift2021.py`, axes from the printed tick labels | §184 |
| `robert2019_trial154.csv` | Robert, Leroyer, Barré, Queutey & Visonneau (2019) J. Mar. Sci. Technol. 24 (CC BY 4.0, HAL hal-02394848), Figs 3 and 7 — the imposed oar angular velocity against sweep angle, and trial 154's measured F_x, F_y, F_z, M_z of a full-size blade on a bending oar; extracted from the vector figures by `research/blade/extract_robert2019.py`, axes from the gridlines and printed tick values | §185 |

Not here, deliberately:
- **Commercial or private data** — [BR24] (BioRow), the coxing race recordings, cox-box
  exports and transcripts — live in the gitignored `data/local/`. Only derived numbers
  enter the repository.
- **The papers themselves** (copyright) — downloaded copies sit in `data/local/literature/`.
