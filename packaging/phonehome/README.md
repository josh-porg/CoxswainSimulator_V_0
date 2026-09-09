# Performance reports

"Too slow" is all a tester can tell you, because the numbers are in a
log file on their machine. With reports on, the trainer sends a summary
of that log home at the end of each session, and your spreadsheet
fills with rows like

| renderer | tier | p50 ms | p95 ms | stalls | dropped s |
|---|---|---|---|---|---|
| Intel(R) UHD Graphics | minimal | 19.9 | 25.0 | 0 | 0.00 |

which is enough to tell an integrated part from a driver picking the
wrong GPU from a physics stall.

## What is in a report

Numbers and product names only: version, OS family, CPU count and RAM,
GPU adapter names and the live renderer string, the tier, the settings
the run used (course, boat, quality, physics rate, window size), world
build time, frame-time percentiles, stall and dropped-time counts, and
the *count* of exceptions. Never a user name, home folder, file path or
e-mail — the sender refuses to send if it finds one
(`coxswain/viz/phonehome.py`, `scrub`), and `tests/test_phonehome.py`
holds it to that.

## Consent

Off until switched on: **Send performance reports** in the setup menu
(Graphics and sound). The choice is remembered in the per-user
`settings.json` beside the logs. Whatever happens — sent, refused, no
URL, failed, off — is written into the diagnostics log, so a tester can
see exactly what left their machine.

## The endpoint: a Google Sheet, no server

1. Create a spreadsheet, open *Extensions → Apps Script*, paste
   `Code.gs`, save.
2. *Deploy → New deployment → Web app*, execute as **Me**, access
   **Anyone**. Copy the URL.
3. Give the trainer the URL, any one of:
   - a file named `report_url.txt` beside `Coxswain.exe` (or the app),
     containing the URL — ship it inside the zip;
   - the environment variable `COXSWAIN_REPORT_URL`;
   - `--report-url https://...` on the command line.

Nothing secret is involved. The URL is the handshake; the worst anyone
holding it can do is append a row.

With no URL set, a session with reports on writes
`report-<stamp>.json` beside the diagnostics log and prints its path,
so a tester can paste it to you instead.

## Reading the sheet

- **p50 well above 16.7 ms on `ultra`** — the machine is below the floor;
  ask for the `renderer` column first: `Microsoft Basic Render Driver`
  means no GPU driver at all.
- **`adapters` lists a dedicated card but `renderer` is the integrated
  one** — Windows chose the wrong GPU; `--prefer-dedicated-gpu` writes
  the per-user preference.
- **`dropped_s` above zero** — the physics could not keep up and the
  loop let time go rather than freeze; the physics rate for that tier
  is too high for that CPU.
- **`stalls` in the tens** — look at the diagnostics log for what each
  stall was doing; the log names the context.
