# Rating site for outside raters (GitHub Pages + Google Sheet)

Invited rowers and coxswains rate coxswain calls on a static site: their name and a short profile on first visit, then
sets of up to 15 calls on four questions and 10 muffled voice clips, up to three sets a day. Design and analysis are
fixed in the study's codebook (addendum of 7 Nov 2026 and its amendments of 7 and 8 Nov).

| file | what |
|---|---|
| `build_site.py` | encrypts every phrase of the 15 pieces (names redacted) and a muffled clip per call into `site_build/` |
| `web/index.html`, `web/app.js`, `web/guide.html` | the site (static; decrypts in the browser with the key in the link) |
| `web/Code.gs` | the Google Apps Script backend: raters, daily batches, ratings in your Sheet |
| `web/test.html` | local test harness that runs the real `Code.gs` with stand-ins for Google's services; never deployed |
| `analyse.py` | the agreement analysis on the ratings the admin view downloads |

**How it stays private with a just-send-a-link workflow.** Phrases and clips are published only encrypted
(AES-256-GCM). The key is in the link after `#`, which browsers never send to GitHub, Google or any log, so the files on
GitHub are unreadable without the link. The Apps Script accepts ratings only with a token derived from the key. The
public `manifest.json` holds only opaque codes. Limits: anyone you forward a link to can read and rate; usernames are
not verified (the admin view shows when one name is used from several devices). If a link leaks, rebuild with
`--rotate` and send new links: old links stop working.

## One-time setup

1. **Review names.** Run `.venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py`, open
   `data/local/coxing/rating_site/names_review.txt` and mark each `?` line `name` (replaced by "[name]") or `keep`.
   Repeat until it reports none left. Get the crews' OK for outside raters reading their calls (without names).
2. **Build the shareable site** with a fresh key:
   `.venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py --final --rotate`.
   It writes the rater link, your admin link, the three script properties and the race codes to
   `data/local/coxing/rating_site/SETUP_PRIVATE.md` (the key itself is in `site_key.json`).
3. **Google Sheet and script.** Create a new Google Sheet. Extensions > Apps Script. Replace the editor's contents
   with `site_build/Code.gs`. Project settings: set the time zone (it decides when a rater's day starts) and add the
   script properties `WRITE_TOKEN`, `ADMIN_KEY` and `MANIFEST_URL` from step 2. Deploy > New deployment > Web app,
   Execute as: Me, Who has access: Anyone. Authorise it, and copy the web app URL.
4. **Point the site at the script and publish:**
   `build_site.py --final --script-url <web app URL>` (no `--rotate`: the key stays; unchanged encrypted files are
   kept, so only what changed is published), then in `data/local/coxing/rating_site/site_build` run `git add -A`,
   `git commit -m "Update"` and `git push origin main` (one at a time in Windows PowerShell 5, which has no `&&`).
   GitHub Pages serves it at https://josh-porg.github.io/coxing-ratings/ within a minute or two.
5. **Send the rater link** (one link for everyone; each person gives their name). Keep the admin link to yourself.

**After changing `web/Code.gs`**: paste the new `site_build/Code.gs` into the Apps Script editor, save, then Deploy >
Manage deployments > edit (pencil) > Version: New version > Deploy. Editing the existing deployment keeps the web app
URL; a new deployment would change it.

## Running it

Raters sign up once per device with their name and profile, and can edit the profile from "my details". To keep a
rater away from races they were in, type the race codes (table in `SETUP_PRIVATE.md`) into their `own_pieces` cell in
the Sheet's `raters` tab; it applies from their next set, and the analysis drops any earlier ratings of those races.
The admin link shows raters, sets, coverage, check-item accuracy and Krippendorff's alpha, and downloads all ratings
as JSON.
For the full analysis: `.venv/Scripts/python.exe research/coxing/rating_site/analyse.py ratings_<day>.json`
(it also reports check accuracy by set of the day, for fatigue).
To delete a rater's answers at their request, delete their rows in the Sheet's `ratings` tab and note it in the study log.

## Testing locally

`build_site.py` without `--final` writes `site_build_test/` (with `test.html`) under a test key of its own
(`site_key_test.json`, never the live key) and prints the harness link. Serve that folder
(`python -m http.server 8768`) and open the harness link: the real `Code.gs` runs in the page with in-memory sheets, with buttons for the next day and the admin view.
