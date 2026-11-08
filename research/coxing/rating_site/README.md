# Rating site for outside raters (GitHub Pages + Google Sheet)

Invited rowers and coxswains rate coxswain calls on a static site: a username on first visit, then each day up to 15
calls on four questions and 10 muffled voice clips. Design and analysis are fixed in the study's codebook (addendum
of 7 Nov 2026 and its hosting amendment).

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
not verified (the admin view shows when one username is used from several devices). If a link leaks, rebuild with
`--rotate` and send new links: old links stop working.

## One-time setup

1. **Review names.** Run `.venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py`, open
   `data/local/coxing/rating_site/names_review.txt` and mark each `?` line `name` (replaced by "[name]") or `keep`.
   Repeat until it reports none left. Get the crews' OK for outside raters reading their calls (without names).
2. **Build the shareable site** with a fresh key:
   `.venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py --final --rotate`.
   It prints the rater link, your admin link, and three script properties (also saved in `site_key.json`).
3. **Google Sheet and script.** Create a new Google Sheet. Extensions > Apps Script. Replace the editor's contents
   with `site_build/Code.gs`. Project settings: set the time zone (it decides when a rater's day starts) and add the
   script properties `WRITE_TOKEN`, `ADMIN_KEY` and `MANIFEST_URL` from step 2. Deploy > New deployment > Web app,
   Execute as: Me, Who has access: Anyone. Authorise it, and copy the web app URL.
4. **Point the site at the script and publish:**
   `build_site.py --final --script-url <web app URL>` (no `--rotate`: the key stays), then
   `cd data/local/coxing/rating_site/site_build && git add -A && git commit -m "Update data" && git push`.
   GitHub Pages serves it at https://josh-porg.github.io/coxing-ratings/ within a minute or two.
5. **Send the rater link** (one link for everyone; each person picks a username). Keep the admin link to yourself.

## Running it

Raters sign up once per device with a username, profile and the races they were in (excluded from their queue). The
admin link shows raters, coverage, check-item accuracy and Krippendorff's alpha, and downloads all ratings as JSON.
For the full analysis: `.venv/Scripts/python.exe research/coxing/rating_site/analyse.py ratings_<day>.json`.
To delete a rater's answers at their request, delete their rows in the Sheet's `ratings` tab and note it in the study log.

## Testing locally

`build_site.py` without `--final` writes `site_build_test/` (with `test.html`). Serve that folder
(`python -m http.server 8768`) and open `http://127.0.0.1:8768/test.html#k=<key>&admin=<admin key>` (both from
`site_key.json`): the real `Code.gs` runs in the page with in-memory sheets, with buttons for the next day and the admin view.
