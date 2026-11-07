# Rating site for outside raters

A small self-hosted site where invited raters (other coxswains, rowers) rate coxswain calls: up to 15 phrases a day
on four questions, then 10 muffled voice clips, then the day closes. Design and analysis are fixed in the study's
codebook (addendum of 7 Nov 2026). Python 3.9+, standard library only, one SQLite file.

| file | what |
|---|---|
| `pack.py` | builds `pack.json` and `clips/` from the human coding sample; names redacted, with a review list |
| `app.py` | the server, plus `invite` and `raters` commands |
| `page.html`, `guide.html` | the rater's page and the short "how to rate" guide |
| `admin.html` | the study lead's dashboard: raters, coverage, check items, Krippendorff's alpha, export |
| `analyse.py` | the full agreement analysis on a downloaded `ratings.db` |

The data folder (`pack.json`, `clips/`, `ratings.db`) holds crew speech and stays private: it is never committed to
this repository.

## Before sharing any link

1. **Names.** Run `pack.py`, open `data/local/coxing/rating_site/names_review.txt`, and mark every line `name` (to be
   replaced by "[name]") or `keep`. Re-run with `--final`; it refuses while any line is unmarked. Then skim a few
   dozen phrases in `pack.json`: words that are both names and ordinary words can slip through.
2. **Crews' consent.** Outside raters will read what was said in the boat (without names). Make sure the crews are
   comfortable with that.

## Running it locally (to try it)

```
python research/coxing/rating_site/app.py invite "me" --admin
python research/coxing/rating_site/app.py serve
```
Open the printed link: it signs the browser in. `/admin` is the dashboard (admins only).

## Deploying

Any machine that can run Python and keep a file between restarts works. Two simple options:

**A small server (VPS) with Caddy for HTTPS.** Copy the code files and the data folder to the server, e.g.
`/srv/ratings/code` and `/srv/ratings/data`. Run it as a service:
```
RATING_DATA=/srv/ratings/data RATING_SECURE=1 RATING_TZ_HOURS=-8 \
  python3 /srv/ratings/code/app.py serve --host 127.0.0.1 --port 8080
```
and point Caddy at it (`/etc/caddy/Caddyfile`):
```
ratings.example.org {
    reverse_proxy 127.0.0.1:8080
}
```
Caddy obtains the HTTPS certificate itself.

**Your own computer behind a tunnel** (Cloudflare Tunnel or Tailscale Funnel): run `serve` locally with
`RATING_SECURE=1` and expose port 8080 through the tunnel, which provides HTTPS. Simple, but the site is down
whenever the computer is off.

Whichever you use, set `RATING_SECURE=1` only when the site is served over HTTPS (the sign-in cookie is then never
sent in the clear), and set `RATING_TZ_HOURS` to the time zone that should define "a day" (-8 is Pacific standard
time).

## Inviting raters

On the machine where the site runs (the invite is written to its database):
```
python3 app.py invite "label for you only" --base-url https://ratings.example.org
python3 app.py invite "label" --exclude race_07 --base-url https://ratings.example.org
```
Each command prints one personal link; send it to that rater only. `--exclude` takes the sessions the rater coxed
or rowed in, so they never rate their own race. The label is only shown to admins. `raters` lists everyone.

## Data and analysis

Ratings are append-only: raters cannot change or delete what they saved. Back up `ratings.db` regularly (copy the
file). For the analysis, copy it into `data/local/coxing/rating_site/` and run `analyse.py`, or download all
ratings as JSON from the dashboard. To delete a rater's answers at their request, remove their rows from the
`ratings` table and note it in the study log.
