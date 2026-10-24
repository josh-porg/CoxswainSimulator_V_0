"""Write the 6 October update into the forecast page (data/local/artifacts/mid_morning_eight_forecast.html)
from charles_update_2026.log, so every number on the page is the script's."""
import html
import os
import re

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "crew", "feeders"))   # its data and outputs (gitignored)
PAGE = os.path.join(HERE, "..", "..", "artifacts", "mid_morning_eight_forecast.html")
log = open(os.path.join(HERE, "charles_update_2026.log"), encoding="utf-8").read()


def field_rows(title_start):
    block = log[log.index(title_start):]
    block = block[:block.index("THE SQUAD")]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\s+(\d+)\s+(.+?)\s{2,}(2025 Charles|TOTL 2026|new|ours)\s+(\S+)\s+(\d+)\s+P\(top \d+\)\s+(\d+)%", line)
        if m:
            rows.append(m.groups())
    return rows


def table(rows, nq):
    label = {"2025 Charles": "2025 Charles result", "TOTL 2026": "Tail of the Lake 2026", "new": "new to the event",
             "ours": "Tail of the Lake + CoxBox"}
    out = []
    for i, (bow, crew, ev, t, med, p) in enumerate(rows):
        cls = ' class="pick"' if ev == "ours" else ""
        crew = "Sammamish (the squad)" if ev == "ours" else crew
        out.append('<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s%%</td></tr>'
                   % (cls, html.escape(crew), bow, label[ev], "—" if t == "-" else t, med, p))
    return "\n".join(out)


four = field_rows("WOMEN'S VETERAN FOURS")
eight = field_rows("WOMEN'S GRAND MASTER EIGHTS")

UPDATE = """
  <div class="headline">
    <div class="kv"><span class="k">The eight, updated 6 Oct</span><span class="big">18:26</span></div>
    <div class="kv"><span class="k">Likely place</span><span class="v">11th of 31</span></div>
    <div class="kv"><span class="k">Requalify (top 15)</span><span class="v">87%</span></div>
    <div class="kv"><span class="k">Top 10</span><span class="v">47%</span></div>
    <p class="sub">The 2026 entries are out and Tail of the Lake has been raced. The eight holds its
    line; the four, at 22:17–22:41 and about 22nd of 26, has about a 1% chance of the top 13.</p>
  </div>

  <section class="sec" id="update-6-oct">
    <p class="eyebrow">Update · 6 October 2026</p>
    <h2>Ten days out: the real fields and a fresh feeder</h2>
    <p>Two things changed. The Charles published its 2026 entries — <strong>31 Grand Master Eights
    and 26 Veteran Fours</strong> — so the placing odds no longer borrow last year's field: every
    crew that is actually racing is in the simulation, by name. And <strong>Tail of the Lake</strong>
    was raced on 4 October, with the four's own CoxBox running, which measures what the four is
    rowing now rather than what its ergs said in September.</p>

    <div class="tablewrap">
      <table>
        <thead><tr><th>Outcome</th><th>The eight</th><th>The four</th></tr></thead>
        <tbody>
          <tr><td>Forecast time</td><td>18:26 <span style="color:var(--muted)">(±14 s)</span></td><td>22:17 – 22:41</td></tr>
          <tr><td>Median place</td><td>11th of 31</td><td>21st – 23rd of 26</td></tr>
          <tr><td>Middle 80% of places</td><td>6th – 16th</td><td>16th – 25th</td></tr>
          <tr class="pick"><td>Guaranteed entry next year (top 50%)</td><td>87%</td><td class="mono" style="color:var(--warn)">≈ 1% <span style="color:var(--muted)">(0–4%)</span></td></tr>
          <tr><td>Top 10</td><td>47%</td><td>≈ 1%</td></tr>
          <tr><td>Top 5</td><td>5%</td><td>&lt;0.1%</td></tr>
          <tr><td>Medal</td><td>0.6%</td><td>&lt;0.1%</td></tr>
          <tr><td style="color:var(--muted)">18 September said</td><td style="color:var(--muted)">18:32, 9th–10th, &gt;99%</td><td style="color:var(--muted)">21:40, 13th, 64%</td></tr>
        </tbody>
      </table>
    </div>
    <figcaption>Each crew's time is drawn 40,000 times: the river shared by everyone (±1.3%), each
    crew's own autumn-to-autumn change (±2.4%) or its measured uncertainty, and the squad's
    race-day spread. Requalification is the top half of the finishers, rounded down.</figcaption>

    <h3 style="margin-top:28px">Why the eight's odds moved, though its time did not</h3>
    <p>The eight's time barely changed: 18:26, against 18:32–18:37 before. What changed is the
    company. The 2026 field is three boats smaller than 2025's, so the top half is 15 boats rather
    than 17, and nearly every crew that finished ahead of the squad last year is back. Bows 1 to 18
    are last year's finishing order, and the squad is bow 7 — eighth last year, with Toronto (1st
    in 2025) not returning. Requalifying at 87% means finishing ahead of most of the eleven crews
    from 9th to 19th last year, and the forecast puts the squad in the middle of that pack.</p>
    <p><strong>An assumption to check:</strong> the Charles eight is taken to be the squad's fastest
    women's eight at Tail of the Lake (15:09.3, average age 56). If the Charles crew raced as a
    different eight there, its Tail of the Lake reading changes, though the erg-based half of the
    forecast does not.</p>

    <h3 style="margin-top:28px">Why the four's forecast fell by a minute</h3>
    <p>The September number came from the four's erg average multiplied by the
    <em>eight's</em> rate of exchange between ergs and boat speed, scaled by a boat factor — one
    calibration point, and not the four's own. Pink Ribbon Row first suggested that rate was
    3–4% too kind, and Tail of the Lake now measures it twice over:</p>

    <div class="tablewrap">
      <table>
        <thead><tr><th>Route to a Charles time for the four</th><th>Charles</th><th>±</th></tr></thead>
        <tbody>
          <tr><td>Tail of the Lake 18:19.7, through the Pacific Northwest fours that raced both Tail of the Lake 2025 and the Charles 2025</td><td>22:17</td><td>31 s</td></tr>
          <tr><td>Tail of the Lake on the four's CoxBox: 2:22.4 per 500 over 3,834 m</td><td>22:35</td><td>15 s</td></tr>
          <tr><td>Pink Ribbon Row, 2:24.0 average over 4,000 m</td><td>22:47</td><td>15 s</td></tr>
          <tr><td>Pink Ribbon Row, 2:22.0 sustained</td><td>22:28</td><td>15 s</td></tr>
          <tr style="color:var(--muted)"><td>18 September, ergs × the eight's rate</td><td>21:40</td><td>17 s</td></tr>
        </tbody>
      </table>
    </div>
    <figcaption>The two Tail of the Lake routes are independent: one goes through other crews' times
    at both regattas, the other through the four's own speed and the flat-water-to-Charles ratio
    from the squad's eights. They agree to 24 seconds. The September number is a minute off both.</figcaption>

    <p><strong>The second route checked on a four.</strong> Its ratio (a crew's Charles time is about
    0.978 of its Pink Ribbon Row time, per metre) came from the squad's eights. One 60+ four raced
    both in 2025: Martha's Moms, 22:12.1 at Pink Ribbon Row and 21:46.4 at the Charles — 0.981. The
    ratio holds for a four, which makes the CoxBox route (22:35–22:41, median place 23rd, top 13
    under 1%) the better-founded of the two, and the cross-crew route (22:17, 21st, 4%) the
    optimistic bound.</p>

    <p><strong>How the first route works, and the trap in it.</strong> Five Pacific Northwest 60+ fours
    raced both Tail of the Lake 2025 and the 2025 Charles Veteran Fours; their Charles times were
    1.105× their Tail of the Lake times (±1.8%). But 2025 at Tail of the Lake was a rough day and
    2026 a quick one, so a 2026 time has to be taken back to 2025's conditions first — and
    <strong>the rough water cost slow boats more</strong>. The same scullers were 17% slower in 2025;
    the same doubles 3–8%; the same quads and fours 0–11%, the slower the crew the bigger the loss.
    Fitted on the crews and doubles that raced both years, the conditions index runs from 0.96 for a
    16-minute crew to 0.91 for a 20-minute one. Using one index for every boat made the route say
    21:13 and put two new Pacific Northwest fours ahead of last year's winner; using the index for a
    boat of the four's speed brings it to 22:17, beside the other route.</p>

    <div class="note">
      <p style="margin:0"><strong>What the four would need.</strong> Even odds of the top 13 sit at
      about <strong>21:16</strong> on the Charles — about 2:14–2:15 per 500 on flat water, against
      the 2:22.4 measured at Tail of the Lake. Seven to eight seconds per 500 is not what a ten-day
      taper finds. The realistic race for the four is with the crews around it: Three Rivers
      (22:14 last year), Martha's Moms if its Charles crew is the slower one, and the fourteen
      crews new to the event, whose speed nobody yet knows.</p>
    </div>

    <h3 style="margin-top:28px">The 2026 Grand Master Eights</h3>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Crew</th><th>Bow</th><th>Evidence</th><th>Time used</th><th>Median place</th><th>P(top 15)</th></tr></thead>
        <tbody>
%(eight)s
        </tbody>
      </table>
    </div>
    <figcaption>A crew new to the event is drawn from the whole 2025 field. Pocock's eight (average 50)
    is the one other crew with a 2026 time, from Tail of the Lake.</figcaption>

    <h3 style="margin-top:28px">The 2026 Veteran Fours</h3>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Crew</th><th>Bow</th><th>Evidence</th><th>Time used</th><th>Median place</th><th>P(top 13)</th></tr></thead>
        <tbody>
%(four)s
        </tbody>
      </table>
    </div>
    <figcaption>Shown with the four at the cross-crew route's 22:17; on the CoxBox route (22:41) its
    median place is 23rd and the top-13 chance is under 1%. Conibear's and Mt. Baker's 2026 crews
    (both averaging 60–61 at Tail of the Lake) come out near last year's winner, which is why the
    top half is harder to reach than last year's cut suggests.</figcaption>

    <h3 style="margin-top:28px">The feeder scan, and what it could not reach</h3>
    <ul>
      <li><strong>Tail of the Lake 2026 and 2025</strong> (HereNow): the four, the eight, Pocock,
      Conibear and Mt. Baker in 2026; the five 60+ fours and the squad's eight that calibrate it
      in 2025.</li>
      <li><strong>Textile River 2026</strong> (Lowell, 4 Oct): no crew entered in either event raced
      at the right age — Riverside's eight there averaged 49.</li>
      <li><strong>Quinsigamond Snake 2026</strong>: collegiate events only.</li>
      <li><strong>Pink Ribbon Row 2026 and 2025</strong> (CrewTimer): the four's 19:08.4 over the new
      4,000 m course; the eights, where Pocock was 4.5 s quicker than the squad's eight (at Tail of
      the Lake it was 18 s slower — the model has them close); and the 2025 check on the four's
      conversion above. None of the other entered 60+ fours raced there in 2026.</li>
      <li>Most East Coast and Midwest masters crews race their feeders after the Charles. For them
      the 2025 Charles is the only evidence, and that is the largest remaining uncertainty in the
      placing odds.</li>
    </ul>
  </section>

  <hr class="rule">
  <p class="eyebrow" style="margin-top:36px">The 18 September analysis, kept as it was</p>
"""

page = open(PAGE, encoding="utf-8").read()
start = page.index('  <div class="headline">')
end = page.index("  <section class=\"sec\">\n    <h2>The two things that move the time</h2>")
old_headline = page[start:end]
kept = old_headline.replace('<div class="headline">', '<div class="headline" style="opacity:.75">', 1)
new = UPDATE.replace("%(eight)s", table(eight, 15)).replace("%(four)s", table(four, 13))
page = page[:start] + new + kept + page[end:]
page = page.replace("and a falsifiable prediction for this Sunday's feeder race.",
                    "and a falsifiable prediction for this Sunday's feeder race. Updated 6 October with "
                    "the published 2026 entries and Tail of the Lake.", 1)
page = page.replace("Sources: the squad's own results sheet 2007–2026;",
                    "Sources: the squad's own results sheet 2007–2026; Head of the Charles 2026 entries "
                    "(RegattaCentral); Tail of the Lake 2025 and 2026, Textile River and Quinsigamond Snake "
                    "2026 (HereNow); the four's CoxBox at Tail of the Lake;", 1)
open(PAGE, "w", encoding="utf-8").write(page)
print("eight rows", len(eight), "four rows", len(four), "page", len(page))
