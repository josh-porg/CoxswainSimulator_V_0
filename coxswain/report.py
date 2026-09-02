"""Everything a coach needs to see, assembled by the code that produced it.

A simulator earns its keep when someone who was not in the room can look
at the output and decide something.  That means the numbers, the pictures
and the caveats have to arrive together, and they have to be generated
rather than transcribed -- a figure pasted into a document is out of date
the moment the model changes, and a table typed by hand is a claim nobody
can check.

So this builds one self-contained HTML page from the live model: it runs
the analyses, writes the figures, embeds them, and states beside every
number where that number came from and how much it should be trusted.

What goes in
------------
Findings are ordered by how much they should change what a crew does,
not by how interesting they were to derive.  The depth loss outranks the
racing line because it is fifty times larger.  Every claim carries its
provenance -- measured, derived, fitted, or reported -- because on this
project the difference has repeatedly been the difference between a
result and an artefact.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["Finding", "Figure", "Embed", "Report"]


@dataclass
class Finding:
    """One thing the model says, with what it rests on."""

    title: str
    headline: str
    detail: str = ""
    #: ``measured`` | ``derived`` | ``fitted`` | ``reported`` | ``open``
    provenance: str = "derived"
    source: str = ""
    #: Rough ranking of how much this should change what a crew does.
    weight: int = 0

    @property
    def badge(self) -> str:
        return {
            "measured": "measured",
            "derived": "derived",
            "fitted": "fitted",
            "reported": "reported",
            "open": "unresolved",
        }.get(self.provenance, self.provenance)


@dataclass
class Figure:
    """An image, its caption, and how to read it."""

    path: str
    title: str
    caption: str = ""
    reading: str = ""
    #: Which tab this belongs under; see :class:`Table`.
    group: str = "Findings"

    def data_uri(self) -> Optional[str]:
        if not os.path.isfile(self.path):
            return None
        lower = self.path.lower()
        if lower.endswith(".mp4"):
            mime = "video/mp4"
        elif lower.endswith(".gif"):
            mime = "image/gif"
        else:
            mime = "image/png"
        with open(self.path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("ascii")
        return "data:%s;base64,%s" % (mime, payload)

    @property
    def is_video(self) -> bool:
        return self.path.lower().endswith(".mp4")


@dataclass
class Table:
    title: str
    columns: Sequence[str]
    rows: Sequence[Sequence[object]]
    note: str = ""
    highlight: Optional[int] = None      # row index to emphasise
    #: Which tab this belongs under.  Items sharing a group are rendered
    #: together in the order they were added, so a table and the figure
    #: that explains it stay side by side.
    group: str = "Findings"


@dataclass
class Embed:
    """An interactive page carried inside a tab.

    Static figures answer the question they were plotted for.  An embed
    is for the ones where the reader's question is not known in advance --
    "what if it is a dry year", "what if the wind backs" -- and the honest
    answer is to hand them the controls.
    """

    path: str                     # relative to the report, e.g. "map.html"
    title: str
    caption: str = ""
    height: int = 1180
    group: str = "Course explorer"


@dataclass
class Report:
    """A page of findings, tables and figures."""

    title: str = "Head of the Charles — what the model says"
    subtitle: str = ""
    findings: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    embeds: list = field(default_factory=list)
    caveats: list = field(default_factory=list)

    def write(self, path: str) -> str:
        html = self._render()
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with io.open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return path

    # -- rendering --------------------------------------------------------
    def _render(self) -> str:
        stamp = _dt.datetime.now().strftime("%d %B %Y")
        # The stylesheet is full of per-cent signs, so %-formatting
        # cannot be used on it.
        parts = [_HEAD.replace("{{TITLE}}", _escape(self.title))]
        parts.append('<header><h1>%s</h1><p class="sub">%s</p>'
                     '<p class="stamp">generated %s</p></header>'
                     % (_escape(self.title), _escape(self.subtitle), stamp))

        # Group the content.  A single flat page of every table followed
        # by every figure was becoming unreadable, and the fix is not
        # smaller sections but *ordering by subject*: a table and the
        # figure that explains it belong together, and the reader wants
        # one topic at a time.
        panels = self._panels()
        parts.append(self._nav(panels))
        for index, (name, body) in enumerate(panels):
            parts.append('<div class="panel" id="panel-%d" '
                         'data-panel="%d"%s>'
                         % (index, index, "" if index == 0 else ""))
            parts.extend(body)
            parts.append('</div>')

        parts.append(_FOOT)
        return "\n".join(parts)

    def _panels(self):
        """``[(tab name, [html, ...]), ...]`` in declared order.

        Findings always lead: they are the summary and the reason anyone
        opens the page.  Everything else follows in the order its group
        was first mentioned, so the builder controls the running order
        without needing to sort anything.
        """
        order = []
        buckets = {}

        def bucket(name):
            if name not in buckets:
                buckets[name] = []
                order.append(name)
            return buckets[name]

        if self.findings:
            body = bucket("What matters")
            body.append('<section><h2>What matters, in order</h2>')
            for finding in sorted(self.findings, key=lambda f: -f.weight):
                body.append(self._finding(finding))
            body.append('</section>')

        for table in self.tables:
            bucket(getattr(table, "group", "Findings")).append(
                self._table(table))
        for figure in self.figures:
            bucket(getattr(figure, "group", "Findings")).append(
                self._figure(figure))
        for embed in self.embeds:
            bucket(getattr(embed, "group", "Findings")).append(
                self._embed(embed))

        if self.caveats:
            body = bucket("Caveats")
            body.append('<section class="caveats"><h2>What this does not '
                        'know</h2><ul>')
            for item in self.caveats:
                body.append('<li>%s</li>' % _escape(item))
            body.append('</ul></section>')

        return [(name, buckets[name]) for name in order]

    @staticmethod
    def _nav(panels) -> str:
        """The tab strip.

        Buttons rather than anchors, because these switch a view rather
        than navigate; ``aria-selected`` carries the state for a screen
        reader.  Without JavaScript every panel stays visible and the
        strip hides itself, so the page degrades to the long scroll it
        used to be rather than to a blank screen.
        """
        if len(panels) < 2:
            return ""
        buttons = "".join(
            '<button role="tab" data-tab="%d" aria-selected="%s" '
            'aria-controls="panel-%d">%s</button>'
            % (index, "true" if index == 0 else "false", index,
               _escape(name))
            for index, (name, _body) in enumerate(panels))
        return ('<nav class="tabs" role="tablist" hidden>%s</nav>' % buttons)

    def _embed(self, embed: "Embed") -> str:
        """An iframe, with a link out for when the frame is inconvenient.

        The frame is loaded lazily: the explorer carries the whole
        bathymetry and every precomputed line, and paying for that on a
        tab the reader may never open would slow the page for everyone.
        """
        return ('<section class="embed">'
                '<h2>%s</h2>'
                '%s'
                '<iframe src="%s" loading="lazy" title="%s" '
                'style="width:100%%;height:%dpx;border:1px solid '
                'var(--edge,#dde4e8);border-radius:12px;'
                'background:var(--panel,#fff)"></iframe>'
                '<p class="source"><a href="%s" target="_blank" '
                'rel="noopener">Open the explorer on its own</a></p>'
                '</section>'
                % (_escape(embed.title),
                   ('<p class="detail">%s</p>' % _escape(embed.caption))
                   if embed.caption else "",
                   _escape(embed.path), _escape(embed.title),
                   int(embed.height), _escape(embed.path)))

    def _finding(self, finding: Finding) -> str:
        source = ('<p class="source">%s</p>' % _escape(finding.source)
                  if finding.source else "")
        detail = ('<p class="detail">%s</p>' % _escape(finding.detail)
                  if finding.detail else "")
        return ('<article class="finding %s">'
                '<div class="rail"></div>'
                '<div class="body"><h3>%s <span class="badge %s">%s</span></h3>'
                '<p class="headline">%s</p>%s%s</div></article>'
                % (finding.provenance, _escape(finding.title),
                   finding.provenance, _escape(finding.badge),
                   _escape(finding.headline), detail, source))

    def _table(self, table: Table) -> str:
        head = "".join("<th>%s</th>" % _escape(str(c)) for c in table.columns)
        body = []
        for index, row in enumerate(table.rows):
            cells = "".join("<td>%s</td>" % _escape(str(v)) for v in row)
            klass = ' class="row-mark"' if index == table.highlight else ""
            body.append("<tr%s>%s</tr>" % (klass, cells))
        note = ('<p class="note">%s</p>' % _escape(table.note)
                if table.note else "")
        return ('<section><h2>%s</h2><div class="scroll"><table>'
                '<thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>%s'
                '</section>'
                % (_escape(table.title), head, "".join(body), note))

    def _figure(self, figure: Figure) -> str:
        uri = figure.data_uri()
        if uri is None:
            return ('<section><h2>%s</h2><p class="missing">not generated'
                    '</p></section>' % _escape(figure.title))
        reading = ('<p class="reading"><strong>How to read it.</strong> %s</p>'
                   % _escape(figure.reading) if figure.reading else "")
        caption = ('<p class="caption">%s</p>' % _escape(figure.caption)
                   if figure.caption else "")
        if figure.is_video:
            media = ('<video src="%s" controls loop muted playsinline>'
                     '</video>' % uri)
        else:
            media = '<img src="%s" alt="%s"/>' % (uri, _escape(figure.title))
        return ('<section><h2>%s</h2><figure>%s%s</figure>%s</section>'
                % (_escape(figure.title), media, caption, reading))


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{TITLE}}</title>
<style>
 :root{
   --ink:#16211f; --muted:#5c6968; --rule:#dce2e0; --page:#fbfaf7;
   --card:#ffffff; --accent:#1f5673; --good:#1f7a4d; --warn:#a2382a;
   --fit:#c8901a; --open:#6b3fa0;
 }
 @media (prefers-color-scheme: dark){
   :root:not([data-theme="light"]){
     --ink:#e8eceb; --muted:#9aa8a6; --rule:#2b3634; --page:#111817;
     --card:#18201f; --accent:#7fb4d0; --good:#5cc08d; --warn:#e0776a;
     --fit:#e0b45c; --open:#b294d8;
   }
 }
 :root[data-theme="dark"]{
   --ink:#e8eceb; --muted:#9aa8a6; --rule:#2b3634; --page:#111817;
   --card:#18201f; --accent:#7fb4d0; --good:#5cc08d; --warn:#e0776a;
   --fit:#e0b45c; --open:#b294d8;
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);
   font:16px/1.6 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
   padding:0 1.2rem 5rem}
 header{max-width:62rem;margin:0 auto;padding:3.4rem 0 1.6rem;
   border-bottom:1px solid var(--rule)}
 h1{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.15;margin:0 0 .5rem;
   text-wrap:balance;letter-spacing:-.01em}
 .sub{margin:0;color:var(--muted);font-size:1.05rem;max-width:46rem}
 .stamp{margin:.9rem 0 0;color:var(--muted);font-size:.8rem;
   text-transform:uppercase;letter-spacing:.09em}
 section{max-width:62rem;margin:0 auto;padding:2.2rem 0 .4rem}
 h2{font-size:1.32rem;margin:0 0 1.1rem;letter-spacing:-.005em}
 h3{font-size:1.02rem;margin:0 0 .35rem;display:flex;gap:.6rem;
   align-items:baseline;flex-wrap:wrap}
 .finding{display:flex;gap:0;background:var(--card);border:1px solid var(--rule);
   border-radius:3px;margin:0 0 .8rem;overflow:hidden}
 .finding .rail{width:4px;flex:0 0 4px;background:var(--accent)}
 .finding.measured .rail{background:var(--good)}
 .finding.fitted .rail{background:var(--fit)}
 .finding.reported .rail{background:var(--accent)}
 .finding.open .rail{background:var(--open)}
 .finding .body{padding:.95rem 1.15rem}
 .headline{margin:0;font-size:1.14rem;font-weight:600}
 .detail{margin:.5rem 0 0;color:var(--muted);font-size:.94rem}
 .source{margin:.5rem 0 0;color:var(--muted);font-size:.82rem;
   font-style:italic}
 .badge{font:600 .66rem/1 ui-sans-serif,system-ui,sans-serif;
   text-transform:uppercase;letter-spacing:.1em;padding:.28em .55em;
   border-radius:2px;border:1px solid currentColor}
 .badge.measured{color:var(--good)} .badge.derived{color:var(--accent)}
 .badge.fitted{color:var(--fit)} .badge.reported{color:var(--accent)}
 .badge.open{color:var(--open)}
 .scroll{overflow-x:auto}
 table{border-collapse:collapse;width:100%;font-size:.93rem;
   font-variant-numeric:tabular-nums}
 th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--rule)}
 th:first-child,td:first-child{text-align:left}
 thead th{font:600 .72rem/1 ui-sans-serif,system-ui,sans-serif;
   text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
 tr.row-mark td{background:color-mix(in srgb,var(--good) 12%, transparent);
   font-weight:600}
 .note,.caption{color:var(--muted);font-size:.87rem;margin:.7rem 0 0}
 .reading{margin:.7rem 0 0;font-size:.93rem;
   border-left:2px solid var(--rule);padding-left:.9rem}
 figure{margin:0;background:var(--card);border:1px solid var(--rule);
   border-radius:3px;padding:.7rem}
 img,video{width:100%;height:auto;display:block;border-radius:2px}
 .caveats{border-top:1px solid var(--rule);margin-top:2rem}
 .caveats li{margin:0 0 .55rem;color:var(--muted)}
 .missing{color:var(--warn)}
 footer{max-width:62rem;margin:2rem auto 0;padding-top:1.2rem;
   border-top:1px solid var(--rule);color:var(--muted);font-size:.82rem}
 /* Tab strip.  Sticky so the reader can move between topics from
    anywhere in a long panel, and scrollable sideways on a phone rather
    than wrapping into a block that eats the viewport. */
 .tabs{position:sticky;top:0;z-index:5;display:flex;gap:.15rem;
   max-width:62rem;margin:0 auto;padding:.55rem 0;overflow-x:auto;
   background:var(--page);border-bottom:1px solid var(--rule);
   scrollbar-width:none}
 .tabs::-webkit-scrollbar{display:none}
 .tabs button{flex:0 0 auto;appearance:none;cursor:pointer;
   background:transparent;color:var(--muted);border:0;
   border-bottom:2px solid transparent;border-radius:2px 2px 0 0;
   padding:.5rem .85rem;white-space:nowrap;
   font:600 .78rem/1.2 ui-sans-serif,system-ui,sans-serif;
   text-transform:uppercase;letter-spacing:.07em}
 .tabs button:hover{color:var(--ink)}
 .tabs button:focus-visible{outline:2px solid var(--accent);
   outline-offset:-2px}
 .tabs button[aria-selected="true"]{color:var(--ink);
   border-bottom-color:var(--accent)}
 .panel[hidden]{display:none}
 .panel>section:first-child{padding-top:1.4rem}
</style></head><body>"""

_FOOT = """<footer>Generated from the live model by
<code>coxswain.report</code>. Every figure and table on this page was
produced by the run that wrote it; nothing is transcribed.</footer>
<script>
/* Progressive enhancement.  The markup ships with every panel visible
   and the tab strip hidden, so a browser with no JavaScript -- or one
   that fails to run this -- gets the long scrolling page rather than a
   blank one.  Only once this runs does the page become tabbed. */
(function () {
  var strip = document.querySelector('.tabs');
  var panels = Array.prototype.slice.call(
    document.querySelectorAll('.panel'));
  if (!strip || panels.length < 2) { return; }
  var tabs = Array.prototype.slice.call(strip.querySelectorAll('button'));

  function show(index) {
    panels.forEach(function (panel, i) { panel.hidden = i !== index; });
    tabs.forEach(function (tab, i) {
      tab.setAttribute('aria-selected', i === index ? 'true' : 'false');
    });
    try { history.replaceState(null, '', '#panel-' + index); } catch (e) {}
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener('click', function () {
      show(index);
      window.scrollTo({top: 0, behavior: 'auto'});
    });
    /* Arrow keys move between tabs, which is what a tablist owes a
       keyboard user. */
    tab.addEventListener('keydown', function (event) {
      var step = event.key === 'ArrowRight' ? 1
               : event.key === 'ArrowLeft' ? -1 : 0;
      if (!step) { return; }
      event.preventDefault();
      var next = (index + step + tabs.length) % tabs.length;
      tabs[next].focus();
      show(next);
    });
  });

  strip.hidden = false;
  var opening = parseInt((location.hash.match(/^#panel-(\\d+)$/) || [])[1],
                         10);
  show(isNaN(opening) || opening >= panels.length ? 0 : opening);
})();
</script>
</body></html>"""
