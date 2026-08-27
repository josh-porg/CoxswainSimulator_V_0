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

__all__ = ["Finding", "Figure", "Report"]


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


@dataclass
class Report:
    """A page of findings, tables and figures."""

    title: str = "Head of the Charles — what the model says"
    subtitle: str = ""
    findings: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    figures: list = field(default_factory=list)
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

        if self.findings:
            parts.append('<section><h2>What matters, in order</h2>')
            for finding in sorted(self.findings, key=lambda f: -f.weight):
                parts.append(self._finding(finding))
            parts.append('</section>')

        for table in self.tables:
            parts.append(self._table(table))
        for figure in self.figures:
            parts.append(self._figure(figure))

        if self.caveats:
            parts.append('<section class="caveats"><h2>What this does not '
                         'know</h2><ul>')
            for item in self.caveats:
                parts.append('<li>%s</li>' % _escape(item))
            parts.append('</ul></section>')

        parts.append(_FOOT)
        return "\n".join(parts)

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
</style></head><body>"""

_FOOT = """<footer>Generated from the live model by
<code>coxswain.report</code>. Every figure and table on this page was
produced by the run that wrote it; nothing is transcribed.</footer>
</body></html>"""
