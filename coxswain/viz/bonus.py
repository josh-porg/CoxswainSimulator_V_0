r"""The bonus run: coins along the course, and boosts.

A secret, unlocked by typing ``boost`` on the setup menu.  Not a
training mode -- a thing to do on a Friday with the crew watching.
Coins sit on the racing line and a few metres either side of it, so
collecting them is steering practice wearing a disguise; boosts are
rarer and sit where a coxswain would want one, on the straights.

Everything here is geometry and bookkeeping.  The renderer draws what
:meth:`BonusRun.visible` hands it, the game loop calls
:meth:`BonusRun.collect` with the bow's position each frame, and
``call_bonus`` is what a live boost adds to the coxswain's power call
-- the same lever Up and Down move, so a boost is a crew briefly
rowing above themselves rather than a different physics.

Physics is untouched: a boost is ``+0.30`` on the call for eight
seconds, which the reserve pays for like any other call.  A coin is a
coin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

#: The word that unlocks it, typed anywhere on the setup menu.
SECRET = "boost"

#: Coins every this many metres along the course, boosts every this
#: many.  A 4.8 km course gives about 80 coins and 9 boosts.
COIN_EVERY = 60.0
BOOST_EVERY = 520.0
#: How far off the line the coins wander, metres, alternating sides;
#: every third coin is dead on the line.
COIN_OFFSET = 6.0
#: Collection radius from the bow, metres -- a blade's length, so a
#: coin under the riggers counts.
REACH = 3.5
#: What a boost does, and for how long.
BOOST_CALL = 0.30
BOOST_SECONDS = 8.0
#: Height above the water the pickups float at, for the renderer.
HEIGHT = 0.55


@dataclass
class Pickup:
    east: float
    north: float
    kind: str                      # "coin" or "boost"
    taken: bool = False


@dataclass
class BonusRun:
    """State of one bonus run."""

    pickups: List[Pickup] = field(default_factory=list)
    coins: int = 0
    boosts: int = 0
    boost_until: float = -1.0
    started: float = field(default_factory=time.monotonic)
    _xy: Optional[np.ndarray] = None

    @classmethod
    def along(cls, course, coin_every: float = COIN_EVERY,
              boost_every: float = BOOST_EVERY,
              offset: float = COIN_OFFSET) -> "BonusRun":
        """Lay the pickups out along a course polyline.

        Coins alternate left, on, right of the line so a straight-line
        crew collects a third and a steered one collects them all;
        boosts are on the line and never within a coin's spacing of
        the start, where the crew are still finding their rhythm.
        """
        course = np.asarray(course, dtype=float)[:, :2]
        if len(course) < 2:
            return cls()
        seg = np.diff(course, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1])
        station = np.concatenate([[0.0], np.cumsum(length)])
        total = float(station[-1])

        def at(s):
            i = int(np.clip(np.searchsorted(station, s) - 1, 0, len(seg) - 1))
            u = (s - station[i]) / max(length[i], 1e-9)
            point = course[i] + u * seg[i]
            tangent = seg[i] / max(length[i], 1e-9)
            normal = np.array([-tangent[1], tangent[0]])
            return point, normal

        pickups = []
        k = 0
        s = coin_every
        while s < total - coin_every * 0.5:
            point, normal = at(s)
            side = (-1.0, 0.0, 1.0)[k % 3]
            p = point + side * offset * normal
            pickups.append(Pickup(float(p[0]), float(p[1]), "coin"))
            k += 1
            s += coin_every
        s = boost_every
        while s < total - coin_every:
            point, _normal = at(s)
            pickups.append(Pickup(float(point[0]), float(point[1]), "boost"))
            s += boost_every
        run = cls(pickups=pickups)
        run._xy = np.array([[p.east, p.north] for p in pickups], dtype=float)
        return run

    # -- play -------------------------------------------------------------
    def collect(self, east: float, north: float, now: float = None,
                reach: float = REACH) -> List[Pickup]:
        """Take every untaken pickup within ``reach`` of ``(east, north)``.

        Returns what was taken this call, for the sound and the HUD.
        """
        if self._xy is None or not len(self._xy):
            return []
        now = time.monotonic() if now is None else float(now)
        d = np.hypot(self._xy[:, 0] - east, self._xy[:, 1] - north)
        taken = []
        for i in np.flatnonzero(d <= reach):
            p = self.pickups[int(i)]
            if p.taken:
                continue
            p.taken = True
            taken.append(p)
            if p.kind == "coin":
                self.coins += 1
            else:
                self.boosts += 1
                # a boost taken during a boost extends it from NOW, so
                # two in a row is sixteen seconds, not eight twice over
                self.boost_until = now + BOOST_SECONDS
        return taken

    def call_bonus(self, now: float = None) -> float:
        now = time.monotonic() if now is None else float(now)
        return BOOST_CALL if now < self.boost_until else 0.0

    def boost_left(self, now: float = None) -> float:
        now = time.monotonic() if now is None else float(now)
        return max(0.0, self.boost_until - now)

    def visible(self, east: float, north: float, within: float = 400.0):
        """Untaken pickups within ``within`` metres: what to draw."""
        if self._xy is None or not len(self._xy):
            return []
        d = np.hypot(self._xy[:, 0] - east, self._xy[:, 1] - north)
        return [self.pickups[int(i)] for i in np.flatnonzero(d <= within)
                if not self.pickups[int(i)].taken]

    @property
    def total_coins(self) -> int:
        return sum(1 for p in self.pickups if p.kind == "coin")

    def score(self) -> int:
        """Coins, plus five for every boost -- a boost is worth going
        out of your line for."""
        return self.coins + 5 * self.boosts

    def hud_line(self, now: float = None) -> str:
        left = self.boost_left(now)
        return ("BONUS   coins %d/%d   boosts %d%s"
                % (self.coins, self.total_coins, self.boosts,
                   ("   BOOST %.0f s" % left) if left > 0 else ""))


# -- the secret ------------------------------------------------------------
class SecretTyper:
    """Watches typed characters for the unlock word.

    Keeps only the last ``len(SECRET)`` printable characters, so
    nothing else typed on the menu is remembered.
    """

    def __init__(self, word: str = SECRET):
        self.word = word.lower()
        self.buffer = ""

    def feed(self, char: str) -> bool:
        """True once, when the word completes."""
        if not char or not char.isprintable() or len(char) != 1:
            return False
        self.buffer = (self.buffer + char.lower())[-len(self.word):]
        if self.buffer == self.word:
            self.buffer = ""
            return True
        return False


def pickup_solids(pickups, t: float):
    """Small spinning octahedra for the renderer, world frame.

    Returns ``(vertices, colours)`` float32 arrays, or ``None``.  Gold
    for coins, green for boosts; they spin on ``t`` so they read as
    pickups and not as buoys.
    """
    if not pickups:
        return None
    verts, cols = [], []
    for p in pickups:
        r = 0.45 if p.kind == "coin" else 0.6
        colour = (1.0, 0.82, 0.2) if p.kind == "coin" else (0.3, 1.0, 0.45)
        a = t * (2.2 if p.kind == "coin" else 1.4)
        c, s = np.cos(a), np.sin(a)
        cx, cy, cz = p.east, p.north, HEIGHT
        top = (cx, cy, cz + r)
        bottom = (cx, cy, cz - r)
        ring = [(cx + r * c, cy + r * s, cz), (cx - r * s, cy + r * c, cz),
                (cx - r * c, cy - r * s, cz), (cx + r * s, cy - r * c, cz)]
        for i in range(4):
            j = (i + 1) % 4
            verts += [top, ring[i], ring[j], bottom, ring[j], ring[i]]
            cols += [colour] * 6
    return (np.asarray(verts, dtype="f4"), np.asarray(cols, dtype="f4"))
