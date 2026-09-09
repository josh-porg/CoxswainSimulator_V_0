r"""Code signing and the update check: what the repository promises.

The signing step must exist, must be gated so its absence skips rather
than fails, and must verify the signature before zipping -- a silent
miss would ship unsigned under a green tick.  The update check must be
one switch from off, everywhere it is offered.
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_the_windows_job_signs_through_signpath_when_the_secrets_exist():
    text = read(".github", "workflows", "release.yml")
    assert "signpath/github-action-submit-signing-request@v1" in text
    # gated on the token, on the Windows leg only, three times over
    gates = re.findall(r"if: matrix\.name == 'windows' && env\.SIGNPATH_API_TOKEN != ''",
                       text)
    assert len(gates) == 3, len(gates)
    # the four secrets the SignPath action needs
    for secret in ("SIGNPATH_API_TOKEN", "SIGNPATH_ORGANIZATION_ID",
                   "SIGNPATH_PROJECT_SLUG", "SIGNPATH_POLICY_SLUG"):
        assert "secrets.%s" % secret in text, secret
    # the signed file goes back BEFORE the zip, and is verified
    assert text.index("Put the signed executable back") < text.index("- name: Zip it")
    assert "Get-AuthenticodeSignature" in text
    assert "if (\\$s.Status -ne 'Valid') { exit 1 }" in text
    assert "wait-for-completion: true" in text


def test_the_signing_doc_says_what_the_owner_must_do_first():
    text = read("docs", "SIGNING.md")
    assert "LICENSE" in text, "the licence is the prerequisite"
    for secret in ("SIGNPATH_API_TOKEN", "SIGNPATH_ORGANIZATION_ID",
                   "SIGNPATH_PROJECT_SLUG", "SIGNPATH_POLICY_SLUG"):
        assert secret in text, secret
    assert "signpath.org/open-source" in text
    assert "Developer ID" in text and "notariz" in text.lower()


def test_the_update_check_is_offered_and_switchable_everywhere():
    fpv = read("scripts", "fpv.py")
    menu = read("coxswain", "viz", "menu.py")
    assert '"--no-update-check"' in fpv
    assert 'Choice("updates", "Check for updates"' in menu
    assert 'updates: str = "on"' in menu
    # remembered beside the report setting, and started before the build
    assert '_settings.load().get("updates") == "off"' in fpv
    assert "args.update_check = _UpdateCheck(" in fpv
    assert fpv.index("args.update_check = _UpdateCheck(") < fpv.index("def build_everything")
    # the pick is remembered wherever the report pick is
    assert fpv.count('_settings.update(updates=args.updates)') == fpv.count(
        '_settings.update(report=args.report)')
    # and it shows on the setup menu as a footer, never as a blocking prompt
    assert 'footer="NEWER RELEASE: " + _update.line()' in fpv
