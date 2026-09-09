# Code signing

What the "unknown publisher" and "cannot be verified" dialogs are, why
ad-hoc signing cannot remove them, and what does.

## What the dialogs check

Both SmartScreen (Windows) and Gatekeeper (macOS) are **identity**
checks, not integrity checks. Ad-hoc signing proves a file has not been
modified since it was built; it says nothing about *who* built it,
because no identity is attached. The dialogs warn about exactly that
absence, so no build change on this side can remove them. Only a
signature that chains to an identity the OS trusts does.

## Windows: SignPath (free for open source)

The release workflow already contains the signing step
(`.github/workflows/release.yml`, "Hand the executable to SignPath" →
"Sign it" → "Put the signed executable back"). It runs only when the
repository secrets below exist; without them it is skipped, not failed,
and the release ships the unsigned exe as before.

Only `Coxswain.exe` is signed. SmartScreen judges the file that is
launched; signing the four hundred DLLs beside it is slow for no gain.

### What has to exist first — the owner's part

1. **A licence file.** SignPath's free plan is for OSI-licensed open
   source, and the repository has no `LICENSE` yet. MIT is the usual
   choice for a project like this; it is a decision, not a build step,
   so it is not made here.
2. **Apply** at <https://signpath.org/open-source> with the repository
   URL. Approval is manual and takes days to a couple of weeks.
3. In the SignPath organisation they create: a **project** for this
   repository (the *project slug*), a **signing policy** named
   `release-signing` (the *policy slug*), and an **API token** for
   CI submissions. The organisation id is on the organisation page.
4. Add four repository secrets (Settings → Secrets and variables →
   Actions):

   | secret | value |
   |---|---|
   | `SIGNPATH_API_TOKEN` | the CI submitter token |
   | `SIGNPATH_ORGANIZATION_ID` | the organisation id |
   | `SIGNPATH_PROJECT_SLUG` | the project slug |
   | `SIGNPATH_POLICY_SLUG` | `release-signing` |

5. Push the next tag. The Windows job uploads the unsigned exe as an
   artifact, submits it, waits for the signed one, puts it back, and
   **verifies the Authenticode signature is `Valid` before zipping** —
   a silent miss would ship unsigned with a green tick, which is worse
   than shipping unsigned on purpose.

SmartScreen's warning disappears immediately with a SignPath-issued
certificate; there is no reputation period to wait out.

## macOS: Apple Developer ID + notarization

The only route: an Apple Developer account (US$99 a year), a Developer
ID Application certificate, and notarization of the `.app` after
signing. Then a double-click just opens it.

Not wired yet. When the account exists it needs: the certificate as a
base64 `.p12` and its password, the team id, and an app-specific
password for `notarytool`, all as secrets; the Mac job then replaces
the ad-hoc `codesign` with the identity, submits with `notarytool
submit --wait`, and staples. It adds a few minutes to the Mac build.

Until then, the README's right-click → Open instruction stands, and
`packaging/README-mac.txt` distinguishes "cannot be verified" (expected;
open it) from "damaged" (the download really is corrupt; fetch again).
