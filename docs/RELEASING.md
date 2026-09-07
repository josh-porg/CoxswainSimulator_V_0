# Putting a build where people can download it

The packaged trainer is a few hundred megabytes. **It cannot go in the
repository**: GitHub rejects any single file over 100 MB, and even under
that limit a binary in git history is permanent and makes every clone
carry it forever. That is what blocked the first push here — a commit
had swept up 1,483 files of PyInstaller output including a 263 MB zip.

Use a **release**. Release assets live outside git history, allow up to
2 GB each, and give a plain download link you can send to anyone.

## Making one

Build and zip:

```bash
python tools/build_exe.py --onedir
```

Then zip `dist/Coxswain` so the archive contains a single `Coxswain/`
folder, and attach it to a release.

### Three platforms, one tag

`.github/workflows/release.yml` builds Windows, macOS and Linux from the
same tag and the same pinned requirements, and attaches all three to the
release. Pushing a `v*` tag runs it; the release is created if it does
not exist yet, so a tag push is all it takes.

The link to hand out never changes and never needs to:

    https://github.com/josh-porg/CoxswainSimulator_V_0/releases/latest

`/releases/latest` redirects to whatever the newest release is.

| Platform | Runner | Ships as | Why that runner |
|---|---|---|---|
| Windows | `windows-latest` | `Coxswain-windows.zip` | |
| macOS | `macos-13` | `Coxswain-mac.zip` | last Intel runner; x86_64 runs on Apple Silicon under Rosetta, arm64 will not start on Intel at all |
| Linux | `ubuntu-22.04` | `Coxswain-linux.tar.gz` | glibc 2.35; a binary will not start against an older glibc than it was built on |

Linux ships as `tar.gz`, not `zip`, because the executable bit has to
survive. Its smoke test runs under `xvfb`, because the headless path
still makes a real standalone GL context and a bare runner has no GLX.

### The Mac build, and why it is not made here

PyInstaller does not cross-compile. It bundles the running interpreter
and the actual binary extension modules from the machine it is on, so a
macOS build has to be made on macOS. There is no Mac in this project, so
`.github/workflows/release.yml` builds both platforms on GitHub runners
and attaches them to the release:

```bash
gh workflow run release.yml -f tag=v0.2
gh run watch <id> --exit-status
```

Pushing a `v*` tag runs it too. It builds on **macos-13**, the last
Intel runner, on purpose: an x86_64 build runs on Apple Silicon through
Rosetta 2, but an arm64 build will not start at all on an Intel Mac, and
plenty of crews are still on 2019 MacBooks. One download that works
everywhere beats two that people have to choose between.

Three macOS-specific things are easy to get wrong and none of them shows
up as a build failure:

* **The GL context.** macOS will not give a 3.3 core profile unless you
  also request forward-compatible. Ask for core alone and it hands back
  a 2.1 legacy context without complaining, and every `#version 330`
  shader then fails to compile -- so it dies at the first draw with a
  message about a shader.
* **Retina.** The drawable is twice the window. Framebuffers and the
  water's `viewport` uniform are in pixels; the mouse and HUD layout are
  in points, which is what SDL reports them in.
* **`ditto`, not `zip`.** A plain zip drops the symlinks and executable
  bits inside a `.app`, and the bundle then will not launch.

The app is **ad-hoc signed** in CI (`codesign --force --deep --sign -`),
which is free and needs no Apple account. That is not notarisation and
does not remove Gatekeeper's prompt -- it changes which prompt. An
unsigned quarantined bundle fails signature validation outright and
macOS calls it "damaged" with a single Cancel button, which reads as a
corrupt download and is the likeliest reason someone gives up. Ad-hoc
signed, the signature is valid and merely unknown, so it says the
developer "cannot be verified" and right-click then Open works cleanly.

Notarisation would remove the prompt entirely and costs 99 dollars a
year for an Apple Developer account.

The first launch still needs right-click then Open.
`packaging/README-mac.txt` leads with that, because macOS's own message
says the app is "damaged" and offers only a Cancel button -- anyone not
told will report it as broken.

### In the browser

1. Go to the repository, then **Releases**, then **Draft a new release**.
2. Give it a tag, say `v0.1`, and a title.
3. **Drag `Coxswain-windows.zip` onto the "Attach binaries" box.**
4. Publish.

The download link is then
`https://github.com/<you>/<repo>/releases/latest`, which is the one
thing to send people.

### From the command line

The GitHub CLI is installed here already (`winget install GitHub.cli`
put it in `C:\Program Files\GitHub CLI`).

Authentication has one wrinkle worth writing down. `gh auth login
--with-token` **rejects the token Git already has stored**, because it
demands `read:org` scope and the credential manager's token does not
carry it. That token is fine for releases -- `repo` is all they need --
so pass it through the environment instead, where `gh` does not check
scopes:

```bash
export GH_TOKEN="$(printf 'protocol=https\nhost=github.com\n\n' \
    | git credential fill | sed -n 's/^password=//p')"
```

Then **create the release first and upload second**, in two commands,
not one:

```bash
gh release create v0.1 --title "Coxswain v0.1" --notes-file notes.md --latest
gh release upload v0.1 dist/Coxswain-windows.zip --clobber
```

The reason for the split is that `gh release create` with an asset
argument **deletes the release it just made if the upload fails**, so a
dropped connection leaves nothing behind and you start over. Created
separately, the release survives and `upload --clobber` retries into it.

Expect to retry. This upload has failed with

    remote error: tls: bad record MAC

which is the TLS stream being corrupted in transit, not GitHub refusing
anything -- the same failure the browser reports as "Something went
really wrong, and we can't process that file". It is transient: the
identical command succeeded on the next attempt, in 24 seconds. If it
recurs, uploads of 100 MB went through while 159 MB did not, so the
size is worth bisecting before blaming the network.

Because that error is *silent corruption* rather than a refusal, check
the asset rather than trusting the exit code:

```bash
gh release download v0.1 --pattern 'Coxswain-windows.zip' --dir /tmp/check
sha256sum dist/Coxswain-windows.zip /tmp/check/Coxswain-windows.zip
```

Finally, confirm a stranger can actually get it. The release being
published is not the same as the repository being public:

```bash
curl -sIL https://github.com/<you>/<repo>/releases/latest/download/Coxswain-windows.zip \
    | grep -Ei '^(HTTP/|content-length)'
```

That is the link to send people.

## What not to do

- **Do not `git add` the zip.** Over 100 MB it is refused outright, and
  under it you have still put a binary in history for good.
- **Git LFS is a poor fit here.** A free account gets 1 GB of storage
  and 1 GB of bandwidth a month; at 130 MB an asset that is about seven
  downloads before it stops working for everyone.
- **Do not send the `.exe` alone.** A `--onedir` build needs its
  `_internal` folder beside it. Send the zip.

## Keeping the download small

`tools/build_exe.py` excludes what the trainer never runs -- PyVista and
VTK, the ffmpeg binary, CasADi. That is about 194 MB of a 756 MB build,
a quarter of the download for code that never executes.

Numba is **kept**, and at 115 MB of LLVM it is the largest single thing
in there. It buys 23% on the physics step, and although both paths fit
the 10 ms budget for 100 Hz on the development machine, the people this
goes to may be on weaker ones. Headroom on an unknown laptop beats a
smaller download.
