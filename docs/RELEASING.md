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

### In the browser

1. Go to the repository, then **Releases**, then **Draft a new release**.
2. Give it a tag, say `v0.1`, and a title.
3. **Drag `Coxswain-windows.zip` onto the "Attach binaries" box.**
4. Publish.

The download link is then
`https://github.com/<you>/<repo>/releases/latest`, which is the one
thing to send people.

### From the command line

Needs the GitHub CLI (`winget install GitHub.cli`, then `gh auth login`):

```bash
gh release create v0.1 dist/Coxswain-windows.zip \
    --title "Coxswain v0.1" \
    --notes "Windows build. Unzip, keep the folder together, run Coxswain.exe."
```

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
