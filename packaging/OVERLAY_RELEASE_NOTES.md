**SRA CoxBox Overlay** puts your CoxBox numbers over the video from your cox camera, in sync, and
works with no internet connection.

Add the videos and the CoxBox CSV exports, pick a layout and what to show, and press Generate. The app
matches each video to the CoxBox session recorded at the same time, lines them up by the stroke
rhythm in the picture (and, if you like, by the splits you call), and writes a new video with the
overlay. A video with no CoxBox data gets a stroke rate estimated from the camera instead, and the
app asks before doing that.

### New in 1.1

- **Title card.** Select a video and press *Title card…*: give the race a name, a second line
  (event, boat, date) and the lineup, one name per line (bow first and cox last, or cox and stroke
  first; or write the seat yourself, `Stroke: Sam`). It shows for the time you choose, 6 s by
  default. Add a **b-roll clip** and it plays first, sped up or slowed to fill exactly that time,
  with the race video after it; without one, the card lies over the race video's first seconds and
  fades out. Off unless you turn it on; *Preview* shows it before you generate.
- The sync ignores the nonsense motion from near-black frames (a race before sunrise).
- The app no longer needs OpenCV (which kept the Mac build from starting), and every column of the
  video list stays visible at any text size.

### Download

| Your computer | File |
|---|---|
| Windows 10 or 11 (64-bit) | `CoxBoxOverlay-windows.zip` |
| Mac with Apple silicon (M1 or newer, 2020 on) | `CoxBoxOverlay-mac.zip` |
| Linux (x86-64, Ubuntu 22.04 or newer, or similar) | `CoxBoxOverlay-linux.tar.gz` |

An Intel Mac cannot run this build.

### First launch

- **Windows**: unzip, open the `CoxBoxOverlay` folder and run `CoxBoxOverlay.exe`. If SmartScreen
  says it "protected your PC", click *More info*, then *Run anyway* (the app is not code-signed).
- **Mac**: unzip, then right-click `CoxBoxOverlay.app` and choose *Open*, then *Open* again. macOS
  asks this once, because the app is from an unidentified developer.
- **Linux**: `tar -xzf CoxBoxOverlay-linux.tar.gz`, then run `CoxBoxOverlay/CoxBoxOverlay`.

### Good to know

- **Export the CoxBox session as CSV** in NK LiNK Logbook (the app reads the per-stroke data in the
  CSV; FIT and GPX alone are not enough).
- **The camera clock.** Camera clocks drift; the app keeps a setting for how far yours runs fast and
  updates it after every video it syncs. If you use a new camera or reset its clock, untick
  "Camera clock is calibrated" once.
- **Everything stays on your computer.** Nothing is uploaded; the speech recognition for your calls
  runs locally.
- The download is large (the speech model alone is about 480 MB). Videos are written as H.264 MP4
  at about the same size as the camera's own files.
