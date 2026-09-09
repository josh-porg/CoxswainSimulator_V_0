/**
 * Coxswain performance reports -> a Google Sheet.
 *
 * Deploy once, from a Google account you own:
 *   1. sheets.new  -> name the spreadsheet (e.g. "Coxswain reports").
 *   2. Extensions -> Apps Script.  Replace the editor's contents with
 *      this file.  Save.
 *   3. Deploy -> New deployment -> type "Web app".
 *        Execute as: Me.        Who has access: Anyone.
 *      Copy the web-app URL it gives you.
 *   4. Put that URL where the trainer can find it -- see
 *      packaging/phonehome/README.md.
 *
 * No secret is involved: the URL is the whole handshake.  Anyone with
 * the URL could append a row, which is the worst they could do, and the
 * sheet is yours to filter.
 */

function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  var body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return ContentService.createTextOutput("bad json").setMimeType(
      ContentService.MimeType.TEXT);
  }
  var f = body.frames || {};
  var h = body.hardware || {};
  var s = body.settings || {};
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(["received", "version", "os", "arch", "frozen", "tier",
                     "renderer", "adapters", "cpu", "ram_gb", "integrated",
                     "frames", "p50_ms", "p95_ms", "worst_ms", "physics_ms",
                     "draw_ms", "stalls", "dropped_s", "build_s",
                     "race", "boat", "quality", "physics_hz", "window",
                     "exceptions"]);
  }
  sheet.appendRow([
    new Date(), body.version, body.os, body.arch, body.frozen, body.tier,
    h.renderer, (h.adapters || []).join(" | "), h.cpu_count, h.ram_gb,
    h.integrated, f.frames, f.p50_ms, f.p95_ms, f.worst_ms, f.physics_ms,
    f.draw_ms, f.stalls, f.dropped_s, body.build_s,
    s.race, s.boat, s.quality, s.physics, s.window, body.exceptions
  ]);
  return ContentService.createTextOutput("ok").setMimeType(
    ContentService.MimeType.TEXT);
}

// A GET in a browser is the quickest way to check the deployment is live.
function doGet() {
  return ContentService.createTextOutput("coxswain report endpoint").setMimeType(
    ContentService.MimeType.TEXT);
}
