// AgentSparks applications -> Google Sheet.
// Paste this into Extensions > Apps Script of the sheet that should hold
// applications, then deploy it as a web app (see ../README.md).

const SHEET_NAME = "Applications";
const HEADERS = ["Submitted", "Role", "Name", "Email", "Link", "Note", "Status"];

function doPost(e) {
  const p = (e && e.parameter) || {};
  if (p.website) return ok_(); // spam trap: real people never fill this field

  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    let sheet = ss.getSheetByName(SHEET_NAME);
    if (!sheet) {
      sheet = ss.insertSheet(SHEET_NAME);
      sheet.appendRow(HEADERS);
      sheet.setFrozenRows(1);
    }
    sheet.appendRow([
      new Date(),
      clean_(p.role, 120),
      clean_(p.name, 120),
      clean_(p.email, 200),
      clean_(p.link, 500),
      clean_(p.note, 2000),
      "New",
    ]);
  } finally {
    lock.releaseLock();
  }
  return ok_();
}

// Trim length and stop text from running as a spreadsheet formula.
function clean_(value, max) {
  const s = String(value || "").slice(0, max);
  return /^[=+\-@]/.test(s) ? "'" + s : s;
}

function ok_() {
  return ContentService.createTextOutput("ok");
}
