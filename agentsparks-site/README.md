# AgentSparks website

A single-page site for AgentSparks: product overview, a link to Auto Startup,
and a job board with three open roles. It is plain HTML, CSS and JavaScript
with no build step, so any static host can serve it.

## Applications database (Google Sheet)

Applications go into a Google Sheet you own. Setup takes about five minutes:

1. Create a new Google Sheet, for example "AgentSparks applications".
2. In the sheet, open **Extensions → Apps Script**. Replace the editor's
   contents with [`google-sheet/Code.gs`](google-sheet/Code.gs) and save.
3. Click **Deploy → New deployment**, choose type **Web app**, set
   **Execute as: Me** and **Who has access: Anyone**, then **Deploy**.
   Approve the permissions prompt.
4. Copy the web app URL (it ends in `/exec`) and paste it into
   `SHEET_ENDPOINT` near the bottom of `index.html`.

The apply form appears once `SHEET_ENDPOINT` is set. Each application
becomes a row in the **Applications** tab with a `Status` column you can
change to Reviewing, Interview, Offer or Rejected. Until it is set, the form
stays hidden and applicants use the email address shown on each role.

## Things to replace

- `APPLY_EMAIL` in `index.html` (currently `careers@agentsparks.ai`).
- The Auto Startup link: the `href` of `#auto-link` (currently `https://example.com`).
- Product copy and job descriptions, which are placeholders.

## Free hosting

- **GitHub Pages:** put this folder in its own repository, then go to
  **Settings → Pages**, choose **Deploy from a branch**, and pick `main` / root.
  The site goes live at `https://<owner>.github.io/<repo>/`.
- **Netlify Drop:** drag this folder onto https://app.netlify.com/drop.
- **Cloudflare Pages** or **Vercel:** connect the repository; no build command needed.
