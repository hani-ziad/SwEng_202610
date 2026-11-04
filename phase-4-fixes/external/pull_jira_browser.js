// Generic pull of Jira Server issues with their Sprint / Story Points change history,
// through the public REST API (no login). Run in the browser console on a page of the
// Jira instance (the cloud sandbox cannot reach these sites), then save window.__pull
// as JSON. Only issues that were ever in a sprint are kept (current Sprint field set,
// or a Sprint change in the history), because only they can be part of a sprint's scope.
// Usage: await pullJira(['COMPASS','WT',...], 'customfield_10557', 'customfield_10555')
async function pullJira(projects, SPRINT, SP) {
  const base = location.origin, out = { origin: base, pulled: new Date().toISOString(), issues: [], errors: [], scanned: {} };
  const fields = 'issuetype,status,priority,created,resolutiondate,assignee,resolution,timeoriginalestimate,' + SPRINT + ',' + SP;
  for (const P of projects) {
    let start = 0, total = 1, kept = 0;
    while (start < total) {
      let r;
      for (let a = 0; a < 5; a++) {
        try {
          const res = await fetch(base + '/rest/api/2/search?jql=' + encodeURIComponent('project=' + P + ' ORDER BY key ASC') +
            '&maxResults=200&startAt=' + start + '&expand=changelog&fields=' + fields);
          if (res.ok) { r = await res.json(); break; }
          out.errors.push(P + start + ':' + res.status);
        } catch (e) { out.errors.push(P + start + ':' + e); }
        await new Promise(z => setTimeout(z, 3000 * (a + 1)));
      }
      if (!r) break;
      total = r.total;
      for (const is of r.issues) {
        const f = is.fields, ch = [];
        for (const h of (is.changelog ? is.changelog.histories : []))
          for (const it of h.items)
            if (/^sprint$/i.test(it.field) || /^story points$/i.test(it.field))
              ch.push([h.id, it.field, it.from, it.to, it.fromString, it.toString, h.created]);
        const spr = f[SPRINT];
        if (!(spr && spr.length) && !ch.some(c => /^sprint$/i.test(c[1]))) continue;
        kept++;
        out.issues.push({ p: P, id: is.id, k: is.key, t: f.issuetype && f.issuetype.name, s: f.status && f.status.name,
          pr: f.priority && f.priority.name, c: f.created, r: f.resolutiondate, res: f.resolution && f.resolution.name,
          a: f.assignee && f.assignee.name, oe: f.timeoriginalestimate, sp: f[SP], spr: spr,
          clt: is.changelog ? is.changelog.total : null, clm: is.changelog ? is.changelog.maxResults : null, ch: ch });
      }
      start += r.issues.length;
      if (!r.issues.length) break;
    }
    out.scanned[P] = { total: total, kept: kept };
  }
  window.__pull = out;
  const trunc = out.issues.filter(i => i.clt > i.clm).length;
  return 'kept ' + out.issues.length + ' | scanned ' + JSON.stringify(out.scanned) + ' | truncated ' + trunc + ' | errors ' + out.errors.length;
}
