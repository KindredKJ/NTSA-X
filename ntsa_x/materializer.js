"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const seen = new Set();
  const records = [];
  let steps = 0, commits = 0, nodes = 0, agreement = null;
  const forward = [], reverse = [];
  const connection = $("connection");

  function path(points) {
    return points.map(([x,y],i) => (i ? "L" : "M") + x + " " + y).join(" ");
  }

  function draw() {
    $("forward-path").setAttribute("d", path(forward));
    $("reverse-path").setAttribute("d", path(reverse));
    if (forward.length) {
      const [x,y] = forward[forward.length-1];
      $("forward-dot").setAttribute("cx", x);
      $("forward-dot").setAttribute("cy", y);
    }
    if (reverse.length) {
      const [x,y] = reverse[reverse.length-1];
      $("reverse-dot").setAttribute("cx", x);
      $("reverse-dot").setAttribute("cy", y);
    }
  }

  function add(event) {
    if (!event || event.schema_version !== "1.0" || event.redaction_level !== "metadata" || seen.has(event.event_id)) return;
    seen.add(event.event_id);
    const p = event.payload || {};
    if (event.type === "orbit.step" || event.type === "counter_orbit.step") {
      steps += 1;
      const x = 40 + (steps % 20) * 43;
      const y = Math.max(20,Math.min(142,90 - Number(p.axis_y || 0)*7 - Number(p.axis_z || 0)*3));
      forward.push([x,y]);
      reverse.push([x,320-y]);
      if (forward.length > 20) forward.shift();
      if (reverse.length > 20) reverse.shift();
      draw();
    }
    if (event.type === "counter_orbit.compare") agreement = p.agreement;
    if (event.type === "commit.append") commits++;
    if (event.type === "nexus.commit") nodes = Math.max(nodes, Number(p.node_count || 0));
    $("steps").textContent = String(steps);
    $("commits").textContent = String(commits);
    $("nodes").textContent = String(nodes);
    $("agreement").textContent = agreement === null ? "—" : agreement ? "MATCH" : "DIVERGENT";
    const li = document.createElement("li");
    const time = document.createElement("span");
    time.className = "timestamp";
    time.textContent = String(event.timestamp || "").slice(11,19);
    const type = document.createElement("span");
    type.className = "event-type";
    type.textContent = event.type;
    li.append(time,type,document.createTextNode(JSON.stringify(p)));
    records.unshift(li);
    $("events").prepend(li);
    if (records.length > 80) {
      const removed = records.pop();
      removed.remove();
    }
  }

  fetch("/api/events").then(r=>r.json()).then(snapshot => {
    (snapshot.events || []).forEach(add);
  }).catch(() => { connection.textContent = "RECONNECTING"; });

  const feed = new EventSource("/events");
  feed.onopen = () => { connection.textContent = "LIVE / LOOPBACK"; connection.classList.add("live"); };
  feed.onerror = () => { connection.textContent = "RECONNECTING"; connection.classList.remove("live"); };
  feed.onmessage = e => {
    try { add(JSON.parse(e.data)); } catch { /* ignore bad telemetry */ }
  };
})();
