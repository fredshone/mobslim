/* Side-by-side animation of the showcase corridor, one row per simulation.
 *
 * Data: window.MOBSLIM_CORRIDOR[<data-scenario of the element>], written by
 * `python -m showcase correctness`.
 * Vehicles are placed from their link entry and exit times only: a vehicle
 * moves at the link's free speed, then waits at the end of the link (behind
 * the vehicles in front of it) until its recorded exit time.
 */
(function () {
  "use strict";

  const CAR_LEN = 7.5, BUS_LEN = 15; // metres of queue each takes
  const ROW_GAP = 8, LABEL_H = 18, PAD_X = 14;
  const Y_SQUASH = 0.6; // draw the side road flatter, to keep rows short
  const VIEW = [350, 1800]; // x range (m) shown: the junction and the bottleneck
  const SPEED = 20; // simulated seconds per second of playback

  function css(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function hms(t) {
    t = Math.max(0, Math.floor(t));
    const p = (n) => String(n).padStart(2, "0");
    return `${p(Math.floor(t / 3600))}:${p(Math.floor(t / 60) % 60)}:${p(t % 60)}`;
  }

  function init(root) {
    const D = (window.MOBSLIM_CORRIDOR || {})[root.dataset.scenario];
    if (!D) { root.textContent = "Animation data failed to load."; return; }

    // ---- geometry
    const nodes = D.nodes;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const grow = (x, y) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); };
    Object.values(nodes).forEach(([x, y]) => grow(x, y));
    D.pax.forEach((p) => { grow(p.home[0], p.home[1]); grow(p.work[0], p.work[1]); });
    y0 -= 25; y1 += 15;
    const links = D.links.map((l) => {
      const [ax, ay] = nodes[l.from], [bx, by] = nodes[l.to];
      return { ...l, ax, ay, dx: (bx - ax) / l.length, dy: (by - ay) / l.length,
               fftime: l.length / l.freespeed };
    });
    const minCap = Math.min(...links.map((l) => l.capacity));
    const maxCap = Math.max(...links.map((l) => l.capacity));
    const roadWidth = (cap) => 1.5 + 6 * cap / maxCap; // px, by capacity

    // ---- time range
    let tEnd = D.t0;
    D.runs.forEach((r) => {
      r.cars.concat(r.buses).forEach((v) => v.l.forEach((s) => { if (s[2] != null) tEnd = Math.max(tEnd, s[2]); }));
      r.pax.forEach((p) => { if (p[4] != null) tEnd = Math.max(tEnd, p[4]); });
    });
    tEnd += 20;
    const runByKey = Object.fromEntries(D.runs.map((r) => [r.key, r]));

    // ---- DOM
    root.innerHTML = `
      <div class="ma-controls">
        <button type="button" class="ma-play" aria-label="Play">▶</button>
        <input type="range" class="ma-time" min="${D.t0}" max="${tEnd}" step="0.5" value="${D.t0}" aria-label="Time">
        <span class="ma-clock">--:--:--</span>
      </div>
      <canvas class="ma-canvas"></canvas>
      <div class="ma-legend">
        <span><i class="ma-dot ma-main"></i>car, main road</span>
        <span><i class="ma-dot ma-side"></i>car, side road</span>
        <span><i class="ma-bus"></i>bus (passengers aboard)</span>
        <span><i class="ma-dot ma-pax"></i>bus passenger</span>
        <span><i class="ma-dot ma-hl"></i>selected car (click one)</span>
        <span><i class="ma-stop"></i>bus stop</span>
        <span class="ma-note">Road width shows capacity. The thinnest section is the bottleneck: one car every ${Math.round(3600 / minCap)} s. Playback at ${SPEED}× real time.</span>
      </div>
      <div class="ma-info" aria-live="polite"></div>`;
    const canvas = root.querySelector(".ma-canvas");
    const ctx = canvas.getContext("2d");
    const slider = root.querySelector(".ma-time");
    const clockEl = root.querySelector(".ma-clock");
    const playBtn = root.querySelector(".ma-play");
    let vx0 = x0, vx1 = x1;
    const info = root.querySelector(".ma-info");

    let t = D.t0, playing = false, last = null, selected = null;
    let scale = 1, rowH = 100, W = 600;
    let positions = []; // [{row, x, y, car}] for click hit tests

    function rows() {
      return D.sims;
    }

    function layout() {
      W = root.clientWidth || 700;
      [vx0, vx1] = VIEW;
      scale = (W - 2 * PAD_X) / (vx1 - vx0);
      rowH = LABEL_H + (y1 - y0) * scale * Y_SQUASH;
      const n = rows().length;
      const H = n * rowH + (n - 1) * ROW_GAP;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width = W + "px";
      canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    const px = (x) => PAD_X + (x - vx0) * scale;
    const py = (y, row) => row * (rowH + ROW_GAP) + LABEL_H + (y1 - y) * scale * Y_SQUASH;

    // vehicle positions on every link at time t: [{link, pos, kind, idx}]
    function placeVehicles(run) {
      const onLink = links.map(() => []);
      const add = (v, kind, idx) => {
        for (const [li, enter, leave, start] of v.l) {
          if (enter > t) break;
          if (leave != null && leave <= t) continue;
          const L = links[li];
          let free;
          if (start) free = L.length;
          else {
            const tt = leave == null ? L.fftime : Math.min(L.fftime, Math.max(leave - enter, 1e-6));
            free = L.length * Math.min(1, (t - enter) / tt);
          }
          onLink[li].push({ kind, idx, enter, free, len: kind === "bus" ? BUS_LEN : CAR_LEN });
          break;
        }
      };
      run.cars.forEach((v, i) => add(v, "car", i));
      run.buses.forEach((v, i) => add(v, "bus", i));
      const out = [];
      onLink.forEach((vs, li) => {
        vs.sort((a, b) => a.enter - b.enter || b.free - a.free);
        let room = links[li].length;
        for (const v of vs) {
          const pos = Math.max(0, Math.min(v.free, room));
          room = pos - v.len;
          out.push({ li, pos, ...v });
        }
      });
      return out;
    }

    function draw() {
      const col = {
        road: css(root, "--ma-road", "#c9ccd1"),
        text: css(root, "--ma-text", "#333"),
        muted: css(root, "--ma-muted", "#777"),
        main: css(root, "--ma-main", "#2a6fdb"),
        side: css(root, "--ma-side", "#e8833a"),
        bus: css(root, "--ma-busc", "#1b9e5a"),
        pax: css(root, "--ma-paxc", "#8e44ad"),
        hl: css(root, "--ma-hlc", "#d81b60"),
        bg: css(root, "--ma-bg", "#fff"),
        band: css(root, "--ma-band", "#f5f6f8"),
      };
      ctx.clearRect(0, 0, W, canvas.height);
      positions = [];
      const R = rows();
      R.forEach((sim, row) => {
        const run = runByKey[sim.key];
        const top = row * (rowH + ROW_GAP);
        ctx.fillStyle = sim.family === "matsim" ? col.band : col.bg;
        ctx.fillRect(0, top, W, rowH);
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, top + LABEL_H - 2, W, rowH - LABEL_H + 2);
        ctx.clip();

        // roads
        ctx.lineCap = "round";
        links.forEach((L) => {
          ctx.strokeStyle = col.road;
          ctx.lineWidth = roadWidth(L.capacity);
          ctx.beginPath();
          ctx.moveTo(px(L.ax), py(L.ay, row));
          ctx.lineTo(px(L.ax + L.dx * L.length), py(L.ay + L.dy * L.length, row));
          ctx.stroke();
        });
        // stops
        ctx.font = "10px system-ui, sans-serif";
        D.stops.forEach((s) => {
          // a small empty shelter beneath the end of the stop's link, where passengers wait
          const sx = px(s.x), sy = py(s.y, row);
          ctx.strokeStyle = col.bus;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(sx - 13, sy + 5, 10, 7);
        });

        // passengers (placed in pixels: they wait in a line beside the stop's shelter)
        const aboard = run.buses.map(() => 0);
        const waiting = D.stops.map(() => 0);
        const queueSpot = (s, k) => [px(s.x) - 17 - (k % 8) * 5, py(s.y, row) + 8.5 + Math.floor(k / 8) * 5];
        run.pax.forEach((p, i) => {
          const [leave, wait, board, alight, work, stop, bus, astop] = p;
          if (leave == null || t < leave) return;
          let x, y;
          const meta = D.pax[i];
          if (wait != null && t < wait) { // walking from home to the stop
            const f = (t - leave) / Math.max(wait - leave, 1);
            const [hx, hy] = [px(meta.home[0]), py(meta.home[1], row)];
            const [tx, ty] = queueSpot(D.stops[stop], 0);
            x = hx + (tx - hx) * f; y = hy + (ty - hy) * f;
          } else if (board == null || t < board) { // waiting
            [x, y] = queueSpot(D.stops[stop], waiting[stop]++);
          } else if (alight == null || t < alight) { // on the bus
            if (bus >= 0) aboard[bus]++;
            return;
          } else if (work != null && t < work) { // walking from the stop to work
            const f = (t - alight) / Math.max(work - alight, 1);
            const [sx, sy] = queueSpot(D.stops[astop], 0);
            const [wx, wy] = [px(meta.work[0]), py(meta.work[1], row)];
            x = sx + (wx - sx) * f; y = sy + (wy - sy) * f;
          } else return;
          ctx.fillStyle = col.pax;
          ctx.beginPath();
          ctx.arc(x, y, 2.2, 0, 2 * Math.PI);
          ctx.fill();
        });

        // vehicles
        let arrived = 0, onRoad = 0;
        run.cars.forEach((v) => {
          const lastSeg = v.l[v.l.length - 1];
          if (lastSeg && lastSeg[2] != null && lastSeg[2] <= t) arrived++;
        });
        placeVehicles(run).forEach((v) => {
          const L = links[v.li];
          const cx = px(L.ax + L.dx * v.pos), cy = py(L.ay + L.dy * v.pos, row);
          if (v.kind === "bus") {
            const len = BUS_LEN * scale + 4;
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(Math.atan2(-L.dy * Y_SQUASH, L.dx));
            ctx.fillStyle = col.bus;
            ctx.fillRect(-len, -4, len, 8);
            ctx.restore();
            ctx.fillStyle = col.text;
            ctx.textAlign = "center";
            ctx.font = "bold 10px system-ui, sans-serif";
            ctx.fillText(String(aboard[v.idx]), cx - len / 2, cy - 7);
            return;
          }
          onRoad++;
          const car = D.cars[v.idx];
          const isSel = selected === v.idx;
          ctx.fillStyle = isSel ? col.hl : car.group === "side" ? col.side : col.main;
          ctx.beginPath();
          ctx.arc(cx, cy, isSel ? 6 : Math.max(3.3, Math.min(4.8, CAR_LEN * scale * 0.68)), 0, 2 * Math.PI);
          ctx.fill();
          ctx.lineWidth = 1; // a ring in the row's background keeps queued cars apart
          ctx.strokeStyle = sim.family === "matsim" ? col.band : col.bg;
          ctx.stroke();
          positions.push({ x: cx, y: cy, idx: v.idx });
        });

        ctx.restore();
        // label
        ctx.textAlign = "left";
        ctx.fillStyle = col.text;
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.fillText(sim.label, 6, top + 13);
        const lw = ctx.measureText(sim.label).width;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = col.muted;
        ctx.fillText(sim.source, 12 + lw, top + 13);
        ctx.textAlign = "right";
        ctx.fillText(`on the road ${onRoad} · arrived ${arrived}/${run.cars.length}`, W - 6, top + 13);
      });
      clockEl.textContent = hms(t);
      slider.value = t;
    }

    function describe(idx) {
      const car = D.cars[idx];
      const parts = rows().map((sim) => {
        const v = runByKey[sim.key].cars[idx];
        const first = v.l[0], lastSeg = v.l[v.l.length - 1];
        if (!first || !lastSeg || lastSeg[2] == null) return `${sim.label}: did not arrive`;
        return `${sim.label}: ${Math.round(lastSeg[2] - first[1])} s`;
      });
      info.innerHTML = `<strong>Car ${car.id}</strong> (${car.group} road), door to door: ` +
        parts.map((p) => `<span>${p}</span>`).join(" · ");
    }

    function frame(now) {
      if (!playing) return;
      if (last != null) t += ((now - last) / 1000) * SPEED;
      last = now;
      if (t >= tEnd) { t = tEnd; setPlaying(false); }
      draw();
      if (playing) requestAnimationFrame(frame);
    }

    function setPlaying(p) {
      playing = p;
      playBtn.textContent = p ? "❚❚" : "▶";
      playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
      last = null;
      if (p) {
        if (t >= tEnd) t = D.t0;
        requestAnimationFrame(frame);
      }
    }

    playBtn.addEventListener("click", () => setPlaying(!playing));
    slider.addEventListener("input", () => { t = Number(slider.value); draw(); });
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      let best = null, bd = 64;
      positions.forEach((p) => {
        const d = (p.x - x) ** 2 + (p.y - y) ** 2;
        if (d < bd) { bd = d; best = p.idx; }
      });
      selected = best;
      if (best == null) info.textContent = ""; else describe(best);
      draw();
    });
    new ResizeObserver(() => { layout(); draw(); }).observe(root);
    // redraw when the docs theme switches between light and dark
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    t = D.t0 + 300;
    const m = /(?:^|[#&])t=(\d+(?:\.\d+)?)/.exec(location.hash); // e.g. #t=25500 (seconds)
    if (m) t = Math.min(tEnd, Math.max(D.t0, Number(m[1])));
    layout();
    draw();
  }

  function start() { document.querySelectorAll(".mobsim-anim").forEach(init); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
