/* Observed against simulated vehicles, replayed side by side (or one above
 * the other, for a road drawn wide).
 *
 * <div class="mobsim-anim replay-anim" data-key="opendd-rdb1">. Data, per
 * key: window.MOBSLIM_REPLAY[key] (replay-<key>-sim.js: the roads, the
 * simulated vehicles and the in-view counts) and
 * window.MOBSLIM_REPLAY_OBS[key] (replay-<key>-obs.js: the observed
 * tracks, built locally from the data and not committed), both written by
 * `python -m showcase <opendd|tgsim|tgsim-i90>` (see showcase/replay.py).
 * Positions are precomputed at `fps` frames per second, in decimetres,
 * in drawing coordinates (u right, v up); `stretch` magnifies v.
 *
 * The replay starts when it scrolls into view, loops, and pauses when it
 * scrolls out. Click the strip beneath the panels (vehicles in view over
 * the whole replay) to jump to a moment.
 */
(function () {
  "use strict";

  const PAD = 8, STRIP_H = 54;

  function css(el, name, fallback) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function clock(t) {
    t = Math.max(0, t);
    const m = Math.floor(t / 60), s = t - 60 * m;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
  }

  function init(root) {
    const key = root.dataset.key;
    const D = (window.MOBSLIM_REPLAY || {})[key];
    if (!D) { root.textContent = "Animation data failed to load."; return; }
    const O = (window.MOBSLIM_REPLAY_OBS || {})[key];
    const STRETCH = D.stretch || 1, T0 = D.t0 || 0, FPS = D.fps, N = D.frames;

    // ---- geometry (metres): the roads' extent, in view and beyond
    const roads = D.roads.map((r) => {
      const p = [];
      for (let i = 0; i < r.xy.length; i += 2) p.push([r.xy[i] / 10, r.xy[i + 1] / 10]);
      return { p, w: r.w, view: r.view };
    });
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    roads.forEach((r) => r.p.forEach(([u, v]) => {
      const h = r.w / 2;
      u0 = Math.min(u0, u - h); u1 = Math.max(u1, u + h); v0 = Math.min(v0, v - h); v1 = Math.max(v1, v + h);
    }));
    u0 -= 2; u1 += 2; v0 -= 2; v1 += 2;

    root.classList.add(D.layout === "stack" ? "rp-stack" : "rp-side");
    root.innerHTML = `
      <div class="ma-controls">
        <button type="button" class="ma-play" aria-label="Pause">❚❚</button>
        <input type="range" class="ma-time" min="0" max="${N - 1}" step="1" value="0" aria-label="Time">
        <span class="ma-clock">${clock(T0)}</span>
        <span class="rp-title">${D.title}</span>
      </div>
      <div class="od-panels">
        <figure class="od-panel"><figcaption class="rp-obs-label">Observed</figcaption><canvas class="rp-obs"></canvas></figure>
        <figure class="od-panel"><figcaption class="rp-sim-label">${D.sim.label}</figcaption><canvas class="rp-sim"></canvas></figure>
      </div>
      <canvas class="rp-strip" aria-label="Vehicles in view over the replay"></canvas>
      <div class="ma-legend">
        <span><i class="ma-dot rp-c-obs"></i>observed</span>
        <span><i class="ma-dot rp-c-sim"></i>simulated</span>
        <span class="ma-note">${D.note} ${D.attribution}</span>
      </div>`;
    const play = root.querySelector(".ma-play"), slider = root.querySelector(".ma-time");
    const clockEl = root.querySelector(".ma-clock");
    const cv = { obs: root.querySelector(".rp-obs"), sim: root.querySelector(".rp-sim") };
    const strip = root.querySelector(".rp-strip");
    const labels = { obs: root.querySelector(".rp-obs-label"), sim: root.querySelector(".rp-sim-label") };

    let frame = 0, pos = 0, playing = false, wanted = true, last = null, W = 300, H = 300, scale = 1, SW = 600;

    function layout() {
      W = cv.obs.parentElement.clientWidth || 320;
      scale = (W - 2 * PAD) / (u1 - u0);
      H = Math.round((v1 - v0) * STRETCH * scale + 2 * PAD);
      const dpr = window.devicePixelRatio || 1;
      for (const c of Object.values(cv)) {
        c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
        c.style.width = W + "px"; c.style.height = H + "px";
        c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      SW = root.clientWidth || 600;
      strip.width = Math.round(SW * dpr); strip.height = Math.round(STRIP_H * dpr);
      strip.style.width = SW + "px"; strip.style.height = STRIP_H + "px";
      strip.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    const px = (u) => PAD + (u - u0) * scale;
    const py = (v) => PAD + (v1 - v) * STRETCH * scale;
    const dot = () => Math.max(2.6, Math.min(4.4, 2.2 * scale));

    function drawPanel(canvas, tracks, colour, col) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = "butt"; ctx.lineJoin = "round";
      for (const pass of [false, true]) { // faded roads first, in-view roads over them
        roads.filter((r) => r.view === pass).forEach((r) => {
          ctx.strokeStyle = r.view ? col.road : col.faded;
          ctx.lineWidth = Math.max(2, r.w * STRETCH * scale);
          ctx.beginPath();
          r.p.forEach(([u, v], i) => (i ? ctx.lineTo(px(u), py(v)) : ctx.moveTo(px(u), py(v))));
          ctx.stroke();
        });
      }
      if (!tracks) {
        ctx.fillStyle = col.muted; ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("Observed tracks are not part of this build (see the page's data licence).", W / 2, H / 2 + 4);
        return 0;
      }
      let n = 0;
      const r = dot();
      ctx.fillStyle = colour; ctx.strokeStyle = col.bg; ctx.lineWidth = 1.2;
      for (const t of tracks) {
        const k = frame - t.f;
        if (k < 0 || 2 * k >= t.xy.length) continue;
        n++;
        ctx.beginPath();
        ctx.arc(px(t.xy[2 * k] / 10), py(t.xy[2 * k + 1] / 10), r, 0, 2 * Math.PI);
        ctx.fill(); ctx.stroke();
      }
      return n;
    }

    function drawStrip(col) {
      const ctx = strip.getContext("2d");
      ctx.clearRect(0, 0, SW, STRIP_H);
      const obs = O ? D.obs_inview : null, sim = D.sim.inview;
      const top = Math.max(1, ...(obs || []), ...sim) * 1.1;
      const x = (f) => (f / (N - 1)) * SW, y = (n) => STRIP_H - 4 - (n / top) * (STRIP_H - 16);
      ctx.fillStyle = col.band; ctx.fillRect(0, 0, SW, STRIP_H);
      const line = (a, c) => {
        ctx.strokeStyle = c; ctx.lineWidth = 1.4; ctx.beginPath();
        a.forEach((n, f) => (f ? ctx.lineTo(x(f), y(n)) : ctx.moveTo(x(f), y(n))));
        ctx.stroke();
      };
      if (obs) line(obs, col.obs);
      line(sim, col.sim);
      ctx.strokeStyle = col.text; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(x(frame), 2); ctx.lineTo(x(frame), STRIP_H - 2); ctx.stroke();
      ctx.fillStyle = col.muted; ctx.font = "10px system-ui, sans-serif"; ctx.textAlign = "left";
      ctx.fillText("vehicles in view", 6, 11);
      ctx.textAlign = "right";
      const now = (obs ? `observed ${obs[frame]} · ` : "") + `simulated ${sim[frame]}`;
      ctx.fillText(now, SW - 6, 11);
    }

    function draw() {
      const col = {
        road: css(root, "--ma-road", "#c9ccd1"), faded: css(root, "--ma-road-faded", "#e6e8eb"),
        muted: css(root, "--ma-muted", "#6b7078"), text: css(root, "--ma-text", "#222"),
        bg: css(root, "--ma-band", "#f3f5f8"), band: css(root, "--ma-band", "#f3f5f8"),
        obs: css(root, "--rp-obs", "#2a78d6"), sim: css(root, "--rp-sim", "#1baf7a"),
      };
      const nObs = drawPanel(cv.obs, O ? O.tracks : null, col.obs, col);
      const nSim = drawPanel(cv.sim, D.sim.tracks, col.sim, col);
      labels.obs.textContent = O ? `Observed · ${nObs} on screen` : "Observed";
      labels.sim.textContent = `${D.sim.label} · ${nSim} on screen`;
      drawStrip(col);
      clockEl.textContent = clock(T0 + frame / FPS);
      slider.value = frame;
    }

    function tick(ts) {
      if (!playing) return;
      if (last != null) pos += ((ts - last) / 1000) * D.speed * FPS;
      last = ts;
      if (pos >= N - 1) pos = 0; // loop
      frame = Math.floor(pos);
      draw();
      requestAnimationFrame(tick);
    }

    function setPlaying(p) {
      playing = p;
      play.textContent = p ? "❚❚" : "▶";
      play.setAttribute("aria-label", p ? "Pause" : "Play");
      last = null;
      if (p) requestAnimationFrame(tick);
    }
    play.addEventListener("click", () => { wanted = !playing; setPlaying(wanted); });
    slider.addEventListener("input", () => { frame = pos = +slider.value; draw(); });
    const seek = (e) => {
      const r = strip.getBoundingClientRect();
      frame = pos = Math.max(0, Math.min(N - 1, Math.round(((e.clientX - r.left) / r.width) * (N - 1))));
      draw();
    };
    strip.addEventListener("pointerdown", (e) => { seek(e); strip.setPointerCapture(e.pointerId); });
    strip.addEventListener("pointermove", (e) => { if (e.buttons) seek(e); });
    new ResizeObserver(() => { layout(); draw(); }).observe(root);
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    // play while on screen, pause when scrolled away (and when the tab is hidden)
    new IntersectionObserver((es) => es.forEach((e) => setPlaying(e.isIntersecting && wanted)), { threshold: 0.3 })
      .observe(root);
    document.addEventListener("visibilitychange", () => { if (document.hidden) setPlaying(false); else if (wanted) setPlaying(true); });
    layout();
    draw();
  }

  function start() { document.querySelectorAll(".replay-anim").forEach(init); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
