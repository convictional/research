/* ── Inline SVG diagrams for the Knowledge Collapse deck ──────────
   All diagrams render at a natural size and scale via CSS.
   Brand colors: ink #19140e, muted #514a44, faint #a89f91,
                 rule #d6cfbf, blue #2b7fff, paper #fffdf5
   Hand-drawn feel: slightly imperfect strokes via roughness filter.
*/

/* ===========================================================
   1. COST CURVES — c(e) = (1/α)·e^α for α ∈ {0.5, 0.7, 1.1, 1.5, 2, 3}
   =========================================================== */
function costCurvesSVG() {
  const W = 900, H = 560;
  const padL = 90, padR = 40, padT = 40, padB = 80;
  const iw = W - padL - padR, ih = H - padT - padB;

  // Sub-1 alphas (concave) in warm amber; super-1 in dark-to-blue
  const alphas = [
    { a: 0.5, label: "α = 0.5", color: "#c08040", accent: true },
    { a: 0.7, label: "α = 0.7", color: "#d4b07a", accent: true },
    { a: 1.1, label: "α = 1.1", color: "#19140e", accent: false },
    { a: 1.5, label: "α = 1.5", color: "#514a44", accent: false },
    { a: 2.0, label: "α = 2.0", color: "#7a94ad", accent: false },
    { a: 3.0, label: "α = 3.0", color: "#2b7fff", accent: false },
  ];

  const xMax = 2.5, yMax = 2.5;
  const x2p = (x) => padL + (x / xMax) * iw;
  const y2p = (y) => padT + ih - (y / yMax) * ih;

  const curve = (a) => {
    const pts = [];
    for (let i = 0; i <= 200; i++) {
      const e = (i / 200) * xMax;
      const y = (1 / a) * Math.pow(e, a);
      if (y > yMax) break; // stop drawing when curve leaves the frame
      pts.push(`${x2p(e).toFixed(1)},${y2p(y).toFixed(1)}`);
    }
    return pts.join(" ");
  };

  const gridX = [0, 0.5, 1, 1.5, 2, 2.5];
  const gridY = [0, 0.5, 1, 1.5, 2, 2.5];

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cost curves c(e)=e^α/α for six alphas including sub-1">
    <!-- axes + grid -->
    ${gridY.map(y => `<line x1="${padL}" y1="${y2p(y)}" x2="${padL+iw}" y2="${y2p(y)}" stroke="#e6dfce" stroke-width="1" />`).join("")}
    ${gridX.map(x => `<line x1="${x2p(x)}" y1="${padT}" x2="${x2p(x)}" y2="${padT+ih}" stroke="#e6dfce" stroke-width="1" />`).join("")}

    <!-- axis lines -->
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT+ih}" stroke="#19140e" stroke-width="1.6" />
    <line x1="${padL}" y1="${padT+ih}" x2="${padL+iw}" y2="${padT+ih}" stroke="#19140e" stroke-width="1.6" />

    <!-- x tick labels -->
    ${gridX.map(x => `<text x="${x2p(x)}" y="${padT+ih+28}" text-anchor="middle" font-family="Inter, sans-serif" font-size="18" fill="#a89f91">${x}</text>`).join("")}
    <!-- y tick labels -->
    ${gridY.map(y => `<text x="${padL-14}" y="${y2p(y)+6}" text-anchor="end" font-family="Inter, sans-serif" font-size="18" fill="#a89f91">${y.toFixed(1)}</text>`).join("")}

    <!-- axis labels -->
    <text x="${padL+iw/2}" y="${H - 18}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="24" fill="#19140e">effort e</text>
    <text x="30" y="${padT+ih/2}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="24" fill="#19140e" transform="rotate(-90 30 ${padT+ih/2})">cost  c(e)</text>

    <!-- curves — draw sub-1 first (behind) then super-1 on top -->
    ${alphas.map(a => `
      <polyline points="${curve(a.a)}" fill="none" stroke="${a.color}" stroke-width="${a.a === 0.5 || a.a === 3.0 ? 3 : 2.2}" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="${a.a < 1 ? '8 4' : 'none'}" />
    `).join("")}

    <!-- legend — right side to avoid overlap with concave curves -->
    <g transform="translate(${padL + iw - 200}, ${padT + 30})">
      <rect width="196" height="${alphas.length * 32 + 16}" fill="#fffdf5" fill-opacity="0.92" stroke="#d6cfbf" />
      ${alphas.map((a, i) => `
        <g transform="translate(14, ${14 + i*32})">
          <line x1="0" y1="9" x2="26" y2="9" stroke="${a.color}" stroke-width="2.5" stroke-linecap="round" ${a.a < 1 ? 'stroke-dasharray="6 3"' : ''} />
          <text x="36" y="14" font-family="Inter, sans-serif" font-size="18" fill="${a.a < 1 ? a.color : '#19140e'}" font-weight="${a.a < 1 ? '600' : '400'}">${a.label}</text>
        </g>
      `).join("")}
    </g>

    <!-- annotation: concave (sub-1) -->
    <text x="${x2p(1.55)}" y="${y2p(2.35)}" text-anchor="start" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#c08040">
      <tspan x="${x2p(1.55)}" dy="0">concave — cost grows</tspan>
      <tspan x="${x2p(1.55)}" dy="22">slower than effort</tspan>
    </text>

    <!-- annotation: steep (super-1) -->
    <text x="${x2p(0.72)}" y="${y2p(2.38)}" text-anchor="start" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#2b7fff">
      <tspan x="${x2p(0.72)}" dy="0">compounds steeply —</tspan>
      <tspan x="${x2p(0.72)}" dy="22">high effort has to earn it</tspan>
    </text>
  </svg>
  `;
}

/* ===========================================================
   2. PAYOFF LATTICE — axonometric lattice surface over unit
   square in the G–I plane, with heights at each corner
   reflecting the payoff value f(G, I).
      Origin A(0,0) at (ox, oy); G → right, depth → back-right,
      I value → up. Heights (in px):
        f(0,0) = 0     → stays on floor
        f(1,0) = 60    → small lift (common right alone)
        f(0,1) = 170   → mid (private right without common)
        f(1,1) = 260   → peak (both right)
   =========================================================== */
function payoffLatticeSVG() {
  const W = 900, H = 640;

  // Coord system (floor plane)
  const ox = 180, oy = 500;
  // Floor corners
  const A = { x: ox,       y: oy       }; // (0,0)
  const B = { x: ox + 340, y: oy       }; // (1,0) floor
  const D = { x: ox + 140, y: oy - 70  }; // (0,1) floor (back-left)
  const C = { x: ox + 480, y: oy - 70  }; // (1,1) floor (back-right)
  // Lifted surface corners
  const Ap = A;                                   // f(0,0) = 0
  const Bp = { x: B.x, y: B.y - 60  };            // f(1,0)
  const Cp = { x: C.x, y: C.y - 260 };            // f(1,1)
  const Dp = { x: D.x, y: D.y - 170 };            // f(0,1)

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Payoff lattice surface over the unit square in the G–I plane">
    <defs>
      <pattern id="pay-hatch" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(60)">
        <line x1="0" y1="0" x2="0" y2="10" stroke="#2b7fff" stroke-width="1" stroke-opacity="0.55"/>
      </pattern>
      <marker id="arr-pay" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 Z" fill="#19140e"/>
      </marker>
    </defs>

    <!-- ===== FLOOR construction ===== -->
    <!-- Depth axis from origin back to D (solid thin) -->
    <line x1="${A.x}" y1="${A.y}" x2="${D.x}" y2="${D.y}" stroke="#19140e" stroke-width="1.1" fill="none"/>
    <!-- Dashed floor lines: B→C, D→C -->
    <line x1="${B.x}" y1="${B.y}" x2="${C.x}" y2="${C.y}" stroke="#a89f91" stroke-width="1" stroke-dasharray="5 5"/>
    <line x1="${D.x}" y1="${D.y}" x2="${C.x}" y2="${C.y}" stroke="#a89f91" stroke-width="1" stroke-dasharray="5 5"/>
    <!-- Vertical dashed risers from floor to surface -->
    <line x1="${B.x}" y1="${B.y}" x2="${Bp.x}" y2="${Bp.y}" stroke="#a89f91" stroke-width="1" stroke-dasharray="5 5"/>
    <line x1="${C.x}" y1="${C.y}" x2="${Cp.x}" y2="${Cp.y}" stroke="#a89f91" stroke-width="1" stroke-dasharray="5 5"/>
    <line x1="${D.x}" y1="${D.y}" x2="${Dp.x}" y2="${Dp.y}" stroke="#a89f91" stroke-width="1" stroke-dasharray="5 5"/>

    <!-- (0,1) floor corner node, drawn before surface so it layers behind -->
    <circle cx="${D.x}" cy="${D.y}" r="4.5" fill="#fffdf5" stroke="#19140e" stroke-width="1.4"/>
    <!-- "I" axis label tucked at the back-left, behind the plane -->
    <text x="${D.x - 12}" y="${D.y - 8}" text-anchor="end" font-family="loretta, Georgia, serif" font-style="italic" font-size="30" fill="#19140e">I</text>

    <!-- ===== AXES ===== -->
    <line x1="${A.x}" y1="${A.y}" x2="${A.x + 580}" y2="${A.y}" stroke="#19140e" stroke-width="1.6" fill="none" stroke-linecap="round" marker-end="url(#arr-pay)"/>
    <line x1="${A.x}" y1="${A.y}" x2="${A.x}" y2="${A.y - 430}" stroke="#19140e" stroke-width="1.6" fill="none" stroke-linecap="round" marker-end="url(#arr-pay)"/>

    <!-- ===== SURFACE ===== -->
    <path d="M ${Ap.x} ${Ap.y} L ${Bp.x} ${Bp.y} L ${Cp.x} ${Cp.y} L ${Dp.x} ${Dp.y} Z"
          fill="#2b7fff" fill-opacity="0.08" stroke="none"/>
    <path d="M ${Ap.x} ${Ap.y} L ${Bp.x} ${Bp.y} L ${Cp.x} ${Cp.y} L ${Dp.x} ${Dp.y} Z"
          fill="url(#pay-hatch)" stroke="none"/>
    <path d="M ${Ap.x} ${Ap.y} L ${Bp.x} ${Bp.y} L ${Cp.x} ${Cp.y} L ${Dp.x} ${Dp.y} Z"
          stroke="#2b7fff" stroke-width="2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>

    <!-- ===== NODES ===== -->
    <circle cx="${A.x}" cy="${A.y}" r="3" fill="#19140e"/>       <!-- (0,0) -->
    <circle cx="${B.x}" cy="${B.y}" r="4.5" fill="#fffdf5" stroke="#19140e" stroke-width="1.4"/>   <!-- (1,0) -->
    <circle cx="${C.x}" cy="${C.y}" r="4.5" fill="#fffdf5" stroke="#19140e" stroke-width="1.4"/>   <!-- (1,1) -->

    <!-- ===== AXIS + CORNER LABELS ===== -->
    <text x="${A.x + 595}" y="${A.y + 8}" font-family="loretta, Georgia, serif" font-style="italic" font-size="30" fill="#19140e">G</text>
    <text x="${A.x}" y="${A.y - 445}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e">f(general, idiosyncratic)</text>

    <text x="${A.x - 18}" y="${A.y + 30}" text-anchor="end" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#514a44">(0, 0)</text>
    <text x="${B.x}" y="${B.y + 30}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#514a44">(1, 0)</text>
    <text x="${C.x + 22}" y="${C.y + 22}" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#514a44">(1, 1)</text>
    <text x="${D.x + 18}" y="${D.y - 12}" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#514a44">(0, 1)</text>
  </svg>
  `;
}

/* ===========================================================
   3. F-MAP BIFURCATION — three panels showing F(X) under
      different τ_A regimes (elastic α).
      Left: τ_A < τ_A^c — S-curve with X_m (unstable) and X_h (stable)
      Mid:  τ_A = τ_A^c — tangent to 45° line (saddle-node)
      Right: τ_A > τ_A^c — F entirely below 45° → collapse to 0
   =========================================================== */
function fMapTripleSVG() {
  const panelW = 460, panelH = 520;
  const gap = 40;
  const W = panelW * 3 + gap * 2;
  const H = panelH + 80;

  const pad = { l: 60, r: 24, t: 50, b: 110 };
  const iw = panelW - pad.l - pad.r;
  const ih = panelH - pad.t - pad.b;

  // Domain [0, 10]
  const xMax = 10, yMax = 10;
  const x2p = (x) => pad.l + (x / xMax) * iw;
  const y2p = (y) => pad.t + ih - (y / yMax) * ih;

  // F functions
  // Left: S-curve crossing 45° twice plus origin:
  //   F(X) = A * X^2 / (B^2 + X^2) * (1 + 0.15*tanh((X-mid)))  — approx Hill + slight S
  // Use a logistic-ish form: F = 8 * x^2 / (x^2 + 9) shifted — tweak for visual.
  // We'll define curves empirically so they match the sketch.

  // LEFT panel — two positive fixed points
  const FL = (x) => {
    // S-curve peaking near 6.5
    const k = 1.6;
    const mid = 3.2;
    return 8.0 * (1 / (1 + Math.exp(-k * (x - mid)))) - 8.0 * (1 / (1 + Math.exp(k * mid)));
  };

  // MIDDLE — saddle-node: tangent to 45° near x≈2
  const FM = (x) => {
    // Lower the whole curve slightly
    const k = 1.5;
    const mid = 3.5;
    const val = 7.0 * (1 / (1 + Math.exp(-k * (x - mid)))) - 7.0 * (1 / (1 + Math.exp(k * mid)));
    return val - 0.5;
  };

  // RIGHT — entirely below y=x but with a knee (collapse)
  const FR = (x) => {
    const k = 1.2;
    const mid = 4.0;
    const val = 5.5 * (1 / (1 + Math.exp(-k * (x - mid)))) - 5.5 * (1 / (1 + Math.exp(k * mid)));
    return val;
  };

  const makeCurve = (F) => {
    const pts = [];
    for (let i = 0; i <= 200; i++) {
      const x = (i / 200) * xMax;
      const y = F(x);
      pts.push(`${x2p(x).toFixed(1)},${y2p(Math.max(0, Math.min(yMax, y))).toFixed(1)}`);
    }
    return pts.join(" ");
  };

  // Find fixed points numerically where F(x) ≈ x
  const findFixedPoints = (F) => {
    const roots = [];
    let prev = F(0) - 0;
    for (let i = 1; i <= 400; i++) {
      const x = (i / 400) * xMax;
      const v = F(x) - x;
      if ((prev <= 0 && v > 0) || (prev >= 0 && v < 0)) {
        // bisect
        let lo = (i - 1) / 400 * xMax, hi = i / 400 * xMax;
        for (let k = 0; k < 40; k++) {
          const m = (lo + hi) / 2;
          if ((F(lo) - lo) * (F(m) - m) < 0) hi = m; else lo = m;
        }
        roots.push((lo + hi) / 2);
      }
      prev = v;
    }
    return roots;
  };

  const drawPanel = (title, F, annotations = [], i) => {
    const ox = (panelW + gap) * i;
    const curve = makeCurve(F);
    const fixed = findFixedPoints(F);
    // Stability: F'(x*) < 1 → stable (solid), > 1 → unstable (hollow)
    const dF = (F, x) => (F(x + 0.01) - F(x - 0.01)) / 0.02;

    return `
    <g transform="translate(${ox}, 0)">
      <text x="${panelW / 2}" y="${pad.t - 18}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="26" fill="#19140e">${title}</text>

      <!-- grid -->
      ${[0, 2, 4, 6, 8, 10].map(v => `
        <line x1="${x2p(v)}" y1="${pad.t}" x2="${x2p(v)}" y2="${pad.t + ih}" stroke="#f2ecdb" stroke-width="1"/>
        <line x1="${pad.l}" y1="${y2p(v)}" x2="${pad.l + iw}" y2="${y2p(v)}" stroke="#f2ecdb" stroke-width="1"/>
      `).join("")}

      <!-- 45° reference line -->
      <line x1="${x2p(0)}" y1="${y2p(0)}" x2="${x2p(yMax)}" y2="${y2p(yMax)}" stroke="#a89f91" stroke-width="1.5" stroke-dasharray="6 6"/>
      <text x="${x2p(yMax) - 8}" y="${y2p(yMax) + 22}" text-anchor="end" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#a89f91">45°</text>

      <!-- axes -->
      <line x1="${pad.l}" y1="${pad.t + ih}" x2="${pad.l + iw}" y2="${pad.t + ih}" stroke="#19140e" stroke-width="1.6"/>
      <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ih}" stroke="#19140e" stroke-width="1.6"/>

      <!-- F curve -->
      <polyline points="${curve}" fill="none" stroke="#2b7fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>

      <!-- Fixed points -->
      ${fixed.map(x => {
        const slope = dF(F, x);
        const stable = slope < 1;
        return `<circle cx="${x2p(x)}" cy="${y2p(x)}" r="8" fill="${stable ? '#2b7fff' : '#fffdf5'}" stroke="#2b7fff" stroke-width="2.5"/>`;
      }).join("")}

      <!-- origin dot always -->
      <circle cx="${x2p(0)}" cy="${y2p(0)}" r="8" fill="#2b7fff" stroke="#2b7fff" stroke-width="2.5"/>

      <!-- axis labels -->
      <text x="${pad.l + iw / 2}" y="${pad.t + ih + 78}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e">X<tspan font-size="15" dy="6">t</tspan>  (knowledge now)</text>
      <text x="18" y="${pad.t + ih / 2}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e" transform="rotate(-90 18 ${pad.t + ih / 2})">X<tspan font-size="15" dy="6">t+1</tspan>  (next)</text>

      <!-- F label -->
      <text x="${x2p(9)}" y="${y2p(F(9)) - 18}" text-anchor="end" font-family="loretta, Georgia, serif" font-style="italic" font-size="26" fill="#2b7fff">F</text>

      ${annotations}
    </g>`;
  };

  // Annotations for left panel — X_m (hollow, unstable) and X_h (filled, stable)
  const fixedL = findFixedPoints(FL);
  // roots ≈ [small unstable, large stable]
  const xm = fixedL[0] ?? 1.5;
  const xh = fixedL[1] ?? 6.5;

  const annotL = `
    <line x1="${x2p(xm)}" y1="${y2p(xm)}" x2="${x2p(xm)}" y2="${pad.t + ih + 8}" stroke="#19140e" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="${x2p(xm)}" y="${pad.t + ih + 32}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e">X<tspan font-size="15" dy="6">m</tspan></text>
    <line x1="${x2p(xh)}" y1="${y2p(xh)}" x2="${x2p(xh)}" y2="${pad.t + ih + 8}" stroke="#19140e" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="${x2p(xh)}" y="${pad.t + ih + 32}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e">X<tspan font-size="15" dy="6">h</tspan></text>
    <text x="${x2p(3.4)}" y="${y2p(1.4)}" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">unstable</text>
    <text x="${x2p(7.8)}" y="${y2p(5.8)}" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">stable</text>
  `;

  // Middle — single touching point
  const fixedM = findFixedPoints(FM);
  const xt = fixedM.length ? fixedM[fixedM.length - 1] : 3.5;
  const annotM = `
    <line x1="${x2p(xt)}" y1="${y2p(xt)}" x2="${x2p(xt)}" y2="${pad.t + ih + 8}" stroke="#19140e" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="${x2p(xt)}" y="${pad.t + ih + 32}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="22" fill="#19140e">X*</text>
    <text x="${x2p(6)}" y="${y2p(7.2)}" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">tangent</text>
  `;

  // Right — no positive fixed points
  const annotR = `
    <text x="${x2p(5)}" y="${y2p(8.2)}" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">F entirely below 45°</text>
    <text x="${x2p(5)}" y="${y2p(7.4)}" font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">— collapse to 0</text>
  `;

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three F(X) panels showing bifurcation as AI capability crosses threshold">
    ${drawPanel('τ_A < τ_A<tspan font-size="18" dy="-10">c</tspan>', FL, annotL, 0).replace('τ_A < τ_A<tspan font-size="18" dy="-10">c</tspan>', '<tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan dy="-8"> &lt; </tspan><tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan font-size="18" dy="-18">c</tspan>')}
    ${drawPanel('saddle', FM, annotM, 1).replace('saddle', '<tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan dy="-8"> = </tspan><tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan font-size="18" dy="-18">c</tspan>')}
    ${drawPanel('right', FR, annotR, 2).replace('right', '<tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan dy="-8"> &gt; </tspan><tspan font-family="loretta, Georgia, serif" font-style="italic" font-size="26">τ</tspan><tspan font-size="18" dy="8" font-family="Inter">A</tspan><tspan font-size="18" dy="-18">c</tspan>')}

    <!-- Legend -->
    <g transform="translate(${W / 2 - 240}, ${H - 26})">
      <circle cx="0" cy="0" r="7" fill="#2b7fff" stroke="#2b7fff" stroke-width="2.5"/>
      <text x="14" y="6" font-family="Inter, sans-serif" font-size="17" fill="#514a44">stable fixed point</text>
      <circle cx="200" cy="0" r="7" fill="#fffdf5" stroke="#2b7fff" stroke-width="2.5"/>
      <text x="214" y="6" font-family="Inter, sans-serif" font-size="17" fill="#514a44">unstable fixed point</text>
    </g>
  </svg>`;
}

/* ===========================================================
   3b. F-MAP — two regimes, three panels (replaces fMapTripleSVG)
       Left:   inelastic α (α−1 > ¼) — concave F, single stable X_h
       Middle: elastic α, τ_A < τ_A^c — S-curve, three fixed points
       Right:  elastic α, τ_A > τ_A^c — F below 45°, only collapse
   =========================================================== */
function fMapNewSVG() {
  const pW = 444, pH = 480;
  const gap = 46;
  const W = pW * 3 + gap * 2;
  const GH = 52;   // group-label header height
  const BH = 36;   // bottom legend strip
  const H = GH + pH + BH;

  const pad = { l: 56, r: 20, t: 46, b: 86 };
  const iw = pW - pad.l - pad.r;   // 368
  const ih = pH - pad.t - pad.b;   // 348

  const xMax = 10, yMax = 10;
  const xp = (x) => pad.l + (x / xMax) * iw;
  const yp = (y) => pad.t + ih - (y / yMax) * ih;

  // ── F curves ──────────────────────────────────────────────
  // Inelastic: concave hyperbolic, crosses 45° once at X_h ≈ 6.3.
  // F'(0)=6.25>1 → origin unstable; F'(X_h)<1 → X_h stable.
  const F_inel = (x) => 7.5 * x / (1.2 + x);

  // Elastic, low τ_A: logistic S-curve, X_l=0 stable, X_m unstable, X_h stable.
  const F_lo = (x) =>  8.0 / (1 + Math.exp(-1.6 * (x - 3.2)))
                     - 8.0 / (1 + Math.exp( 1.6 * 3.2));

  // Elastic, high τ_A: S-curve entirely below 45°, only X_l=0 stable.
  const F_hi = (x) =>  5.5 / (1 + Math.exp(-1.2 * (x - 4.0)))
                     - 5.5 / (1 + Math.exp( 1.2 * 4.0));

  const curve = (F) => {
    const pts = [];
    for (let i = 0; i <= 240; i++) {
      const x = (i / 240) * xMax;
      pts.push(`${xp(x).toFixed(1)},${yp(Math.max(0, Math.min(yMax, F(x)))).toFixed(1)}`);
    }
    return pts.join(' ');
  };

  // ── Fixed-point finder (bisection, skips origin) ──────────
  const findFP = (F) => {
    const roots = [];
    let prev = F(0.02) - 0.02;
    for (let i = 1; i <= 500; i++) {
      const x = i / 500 * xMax;
      const v = F(x) - x;
      if (prev * v < 0) {
        let lo = (i - 1) / 500 * xMax, hi = x;
        for (let k = 0; k < 50; k++) {
          const m = (lo + hi) / 2;
          (F(lo) - lo) * (F(m) - m) < 0 ? hi = m : lo = m;
        }
        roots.push((lo + hi) / 2);
      }
      prev = v;
    }
    return roots;
  };

  // ── SVG helpers ───────────────────────────────────────────
  const dot = (x, stable) =>
    `<circle cx="${xp(x).toFixed(1)}" cy="${yp(x).toFixed(1)}" r="8"
             fill="${stable ? '#2b7fff' : '#fffdf5'}" stroke="#2b7fff" stroke-width="2.5"/>`;

  const vdrop = (x) =>
    `<line x1="${xp(x).toFixed(1)}" y1="${yp(x).toFixed(1)}"
           x2="${xp(x).toFixed(1)}" y2="${(pad.t + ih + 10).toFixed(1)}"
           stroke="#a89f91" stroke-width="1" stroke-dasharray="3 3"/>`;

  const xlabel = (x, sub, muted) =>
    `<text x="${xp(x).toFixed(1)}" y="${pad.t + ih + 30}" text-anchor="middle"
           font-family="loretta, Georgia, serif" font-style="italic" font-size="20"
           fill="${muted ? '#a89f91' : '#19140e'}">X<tspan font-size="13" dy="5">${sub}</tspan></text>`;

  // Shared panel chrome (grid, 45° dashed, axes, axis labels)
  const chrome = `
    ${[0,2,4,6,8,10].map(v => `
      <line x1="${xp(v).toFixed(1)}" y1="${pad.t}" x2="${xp(v).toFixed(1)}" y2="${pad.t+ih}" stroke="#ede8dc" stroke-width="1"/>
      <line x1="${pad.l}" y1="${yp(v).toFixed(1)}" x2="${pad.l+iw}" y2="${yp(v).toFixed(1)}" stroke="#ede8dc" stroke-width="1"/>
    `).join('')}
    <line x1="${xp(0)}" y1="${yp(0)}" x2="${xp(10)}" y2="${yp(10)}"
          stroke="#b8b0a0" stroke-width="1.4" stroke-dasharray="7 5"/>
    <line x1="${pad.l}" y1="${pad.t+ih}" x2="${pad.l+iw}" y2="${pad.t+ih}" stroke="#19140e" stroke-width="1.6"/>
    <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+ih}" stroke="#19140e" stroke-width="1.6"/>
    <text x="${pad.l + iw/2}" y="${pad.t+ih+62}" text-anchor="middle"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="19" fill="#19140e">
      X<tspan font-size="12" dy="5">t</tspan><tspan dy="-5" font-size="14"> (knowledge now)</tspan>
    </text>
    <text x="15" y="${pad.t+ih/2}" text-anchor="middle"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="19" fill="#19140e"
          transform="rotate(-90 15 ${pad.t+ih/2})">
      X<tspan font-size="12" dy="5">t+1</tspan><tspan dy="-5" font-size="14"> (next)</tspan>
    </text>`;

  // ── Panel 1: Inelastic ────────────────────────────────────
  const fps1 = findFP(F_inel);
  const xh1  = fps1[0] ?? 6.3;
  const arrY  = pad.t + ih + 20; // arrow row between x-axis and x-label

  const panel1 = `
  <g transform="translate(0,${GH})">
    ${chrome}
    <polyline points="${curve(F_inel)}" fill="none" stroke="#2b7fff"
              stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="${xp(9.5)}" y="${(yp(F_inel(9.5)) - 14).toFixed(1)}" text-anchor="end"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="24" fill="#2b7fff">F</text>
    ${dot(0, false)}
    ${dot(xh1, true)}
    ${xlabel(0, 'l', true)}
    ${vdrop(xh1)}${xlabel(xh1, 'h', false)}
    <!-- dynamics arrows: → left of X_h, ← right of X_h -->
    <line x1="${xp(1.0)}" y1="${arrY}" x2="${xp(2.8)}" y2="${arrY}"
          stroke="#8a7560" stroke-width="1.8" marker-end="url(#arr-dyn)"/>
    <line x1="${xp(9.2)}" y1="${arrY}" x2="${xp(7.4)}" y2="${arrY}"
          stroke="#8a7560" stroke-width="1.8" marker-end="url(#arr-dyn)"/>
    <text x="${xp(4.8)}" y="${yp(2.6)}" text-anchor="middle"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="16" fill="#514a44">stable</text>
  </g>`;

  // ── Panel 2: Elastic, low τ_A ─────────────────────────────
  const fps2 = findFP(F_lo);
  const xm2  = fps2[0] ?? 2.85;
  const xh2  = fps2[1] ?? 7.9;
  const ox2  = pW + gap;

  const panel2 = `
  <g transform="translate(${ox2},${GH})">
    ${chrome}
    <polyline points="${curve(F_lo)}" fill="none" stroke="#2b7fff"
              stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="${xp(8.6)}" y="${(yp(F_lo(8.6)) - 14).toFixed(1)}" text-anchor="end"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="24" fill="#2b7fff">F</text>
    ${dot(0, true)}
    ${dot(xm2, false)}
    ${dot(xh2, true)}
    ${xlabel(0, 'l', false)}
    ${vdrop(xm2)}${xlabel(xm2, 'm', false)}
    ${vdrop(xh2)}${xlabel(xh2, 'h', false)}
    <text x="${xp(3.6)}" y="${yp(1.3)}"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="15" fill="#514a44">unstable</text>
    <text x="${xp(7.8)}" y="${yp(5.6)}"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="15" fill="#514a44">stable</text>
  </g>`;

  // ── Panel 3: Elastic, high τ_A ────────────────────────────
  const ox3 = 2 * (pW + gap);

  const panel3 = `
  <g transform="translate(${ox3},${GH})">
    ${chrome}
    <polyline points="${curve(F_hi)}" fill="none" stroke="#2b7fff"
              stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    <text x="${xp(9.2)}" y="${(yp(F_hi(9.2)) - 14).toFixed(1)}" text-anchor="end"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="24" fill="#2b7fff">F</text>
    ${dot(0, true)}
    ${xlabel(0, 'l', false)}
    <text x="${xp(5)}" y="${yp(8.0)}" text-anchor="middle"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="16" fill="#514a44">F entirely below</text>
    <text x="${xp(5)}" y="${yp(7.2)}" text-anchor="middle"
          font-family="loretta, Georgia, serif" font-style="italic" font-size="16" fill="#514a44">→ only collapse to 0</text>
  </g>`;

  // τ_A sub-label inside the GH strip, above each elastic panel
  const tauSub = (ox, cmp) =>
    `<text x="${ox + pW/2}" y="${GH - 8}" text-anchor="middle"
           font-family="Inter, sans-serif" font-size="17" fill="#514a44">
       τ<tspan font-size="12" dy="5">A</tspan><tspan dy="-5"> ${cmp} τ</tspan><tspan font-size="12" dy="5">A</tspan><tspan font-size="11" dy="-8">c</tspan>
     </text>`;

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"
       role="img" aria-label="F-map panels: inelastic and elastic alpha regimes">
    <defs>
      <marker id="arr-dyn" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,1 L9,5 L0,9 Z" fill="#8a7560"/>
      </marker>
    </defs>

    <!-- ── Group headers ── -->
    <text x="${pW/2}" y="20" text-anchor="middle"
          font-family="Inter, sans-serif" font-weight="600" font-size="17"
          letter-spacing="0.06em" fill="#514a44">α − 1 &gt; ¼  ·  inelastic</text>
    <line x1="${pW/2 - 128}" y1="28" x2="${pW/2 + 128}" y2="28"
          stroke="#b8b0a0" stroke-width="1"/>

    <text x="${ox2 + pW + gap/2}" y="20" text-anchor="middle"
          font-family="Inter, sans-serif" font-weight="600" font-size="17"
          letter-spacing="0.06em" fill="#2b7fff">α − 1 &lt; ¼  ·  elastic</text>
    <line x1="${ox2 + 16}" y1="28" x2="${ox3 + pW - 16}" y2="28"
          stroke="#2b7fff" stroke-width="1.2" opacity="0.5"/>

    ${tauSub(ox2, '&lt;')}
    ${tauSub(ox3, '&gt;')}

    ${panel1}
    ${panel2}
    ${panel3}

    <!-- ── Legend ── -->
    <g transform="translate(${W/2 - 210},${H - 20})">
      <circle cx="0" cy="0" r="7" fill="#2b7fff"/>
      <text x="14" y="5" font-family="Inter, sans-serif" font-size="15" fill="#514a44">stable fixed point</text>
      <circle cx="186" cy="0" r="7" fill="#fffdf5" stroke="#2b7fff" stroke-width="2.5"/>
      <text x="200" y="5" font-family="Inter, sans-serif" font-size="15" fill="#514a44">unstable fixed point</text>
    </g>
  </svg>`;
}

/* ===========================================================
   4. ISLANDS — communities of practice, hand-drawn archipelago
   =========================================================== */
function islandsSVG() {
  const W = 1000, H = 620;

  // A few "islands" at different sizes, with a subtle texture
  const islands = [
    { cx: 220, cy: 310, rx: 150, ry: 95, label: "Economics", dots: 22 },
    { cx: 530, cy: 170, rx: 130, ry: 80, label: "Rust community", dots: 16 },
    { cx: 770, cy: 420, rx: 170, ry: 100, label: "Your team", highlight: true, dots: 14 },
    { cx: 300, cy: 500, rx: 75, ry: 45, label: "A niche", dots: 8 },
  ];

  // Deterministic pseudo-random for agents
  let seed = 42;
  const rand = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Islands of knowledge — communities of practice">
    <defs>
      <pattern id="sea-pattern" patternUnits="userSpaceOnUse" width="18" height="18" patternTransform="rotate(12)">
        <line x1="0" y1="9" x2="18" y2="9" stroke="#efebe2" stroke-width="1"/>
      </pattern>
      <filter id="rough" x="-2%" y="-2%" width="104%" height="104%">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="3"/>
        <feDisplacementMap in="SourceGraphic" scale="1.6"/>
      </filter>
    </defs>

    <!-- sea/background -->
    <rect width="${W}" height="${H}" fill="url(#sea-pattern)"/>

    ${islands.map(isl => {
      const fill = isl.highlight ? "#e9f1ff" : "#f7f3ea";
      const stroke = isl.highlight ? "#2b7fff" : "#19140e";
      const labelFill = isl.highlight ? "#2b7fff" : "#19140e";

      // Agent dots scattered on island
      let agents = "";
      for (let i = 0; i < isl.dots; i++) {
        const t = rand() * Math.PI * 2;
        const r = Math.sqrt(rand()) * 0.78;
        const px = isl.cx + Math.cos(t) * isl.rx * r;
        const py = isl.cy + Math.sin(t) * isl.ry * r;
        agents += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3" fill="${isl.highlight ? '#2b7fff' : '#514a44'}" opacity="0.72"/>`;
      }

      // Hand-drawn island outline — blobby ellipse with slight imperfection
      const path = (() => {
        const n = 24;
        const pts = [];
        for (let i = 0; i <= n; i++) {
          const theta = (i / n) * Math.PI * 2;
          const jitter = 0.92 + rand() * 0.16;
          const x = isl.cx + Math.cos(theta) * isl.rx * jitter;
          const y = isl.cy + Math.sin(theta) * isl.ry * jitter;
          pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        }
        return pts.join(" ");
      })();

      return `
        <polygon points="${path}" fill="${fill}" stroke="${stroke}" stroke-width="${isl.highlight ? 2.4 : 1.8}" stroke-linejoin="round"/>
        ${agents}
        <text x="${isl.cx}" y="${isl.cy + isl.ry + 32}" text-anchor="middle"
              font-family="loretta, Georgia, serif" font-style="italic" font-size="26" fill="${labelFill}">${isl.label}</text>
      `;
    }).join("")}

    <!-- legend/key -->
    <g transform="translate(30, 40)">
      <circle cx="0" cy="0" r="3" fill="#514a44"/>
      <text x="12" y="5" font-family="Inter, sans-serif" font-size="18" fill="#514a44">one agent — a doctor, a developer, a salesperson</text>
    </g>
  </svg>
  `;
}

/* ===========================================================
   5. FEEDBACK LOOP — F map as a cycle diagram
   =========================================================== */
function feedbackLoopSVG() {
  const W = 1100, H = 680;
  const cx = W / 2, cy = H / 2 - 40;
  const r = 200;

  // Four nodes around a loop
  const nodes = [
    { angle: -Math.PI/2, title: "General knowledge", sub: "X_t", w: 280 },
    { angle: 0, title: "Agents solve problems", sub: "using X_t as context", w: 320 },
    { angle: Math.PI/2, title: "Public signals", sub: "best practices, docs, posts", w: 340 },
    { angle: Math.PI, title: "Update", sub: "X_t+1", w: 260 },
  ];

  const nodePos = nodes.map(n => ({
    ...n,
    x: cx + Math.cos(n.angle) * r,
    y: cy + Math.sin(n.angle) * r,
  }));

  // Arcs between nodes (slightly curved)
  const arcs = [];
  for (let i = 0; i < nodePos.length; i++) {
    const a = nodePos[i], b = nodePos[(i + 1) % nodePos.length];
    arcs.push({ a, b });
  }

  return `
  <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The F feedback loop — how common knowledge evolves period to period">
    <defs>
      <marker id="arrow-fb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 Z" fill="#2b7fff"/>
      </marker>
    </defs>

    <!-- arcs between nodes -->
    ${arcs.map(({a, b}, i) => {
      // Midpoint pushed outward from center for a nice curve
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const dx = mx - cx, dy = my - cy;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const mag = r * 0.22;
      const cmx = mx + (dx / dist) * mag;
      const cmy = my + (dy / dist) * mag;

      // Shorten endpoints so arrow doesn't overlap node circles
      const shrink = 0.84;
      const sx = a.x + (b.x - a.x) * (1 - shrink) / 2;
      const sy = a.y + (b.y - a.y) * (1 - shrink) / 2;
      const ex = a.x + (b.x - a.x) * (1 + shrink) / 2;
      const ey = a.y + (b.y - a.y) * (1 + shrink) / 2;

      return `<path d="M${sx} ${sy} Q${cmx} ${cmy} ${ex} ${ey}" fill="none" stroke="#2b7fff" stroke-width="2" marker-end="url(#arrow-fb)"/>`;
    }).join("")}

    <!-- center label -->
    <text x="${cx}" y="${cy - 8}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="48" fill="#2b7fff">F</text>
    <text x="${cx}" y="${cy + 26}" text-anchor="middle" font-family="Inter, sans-serif" font-size="17" letter-spacing="0.14em" fill="#a89f91" text-transform="uppercase">THE MAP</text>

    <!-- nodes -->
    ${nodePos.map(n => {
      const isTop  = n.angle === -Math.PI/2;
      const isBot  = n.angle ===  Math.PI/2;
      const isEast = n.angle === 0;

      const edge = 22 + 12; // circle radius + gap

      // Per-direction label placement.
      // tb/sb = dominant-baseline for title/sub.
      // North/south use 'hanging' on both so y = top-of-text (predictable stacking).
      // East/west use 'auto' (baseline) for natural side-by-side feel.
      let tx, ty, ta, tb, sx, sy, sa, sb;
      if (isTop) {
        // Both lines sit above the circle. Title baseline at -50; sub top at -38
        // (giving ~8 px clearance above the circle edge at y=-22).
        tx = 0;  ty = -50; ta = 'middle'; tb = 'auto';
        sx = 0;  sy = -30; sa = 'middle'; sb = 'auto';
      } else if (isBot) {
        // Title top at circle edge + gap; sub top = title top + title height + gap.
        tx = 0;  ty = edge;      ta = 'middle'; tb = 'hanging';
        sx = 0;  sy = edge + 34; sa = 'middle'; sb = 'hanging';
      } else if (isEast) {
        tx = edge;  ty = -14; ta = 'start'; tb = 'auto';
        sx = edge;  sy =  14; sa = 'start'; sb = 'auto';
      } else {
        tx = -edge; ty = -14; ta = 'end'; tb = 'auto';
        sx = -edge; sy =  14; sa = 'end'; sb = 'auto';
      }

      return `
      <g transform="translate(${n.x}, ${n.y})">
        <circle cx="0" cy="0" r="22" fill="#fffdf5" stroke="#19140e" stroke-width="2"/>
        <text x="${tx}" y="${ty}"
              text-anchor="${ta}" dominant-baseline="${tb}"
              font-family="loretta, Georgia, serif" font-size="26" fill="#19140e">${n.title}</text>
        <text x="${sx}" y="${sy}"
              text-anchor="${sa}" dominant-baseline="${sb}"
              font-family="loretta, Georgia, serif" font-style="italic" font-size="18" fill="#514a44">${n.sub}</text>
      </g>
    `;}).join("")}

    <!-- Caption under the loop -->
    <text x="${cx}" y="${H - 15}" text-anchor="middle" font-family="loretta, Georgia, serif" font-style="italic" font-size="20" fill="#a89f91">each period's common knowledge becomes next period's input</text>
  </svg>
  `;
}

/* Export helpers */
window.KCDiagrams = {
  costCurvesSVG,
  payoffLatticeSVG,
  fMapTripleSVG,
  fMapNewSVG,
  islandsSVG,
  feedbackLoopSVG,
};
