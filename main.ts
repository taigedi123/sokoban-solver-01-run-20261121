// Sokoban solver: decompose the level into rooms (each room hangs off the
// 1-wide hallway through a single door, so boxes never leave their room),
// solve each room push-optimally with A* over pushes, then stitch the
// player walks between rooms. Written from scratch for this competition.
// Deterministic: budgets are counted in node expansions, never wall time.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions: 0=up, 1=down, 2=left, 3=right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];

// ---------------------------------------------------------------- parsing

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall (includes out of bounds padding via index checks)
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch !== '#') wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ------------------------------------------------------------- decomposition

type RoomSpec = {
  // Global cell indices of the room floor: the room component plus the
  // hallway cell(s) touching its door, so the player may stand there to push.
  floor: number[];
  boxes: number[];
  goals: number[];
  door: number; // global cell of the door (player enters here)
  entry: number; // global hallway cell touching the door (where the player stands to enter)
};

// Split the level into rooms by removing the hallway row (the player's row).
// Returns null when the shape does not decompose cleanly.
function decompose(lv: Level): RoomSpec[] | null {
  const { w, h, wall, goals, boxes, player } = lv;
  const py = Math.floor(player / w);
  const goalSet = new Set(goals);
  const boxSet = new Set(boxes);
  const inHallRow = (c: number): boolean => Math.floor(c / w) === py;
  const seen = new Uint8Array(w * h);
  const rooms: RoomSpec[] = [];
  let coveredBoxes = 0;
  let coveredGoals = 0;
  for (let start = 0; start < w * h; start++) {
    if (wall[start] || inHallRow(start) || seen[start]) continue;
    // Flood-fill one component of the floor minus the hallway row.
    const comp: number[] = [];
    const q = [start];
    seen[start] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      comp.push(c);
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || inHallRow(n) || seen[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    const cBoxes = comp.filter((c) => boxSet.has(c));
    if (cBoxes.length === 0) continue; // empty alcove, not a room
    const cGoals = comp.filter((c) => goalSet.has(c));
    if (cGoals.length !== cBoxes.length) return null;
    // Door cells: component cells touching a floor cell of the hallway row.
    const doors = comp.filter((c) => {
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (Math.floor(n / w) === py && !wall[n]) return true;
      }
      return false;
    });
    if (doors.length === 0) return null;
    doors.sort((a, b) => a - b);
    const door = doors[0]!;
    // Staging cells: hallway cells touching a door cell (pusher stance, and
    // where the player stands before entering: a box may sit on the door).
    const floorSet = new Set(comp);
    const staging: number[] = [];
    for (const dc of doors) {
      const x = dc % w;
      const y = Math.floor(dc / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (Math.floor(n / w) === py && !wall[n]) {
          floorSet.add(n);
          staging.push(n);
        }
      }
    }
    staging.sort((a, b) => a - b);
    rooms.push({ floor: [...floorSet].sort((a, b) => a - b), boxes: cBoxes, goals: cGoals, door, entry: staging[0]! });
    coveredBoxes += cBoxes.length;
    coveredGoals += cGoals.length;
  }
  if (coveredBoxes !== boxes.length || coveredGoals !== goals.length) return null;
  return rooms;
}

// ------------------------------------------------------------------ room A*

type Push = { box: number; dir: number }; // box = global cell before the push

type Ctx = {
  n: number;
  nbr: Int32Array; // n*4, -1 when the neighbour is not room floor
  isGoal: Uint8Array;
  goals: number[];
  dead: Uint8Array; // dead-square cells: a box there can never reach a goal
  dist: Int32Array; // n * goals.length: box-push distance ignoring other boxes
  perms: number[][]; // permutations of goal indices (for the matching heuristic)
  quads: Int32Array; // flat list of 2x2 quads, 4 entries each, -1 = wall cell
  quadOf: number[][]; // per cell: quad indices containing it
};

function buildCtx(floor: number[], boxes: number[], goals: number[], w: number): Ctx {
  const n = floor.length;
  const loc = new Map<number, number>();
  floor.forEach((g, i) => loc.set(g, i));
  const nbr = new Int32Array(n * 4).fill(-1);
  for (let i = 0; i < n; i++) {
    const g = floor[i]!;
    const x = g % w;
    const y = Math.floor(g / w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (nx < 0 || ny < 0) continue;
      const m = loc.get(ny * w + nx);
      if (m !== undefined) nbr[i * 4 + d] = m;
    }
  }
  const isGoal = new Uint8Array(n);
  const goalLoc: number[] = [];
  for (const g of goals) {
    const li = loc.get(g)!;
    isGoal[li] = 1;
    goalLoc.push(li);
  }
  const ng = goalLoc.length;
  // BFS distances from each goal over floor (ignoring boxes).
  const dist = new Int32Array(n * ng).fill(1e9);
  for (let gi = 0; gi < ng; gi++) {
    const start = goalLoc[gi]!;
    const dd = new Int32Array(n).fill(-1);
    const q = [start];
    dd[start] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi]!;
      for (let d = 0; d < 4; d++) {
        const m = nbr[c * 4 + d]!;
        if (m >= 0 && dd[m] < 0) {
          dd[m] = dd[c]! + 1;
          q.push(m);
        }
      }
    }
    for (let c = 0; c < n; c++) if (dd[c] >= 0) dist[c * ng + gi] = dd[c]!;
  }
  // Dead squares: cells from which no goal is reachable by pushes. A box at
  // c can step to c+d when c+d can reach a goal and a pusher can stand at c-d.
  const reach = new Uint8Array(n);
  for (const g of goalLoc) reach[g] = 1;
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < n; c++) {
      if (reach[c]) continue;
      for (let d = 0; d < 4; d++) {
        const fwd = nbr[c * 4 + d]!;
        const back = nbr[c * 4 + OPP[d]!]!;
        if (fwd >= 0 && reach[fwd] && back >= 0) {
          reach[c] = 1;
          changed = true;
          break;
        }
      }
    }
  }
  const dead = new Uint8Array(n);
  for (let c = 0; c < n; c++) if (!reach[c]) dead[c] = 1;
  // 2x2 quads (cells that are floor; -1 for a wall cell of the quad).
  const quads: number[] = [];
  const quadOf: number[][] = Array.from({ length: n }, () => []);
  const cellAt = (gx: number, gy: number): number => {
    if (gx < 0 || gy < 0) return -1;
    const m = loc.get(gy * w + gx);
    return m === undefined ? -1 : m;
  };
  for (const g of floor) {
    const x = g % w;
    const y = Math.floor(g / w);
    // g as the top-left corner of a quad
    const q0 = loc.get(g)!;
    const q1 = cellAt(x + 1, y);
    const q2 = cellAt(x, y + 1);
    const q3 = cellAt(x + 1, y + 1);
    const qi = quads.length / 4;
    quads.push(q0, q1, q2, q3);
    for (const c of [q0, q1, q2, q3]) if (c >= 0) quadOf[c]!.push(qi);
  }
  // Goal permutations for the min-matching heuristic.
  const perms: number[][] = [];
  const idx = goalLoc.map((_, i) => i);
  const gen = (a: number[], k: number): void => {
    if (k === 1) {
      perms.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(a, k - 1);
      const j = k % 2 === 0 ? i : 0;
      const t = a[j]!;
      a[j] = a[k - 1]!;
      a[k - 1] = t;
    }
  };
  gen(idx, idx.length);
  return { n, nbr, isGoal, goals: goalLoc, dead, dist, perms, quads: Int32Array.from(quads), quadOf };
}

// Min-cost matching of boxes to goals (admissible: each box needs at least
// its BFS distance in pushes). Returns a large number when unmatchable.
function heuristic(ctx: Ctx, boxes: number[]): number {
  const ng = ctx.goals.length;
  let best = 1e9;
  for (const p of ctx.perms) {
    let s = 0;
    for (let i = 0; i < boxes.length; i++) {
      s += ctx.dist[boxes[i]! * ng + p[i]!]!;
      if (s >= best) break;
    }
    if (s < best) best = s;
  }
  return best;
}

// Sum of per-box nearest-goal distances (weaker, for the whole-level fallback).
function heuristicSum(ctx: Ctx, boxes: number[]): number {
  const ng = ctx.goals.length;
  let s = 0;
  for (const b of boxes) {
    let m = 1e9;
    for (let gi = 0; gi < ng; gi++) {
      const d = ctx.dist[b * ng + gi]!;
      if (d < m) m = d;
    }
    s += m;
    if (s >= 1e9) break;
  }
  return s;
}

function packBoxes(boxes: number[]): number {
  let p = 0;
  for (let i = boxes.length - 1; i >= 0; i--) p = (p << 6) | boxes[i]!;
  return p;
}

function unpackBoxes(packed: number, count: number, out: number[]): void {
  for (let i = 0; i < count; i++) {
    out[i] = packed & 63;
    packed >>= 6;
  }
}

// Player region from `start` over floor minus boxes; returns the smallest
// cell of the region and fills `reg` (1 = reachable).
function regionOf(ctx: Ctx, start: number, boxSet: Uint8Array, reg: Uint8Array): number {
  reg.fill(0);
  const q = new Int32Array(ctx.n);
  let qh = 0;
  let qt = 0;
  q[qt++] = start;
  reg[start] = 1;
  let min = start;
  while (qh < qt) {
    const c = q[qh++]!;
    if (c < min) min = c;
    for (let d = 0; d < 4; d++) {
      const m = ctx.nbr[c * 4 + d]!;
      if (m >= 0 && !reg[m] && !boxSet[m]) {
        reg[m] = 1;
        q[qt++] = m;
      }
    }
  }
  return min;
}

// Deadlock test for a box configuration: dead squares, frozen boxes, 2x2.
function isDeadlocked(ctx: Ctx, boxes: number[], boxSet: Uint8Array): boolean {
  for (const b of boxes) {
    if (ctx.dead[b] && !ctx.isGoal[b]) return true;
  }
  for (const b of boxes) {
    if (ctx.isGoal[b]) continue;
    // Frozen: blocked on both sides of both axes.
    const l = ctx.nbr[b * 4 + 2]!;
    const r = ctx.nbr[b * 4 + 3]!;
    const u = ctx.nbr[b * 4 + 0]!;
    const d = ctx.nbr[b * 4 + 1]!;
    const blockedL = l < 0 || boxSet[l];
    const blockedR = r < 0 || boxSet[r];
    const blockedU = u < 0 || boxSet[u];
    const blockedD = d < 0 || boxSet[d];
    if (blockedL && blockedR && blockedU && blockedD) return true;
    // 2x2 blocks containing this box.
    for (const qi of ctx.quadOf[b]!) {
      let nBox = 0;
      let allBlocked = true;
      let boxOnGoal = false;
      for (let k = 0; k < 4; k++) {
        const c = ctx.quads[qi * 4 + k]!;
        if (c < 0) continue;
        if (boxSet[c]) {
          nBox++;
          if (ctx.isGoal[c]) boxOnGoal = true;
        } else {
          allBlocked = false;
        }
      }
      if (nBox > 0 && allBlocked && !boxOnGoal) return true;
    }
  }
  return false;
}

type SearchOpts = {
  cap: number; // max expansions (deterministic budget)
  greedy: boolean; // order by h only (finds a solution fast, not optimal)
  door: number; // local door cell, or -1 when no exit is required
  useMatching: boolean;
};

// A* over pushes. Returns the push list (local cells) or null.
function searchPushes(
  ctx: Ctx,
  startBoxes: number[],
  startPlayer: number,
  opts: SearchOpts,
): { box: number; dir: number }[] | null {
  const nb = startBoxes.length;
  const startSorted = startBoxes.slice().sort((a, b) => a - b);
  const hFn = opts.useMatching ? heuristic : heuristicSum;
  const hCache = new Map<number, number>();
  const hOf = (packed: number, boxes: number[]): number => {
    const hit = hCache.get(packed);
    if (hit !== undefined) return hit;
    const v = hFn(ctx, boxes);
    hCache.set(packed, v);
    return v;
  };

  const pk: number[] = [];
  const pl: number[] = [];
  const pg: number[] = [];
  const parent: number[] = [];
  const pBox: number[] = [];
  const pDir: number[] = [];
  const exitFlags: number[] = [];

  const startPacked = packBoxes(startSorted);
  const boxSet0 = new Uint8Array(ctx.n);
  for (const b of startSorted) boxSet0[b] = 1;
  const reg0 = new Uint8Array(ctx.n);
  const startCanon = regionOf(ctx, startPlayer, boxSet0, reg0);
  pk.push(startPacked);
  pl.push(startCanon);
  pg.push(0);
  parent.push(-1);
  pBox.push(-1);
  pDir.push(-1);
  exitFlags.push(opts.door < 0 || reg0[opts.door] === 1 ? 1 : 0);

  const best = new Map<number, number>(); // (packed*64 + player) -> best g
  best.set(startPacked * 64 + startCanon, 0);

  // Binary heap on (f, -g, seq): parallel arrays.
  const hf: number[] = [];
  const hg: number[] = [];
  const hi: number[] = [];
  const hseq: number[] = [];
  let seq = 0;
  const heapLess = (a: number, b: number): boolean =>
    hf[a]! < hf[b]! || (hf[a]! === hf[b]! && (hg[a]! > hg[b]! || (hg[a]! === hg[b]! && hseq[a]! < hseq[b]!)));
  const heapPush = (f: number, g: number, idx: number): void => {
    const i = hf.length;
    hf.push(f);
    hg.push(g);
    hi.push(idx);
    hseq.push(seq++);
    let c = i;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (!heapLess(c, p)) break;
      const tf = hf[c]!; hf[c] = hf[p]!; hf[p] = tf;
      const tg = hg[c]!; hg[c] = hg[p]!; hg[p] = tg;
      const ti = hi[c]!; hi[c] = hi[p]!; hi[p] = ti;
      const ts = hseq[c]!; hseq[c] = hseq[p]!; hseq[p] = ts;
      c = p;
    }
  };
  const heapPop = (): number => {
    const top = hi[0]!;
    const lf = hf.pop()!;
    const lg = hg.pop()!;
    const li = hi.pop()!;
    const ls = hseq.pop()!;
    if (hf.length > 0) {
      hf[0] = lf; hg[0] = lg; hi[0] = li; hseq[0] = ls;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= hf.length) break;
        if (c + 1 < hf.length && heapLess(c + 1, c)) c++;
        if (!heapLess(c, i)) break;
        const tf = hf[c]!; hf[c] = hf[i]!; hf[i] = tf;
        const tg = hg[c]!; hg[c] = hg[i]!; hg[i] = tg;
        const ti = hi[c]!; hi[c] = hi[i]!; hi[i] = ti;
        const ts = hseq[c]!; hseq[c] = hseq[i]!; hseq[i] = ts;
        i = c;
      }
    }
    return top;
  };

  const h0 = hOf(startPacked, startSorted);
  if (h0 >= 1e9) return null;
  heapPush(opts.greedy ? h0 : h0, 0, 0);

  const boxes = new Array<number>(nb);
  const reg = new Uint8Array(ctx.n);
  const reg2 = new Uint8Array(ctx.n);
  const boxSet = new Uint8Array(ctx.n);
  let expanded = 0;
  while (hf.length > 0) {
    const s = heapPop();
    const g = pg[s]!;
    // Stale entry (a better g was recorded after this was pushed).
    if (best.get(pk[s]! * 64 + pl[s]!) !== g) continue;
    if (expanded >= opts.cap) return null;
    expanded++;
    unpackBoxes(pk[s]!, nb, boxes);
    // Terminal: every box on a goal and the player can reach the door.
    let allGoals = true;
    for (const b of boxes) if (!ctx.isGoal[b]) { allGoals = false; break; }
    if (allGoals && exitFlags[s] === 1) {
      const out: { box: number; dir: number }[] = [];
      for (let t = s; parent[t]! >= 0; t = parent[t]!) out.push({ box: pBox[t]!, dir: pDir[t]! });
      out.reverse();
      return out;
    }
    // Player region for this state.
    boxSet.fill(0);
    for (const b of boxes) boxSet[b] = 1;
    regionOf(ctx, pl[s]!, boxSet, reg);
    for (let bi = 0; bi < nb; bi++) {
      const b = boxes[bi]!;
      for (let d = 0; d < 4; d++) {
        const dest = ctx.nbr[b * 4 + d]!;
        if (dest < 0 || boxSet[dest]) continue;
        const pusher = ctx.nbr[b * 4 + OPP[d]!]!;
        if (pusher < 0 || !reg[pusher]) continue;
        // Child boxes, kept sorted.
        const child = boxes.slice();
        child[bi] = dest;
        child.sort((x, y) => x - y);
        if (ctx.dead[dest] && !ctx.isGoal[dest]) continue;
        const childPacked = packBoxes(child);
        const childG = g + 1;
        // New player region from the box's old cell.
        const boxSet2 = new Uint8Array(ctx.n);
        for (const cb of child) boxSet2[cb] = 1;
        const childCanon = regionOf(ctx, b, boxSet2, reg2);
        const key = childPacked * 64 + childCanon;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= childG) continue;
        if (isDeadlocked(ctx, child, boxSet2)) continue;
        const hh = hOf(childPacked, child);
        if (hh >= 1e9) continue;
        const t = pk.length;
        pk.push(childPacked);
        pl.push(childCanon);
        pg.push(childG);
        parent.push(s);
        pBox.push(b);
        pDir.push(d);
        exitFlags.push(opts.door < 0 || reg2[opts.door] === 1 ? 1 : 0);
        best.set(key, childG);
        heapPush(opts.greedy ? hh : childG + hh, childG, t);
      }
    }
  }
  return null;
}

// Solve one room push-optimally; greedy fallback when A* runs out of budget.
function solveRoom(ctx: Ctx, boxes: number[], startPlayer: number, door: number): { box: number; dir: number }[] | null {
  const exact = searchPushes(ctx, boxes, startPlayer, { cap: 150_000, greedy: false, door, useMatching: true });
  if (exact) return exact;
  const greedy = searchPushes(ctx, boxes, startPlayer, { cap: 150_000, greedy: true, door, useMatching: true });
  if (greedy) return greedy;
  // Last resort: accept a solution the player cannot exit from (stitching
  // re-checks reachability; usually the door is still reachable).
  const loose = searchPushes(ctx, boxes, startPlayer, { cap: 150_000, greedy: false, door: -1, useMatching: true });
  return loose;
}

// ------------------------------------------------------------------ stitching

// Shortest player walk (u/d/l/r) from `from` to `to` avoiding walls and the
// current box positions; null when unreachable.
function walkPath(lv: Level, boxes: Set<number>, from: number, to: number): string | null {
  if (from === to) return '';
  const { w, h, wall } = lv;
  const prev = new Int32Array(w * h).fill(-2);
  const prevDir = new Int8Array(w * h);
  const q = [from];
  prev[from] = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    if (c === to) break;
    const x = c % w;
    const y = Math.floor(c / w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || boxes.has(n) || prev[n] !== -2) continue;
      prev[n] = c;
      prevDir[n] = d;
      q.push(n);
    }
  }
  if (prev[to] === -2) return null;
  let moves = '';
  for (let c = to; prev[c]! >= 0; c = prev[c]!) moves = LETTER[prevDir[c]!]! + moves;
  return moves;
}

// Solve one whole level; returns the move string or '' when unsolved.
function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const rooms = decompose(lv);
  if (!rooms) return solveWholeLevel(lv);
  // Visit rooms nearest-first from the player's start (push count is
  // unaffected by the order; this only shortens the walks).
  const ordered = rooms.slice().sort((a, b) => {
    const da = Math.abs((a.door % lv.w) - (lv.player % lv.w));
    const db = Math.abs((b.door % lv.w) - (lv.player % lv.w));
    return da - db || a.door - b.door;
  });
  const boxSet = new Set(lv.boxes);
  let player = lv.player;
  let moves = '';
  const goalSet = new Set(lv.goals);
  for (const room of ordered) {
    const ctx = buildCtx(room.floor, room.boxes, room.goals, lv.w);
    const loc = new Map<number, number>();
    room.floor.forEach((g, i) => loc.set(g, i));
    const localBoxes = room.boxes.map((g) => loc.get(g)!);
    const localDoor = loc.get(room.door)!;
    const localEntry = loc.get(room.entry)!;
    const pushes = solveRoom(ctx, localBoxes, localEntry, localDoor);
    if (!pushes) return '';
    // Walk to the entry cell, then replay the pushes with real player walks.
    const toDoor = walkPath(lv, boxSet, player, room.entry);
    if (toDoor === null) return '';
    moves += toDoor;
    player = room.entry;
    for (const p of pushes) {
      const boxGlobal = room.floor[p.box]!;
      const x = boxGlobal % lv.w;
      const y = Math.floor(boxGlobal / lv.w);
      const pusher = (y + DY[OPP[p.dir]!]!) * lv.w + (x + DX[OPP[p.dir]!]!);
      const path = walkPath(lv, boxSet, player, pusher);
      if (path === null) return '';
      moves += path;
      moves += LETTER[p.dir]!;
      boxSet.delete(boxGlobal);
      const dest = (y + DY[p.dir]!) * lv.w + (x + DX[p.dir]!);
      boxSet.add(dest);
      player = boxGlobal;
    }
    // Sanity: this room's boxes must now sit on its goals (checked globally
    // at the end of the level).
  }
  for (const b of boxSet) if (!goalSet.has(b)) return '';
  return moves;
}

// Fallback: push-A* on the whole level (no decomposition).
function solveWholeLevel(lv: Level): string {
  const floor: number[] = [];
  for (let c = 0; c < lv.w * lv.h; c++) if (!lv.wall[c]) floor.push(c);
  const ctx = buildCtx(floor, lv.boxes, lv.goals, lv.w);
  const loc = new Map<number, number>();
  floor.forEach((g, i) => loc.set(g, i));
  const pushes = searchPushes(ctx, lv.boxes.map((g) => loc.get(g)!), loc.get(lv.player)!, {
    cap: 150_000,
    greedy: false,
    door: -1,
    useMatching: false,
  }) ?? searchPushes(ctx, lv.boxes.map((g) => loc.get(g)!), loc.get(lv.player)!, {
    cap: 150_000,
    greedy: true,
    door: -1,
    useMatching: false,
  });
  if (!pushes) return '';
  const boxSet = new Set(lv.boxes);
  let player = lv.player;
  let moves = '';
  for (const p of pushes) {
    const boxGlobal = floor[p.box]!;
    const x = boxGlobal % lv.w;
    const y = Math.floor(boxGlobal / lv.w);
    const pusher = (y + DY[OPP[p.dir]!]!) * lv.w + (x + DX[OPP[p.dir]!]!);
    const path = walkPath(lv, boxSet, player, pusher);
    if (path === null) return '';
    moves += path;
    moves += LETTER[p.dir]!;
    boxSet.delete(boxGlobal);
    boxSet.add((y + DY[p.dir]!) * lv.w + (x + DX[p.dir]!));
    player = boxGlobal;
  }
  const goalSet = new Set(lv.goals);
  for (const b of boxSet) if (!goalSet.has(b)) return '';
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
