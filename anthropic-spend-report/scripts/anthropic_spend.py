#!/usr/bin/env python3
"""Per-person, per-month Anthropic API spend for a Claude Console organization.

WHAT IT DOES
  Pulls the authoritative dollar line-items from the Admin Cost API and allocates
  each one to API keys in proportion to that key's share of the matching tokens
  from the Usage API (joined on day + model + context window + service tier +
  token type). Per-key dollars therefore RECONCILE EXACTLY to the Cost API total --
  no pricing table to maintain. (Matching that total to your actual invoice, which
  may include priority/flex-tier spend the Cost API omits, is a one-time human check
  -- see REFERENCE.md.) Keys are then mapped to people via an editable roster, and
  each person's spend is split into Claude Code vs dev vs service.

PREREQUISITE: an ADMIN API key (sk-ant-admin01-...), NOT a regular sk-ant-api key.
  Create one (Console admin role) at https://platform.claude.com/settings/admin-keys
  then:  export ANTHROPIC_ADMIN_KEY=sk-ant-admin01-...

USAGE
  python3 anthropic_spend.py                            # last 12 full months
  python3 anthropic_spend.py --start 2024-01 --end 2026-06 --out spend.csv

# ============================================================================
# ORG CONFIG -- EDIT THIS for your organization
# ----------------------------------------------------------------------------
# PEOPLE: canonical name -> leading key-name fragments that belong to them.
#   The "leading fragment" is the person token of a key name: for
#   `claude_code_key_<first>.<last>_<suffix>` it's `<first>.<last>`; for a
#   personal key like `adams-local-key` it's the first token `adams`.
#   If new keys land in "uncategorized", add their fragment here.
# BUCKET: exact non-person key names -> a category bucket.
# KEY_NAME_PREFIX_CLAUDE_CODE: prefix marking a Claude Code key (vs a dev key).
# ============================================================================
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

# ---- ORG CONFIG (edit me) --------------------------------------------------
# The entries below are placeholders. The real roster and key-name map were
# removed before open-sourcing; replace them with your own. Anything that does
# not match ends up in "uncategorized", which the report tells you about, so an
# empty config is a valid starting point.
PEOPLE = {
    "Person A": ["persona", "person.a"],
    "Person B": ["personb", "person.b"],
    "Person C": ["personc", "person.c"],
}
BUCKET = {
    "example-prod": "production",
    "example-prod-2026": "production",
    "log_probability_research": "research/projects",
    "research-LLMaaj-key": "research/projects",
    "alignsim-sandbox": "research/projects",
    "alignment_tuning": "research/projects",
    "geo-testing": "research/projects",
    "agentic-experiments": "research/projects",
    "demo-environments": "research/projects",
    "github-ci": "CI/integrations",
    "zapier-testing": "CI/integrations",  # gitleaks:allow (label, not a secret; 'zapier' contains 'api')
}
KEY_NAME_PREFIX_CLAUDE_CODE = "claude_code_key_"
# ---------------------------------------------------------------------------

BASE = "https://api.anthropic.com/v1/organizations"
VERSION = "2023-06-01"
THROTTLE = 1.0  # seconds between Admin API calls -> stay under the per-minute rate limit
ADMIN_KEY = os.environ.get("ANTHROPIC_ADMIN_KEY") or os.environ.get("ANTHROPIC_ADMIN_API_KEY")
# default output lands in the git-ignored output/ folder next to this skill (tidy, local)
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "output" / "anthropic_spend_by_user.csv")

TOKEN_FIELDS = (
    "uncached_input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation.ephemeral_1h_input_tokens",
    "cache_creation.ephemeral_5m_input_tokens",
)
ALIAS = {frag: person for person, frags in PEOPLE.items() for frag in frags}
_all_frags = [frag for frags in PEOPLE.values() for frag in frags]
assert len(_all_frags) == len(set(_all_frags)), "Duplicate fragment in PEOPLE — fix the ALIAS collision"
MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")  # YYYY-MM with a real month 01-12


# ---- Admin API helpers -----------------------------------------------------
def api_get(path: str, params: list[tuple[str, str]]) -> dict:
    """GET BASE+path with x-api-key auth; retry on 429/5xx. params = list of (k, v)."""
    time.sleep(THROTTLE)
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "x-api-key": ADMIN_KEY,
        "anthropic-version": VERSION,
        "User-Agent": "anthropic-spend-report/1.0",
    })
    for attempt in range(9):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 429 or e.code >= 500:
                wait = min(2 ** attempt, 60)
                ra = e.headers.get("retry-after")
                if ra and ra.isdigit():
                    wait = max(wait, int(ra) + 1)
                sys.stderr.write(f"  HTTP {e.code}; retry in {wait}s\n")
                time.sleep(wait)
                continue
            sys.exit(f"HTTP {e.code} on {path}:\n{body}")
        except urllib.error.URLError as e:
            wait = min(2 ** attempt, 60)
            sys.stderr.write(f"  network error ({e.reason}); retry in {wait}s\n")
            time.sleep(wait)
    sys.exit(f"Gave up after retries on {path}")


def paginate(path: str, base_params: list[tuple[str, str]]) -> Iterator[dict]:
    params = list(base_params)
    while True:
        resp = api_get(path, params)
        yield from resp.get("data", [])
        if resp.get("has_more") and resp.get("next_page"):
            params = list(base_params) + [("page", resp["next_page"])]
        else:
            return


def fetch_key_names() -> dict[str, str]:
    names, after = {}, None
    while True:
        params = [("limit", "1000")]
        if after:
            params.append(("after_id", after))
        resp = api_get("/api_keys", params)
        for k in resp.get("data", []):
            names[k["id"]] = k.get("name") or k["id"]
        if resp.get("has_more") and resp.get("last_id"):
            after = resp["last_id"]
        else:
            return names


def token_counts(r: dict) -> dict[str, int]:
    cc = r.get("cache_creation") or {}
    return {
        "uncached_input_tokens": r.get("uncached_input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        "cache_read_input_tokens": r.get("cache_read_input_tokens") or 0,
        "cache_creation.ephemeral_1h_input_tokens": cc.get("ephemeral_1h_input_tokens") or 0,
        "cache_creation.ephemeral_5m_input_tokens": cc.get("ephemeral_5m_input_tokens") or 0,
    }


# ---- key-name -> person / bucket / key_type --------------------------------
def lead_token(name: str) -> str | None:
    if name.startswith("("):
        return None
    n = name.lower().strip()
    if n.startswith(KEY_NAME_PREFIX_CLAUDE_CODE):
        return n[len(KEY_NAME_PREFIX_CLAUDE_CODE):].rsplit("_", 1)[0]
    return re.split(r"[-_./:@\s]+", n)[0]


def attribute(name: str) -> str:
    # An exact-name BUCKET match wins over a fuzzy lead-token match, so a project
    # key whose first token happens to collide with a person fragment (e.g. a
    # "ben-*" project key) is not silently folded into that person's total.
    if name in BUCKET:
        return BUCKET[name]
    t = lead_token(name)
    if t and t in ALIAS:
        return ALIAS[t]
    if name == "(no API key - Console/Workbench)":
        return "console/workbench"
    return "uncategorized"


def key_type(name: str) -> str:
    # A claude_code_key_ is a Claude Code key regardless of whether we've mapped the
    # person yet, so its spend still lands in the Claude Code column (not service).
    if name.lower().startswith(KEY_NAME_PREFIX_CLAUDE_CODE):
        return "claude_code"
    if lead_token(name) in ALIAS:
        return "dev"
    return "service"


# ---- date windows ----------------------------------------------------------
def month_windows(start_ym: str, end_ym: str) -> Iterator[tuple[str, str, str]]:
    y, m = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        start = datetime(y, m, 1, tzinfo=timezone.utc)
        # ending_at is exclusive on the bucket END: the last day's bucket ends at
        # next-month-01 00:00, so push ending_at one day past that to keep it. The
        # (b["starting_at"][:7] != month) filter below discards the extra day pulled in.
        end = datetime(ny, nm, 1, tzinfo=timezone.utc) + timedelta(days=1)
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        yield f"{y:04d}-{m:02d}", start.strftime(fmt), end.strftime(fmt)
        y, m = ny, nm


def default_range() -> tuple[str, str]:
    today = datetime.now(timezone.utc)
    end = today.replace(day=1) - timedelta(days=1)
    end_ym = f"{end.year:04d}-{end.month:02d}"
    sy, sm = (end.year - 1, end.month + 1) if end.month < 12 else (end.year, 1)
    return f"{sy:04d}-{sm:02d}", end_ym


def collect(windows: list[tuple[str, str, str]]) -> tuple[dict, dict, Decimal, dict, dict]:
    """Pull cost + usage for each (month, start, end) window from the Admin API.

    Returns (cost_token, cost_nontoken, cost_total, usage_tok, key_tokens).
    """
    cost_token: dict = defaultdict(Decimal)          # (day, model, cw, tier, token_type) -> cents
    cost_nontoken: dict = defaultdict(Decimal)        # (month, cost_type) -> cents
    cost_total = Decimal(0)
    usage_tok: dict = defaultdict(lambda: defaultdict(int))  # (day,model,cw,tier,ttype) -> {api_key_id: tokens}
    key_tokens: dict = defaultdict(lambda: defaultdict(int))  # (month, api_key_id) -> {token_field: tokens}

    for month, s, e in windows:
        sys.stderr.write(f"[{month}] cost...")
        for b in paginate("/cost_report", [
            ("starting_at", s), ("ending_at", e), ("bucket_width", "1d"),
            ("limit", "31"), ("group_by[]", "description"),
        ]):
            if b["starting_at"][:7] != month:
                continue
            day = b["starting_at"][:10]
            for r in b.get("results", []):
                amt = Decimal(r.get("amount") or "0")
                if amt == 0:
                    continue
                cost_total += amt
                if r.get("cost_type") == "tokens":
                    cost_token[(day, r.get("model"), r.get("context_window"),
                                r.get("service_tier"), r.get("token_type"))] += amt
                else:
                    cost_nontoken[(month, r.get("cost_type") or "other")] += amt
        sys.stderr.write(" usage...")
        for b in paginate("/usage_report/messages", [
            ("starting_at", s), ("ending_at", e), ("bucket_width", "1d"), ("limit", "31"),
            ("group_by[]", "api_key_id"), ("group_by[]", "model"),
            ("group_by[]", "context_window"), ("group_by[]", "service_tier"),
        ]):
            if b["starting_at"][:7] != month:
                continue
            day = b["starting_at"][:10]
            for r in b.get("results", []):
                akid, model = r.get("api_key_id"), r.get("model")
                cw, tier = r.get("context_window"), r.get("service_tier")
                for ttype, n in token_counts(r).items():
                    if n:
                        usage_tok[(day, model, cw, tier, ttype)][akid] += n
                        key_tokens[(month, akid)][ttype] += n
        sys.stderr.write(" done\n")
    return cost_token, cost_nontoken, cost_total, usage_tok, key_tokens


def allocate(cost_token: dict, usage_tok: dict) -> tuple[dict, dict]:
    """Split each token cost line across keys by token share (pure -- no I/O).

    Returns (alloc[(month, api_key_id)] -> cents, unattributed[month] -> cents).
    Per cost line the parts sum to the line amount exactly, with the rounding
    remainder given to the largest holder.
    """
    alloc: dict = defaultdict(Decimal)
    unattributed: dict = defaultdict(Decimal)
    for (day, model, cw, tier, ttype), amt in cost_token.items():
        month = day[:7]
        shares = usage_tok.get((day, model, cw, tier, ttype), {})
        total = sum(shares.values())
        if total <= 0:
            unattributed[month] += amt
            continue
        # ascending by tokens, then by key id so ties break deterministically and
        # re-runs of the same month are byte-identical; the largest holder is last.
        items = sorted(shares.items(), key=lambda kv: (kv[1], str(kv[0])))
        used = Decimal(0)
        for i, (akid, toks) in enumerate(items):
            if i == len(items) - 1:
                part = amt - used                       # remainder to the largest holder
            else:
                # round DOWN so `used` can never exceed the line amount; the
                # accumulated floor remainder then goes to the largest holder,
                # keeping every share non-negative.
                part = (amt * Decimal(toks) / Decimal(total)).quantize(
                    Decimal("0.000001"), rounding=ROUND_DOWN)
                used += part
            alloc[(month, akid)] += part
    return alloc, unattributed


def reconcile(cost_total: Decimal, alloc: dict, unattributed: dict, cost_nontoken: dict) -> Decimal:
    """Residual (in cents) that should be ~0: the Cost API total minus everything we
    placed -- allocated to keys, left unattributed, or booked as non-token cost.
    This is the tool's headline invariant; kept pure so it can be checked offline."""
    return cost_total - (sum(alloc.values()) + sum(unattributed.values()) + sum(cost_nontoken.values()))


def self_test(month: str) -> int:
    """Live smoke test: pull ONE month (2 API calls) and validate the cost pull +
    per-key allocation against the current API with NO stored golden values.
    Asserts data came back, the fields we parse are present (catches response-shape
    drift), the allocation reconciles to the Cost API total, unattributed cost stays
    under $1.00 (normally $0), and every cost line's implied $/Mtok is in a sane band
    (catches a silent units change, e.g. cents<->dollars). It does NOT exercise the
    attribution/CSV half of the pipeline (that logic is covered by the offline
    tests). Failures are loud; an empty month fails rather than passing quietly.
    Returns an exit code (0 = pass). One month only, to respect the rate limit.
    NOTE: this proves internal consistency, not absolute correctness -- for that, do
    the one-time invoice reconciliation in REFERENCE.md.
    """
    windows = list(month_windows(month, month))
    sys.stderr.write(f"[self-test] pulling {month} (2 API calls)...\n")
    cost_token, cost_nontoken, cost_total, usage_tok, _ = collect(windows)
    alloc, unattributed = allocate(cost_token, usage_tok)

    unattr = sum(unattributed.values())
    residual = reconcile(cost_total, alloc, unattributed, cost_nontoken)
    lo, hi = Decimal("0.02"), Decimal("150")            # sane implied $/Mtok bounds (real range ~0.05-50)
    out_of_band = []
    for (day, model, cw, tier, ttype), amt in cost_token.items():
        toks = sum(usage_tok.get((day, model, cw, tier, ttype), {}).values())
        if toks <= 0 or amt <= 0:
            continue
        implied = amt * Decimal(10000) / Decimal(toks)  # (cents/100) / (tokens/1e6)
        if not (lo <= implied <= hi):
            out_of_band.append(f"{model}/{ttype}/{tier}: ${implied:.4f}/Mtok")

    checks = [
        ("cost API returned spend for this month", cost_total > 0),
        ("usage API returned token rows", bool(usage_tok)),
        ("cost lines carry model + token_type (schema ok)",
         bool(cost_token) and all(k[1] and k[4] for k in cost_token)),
        (f"reconciles to Cost API total (residual ${residual / 100:.4f})", abs(residual) < Decimal("0.01")),
        (f"unattributed under $1.00 (${unattr / 100:.2f})", unattr < Decimal("100")),
        (f"implied unit prices within ${lo}-${hi}/Mtok ({len(out_of_band)} outliers)", not out_of_band),
    ]
    for desc, passed in checks:
        sys.stderr.write(f"  [{'PASS' if passed else 'FAIL'}] {desc}\n")
    for o in out_of_band[:5]:
        sys.stderr.write(f"        out-of-band: {o}\n")
    if cost_total == 0:
        sys.stderr.write("  NOTE: choose a --self-test month that had activity.\n")
    ok = all(p for _, p in checks)
    sys.stderr.write(f"\nself-test {'PASSED' if ok else 'FAILED'} for {month}. This checks internal "
                     "consistency; run the invoice reconciliation in REFERENCE.md for absolute truth.\n")
    return 0 if ok else 1


def main() -> None:
    ds, de = default_range()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=ds, help=f"first month YYYY-MM (default {ds})")
    ap.add_argument("--end", default=de, help=f"last month YYYY-MM inclusive (default {de})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output CSV path (default: the git-ignored output/ folder next to the skill)")
    ap.add_argument("--self-test", nargs="?", const=de, metavar="YYYY-MM",
                    help="live smoke test on one month (2 API calls), then exit; "
                         "defaults to the most recent full month")
    args = ap.parse_args()
    # Validate every month-shaped value that was supplied (defaults are always
    # valid). The regex enforces month 01-12 so a value like 2026-13 is rejected
    # here with a friendly message instead of blowing up later in datetime().
    to_check = [("--start", args.start), ("--end", args.end)]
    if args.self_test is not None:
        to_check.append(("--self-test", args.self_test))
    for flag, val in to_check:
        if not MONTH_RE.fullmatch(val):
            sys.exit(f"{flag} must be YYYY-MM with month 01-12, got: {val!r}")

    if not ADMIN_KEY:
        sys.exit("Set ANTHROPIC_ADMIN_KEY (an sk-ant-admin01-... key) first.")
    if not ADMIN_KEY.startswith("sk-ant-admin"):
        sys.stderr.write("WARNING: key is not sk-ant-admin...; the Admin API rejects regular keys.\n")

    if args.self_test is not None:
        sys.exit(self_test(args.self_test))

    windows = list(month_windows(args.start, args.end))
    sys.stderr.write(f"Range {args.start}..{args.end} ({len(windows)} months); fetching key names...\n")
    names = fetch_key_names()
    sys.stderr.write(f"  {len(names)} keys\n")

    cost_token, cost_nontoken, cost_total, usage_tok, key_tokens = collect(windows)
    alloc, unattributed = allocate(cost_token, usage_tok)

    def label(akid: str | None) -> str:
        return "(no API key - Console/Workbench)" if akid is None else names.get(akid, akid)

    out_fields = ["month", "user", "key_type", "api_key_name", "spend_usd",
                  "uncached_input_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens", "output_tokens"]
    rows = []
    for (month, akid) in sorted(set(alloc) | set(key_tokens), key=lambda x: (x[0], str(label(x[1])).lower())):
        nm = label(akid)
        tk = key_tokens.get((month, akid), {})
        rows.append({
            "month": month, "user": attribute(nm), "key_type": key_type(nm),
            "api_key_name": nm,
            "spend_usd": f"{alloc.get((month, akid), Decimal(0)) / 100:.2f}",
            "uncached_input_tokens": tk.get("uncached_input_tokens", 0),
            "cache_read_input_tokens": tk.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": (tk.get("cache_creation.ephemeral_1h_input_tokens", 0)
                                            + tk.get("cache_creation.ephemeral_5m_input_tokens", 0)),
            "output_tokens": tk.get("output_tokens", 0),
        })
    for (month, ctype), cents in sorted(cost_nontoken.items()):
        rows.append({"month": month, "user": "uncategorized", "key_type": "service",
                     "api_key_name": f"(non-token: {ctype})",
                     "spend_usd": f"{cents / 100:.2f}", "uncached_input_tokens": 0,
                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    # summary -- accumulate EXACT cents from alloc (not the per-row rounded CSV
    # strings), so the printed GRAND TOTAL reconciles to the Cost API total.
    cc, dev, svc = defaultdict(Decimal), defaultdict(Decimal), defaultdict(Decimal)
    for (month, akid), cents in alloc.items():
        nm = label(akid)
        kt = key_type(nm)
        (cc if kt == "claude_code" else dev if kt == "dev" else svc)[attribute(nm)] += cents
    for (month, ctype), cents in cost_nontoken.items():
        svc["uncategorized"] += cents
    sys.stderr.write("\n=== spend by person (USD) ===\n")
    sys.stderr.write(f"{'PERSON':<16}{'ClaudeCode':>12}{'Dev':>11}{'Total':>11}\n")
    for u in sorted(set(cc) | set(dev), key=lambda u: -(cc[u] + dev[u])):
        sys.stderr.write(f"{u:<16}{cc[u] / 100:>12,.2f}{dev[u] / 100:>11,.2f}{(cc[u] + dev[u]) / 100:>11,.2f}\n")
    sys.stderr.write("\n=== non-person buckets (USD) ===\n")
    for u in sorted(svc, key=lambda u: -svc[u]):
        sys.stderr.write(f"  {u:<18}{svc[u] / 100:>11,.2f}\n")
    grand = sum(cc.values()) + sum(dev.values()) + sum(svc.values())
    unattr = sum(unattributed.values())
    residual = reconcile(cost_total, alloc, unattributed, cost_nontoken)
    sys.stderr.write(f"\nGRAND TOTAL ${grand / 100:,.2f}   (Cost API total ${cost_total / 100:,.2f}, "
                     f"unattributed ${unattr / 100:,.2f})\n")
    if abs(residual) >= Decimal("0.01"):
        sys.stderr.write(f"WARNING: reconciliation residual ${residual / 100:,.4f} -- the split did not "
                         "fully add back to the Cost API total; investigate before trusting per-person figures.\n")
    if unattr > 0:
        sys.stderr.write("NOTE: 'unattributed' = token cost with no matching usage row (rare).\n")
    sys.stderr.write("NOTE: priority/flex-tier spend is omitted by the Cost API (its tokens still show).\n")

    flagged = {r["api_key_name"] for r in rows
               if r["user"] == "uncategorized" and not r["api_key_name"].startswith("(")}
    if flagged:
        # exact per-key spend straight from alloc (not the rounded CSV strings),
        # summed only over the (usually tiny) flagged set.
        spend_by_name: dict = defaultdict(Decimal)
        for (month, akid), cents in alloc.items():
            nm = label(akid)
            if nm in flagged:
                spend_by_name[nm] += cents
        sys.stderr.write(f"\n⚠ {len(flagged)} key(s) could not be categorized — add a fragment to "
                         "PEOPLE (a person) or BUCKET (a project/service) in the ORG CONFIG, then re-run:\n")
        for nm in sorted(flagged, key=lambda n: -spend_by_name[n]):
            sys.stderr.write(f"    {nm}  (${spend_by_name[nm] / 100:,.2f})\n")

    sys.stderr.write(f"\nWrote {len(rows)} rows to {args.out}\n")


if __name__ == "__main__":
    main()
