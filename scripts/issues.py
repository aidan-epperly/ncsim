#!/usr/bin/env python3
"""Issue tracker for ncsim: one Markdown file per issue in issues/, mirrored to GitHub.

Local files work offline and can be filed from any checkout; GitHub Issues
(aidan-epperly/ncsim) is the shared view.  See CONTRIBUTING.md for the
workflow.

    python scripts/issues.py new bug "Relaxation::build panics on ..." --area elimination --severity 2
    python scripts/issues.py list [--all] [--type bug] [--area solver]
    python scripts/issues.py show 6
    python scripts/issues.py close 6 --commit abc1234 --note "Fixed by ..."  [--status wontfix]
    python scripts/issues.py index          # regenerate issues/README.md
    python scripts/issues.py check          # validate files and index
    python scripts/issues.py labels         # create/update the GitHub labels      (needs gh)
    python scripts/issues.py push [IDS...]  # open GitHub issues for unmirrored files (needs gh)
    python scripts/issues.py pull           # copy GitHub open/closed state back    (needs gh)
    python scripts/issues.py sync           # push, then update text and state on GitHub (needs gh)

Standard library only.  Frontmatter is flat `key: value` YAML.
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUES = ROOT / "issues"
INDEX = ISSUES / "README.md"

TYPES = ["bug", "improvement", "docs"]
STATUSES = ["open", "needs-repro", "fixed", "wontfix", "duplicate"]
CLOSED = {"fixed", "wontfix", "duplicate"}
AREAS = [
    "pauli", "clifford", "phasepoly", "gate", "circuit", "state", "pathsum",
    "stabsum", "mps", "bridge", "examples", "docs", "build",
]
SEVERITIES = {
    "1": "wrong result without an error (invalid bound, wrong normal form, wrong certificate)",
    "2": "crash, panic or error on valid input",
    "3": "performance or usability",
}
FIELDS = ["id", "title", "type", "status", "area", "severity", "created",
          "closed", "reporter", "commit", "fixed_in", "github"]

# name -> (color, description); `labels` pushes exactly this set.
LABELS = {
    "bug": ("d73a4a", "Something computes or behaves incorrectly"),
    "improvement": ("a2eeef", "New capability, performance or usability"),
    "docs": ("0075ca", "README, API or example documentation"),
    "needs-repro": ("fbca04", "Not reproduced on the current commit yet"),
    "sev:1-wrong-result": ("b60205", SEVERITIES["1"]),
    "sev:2-crash": ("d93f0b", SEVERITIES["2"]),
    "sev:3-usability": ("fef2c0", SEVERITIES["3"]),
    **{f"area:{a}": ("c5def5", f"Area: {a}") for a in AREAS},
}
SEV_LABEL = {"1": "sev:1-wrong-result", "2": "sev:2-crash", "3": "sev:3-usability"}

TEMPLATES = {
    "bug": """## Summary

<one or two sentences: what is wrong>

## Reproduce

```
<smallest command or code that shows it; attach the input files or inline the code>
```

## Observed

<output, panic message, wrong value; include the value a trusted method gives>

## Expected

<what should happen>

## Environment

- ncsim commit: {commit}
- {env}

## Notes

<suspected cause, location (src/file.rs:line), workaround>
""",
    "improvement": """## Motivation

<what task this would make possible, faster or less error-prone>

## Proposal

<API sketch, algorithm, or doc change>

## Acceptance

- [ ] <test, benchmark or example that shows it is done>

## Context

- ncsim commit: {commit}
""",
    "docs": """## What is missing or wrong

<README section, doc comment or example>

## Where it cost time

<what you were doing when you needed it>

## Suggested text or example

<draft>
""",
}


def today():
    return datetime.date.today().isoformat()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def current_commit():
    sha = git("rev-parse", "--short", "HEAD")
    if not sha:
        return "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "src", "Cargo.toml")
    return sha + (" (uncommitted changes in src/)" if dirty else "")


def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60].rstrip("-") or "issue"


# ---------------------------------------------------------------- files

class Issue:
    def __init__(self, path, meta, body):
        self.path, self.meta, self.body = path, meta, body

    @property
    def id(self):
        return int(self.meta["id"])

    @property
    def is_open(self):
        return self.meta.get("status", "open") not in CLOSED

    def write(self):
        lines = ["---"]
        for k in FIELDS:
            v = self.meta.get(k, "")
            lines.append(f"{k}: {v}".rstrip())
        for k, v in self.meta.items():
            if k not in FIELDS:
                lines.append(f"{k}: {v}".rstrip())
        lines.append("---")
        self.path.write_text("\n".join(lines) + "\n\n" + self.body.strip() + "\n",
                             encoding="utf-8", newline="\n")

    def labels(self):
        out = [self.meta["type"]]
        if self.meta.get("severity") in SEV_LABEL:
            out.append(SEV_LABEL[self.meta["severity"]])
        if self.meta.get("area"):
            out += [f"area:{a.strip()}" for a in self.meta["area"].split(",")]
        if self.meta.get("status") == "needs-repro":
            out.append("needs-repro")
        return out


def parse(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        raise ValueError(f"{path.name}: missing --- frontmatter ---")
    meta = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        k, sep, v = line.partition(":")
        if not sep:
            raise ValueError(f"{path.name}: bad frontmatter line {line!r}")
        meta[k.strip()] = v.strip()
    return Issue(path, meta, m.group(2))


def load_all():
    if not ISSUES.is_dir():
        return []
    issues = [parse(p) for p in sorted(ISSUES.glob("[0-9][0-9][0-9][0-9]-*.md"))]
    return sorted(issues, key=lambda i: i.id)


def find(issues, ident):
    for i in issues:
        if i.id == int(ident):
            return i
    sys.exit(f"no issue {ident}")


def validate(issue):
    errs = []
    m = issue.meta
    for k in ("id", "title", "type", "status", "created"):
        if not m.get(k):
            errs.append(f"missing {k}")
    if m.get("type") not in TYPES:
        errs.append(f"type {m.get('type')!r} not in {TYPES}")
    if m.get("status") not in STATUSES:
        errs.append(f"status {m.get('status')!r} not in {STATUSES}")
    for a in filter(None, (s.strip() for s in m.get("area", "").split(","))):
        if a not in AREAS:
            errs.append(f"area {a!r} not in {AREAS}")
    if m.get("severity") and m["severity"] not in SEVERITIES:
        errs.append(f"severity {m['severity']!r} not in 1/2/3")
    if m.get("type") == "bug" and not m.get("severity"):
        errs.append("bug without severity")
    if m.get("status") in CLOSED and not m.get("closed"):
        errs.append("closed status without closed date")
    if str(m.get("id", "")).isdigit() and not issue.path.name.startswith(f"{int(m['id']):04d}-"):
        errs.append("file name does not start with the zero-padded id")
    return [f"{issue.path.name}: {e}" for e in errs]


# ---------------------------------------------------------------- index

LOCAL = "local"  # `github: local`: never mirrored to GitHub (push, pull and close skip it)


def gh_number(m):
    """The GitHub issue number of an issue's metadata, or None (unmirrored or local-only)."""
    v = m.get("github")
    return v if v and v != LOCAL else None


def gh_label(m):
    v = m.get("github")
    return "" if not v else (LOCAL if v == LOCAL else f"#{v}")


def render_index(issues):
    def row(i):
        m = i.meta
        gh = gh_label(m)
        sev = f"S{m['severity']}" if m.get("severity") else ""
        return (f"| {i.id} | [{m['title']}]({i.path.name}) | {m['type']} | {sev} | "
                f"{m.get('area', '')} | {m['status']} | {m['created']} | {gh} |")

    head = ("| # | Title | Type | Sev | Area | Status | Created | GitHub |\n"
            "|---|---|---|---|---|---|---|---|")
    open_ = [i for i in issues if i.is_open]
    closed = [i for i in issues if not i.is_open]
    open_.sort(key=lambda i: (i.meta.get("severity") or "9", i.id))
    closed.sort(key=lambda i: -i.id)
    sev = "\n".join(f"- **S{k}**: {v}" for k, v in SEVERITIES.items())
    return f"""# ncsim issues

Generated by `python scripts/issues.py index`; do not edit by hand.
How to file, fix and close issues: [CONTRIBUTING.md](../CONTRIBUTING.md).

Severity (bugs):
{sev}

## Open ({len(open_)})

{head}
{chr(10).join(map(row, open_)) or "| | none | | | | | | |"}

## Closed ({len(closed)})

{head}
{chr(10).join(map(row, closed)) or "| | none | | | | | | |"}
"""


def write_index(issues):
    INDEX.write_text(render_index(issues), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- GitHub

def gh(*args, input=None):
    exe = shutil.which("gh")
    if not exe:
        sys.exit("GitHub CLI `gh` not found: install it (winget install GitHub.cli, "
                 "https://cli.github.com) and run `gh auth login`.")
    r = subprocess.run([exe, *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", input=input)
    if r.returncode:
        sys.exit(f"gh {' '.join(args[:2])} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def github_body(issue, repo_url):
    m = issue.meta
    rel = issue.path.relative_to(ROOT).as_posix()
    facts = [f"**Reported** {m['created']}" + (f" by {m['reporter']}" if m.get("reporter") else "")]
    if m.get("commit"):
        facts.append(f"**at commit** `{m['commit']}`")
    if m.get("fixed_in"):
        facts.append(f"**fixed in** `{m['fixed_in']}`")
    return (issue.body.strip() + "\n\n---\n" + ", ".join(facts) +
            f".  Mirrored from [`{rel}`]({repo_url}/blob/main/{rel}).\n")


# ---------------------------------------------------------------- commands

def cmd_new(a):
    if a.type not in TYPES:
        sys.exit(f"type must be one of {TYPES}")
    if a.type == "bug" and not a.severity:
        sys.exit("bugs need --severity (1 wrong result, 2 crash, 3 usability)")
    ISSUES.mkdir(exist_ok=True)
    issues = load_all()
    nid = max((i.id for i in issues), default=0) + 1
    commit = current_commit()
    if a.body_file:
        body = Path(a.body_file).read_text(encoding="utf-8") if a.body_file != "-" else sys.stdin.read()
    else:
        env = f"{sys.platform}, Python {sys.version.split()[0]}"
        body = TEMPLATES[a.type].format(commit=commit, env=env)
    meta = {
        "id": nid, "title": a.title.replace("\n", " "), "type": a.type,
        "status": "needs-repro" if a.needs_repro else "open",
        "area": a.area or "", "severity": a.severity or "", "created": today(),
        "closed": "", "reporter": a.reporter or "", "commit": commit.split()[0],
        "fixed_in": "", "github": "",
    }
    issue = Issue(ISSUES / f"{nid:04d}-{slugify(a.title)}.md", meta, body)
    errs = validate(issue)
    if errs:
        sys.exit("\n".join(errs))
    issue.write()
    write_index(load_all())
    print(issue.path.relative_to(ROOT).as_posix())


def cmd_list(a):
    for i in load_all():
        m = i.meta
        if not a.all and not i.is_open:
            continue
        if a.type and m["type"] != a.type or a.area and a.area not in m.get("area", ""):
            continue
        sev = f"S{m['severity']}" if m.get("severity") else "  "
        gh_ = gh_label(m)
        print(f"{i.id:4d}  {m['status']:<11} {m['type']:<11} {sev}  "
              f"{m.get('area', ''):<12} {m['title']}  {gh_}")


def cmd_show(a):
    i = find(load_all(), a.id)
    print(i.path.read_text(encoding="utf-8"))


def cmd_close(a):
    issues = load_all()
    i = find(issues, a.id)
    if a.status not in CLOSED:
        sys.exit(f"--status must be one of {sorted(CLOSED)}")
    i.meta.update(status=a.status, closed=today())
    if a.commit:
        i.meta["fixed_in"] = a.commit
    note = a.note or ""
    i.body = i.body.rstrip() + f"\n\n## Resolution ({today()})\n\n" + \
        (f"{a.status.capitalize()}" + (f" in `{a.commit}`" if a.commit else "") + ". " + note).strip() + "\n"
    i.write()
    write_index(issues)
    if gh_number(i.meta) and shutil.which("gh"):
        reason = "completed" if a.status == "fixed" else "not planned"
        gh("issue", "close", i.meta["github"], "--reason", reason,
           "--comment", (f"Fixed in {a.commit}. " if a.commit else "") + note)
        print(f"closed GitHub #{i.meta['github']}")
    print(i.path.relative_to(ROOT).as_posix())


def cmd_index(a):
    write_index(load_all())
    print(INDEX.relative_to(ROOT).as_posix())


def cmd_check(a):
    try:
        issues = load_all()
    except ValueError as e:
        sys.exit(str(e))
    errs = [e for i in issues for e in validate(i)]
    ids = [i.id for i in issues]
    if len(ids) != len(set(ids)):
        errs.append("duplicate issue ids")
    if not INDEX.exists() or INDEX.read_text(encoding="utf-8") != render_index(issues):
        errs.append("issues/README.md is stale: run `python scripts/issues.py index`")
    if errs:
        sys.exit("\n".join(errs))
    print(f"{len(issues)} issues OK ({sum(i.is_open for i in issues)} open)")


def cmd_labels(a):
    for name, (color, desc) in LABELS.items():
        gh("label", "create", name, "--color", color, "--description", desc[:100], "--force")
        print(name)


def cmd_push(a):
    issues = load_all()
    repo_url = json.loads(gh("repo", "view", "--json", "url"))["url"]
    todo = [find(issues, x) for x in a.ids] if a.ids else [i for i in issues if not i.meta.get("github")]
    for i in todo:
        if i.meta.get("github") == LOCAL:
            print(f"{i.id}: local only (github: {LOCAL}), not mirrored")
            continue
        if i.meta.get("github"):
            print(f"{i.id}: already #{i.meta['github']}")
            continue
        args = ["issue", "create", "--title", i.meta["title"], "--body-file", "-"]
        for lab in i.labels():
            args += ["--label", lab]
        url = gh(*args, input=github_body(i, repo_url))
        num = url.rstrip("/").rsplit("/", 1)[-1]
        i.meta["github"] = num
        i.write()
        if not i.is_open:
            gh("issue", "close", num, "--reason",
               "completed" if i.meta["status"] == "fixed" else "not planned")
        print(f"{i.id} -> {url}")
    write_index(issues)


def cmd_sync(a):
    """Make GitHub match the issue files: create missing issues (except `github: local`), update the
    title and body of mirrored ones, and open/close them to match the local status."""
    cmd_push(argparse.Namespace(ids=[]))
    issues = load_all()
    repo_url = json.loads(gh("repo", "view", "--json", "url"))["url"]
    for i in issues:
        num = gh_number(i.meta)
        if not num:
            continue
        info = json.loads(gh("issue", "view", str(num), "--json", "title,body,state"))
        body = github_body(i, repo_url)
        changed = []
        norm = lambda t: t.replace("\r\n", "\n").strip()
        if info["title"] != i.meta["title"] or norm(info["body"]) != norm(body):
            gh("issue", "edit", str(num), "--title", i.meta["title"], "--body-file", "-", input=body)
            changed.append("text")
        if i.is_open and info["state"] == "CLOSED":
            gh("issue", "reopen", str(num))
            changed.append("reopened")
        elif not i.is_open and info["state"] == "OPEN":
            gh("issue", "close", str(num), "--reason",
               "completed" if i.meta["status"] == "fixed" else "not planned")
            changed.append("closed")
        if changed:
            print(f"{i.id}: #{num} updated ({', '.join(changed)})")
    write_index(issues)


def cmd_pull(a):
    issues = load_all()
    for i in issues:
        if not gh_number(i.meta):
            continue
        info = json.loads(gh("issue", "view", i.meta["github"], "--json", "state,stateReason,closedAt"))
        if info["state"] == "CLOSED" and i.is_open:
            status = "fixed" if info.get("stateReason") == "COMPLETED" else "wontfix"
            i.meta.update(status=status, closed=(info.get("closedAt") or today())[:10])
            i.body = i.body.rstrip() + f"\n\n## Resolution ({i.meta['closed']})\n\nClosed on GitHub (#{i.meta['github']}).\n"
            i.write()
            print(f"{i.id}: closed on GitHub -> {status}")
        elif info["state"] == "OPEN" and not i.is_open:
            i.meta.update(status="open", closed="")
            i.write()
            print(f"{i.id}: reopened on GitHub")
    write_index(issues)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new", help="create an issue file from a template")
    n.add_argument("type", choices=TYPES)
    n.add_argument("title")
    n.add_argument("--area", help=f"comma-separated, from {AREAS}")
    n.add_argument("--severity", choices=list(SEVERITIES))
    n.add_argument("--reporter", help="name or GitHub handle of the reporter")
    n.add_argument("--body-file", help="Markdown body instead of the template ('-' for stdin)")
    n.add_argument("--needs-repro", action="store_true", help="seen once, not reproduced yet")
    n.set_defaults(fn=cmd_new)

    ls = sub.add_parser("list", help="list open issues")
    ls.add_argument("--all", action="store_true")
    ls.add_argument("--type", choices=TYPES)
    ls.add_argument("--area")
    ls.set_defaults(fn=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("id")
    s.set_defaults(fn=cmd_show)

    c = sub.add_parser("close", help="mark fixed/wontfix/duplicate and append a resolution")
    c.add_argument("id")
    c.add_argument("--status", default="fixed")
    c.add_argument("--commit")
    c.add_argument("--note")
    c.set_defaults(fn=cmd_close)

    sub.add_parser("index", help="regenerate issues/README.md").set_defaults(fn=cmd_index)
    sub.add_parser("check", help="validate issue files and the index").set_defaults(fn=cmd_check)
    sub.add_parser("labels", help="create the GitHub labels (gh)").set_defaults(fn=cmd_labels)

    pu = sub.add_parser("push", help="open GitHub issues for files without one (gh)")
    pu.add_argument("ids", nargs="*")
    pu.set_defaults(fn=cmd_push)
    sub.add_parser("pull", help="copy GitHub open/closed state back (gh)").set_defaults(fn=cmd_pull)
    sub.add_parser("sync", help="push new issues and update text and state of mirrored ones (gh)").set_defaults(
        fn=cmd_sync)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
