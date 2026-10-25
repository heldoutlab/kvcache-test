#!/opt/homebrew/bin/python3.12
"""Programmatic long-context task generator for the KV-cache test (Held Out Lab).

Stdlib only, deterministic. Every document is invented from scratch (fictional organisations, people,
IDs and numbers); no external text is used.

Each document is an "operations archive": reference tables (people, assets, projects) followed by a
dated daily journal (work orders, expense claims, contract amendments). Questions about one document:

  R   multi-needle retrieval  - give a recorded code for 3 journal entries, in order
  M3  3-hop lookup            - asset -> owning project -> project lead -> lead's base site
  M2  2-hop lookup            - journal event -> person who handled it -> person's unit
  AS  aggregation (sum)       - total amount of claims matching unit + category + status
  AC  aggregation (count)     - number of events matching person + priority + status
  L   latest version          - value of a contract clause after all effective amendments

Sets (disjoint IDs, names, seeds and document templates):
  public       theme "public"       (pipe tables, terse journal lines)
  private      theme from private_theme.py (prose crew/vessel log) - NEVER published
  calibration  theme "calibration"  (key=value rows) - difficulty/timing pilot only, never scored

Usage:
  python3.12 gen_tasks.py              # write tasks/{public,private,calibration}.jsonl + tasks/MANIFEST.json
  python3.12 gen_tasks.py --check      # regenerate twice in memory, solve every question from the rendered
                                       # text alone, check sizes/disjointness/files on disk; prints OK
  python3.12 gen_tasks.py --cpt 3.6    # different chars-per-token estimate (only before freezing)

Token counts here are approximate (chars / CHARS_PER_TOKEN). The real Muse Glimmer tokenizer count is
measured at run time (harness/run.py --preflight) and the documents are regenerated with a corrected
--cpt before freezing if they fall outside the allowed band (see PREREG.md).
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import sys
from decimal import Decimal, InvalidOperation

GEN_VERSION = "kvcache-gen 1.0 (2026-10-22)"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "tasks")

CHARS_PER_TOKEN = 4.0
# Target document length in approximate tokens. Leaves room for system prompt, question and answer.
CONTEXTS = {"4k": 3600, "16k": 15000, "48k": 46000}
SIZE_TOL = 0.03  # each document must be within +-3% of its target (approximate tokens)

DOCS_PER_CONTEXT = {"public": 6, "private": 6, "calibration": 1}
PLAN = {
    "public": ["R", "R", "M3", "M2", "AS", "AC", "L", "L"],
    "private": ["R", "R", "M3", "M2", "AS", "AC", "L", "L"],
    "calibration": ["R", "M3", "AS", "L", "M2", "AC"],
}
SEEDS = {"public": 410_771, "calibration": 958_213}  # private seed lives in private_theme.py

# Difficulty knobs (identical across context lengths, so only document length changes between them)
N_NEEDLES = 3
AS_QUALIFYING = (4, 7)
AC_QUALIFYING = (3, 6)
L_CHAIN = (3, 5)
P_LAST_WITHDRAWN = 0.5
REF_SHARE = 0.22                       # share of document chars spent on reference tables
REF_SPLIT = {"people": 0.35, "assets": 0.45, "projects": 0.20}
JOURNAL_MIX = {"event": 0.5, "claim": 0.3, "amend": 0.2}

ANSWER_INSTRUCTION = "Reply with your answer on a final line in the form: ANSWER: <answer>"

# ----------------------------------------------------------------------------------------------
# Themes. A theme is data only: vocabulary, ID formats and line templates. The generator logic is
# shared; the solver re-reads rendered text with regexes built from the same line templates.
# ----------------------------------------------------------------------------------------------

FIRST_NAMES = [
    "Amara", "Bastian", "Celeste", "Dario", "Elif", "Fergus", "Greta", "Hamid", "Ilse", "Joaquin",
    "Kalinda", "Leopold", "Mireille", "Nikhil", "Odette", "Pavel", "Rosalind", "Soren", "Tamsin", "Ulrich",
    "Valeria", "Wendell", "Yusra", "Zoltan", "Anouk", "Bram", "Cosima", "Desmond", "Esme", "Florian",
    "Gideon", "Hester", "Imogen", "Jasper", "Katja", "Lorcan", "Maelle", "Niamh", "Osric", "Priya",
]

PUBLIC = {
    "key": "public",
    "title": "Harbour City Facilities Department - Operations Archive (fictional)",
    "intro": ("This archive is a generated, fictional record kept by a municipal facilities department. "
              "It contains a staff directory, an asset register, a project register and a daily journal "
              "of work orders, expense claims and contract amendments."),
    "first_names": FIRST_NAMES,
    "surnames": ["Abernathy", "Bellweather", "Corrigan", "Dunmore", "Ellsworth", "Farrow", "Galloway", "Hartigan",
                 "Ingleby", "Jessop", "Kincaid", "Lowther", "Merriman", "Norcross", "Ormsby", "Pryce", "Quayle",
                 "Redfern", "Selwyn", "Tolliver", "Upshaw", "Vantongeren", "Whitlock", "Yardley", "Ashcombe",
                 "Brightwell", "Cardew", "Dalgleish", "Emberton", "Fairbairn"],
    "units": ["Grounds", "Electrical", "HVAC", "Plumbing", "Custodial", "Security", "Fleet", "Carpentry",
              "Energy Office", "Events"],
    "sites": ["Cobalt Hall", "Larchmont Annex", "Pier 4 Depot", "Westgate Tower", "Fernhill Yard",
              "Quarry Lane Works", "Old Mint Building", "Saltworks Block", "Beacon House", "Hollins Garage"],
    "roles": ["technician", "supervisor", "planner", "coordinator", "inspector", "apprentice"],
    "asset_kinds": ["chiller unit", "boiler", "air handler", "lift car", "fire panel", "backup generator",
                    "sprinkler riser", "rooftop fan", "water heater", "access gate", "solar inverter", "sump pump"],
    "project_a": ["Riverside", "North Quay", "Civic Centre", "Harbourfront", "Library", "Depot", "Market Hall",
                  "Eastside", "Parkland", "Council Chamber"],
    "project_b": ["Lighting Upgrade", "Roof Renewal", "Accessibility Works", "Heat Pump Retrofit", "Drainage Repair",
                  "Security Refresh", "Window Replacement", "Fire Safety Review"],
    "project_c": ["Phase 1", "Phase 2", "Phase 3"],
    "task_verbs": ["replaced", "inspected", "cleaned", "recalibrated", "tightened", "lubricated", "tested",
                   "repaired", "reset", "serviced"],
    "task_objects": ["drive belt", "pressure sensor", "control board", "door seal", "filter bank", "isolation valve",
                     "thermostat", "emergency light", "cable tray", "drain trap"],
    "claim_categories": ["materials", "travel", "training", "tools", "equipment hire", "safety gear"],
    "priorities": ["P1", "P2", "P3"],
    "event_statuses": ["CLOSED", "OPEN", "DEFERRED"],
    "claim_statuses": ["APPROVED", "REJECTED", "PENDING"],
    "amend_statuses": ["EFFECTIVE", "WITHDRAWN"],
    "clauses": [
        {"name": "late-payment fee (EUR)", "kind": "int", "lo": 40, "hi": 400, "step": 5},
        {"name": "payment term (days)", "kind": "int", "lo": 14, "hi": 90, "step": 1},
        {"name": "service credit (%)", "kind": "dec", "lo": 0.5, "hi": 9.75, "step": 0.25},
        {"name": "response time (hours)", "kind": "int", "lo": 2, "hi": 72, "step": 1},
        {"name": "annual cap (EUR)", "kind": "int", "lo": 5000, "hi": 90000, "step": 500},
    ],
    "ids": {"person": ("S-", 4), "asset": ("AX-", 4), "project": ("PRJ-", 3), "event": ("WO-", 5),
            "claim": ("EC-", 5), "contract": ("C-", 4), "amend": ("AM-", 5)},
    "code": "LD-LLD",  # L = letter, D = digit
    "date_fmt": "### {iso} ({weekday})",
    "people_header": "## Staff directory\nID | Name | Unit | Base building | Role",
    "person_row": "{pid} | {name} | {unit} | {site} | {role}",
    "asset_header": "## Asset register\nAsset | Type | Owning project",
    "asset_row": "{aid} | {kind} | {prj}",
    "project_header": "## Project register\nProject | Title | Lead",
    "project_row": "{prj} | {pname} | {lead}",
    "journal_header": "## Daily journal\nEntries are listed in the order they were recorded.",
    "event_line": ("{time} - Work order {eid} on asset {aid}: {task}. Priority {prio}. Handled by {name}. "
                   "Status {status}. Inspection code {code}."),
    "claim_line": "{time} - Expense claim {cid}: {unit} unit, category {cat}, EUR {amount}, status {status}.",
    "amend_line": "{time} - Amendment {amno} to contract {kid}: {clause} set to {value}. Status {status}.",
    "q": {
        "R": ("In the daily journal, find work orders {ids}. What inspection code was recorded on each of them? "
              "Give the {n_words} codes in that order, separated by commas."),
        "M3": ("Asset {aid} is owned by a project. In which building is that project's lead based, according to "
               "the staff directory? Give the building name."),
        "M2": ("Work order {eid} was handled by one member of staff. Which unit does that person belong to, "
               "according to the staff directory? Give the unit name."),
        "AS": ("What is the total amount in EUR of all expense claims in the journal from the {unit} unit, in "
               "category {cat}, with status {status}? Give the total as a whole number."),
        "AC": ("How many work orders with priority {prio} and status {status} were handled by {name}? "
               "Give the count as a whole number."),
        "L": ("Contract {kid} was amended several times. What is the {clause} currently in force for contract "
              "{kid}? Amendments with status {withdrawn} never took effect. Give the number only."),
    },
}

CALIBRATION = {
    "key": "calibration",
    "title": "Ridgeview Polytechnic Research Equipment Office - Records (fictional)",
    "intro": ("Generated, fictional records of a university equipment office: staff list, instrument list, "
              "grant list and a dated log of service tickets, purchase orders and agreement changes."),
    "first_names": FIRST_NAMES,
    "surnames": ["Achterberg", "Blakemore", "Coulthard", "Drummond", "Etheridge", "Faulkner", "Gresham",
                 "Hollingworth", "Iddon", "Jarrold", "Kettering", "Lavelle", "Mottram", "Nuttall", "Oakshott",
                 "Penhaligon", "Rackham", "Sowerby", "Thistlewood", "Utterson"],
    "units": ["Optics Lab", "Soil Lab", "Robotics Lab", "Cryo Lab", "Imaging Core", "Materials Lab",
              "Acoustics Lab", "Genomics Core"],
    "sites": ["Hale Wing", "Corbel Court", "Ashby Block", "Linden Pavilion", "Marlowe Centre", "Quill House"],
    "roles": ["technician", "research fellow", "lab manager", "postdoc", "engineer"],
    "asset_kinds": ["spectrometer", "centrifuge", "laser cutter", "electron microscope", "freeze dryer",
                    "3D printer", "fume hood", "thermal cycler", "oscilloscope", "vacuum oven"],
    "project_a": ["Coastal", "Urban", "Alpine", "Microbial", "Polymer", "Neural", "Seismic", "Solar"],
    "project_b": ["Sensing Study", "Imaging Pilot", "Materials Survey", "Field Campaign", "Calibration Programme"],
    "project_c": ["Year 1", "Year 2"],
    "task_verbs": ["aligned", "descaled", "re-flashed", "inspected", "replaced", "cleaned", "tested", "recalibrated"],
    "task_objects": ["optics bench", "rotor", "power supply", "vacuum seal", "detector", "cooling loop",
                     "interlock", "firmware"],
    "claim_categories": ["consumables", "service contract", "spare parts", "software licence", "gas supply"],
    "priorities": ["urgent", "normal", "low"],
    "event_statuses": ["RESOLVED", "WAITING", "CANCELLED"],
    "claim_statuses": ["PAID", "DECLINED", "QUERIED"],
    "amend_statuses": ["ACTIVE", "RESCINDED"],
    "clauses": [
        {"name": "callout fee (GBP)", "kind": "int", "lo": 50, "hi": 600, "step": 10},
        {"name": "uptime guarantee (%)", "kind": "dec", "lo": 90.0, "hi": 99.75, "step": 0.25},
        {"name": "on-site visits per year", "kind": "int", "lo": 1, "hi": 24, "step": 1},
        {"name": "parts discount (%)", "kind": "int", "lo": 2, "hi": 40, "step": 1},
    ],
    "ids": {"person": ("R-", 4), "asset": ("IN-", 4), "project": ("GR-", 3), "event": ("ST-", 5),
            "claim": ("PO-", 5), "contract": ("SA-", 4), "amend": ("CHG-", 5)},
    "code": "DL-DD",
    "date_fmt": "[{iso}]",
    "people_header": "STAFF",
    "person_row": "id={pid}; name={name}; lab={unit}; building={site}; role={role}",
    "asset_header": "INSTRUMENTS",
    "asset_row": "id={aid}; instrument={kind}; grant={prj}",
    "project_header": "GRANTS",
    "project_row": "id={prj}; title={pname}; holder={lead}",
    "journal_header": "LOG (chronological)",
    "event_line": ("{time} ticket={eid}; instrument={aid}; work={task}; urgency={prio}; engineer={name}; "
                   "state={status}; tag={code}"),
    "claim_line": "{time} order={cid}; lab={unit}; type={cat}; GBP={amount}; state={status}",
    "amend_line": "{time} change={amno}; agreement={kid}; term={clause}; new value={value}; state={status}",
    "q": {
        "R": ("Find service tickets {ids} in the log. What tag was recorded on each? Give the {n_words} tags in that "
              "order, separated by commas."),
        "M3": ("Instrument {aid} is funded by a grant. In which building is that grant's holder based, according "
               "to the staff list? Give the building name."),
        "M2": "Ticket {eid} was handled by an engineer. Which lab does that engineer belong to? Give the lab name.",
        "AS": ("What is the total GBP of all purchase orders from the {unit} with type {cat} and state {status}? "
               "Give the total as a whole number."),
        "AC": ("How many service tickets with urgency {prio} and state {status} were handled by {name}? "
               "Give the count as a whole number."),
        "L": ("Agreement {kid} was changed several times. What is its {clause} now? Changes with state {withdrawn} "
              "never took effect. Give the number only."),
    },
}


def load_private():
    """The private theme and seed are kept out of the public repo (see .gitignore)."""
    sys.path.insert(0, HERE)
    try:
        import private_theme
    except ImportError:
        return None, None
    finally:
        sys.path.pop(0)
    return private_theme.THEME, private_theme.SEED


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def sub_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def approx_tokens(text, cpt):
    return int(round(len(text) / cpt))


def fmt_id(theme, kind, n):
    prefix, width = theme["ids"][kind]
    return "%s%0*d" % (prefix, width, n)


def id_space(theme, kind):
    return 10 ** theme["ids"][kind][1]


def make_code(rng, pattern):
    out = []
    for ch in pattern:
        if ch == "L":
            out.append(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
        elif ch == "D":
            out.append(rng.choice("23456789"))
        else:
            out.append(ch)
    return "".join(out)


def clause_value(rng, clause):
    lo, hi, step = clause["lo"], clause["hi"], clause["step"]
    n = int(round((hi - lo) / step))
    v = lo + step * rng.randint(0, n)
    if clause["kind"] == "int":
        return str(int(round(v)))
    return "%.2f" % v


def near_miss_id(rng, theme, kind, real, taken):
    """An ID that differs from `real` by a swap of two adjacent digits (or one digit), not already taken."""
    prefix, width = theme["ids"][kind]
    digits = list(real[len(prefix):])
    for _ in range(200):
        d = digits[:]
        i = rng.randrange(width - 1)
        if d[i] != d[i + 1] and rng.random() < 0.7:
            d[i], d[i + 1] = d[i + 1], d[i]
        else:
            j = rng.randrange(width)
            d[j] = rng.choice([c for c in "0123456789" if c != d[j]])
        cand = prefix + "".join(d)
        if cand != real and cand not in taken:
            return cand
    raise RuntimeError("could not make a near-miss id")


NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def human_list(items):
    return ", ".join(items[:-1]) + " and " + items[-1]


# ----------------------------------------------------------------------------------------------
# document builder
# ----------------------------------------------------------------------------------------------

class Doc:
    def __init__(self, theme, rng):
        self.t = theme
        self.rng = rng
        self.ids = {k: set() for k in theme["ids"]}
        self.people = []        # dicts: pid name unit site role
        self.names = set()
        self.assets = []        # aid kind prj
        self.projects = []      # prj pname lead
        self.core_journal = []  # (fraction, entry)
        self.filler_journal = []
        self.forbid_event = []  # predicates on event dicts
        self.forbid_claim = []
        self.forbid_amend = set()  # (kid, clause_idx)
        self.questions = []

    # ---- ids / entities ----
    def new_id(self, kind):
        space = id_space(self.t, kind)
        lo = space // 10 if self.t["ids"][kind][1] > 2 else 1
        for _ in range(10000):
            v = fmt_id(self.t, kind, self.rng.randrange(lo, space))
            if v not in self.ids[kind]:
                self.ids[kind].add(v)
                return v
        raise RuntimeError("id space exhausted for " + kind)

    def new_person(self, unit=None, site=None, surname=None):
        for _ in range(10000):
            name = "%s %s" % (self.rng.choice(self.t["first_names"]), surname or self.rng.choice(self.t["surnames"]))
            if name not in self.names:
                break
        else:
            raise RuntimeError("name space exhausted")
        self.names.add(name)
        p = {"pid": self.new_id("person"), "name": name, "unit": unit or self.rng.choice(self.t["units"]),
             "site": site or self.rng.choice(self.t["sites"]), "role": self.rng.choice(self.t["roles"])}
        self.people.append(p)
        return p

    def new_project(self, lead=None):
        pname = "%s %s %s" % (self.rng.choice(self.t["project_a"]), self.rng.choice(self.t["project_b"]),
                              self.rng.choice(self.t["project_c"]))
        lead = lead or self.rng.choice(self.people)
        pr = {"prj": self.new_id("project"), "pname": pname, "lead": lead["name"]}
        self.projects.append(pr)
        return pr

    def new_asset(self, prj=None, aid=None):
        if aid is None:
            aid = self.new_id("asset")
        else:
            self.ids["asset"].add(aid)
        a = {"aid": aid, "kind": self.rng.choice(self.t["asset_kinds"]),
             "prj": (prj or self.rng.choice(self.projects))["prj"]}
        self.assets.append(a)
        return a

    def task(self):
        return "%s %s" % (self.rng.choice(self.t["task_verbs"]), self.rng.choice(self.t["task_objects"]))

    def event(self, eid=None, name=None, prio=None, status=None, code=None, aid=None):
        if eid is None:
            eid = self.new_id("event")
        else:
            self.ids["event"].add(eid)
        return {"type": "event", "eid": eid, "aid": aid or self.rng.choice(self.assets)["aid"], "task": self.task(),
                "prio": self.rng.randrange(3) if prio is None else prio,
                "name": name or self.rng.choice(self.people)["name"],
                "status": self.rng.randrange(3) if status is None else status,
                "code": code or make_code(self.rng, self.t["code"])}

    def claim(self, unit=None, cat=None, status=None, amount=None):
        return {"type": "claim", "cid": self.new_id("claim"), "unit": unit or self.rng.choice(self.t["units"]),
                "cat": cat or self.rng.choice(self.t["claim_categories"]),
                "amount": amount if amount is not None else self.rng.randint(20, 950),
                "status": self.rng.randrange(3) if status is None else status}

    def amend(self, kid, clause, value=None, status=0):
        return {"type": "amend", "amno": self.new_id("amend"), "kid": kid, "clause": clause,
                "value": value or clause_value(self.rng, self.t["clauses"][clause]), "status": status}

    # ---- rendering ----
    def render_entry(self, e, time):
        t = self.t
        if e["type"] == "event":
            return t["event_line"].format(time=time, eid=e["eid"], aid=e["aid"], task=e["task"],
                                          prio=t["priorities"][e["prio"]], name=e["name"],
                                          status=t["event_statuses"][e["status"]], code=e["code"])
        if e["type"] == "claim":
            return t["claim_line"].format(time=time, cid=e["cid"], unit=e["unit"], cat=e["cat"], amount=e["amount"],
                                          status=t["claim_statuses"][e["status"]])
        return t["amend_line"].format(time=time, amno=e["amno"], kid=e["kid"], clause=t["clauses"][e["clause"]]["name"],
                                      value=e["value"], status=t["amend_statuses"][e["status"]])


def build_core(d, plan, contracts):
    """Create the entities and journal entries every question needs, plus targeted distractors."""
    t, rng = d.t, d.rng
    for qi, fam in enumerate(plan):
        qid = "%s%d" % (fam, qi)
        if fam == "R":
            ids, codes, depths = [], [], []
            # one needle per equal depth band of the journal (3 needles: 0.03-0.30, 0.36-0.63, 0.70-0.97)
            w = 0.94 / N_NEEDLES
            strata = [(0.03 + i * w + (0.03 if i else 0), 0.03 + (i + 1) * w) for i in range(N_NEEDLES)]
            rng.shuffle(strata)
            for k in range(N_NEEDLES):
                e = d.event()
                ids.append(e["eid"])
                codes.append(e["code"])
                f = rng.uniform(*strata[k])
                depths.append(round(f, 3))
                d.core_journal.append((f, e))
                nm = near_miss_id(rng, t, "event", e["eid"], d.ids["event"])
                d.core_journal.append((rng.random(), d.event(eid=nm)))
            q = t["q"]["R"].format(ids=human_list(ids), n_words=NUM_WORDS[N_NEEDLES])
            d.questions.append({"qid": qid, "family": "R", "question": q, "answer": codes, "answer_type": "list",
                                "query": {"event_ids": ids}, "meta": {"journal_depths": depths}})
        elif fam == "M3":
            site = rng.choice(t["sites"])
            lead = d.new_person(site=site)
            prj = d.new_project(lead=lead)
            asset = d.new_asset(prj=prj)
            # distractor: near-miss asset owned by another project whose lead sits elsewhere
            other_site = rng.choice([s for s in t["sites"] if s != site])
            olead = d.new_person(site=other_site)
            oprj = d.new_project(lead=olead)
            d.new_asset(prj=oprj, aid=near_miss_id(rng, t, "asset", asset["aid"], d.ids["asset"]))
            q = t["q"]["M3"].format(aid=asset["aid"])
            d.questions.append({"qid": qid, "family": "M3", "question": q, "answer": site, "answer_type": "choice",
                                "query": {"asset": asset["aid"]}, "meta": {"options": "sites", "hops": 3}})
        elif fam == "M2":
            unit = rng.choice(t["units"])
            handler = d.new_person(unit=unit)
            twin = d.new_person(unit=rng.choice([u for u in t["units"] if u != unit]),
                                surname=handler["name"].split(" ", 1)[1])
            e = d.event(name=handler["name"])
            f = rng.uniform(0.05, 0.95)
            d.core_journal.append((f, e))
            d.core_journal.append((rng.random(), d.event(name=twin["name"])))
            q = t["q"]["M2"].format(eid=e["eid"])
            d.questions.append({"qid": qid, "family": "M2", "question": q, "answer": unit, "answer_type": "choice",
                                "query": {"event": e["eid"]}, "meta": {"options": "units", "hops": 2,
                                                                         "journal_depth": round(f, 3)}})
        elif fam == "AS":
            unit, cat = rng.choice(t["units"]), rng.choice(t["claim_categories"])
            n = rng.randint(*AS_QUALIFYING)
            total = 0
            for _ in range(n):
                c = d.claim(unit=unit, cat=cat, status=0)
                total += c["amount"]
                d.core_journal.append((rng.random(), c))
            for _ in range(2):
                d.core_journal.append((rng.random(), d.claim(unit=unit, cat=cat, status=rng.choice([1, 2]))))
                d.core_journal.append((rng.random(), d.claim(unit=unit, status=0,
                                       cat=rng.choice([c for c in t["claim_categories"] if c != cat]))))
                d.core_journal.append((rng.random(), d.claim(cat=cat, status=0,
                                       unit=rng.choice([u for u in t["units"] if u != unit]))))
            d.forbid_claim.append(lambda c, u=unit, k=cat: c["unit"] == u and c["cat"] == k and c["status"] == 0)
            q = t["q"]["AS"].format(unit=unit, cat=cat, status=t["claim_statuses"][0])
            d.questions.append({"qid": qid, "family": "AS", "question": q, "answer": total, "answer_type": "int",
                                "query": {"unit": unit, "cat": cat, "status": t["claim_statuses"][0]},
                                "meta": {"n_records": n}})
        elif fam == "AC":
            person = d.new_person()
            prio = 0
            n = rng.randint(*AC_QUALIFYING)
            for _ in range(n):
                d.core_journal.append((rng.random(), d.event(name=person["name"], prio=prio, status=0)))
            d.core_journal.append((rng.random(), d.event(name=person["name"], prio=prio, status=rng.choice([1, 2]))))
            d.core_journal.append((rng.random(), d.event(name=person["name"], prio=rng.choice([1, 2]), status=0)))
            d.core_journal.append((rng.random(), d.event(name=person["name"], prio=rng.choice([1, 2]), status=0)))
            d.forbid_event.append(lambda e, nm=person["name"]: e["name"] == nm and e["prio"] == 0 and e["status"] == 0)
            q = t["q"]["AC"].format(prio=t["priorities"][prio], status=t["event_statuses"][0], name=person["name"])
            d.questions.append({"qid": qid, "family": "AC", "question": q, "answer": n, "answer_type": "int",
                                "query": {"name": person["name"], "prio": t["priorities"][prio],
                                          "status": t["event_statuses"][0]}, "meta": {"n_records": n}})
        elif fam == "L":
            for _ in range(1000):
                kid, clause = rng.choice(contracts), rng.randrange(len(t["clauses"]))
                if (kid, clause) not in d.forbid_amend:
                    break
            d.forbid_amend.add((kid, clause))
            k = rng.randint(*L_CHAIN)
            values, statuses = [], []
            for i in range(k):
                for _ in range(100):
                    v = clause_value(rng, t["clauses"][clause])
                    if v not in values:
                        break
                values.append(v)
                statuses.append(0 if i == 0 else (1 if rng.random() < 0.2 else 0))
            if rng.random() < P_LAST_WITHDRAWN:
                statuses[-1] = 1
            else:
                statuses[-1] = 0
            answer = [v for v, s in zip(values, statuses) if s == 0][-1]
            fracs = sorted(rng.uniform(0.02, 0.98) for _ in range(k))
            for f, v, s in zip(fracs, values, statuses):
                d.core_journal.append((f, d.amend(kid, clause, v, s)))
            # distractors: same contract other clause; same clause other contract (late in the log)
            oclause = rng.choice([c for c in range(len(t["clauses"])) if c != clause])
            okid = rng.choice([c for c in contracts if c != kid])
            for pair in ((kid, oclause), (okid, clause)):
                if pair not in d.forbid_amend:
                    d.core_journal.append((rng.uniform(fracs[-1], 1.0), d.amend(*pair)))
                    d.forbid_amend.add(pair)  # later chains and filler leave this pair alone
            q = t["q"]["L"].format(kid=kid, clause=t["clauses"][clause]["name"], withdrawn=t["amend_statuses"][1])
            d.questions.append({"qid": qid, "family": "L", "question": q, "answer": answer, "answer_type": "num",
                                "query": {"contract": kid, "clause": t["clauses"][clause]["name"]},
                                "meta": {"chain_length": k, "last_withdrawn": statuses[-1] == 1,
                                         "n_withdrawn": statuses.count(1)}})
        else:
            raise ValueError(fam)


def build_document(theme, set_name, ctx, index, base_seed, cpt):
    target_tokens = CONTEXTS[ctx]
    target_chars = int(target_tokens * cpt)
    seed = sub_seed(set_name, ctx, index, base_seed)
    rng = random.Random(seed)
    d = Doc(theme, rng)
    plan = PLAN[set_name]

    # rough sizes for reference tables (rows are ~45-75 chars)
    ref_chars = REF_SHARE * target_chars
    n_people = max(10, int(ref_chars * REF_SPLIT["people"] / 62))
    n_projects = max(5, int(ref_chars * REF_SPLIT["projects"] / 58))
    n_assets = max(8, int(ref_chars * REF_SPLIT["assets"] / 40))
    for _ in range(n_people // 2):
        d.new_person()
    for _ in range(max(2, n_projects // 2)):
        d.new_project()
    for _ in range(max(4, n_assets // 2)):
        d.new_asset()
    n_contracts = max(6, int(target_chars * (1 - REF_SHARE) * JOURNAL_MIX["amend"] / 105 / 5))
    contracts = sorted(d.new_id("contract") for _ in range(n_contracts))

    build_core(d, plan, contracts)

    while len(d.people) < n_people:
        d.new_person()
    while len(d.projects) < n_projects:
        d.new_project()
    while len(d.assets) < n_assets:
        d.new_asset()

    # filler journal pool (more than needed; a prefix of it is used)
    def filler(kind):
        for _ in range(1000):
            if kind == "event":
                e = d.event()
                if not any(f(e) for f in d.forbid_event):
                    return e
                d.ids["event"].discard(e["eid"])
            elif kind == "claim":
                c = d.claim()
                if not any(f(c) for f in d.forbid_claim):
                    return c
                d.ids["claim"].discard(c["cid"])
            else:
                kid, cl = rng.choice(contracts), rng.randrange(len(theme["clauses"]))
                if (kid, cl) not in d.forbid_amend:
                    return d.amend(kid, cl, status=0 if rng.random() < 0.85 else 1)
        raise RuntimeError("could not make filler")

    kinds = list(JOURNAL_MIX)
    weights = [JOURNAL_MIX[k] for k in kinds]
    pool_size = int(target_chars / 70) + 50
    pool = [filler(rng.choices(kinds, weights)[0]) for _ in range(pool_size)]

    start = datetime.date(2025, 1, 6) + datetime.timedelta(days=rng.randrange(0, 60))
    layout_seed = sub_seed(seed, "layout")

    def render(n_fill):
        lrng = random.Random(layout_seed)
        entries = [(f, e) for f, e in d.core_journal]
        for i, e in enumerate(pool[:n_fill]):
            entries.append(((i + lrng.random()) / max(n_fill, 1), e))
        entries.sort(key=lambda x: x[0])
        lines = [theme["title"], "", theme["intro"], ""]
        lines.append(theme["people_header"])
        lines += [theme["person_row"].format(**p) for p in sorted(d.people, key=lambda p: p["pid"])]
        lines.append("")
        lines.append(theme["asset_header"])
        lines += [theme["asset_row"].format(**a) for a in sorted(d.assets, key=lambda a: a["aid"])]
        lines.append("")
        lines.append(theme["project_header"])
        lines += [theme["project_row"].format(**p) for p in sorted(d.projects, key=lambda p: p["prj"])]
        lines.append("")
        lines.append(theme["journal_header"])
        day = start
        i = 0
        while i < len(entries):
            k = lrng.randint(5, 9)
            chunk = entries[i:i + k]
            i += k
            minutes = sorted(lrng.sample(range(7 * 60, 19 * 60), len(chunk)))
            lines.append("")
            lines.append(theme["date_fmt"].format(iso=day.isoformat(), weekday=day.strftime("%A"),
                                                  long=day.strftime("%A %d %B %Y").replace(" 0", " ")))
            for (f, e), m in zip(chunk, minutes):
                lines.append(d.render_entry(e, "%02d:%02d" % (m // 60, m % 60)))
            day += datetime.timedelta(days=1)
        return "\n".join(lines) + "\n"

    # pick the number of filler entries so the document lands on target
    lo, hi = 0, len(pool)
    text = render(hi)
    if len(text) < target_chars:
        raise RuntimeError("filler pool too small for %s %s %d" % (set_name, ctx, index))
    while lo < hi:
        mid = (lo + hi) // 2
        if len(render(mid)) < target_chars:
            lo = mid + 1
        else:
            hi = mid
    text = render(lo)
    sites = sorted({p["site"] for p in d.people})
    units = sorted({p["unit"] for p in d.people})
    doc_id = "%s-%s-%02d" % (set_name[:3], ctx, index)
    for q in d.questions:
        q["qid"] = "%s-%s" % (doc_id, q["qid"])
        q["question"] = q["question"] + "\n" + ANSWER_INSTRUCTION
        if q["meta"].get("options") == "sites":
            q["meta"]["options"] = sites
        elif q["meta"].get("options") == "units":
            q["meta"]["options"] = units
    return {"doc_id": doc_id, "set": set_name, "theme": theme["key"], "ctx": ctx, "target_tokens": target_tokens,
            "chars": len(text), "approx_tokens": approx_tokens(text, cpt), "cpt": cpt, "seed": seed,
            "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text, "questions": d.questions}


def build_set(set_name, theme, base_seed, cpt):
    docs = []
    for ctx in CONTEXTS:
        for i in range(DOCS_PER_CONTEXT[set_name]):
            docs.append(build_document(theme, set_name, ctx, i, base_seed, cpt))
    return docs


def build_all(cpt, require_private=True):
    ptheme, pseed = load_private()
    if ptheme is None and require_private:
        raise SystemExit("private_theme.py not found: the private set cannot be generated")
    sets = {"public": build_set("public", PUBLIC, SEEDS["public"], cpt)}
    themes = {"public": PUBLIC}
    if ptheme is not None:  # absent in the public repo; --check then covers public + calibration only
        sets["private"] = build_set("private", ptheme, pseed, cpt)
        themes["private"] = ptheme
    sets["calibration"] = build_set("calibration", CALIBRATION, SEEDS["calibration"], cpt)
    themes["calibration"] = CALIBRATION
    return sets, themes


# ----------------------------------------------------------------------------------------------
# independent solver: reads only the rendered text (plus each question's machine-readable query)
# ----------------------------------------------------------------------------------------------

ID_FIELDS = {"pid": "person", "aid": "asset", "prj": "project", "eid": "event", "cid": "claim", "kid": "contract",
             "amno": "amend"}


def template_regex(template, theme):
    """Regex for one line template. ID fields only match that theme's ID format, so header rows and
    other record types cannot match by accident."""
    parts = re.split(r"(\{[a-z]+\})", template)
    out = []
    for p in parts:
        if re.fullmatch(r"\{[a-z]+\}", p):
            field = p[1:-1]
            if field in ID_FIELDS:
                prefix, width = theme["ids"][ID_FIELDS[field]]
                out.append("(?P<%s>%s\\d{%d})" % (field, re.escape(prefix), width))
            elif field == "time":
                out.append("(?P<time>\\d\\d:\\d\\d)")
            else:
                out.append("(?P<%s>.+?)" % field)
        else:
            out.append(re.escape(p))
    return re.compile("^" + "".join(out) + "$")


def parse_document(text, theme):
    rx = {k: template_regex(theme[k], theme) for k in ("person_row", "asset_row", "project_row", "event_line",
                                                          "claim_line", "amend_line")}
    date_rx = template_regex(theme["date_fmt"], theme)
    parsed = {"people": [], "assets": [], "projects": [], "journal": [], "dates": []}
    key = {"person_row": "people", "asset_row": "assets", "project_row": "projects"}
    for line in text.splitlines():
        hits = [(k, m) for k, r in rx.items() for m in [r.match(line)] if m]
        if len(hits) > 1:
            raise AssertionError("line matches several record types: " + line)
        if hits:
            k, m = hits[0]
            if k in key:
                parsed[key[k]].append(m.groupdict())
            else:
                g = m.groupdict()
                g["type"] = k.split("_")[0]
                parsed["journal"].append(g)
            continue
        m = date_rx.match(line)
        if m:
            parsed["dates"].append(line)
    return parsed


def solve(parsed, q, theme):
    query = q["query"]
    fam = q["family"]
    events = [e for e in parsed["journal"] if e["type"] == "event"]
    if fam == "R":
        out = []
        for eid in query["event_ids"]:
            hits = [e for e in events if e["eid"] == eid]
            assert len(hits) == 1, (eid, len(hits))
            out.append(hits[0]["code"])
        return out
    if fam == "M3":
        a = [x for x in parsed["assets"] if x["aid"] == query["asset"]]
        assert len(a) == 1
        p = [x for x in parsed["projects"] if x["prj"] == a[0]["prj"]]
        assert len(p) == 1
        s = [x for x in parsed["people"] if x["name"] == p[0]["lead"]]
        assert len(s) == 1
        return s[0]["site"]
    if fam == "M2":
        e = [x for x in events if x["eid"] == query["event"]]
        assert len(e) == 1
        s = [x for x in parsed["people"] if x["name"] == e[0]["name"]]
        assert len(s) == 1
        return s[0]["unit"]
    if fam == "AS":
        return sum(int(c["amount"]) for c in parsed["journal"] if c["type"] == "claim" and c["unit"] == query["unit"]
                   and c["cat"] == query["cat"] and c["status"] == query["status"])
    if fam == "AC":
        return sum(1 for e in events if e["name"] == query["name"] and e["prio"] == query["prio"]
                   and e["status"] == query["status"])
    if fam == "L":
        eff = theme["amend_statuses"][0]
        chain = [a for a in parsed["journal"] if a["type"] == "amend" and a["kid"] == query["contract"]
                 and a["clause"] == query["clause"]]
        assert len(chain) >= 2
        return [a["value"] for a in chain if a["status"] == eff][-1]
    raise ValueError(fam)


# ----------------------------------------------------------------------------------------------
# output / check
# ----------------------------------------------------------------------------------------------

def file_payload(docs):
    return "".join(json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n" for d in docs)


def write_all(sets, cpt):
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"generator": GEN_VERSION, "cpt": cpt, "contexts": CONTEXTS, "files": {}}
    for name, docs in sets.items():
        payload = file_payload(docs)
        path = os.path.join(OUT_DIR, "%s.jsonl" % name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
        manifest["files"]["%s.jsonl" % name] = {
            "sha256": hashlib.sha256(payload.encode()).hexdigest(), "documents": len(docs),
            "questions": sum(len(d["questions"]) for d in docs)}
    with open(os.path.join(OUT_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def entity_keys(doc, theme):
    """IDs and full names that must not be shared between sets."""
    p = parse_document(doc["text"], theme)
    keys = set()
    for x in p["people"]:
        keys.add(("name", x["name"]))
        keys.add(("id", x["pid"]))
    keys |= {("id", x["aid"]) for x in p["assets"]}
    keys |= {("id", x["prj"]) for x in p["projects"]}
    for e in p["journal"]:
        for f in ("eid", "cid", "amno", "kid"):
            if f in e:
                keys.add(("id", e[f]))
    return keys


def check(cpt):
    problems = []
    sets_a, themes = build_all(cpt, require_private=False)
    sets_b, _ = build_all(cpt, require_private=False)
    if "private" not in sets_a:
        print("private_theme.py absent: private set not checked")
    for name in sets_a:
        if file_payload(sets_a[name]) != file_payload(sets_b[name]):
            problems.append("non-deterministic output for set " + name)

    summary = []
    all_keys = {}
    for name, docs in sets_a.items():
        theme = themes[name]
        want_docs = DOCS_PER_CONTEXT[name] * len(CONTEXTS)
        if len(docs) != want_docs:
            problems.append("%s: %d documents, expected %d" % (name, len(docs), want_docs))
        keys = set()
        fam_count = {}
        for d in docs:
            dev = d["approx_tokens"] / d["target_tokens"] - 1
            if abs(dev) > SIZE_TOL:
                problems.append("%s: size %d tokens vs target %d" % (d["doc_id"], d["approx_tokens"], d["target_tokens"]))
            if len(d["questions"]) != len(PLAN[name]):
                problems.append("%s: %d questions" % (d["doc_id"], len(d["questions"])))
            parsed = parse_document(d["text"], theme)
            n_lines = sum(1 for line in d["text"].splitlines() if line.strip())
            n_parsed = (len(parsed["people"]) + len(parsed["assets"]) + len(parsed["projects"]) +
                        len(parsed["journal"]) + len(parsed["dates"]))
            # every non-header line must parse as a record or a date line
            n_header = sum(1 for k in ("people_header", "asset_header", "project_header", "journal_header")
                           for _ in theme[k].splitlines()) + 2
            if n_lines - n_parsed != n_header:
                problems.append("%s: %d unparsed lines (expected %d header lines)" % (d["doc_id"], n_lines - n_parsed, n_header))
            for kind, field in (("people", "pid"), ("assets", "aid"), ("projects", "prj")):
                vals = [x[field] for x in parsed[kind]]
                if len(vals) != len(set(vals)):
                    problems.append("%s: duplicate %s" % (d["doc_id"], field))
            names = [x["name"] for x in parsed["people"]]
            if len(names) != len(set(names)):
                problems.append("%s: duplicate person name" % d["doc_id"])
            for f in ("eid", "cid", "amno"):
                vals = [e[f] for e in parsed["journal"] if f in e]
                if len(vals) != len(set(vals)):
                    problems.append("%s: duplicate journal id %s" % (d["doc_id"], f))
            # references resolve
            pnames = set(names)
            prjs = {x["prj"] for x in parsed["projects"]}
            aids = {x["aid"] for x in parsed["assets"]}
            if any(x["lead"] not in pnames for x in parsed["projects"]):
                problems.append("%s: project lead missing from directory" % d["doc_id"])
            if any(x["prj"] not in prjs for x in parsed["assets"]):
                problems.append("%s: asset owner project missing" % d["doc_id"])
            for e in parsed["journal"]:
                if e["type"] == "event" and (e["name"] not in pnames or e["aid"] not in aids):
                    problems.append("%s: event %s references unknown person/asset" % (d["doc_id"], e["eid"]))
                    break
            for q in d["questions"]:
                fam_count[q["family"]] = fam_count.get(q["family"], 0) + 1
                try:
                    got = solve(parsed, q, theme)
                except AssertionError as e:
                    problems.append("%s: solver assertion %s" % (q["qid"], e))
                    continue
                key = q["answer"]
                same = (Decimal(str(got)) == Decimal(str(key))) if q["answer_type"] in ("int", "num") else got == key
                if not same:
                    problems.append("%s: key %r but text gives %r" % (q["qid"], key, got))
                if q["answer_type"] == "choice" and key not in q["meta"]["options"]:
                    problems.append("%s: answer not among options" % q["qid"])
                for v in q["query"].values():
                    for item in (v if isinstance(v, list) else [v]):
                        if str(item) not in q["question"]:
                            problems.append("%s: query value %r not in question text" % (q["qid"], item))
            keys |= entity_keys(d, theme)
        all_keys[name] = keys
        sizes = {}
        for d in docs:
            sizes.setdefault(d["ctx"], []).append(d["approx_tokens"])
        summary.append("%-11s theme=%-11s docs=%2d questions=%3d families=%s sizes=%s" % (
            name, theme["key"], len(docs), sum(len(d["questions"]) for d in docs),
            ",".join("%s:%d" % kv for kv in sorted(fam_count.items())),
            " ".join("%s~%d-%d" % (c, min(v), max(v)) for c, v in sizes.items())))

    names = list(all_keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = all_keys[names[i]] & all_keys[names[j]]
            if inter:
                problems.append("%s and %s share %d ids/names, e.g. %s" % (names[i], names[j], len(inter),
                                                                           sorted(inter)[:3]))
    themes_text = {n: (themes[n]["event_line"], themes[n]["person_row"]) for n in themes}
    if len(set(themes_text.values())) != len(themes_text):
        problems.append("two sets share a document template")
    # within a document every question is different; across sets no question text is shared
    for docs in sets_a.values():
        for d in docs:
            qs = [q["question"] for q in d["questions"]]
            if len(qs) != len(set(qs)):
                problems.append("%s: duplicate question text" % d["doc_id"])
    qsets = {n: {q["question"] for d in docs for q in d["questions"]} for n, docs in sets_a.items()}
    for a in qsets:
        for b in qsets:
            if a < b and qsets[a] & qsets[b]:
                problems.append("%s and %s share question text" % (a, b))
    qids = [q["qid"] for docs in sets_a.values() for d in docs for q in d["questions"]]
    if len(qids) != len(set(qids)):
        problems.append("duplicate question id")

    # files on disk, if present, must match a fresh generation
    mpath = os.path.join(OUT_DIR, "MANIFEST.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
        if manifest.get("cpt") != cpt:
            problems.append("tasks/ on disk were generated with cpt=%s; rerun --check with --cpt %s" % (
                manifest.get("cpt"), manifest.get("cpt")))
        for name, docs in sets_a.items():
            want = hashlib.sha256(file_payload(docs).encode()).hexdigest()
            path = os.path.join(OUT_DIR, "%s.jsonl" % name)
            have = hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None
            if have != want or manifest["files"].get("%s.jsonl" % name, {}).get("sha256") != want:
                problems.append("tasks/%s.jsonl on disk differs from a fresh generation" % name)
        summary.append("files on disk match a fresh generation" if not any("on disk" in p for p in problems)
                       else "files on disk checked")
    else:
        summary.append("no tasks/ on disk yet (run without --check to write them)")

    for line in summary:
        print(line)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        print("FAILED (%d problems)" % len(problems))
        return 1
    print("OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cpt", type=float, default=CHARS_PER_TOKEN, help="chars per token estimate")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.cpt))
    sets, _ = build_all(args.cpt)
    manifest = write_all(sets, args.cpt)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
