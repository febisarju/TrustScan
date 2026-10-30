"""Text normalization, scam-cue extraction, structural features and grouping."""
import re, hashlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_PLACE = re.compile(r"#(URL|EMAIL|PHONE)_[0-9a-f]+#")
_TAG = re.compile(r"<[^>]+>")

def normalise(s):
    s = _TAG.sub(" ", s)
    s = _PLACE.sub(lambda m: f" {m.group(1).lower()}tok ", s)
    return re.sub(r"\s+", " ", s).strip()

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

LEX = { 
 "cue_urgency": r"\b(urgent(ly)?|immediate(ly)?|asap|hurry|limited (seats|slots|positions|openings)|apply (now|today)|start (today|now|immediately))\b",
 "cue_payment": r"\b(fees?|deposit|registration|western union|money ?gram|wire transfer|bank (account|details)|processing|advance payment|invest(ment)?|refundable)\b",
 "cue_easy_money": r"(earn up to|\$ ?\d[\d,]*\s*(per|/|a)\s*(hour|hr|week|day|month)|weekly pay|high (income|salary|pay)|unlimited (income|earning)|no (prior |previous )?experience|easy (money|job|work)|guaranteed|get paid)",
 "cue_personal": r"\b(ssn|social security|passport|id (card|proof)|driver.?s licen[cs]e|credit (card|report|check)|date of birth|aadhaar|pan card)\b",
 "cue_remote": r"\b(work (from|at) home|home[- ]based|work remotely|data entry|typist|typing|online (job|work))\b",
 "cue_vague": r"\b(our client|reputed|leading (company|firm|organi[sz]ation)|well[- ]known|multinational|confidential)\b",
 "cue_contact": r"\b(contact (us|me|hr)|whatsapp|telegram|text us|call (us|now)|reply to this|send (your )?(resume|cv)|email (your|us))\b",
}

def cue_features(texts):
    s = pd.Series(list(texts), dtype=object).fillna("")
    out = pd.DataFrame(index=s.index)
    for k, pat in LEX.items():
        out[k] = s.str.count(pat, flags=re.I)
    for t in ("urltok", "emailtok", "phonetok"):
        out["n_" + t] = s.str.count(rf"\b{t}\b")
    n_chars = s.str.len().clip(lower=1)
    letters = s.str.count(r"[A-Za-z]").clip(lower=1)
    out["caps_ratio"] = s.str.count(r"[A-Z]") / letters
    out["excl"] = s.str.count("!")
    out["dollar"] = s.str.count(r"\$")
    out["digit_ratio"] = s.str.count(r"\d") / n_chars
    words = s.str.lower().str.findall(r"[a-z']+")
    nw = words.map(len).clip(lower=1)
    out["log_words"] = np.log1p(nw)
    out["ttr"] = words.map(lambda w: len(set(w)) / max(len(w), 1))
    out["avg_word_len"] = n_chars / nw
    return out

class D:
    def __init__(self, text, cue, meta):
        self.text = np.asarray(text, dtype=object)
        self.cue = cue.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True)
    def __len__(self): return len(self.text)
    def take(self, idx): return D(self.text[idx], self.cue.iloc[idx], self.meta.iloc[idx])
    def dense(self, cue=True, meta=True):
        parts = ([self.meta] if meta else []) + ([self.cue] if cue else [])
        return pd.concat(parts, axis=1)
    def mask(self, words):
        if not words: return self
        rx = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
        t = np.array([rx.sub(" ", x) for x in self.text], dtype=object)
        return D(t, cue_features(t), self.meta)

class FeatureBuilder:
    OPT = ["department", "salary_range", "company_profile", "requirements", "benefits",
           "employment_type", "required_experience", "required_education", "industry", "function"]
    def fit(self, df):
        self.cats = {}
        for c, top in [("employment_type", None), ("required_experience", None),
                       ("required_education", None), ("function", 15)]:
            vc = df[c].fillna("NA").value_counts()
            self.cats[c] = list(vc.index[:top] if top else vc.index)
        self.cats["country"] = list(self._country(df).value_counts().index[:15])
        return self
    @staticmethod
    def _country(df): return df["location"].fillna("NA").str.split(",").str[0].str.strip()
    def text(self, df):
        parts = [df[c].fillna("") for c in TEXT_FIELDS]
        return np.array([normalise(" . ".join(x)) for x in zip(*parts)], dtype=object)
    def meta(self, df):
        m = pd.DataFrame(index=df.index)
        for c in ("telecommuting", "has_company_logo", "has_questions"): m[c] = df[c].astype(int)
        for c in self.OPT: m["miss_" + c] = df[c].isna().astype(int)
        m["completeness"] = 1 - m[[c for c in m.columns if c.startswith("miss_")]].mean(axis=1)
        for c in ("title", "company_profile", "description", "requirements", "benefits"):
            m["len_" + c] = np.log1p(df[c].fillna("").str.len())
        m["title_upper"] = df["title"].fillna("").str.count(r"[A-Z]") / df["title"].fillna("").str.len().clip(lower=1)
        for c in ("employment_type", "required_experience", "required_education", "function"):
            v = df[c].fillna("NA")
            for cat in self.cats[c]: m[f"{c}={cat}"] = (v == cat).astype(int)
        ctry = self._country(df)
        for cat in self.cats["country"]: m[f"country={cat}"] = (ctry == cat).astype(int)
        m["country=other"] = (~ctry.isin(self.cats["country"])).astype(int)
        m.columns = [re.sub(r"[\[\]<>]", "", c) for c in m.columns]
        return m
    def transform(self, df):
        t = self.text(df)
        return D(t, cue_features(t), self.meta(df))

def make_groups(df):
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    kt = (df.title.fillna("") + df.description.fillna("") + df.requirements.fillna("")).map(norm)
    kp = df.company_profile.fillna("").map(norm)
    n = len(df); rows, cols = [], []
    for off, keys in ((n, kt), (n + kt.nunique() + 1, kp)):
        codes, _ = pd.factorize(keys.map(lambda x: hashlib.md5(x.encode()).hexdigest()))
        valid = (keys != "").values
        rows += list(np.arange(n)[valid]); cols += list(off + codes[valid])
    N = n + kt.nunique() + kp.nunique() + 2
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    return pd.factorize(lab[:n])[0]