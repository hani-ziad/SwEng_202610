"""Parse the CCECE paper's tex into a JSON list of blocks for build_docx.js."""
import json, re, sys

t = open(sys.argv[1], encoding="utf-8").read()
body = t.split("\\maketitle", 1)[1].split("\\end{document}")[0]

keys = re.findall(r"\\bibitem\{(r\d+)\}", t)
CITE = {k: i + 1 for i, k in enumerate(keys)}
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
LETTERS = "ABCDEFGHIJ"

# label numbering, in source order
REF = {}
tabs = figs = 0
sec = sub = 0
for m in re.finditer(r"\\section\*?\{|\\subsection\{|\\begin\{table\*?\}|\\begin\{figure\}|\\label\{([^}]+)\}", body):
    s = m.group(0)
    if s.startswith("\\section{"):
        sec += 1; sub = 0; cur = ROMAN[sec - 1]
    elif s.startswith("\\subsection"):
        sub += 1; cur = f"{ROMAN[sec - 1]}-{LETTERS[sub - 1]}"
    elif s.startswith("\\begin{table"):
        tabs += 1; cur = ROMAN[tabs - 1]
    elif s.startswith("\\begin{figure"):
        figs += 1; cur = str(figs)
    elif m.group(1):
        REF[m.group(1)] = cur

def inline(s):
    """Return list of runs [{t, b?, i?, sup?}] from a tex fragment."""
    s = s.strip()
    s = re.sub(r"\\cite\{([^}]+)\}", lambda m: ", ".join(f"[{CITE[k.strip()]}]" for k in m.group(1).split(",")), s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: REF[m.group(1)], s)
    s = s.replace("Table~", "Table ").replace("Fig.~", "Fig. ").replace("Section~", "Section ").replace("~", " ")
    s = re.sub(r"\\url\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\v\{Z\}", "Ž", s); s = re.sub(r"\\\.\{e\}", "ė", s)
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&").replace("\\ ", " ").replace("vs.\\", "vs.")
    s = s.replace("abs.\\", "abs.")
    # math
    def math(m):
        x = m.group(1)
        x = (x.replace("\\geq", "\u2265").replace("\\pm", "\u00b1").replace("\\times", "\u00d7").replace("\\approx", "\u2248")
               .replace("-", "\u2212").replace("+", "+"))
        return "\x01" + x + "\x02"
    s = re.sub(r"\$([^$]+)\$", math, s)
    runs = []
    pos = 0
    for m in re.finditer(r"\\textbf\{([^}]*)\}|\\emph\{([^}]*)\}|\x01([^\x02]*)\x02", s):
        if m.start() > pos:
            runs.append({"t": s[pos:m.start()]})
        if m.group(1) is not None:
            runs.append({"t": m.group(1), "b": True})
        elif m.group(2) is not None:
            runs.append({"t": m.group(2), "i": True})
        else:
            x = m.group(3)
            mm = re.fullmatch(r"(R)\^(\d)", x)
            if mm:
                runs += [{"t": "R", "i": True}, {"t": mm.group(2), "sup": True}]
            elif x in ("n",):
                runs.append({"t": x, "i": True})
            else:
                runs.append({"t": x})
        pos = m.end()
    if pos < len(s):
        runs.append({"t": s[pos:]})
    for r in runs:
        r["t"] = re.sub(r"\s+", " ", r["t"])
        assert "\\" not in r["t"], r["t"]
    return runs

blocks = []
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
blocks.append({"k": "abstract", "runs": inline(abstract)})
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", body, re.S).group(1)
blocks.append({"k": "keywords", "runs": inline(kw)})
rest = body.split("\\end{IEEEkeywords}", 1)[1]
main, bib = rest.split("\\begin{thebibliography}", 1)

def table(env):
    cap = re.search(r"\\caption\{(.*?)\}\n\\label", env, re.S).group(1)
    lab = re.search(r"\\label\{([^}]+)\}", env).group(1)
    tab = re.search(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", env, re.S)
    rows = [r.strip() for r in tab.group(2).replace("\\hline", "").split("\\\\") if r.strip()]
    cells = [[inline(c) for c in r.split("&")] for r in rows]
    return {"k": "table", "wide": env.startswith("\\begin{table*}"), "num": REF[lab], "cap": inline(cap),
            "rows": cells, "align": tab.group(1)}

for part in re.split(r"(\\begin\{table\*?\}.*?\\end\{table\*?\}|\\begin\{figure\}.*?\\end\{figure\}|\\section\{[^}]*\}|\\subsection\{[^}]*\}|\\label\{[^}]*\})", main, flags=re.S):
    if not part or not part.strip():
        continue
    p = part.strip()
    if p.startswith("\\begin{table"):
        blocks.append(table(p))
    elif p.startswith("\\begin{figure"):
        img = re.search(r"includegraphics\[width=([\d.]*)\\columnwidth\]\{([^}]+)\}", p)
        cap = re.search(r"\\caption\{(.*?)\}\n\\label", p, re.S).group(1)
        lab = re.search(r"\\label\{([^}]+)\}", p).group(1)
        blocks.append({"k": "figure", "file": img.group(2), "width": float(img.group(1) or 1), "num": REF[lab], "cap": inline(cap)})
    elif p.startswith("\\section{"):
        blocks.append({"k": "h1", "text": p[9:-1]})
    elif p.startswith("\\subsection{"):
        blocks.append({"k": "h2", "text": p[12:-1]})
    elif p.startswith("\\label"):
        continue
    else:
        for para in re.split(r"\n\s*\n", p):
            para = para.strip().replace("\\balance", "").strip()
            if para:
                blocks.append({"k": "p", "runs": inline(para)})

blocks.append({"k": "h1", "text": "References", "nonum": True})
for m in re.finditer(r"\\bibitem\{(r\d+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", bib, re.S):
    blocks.append({"k": "ref", "num": CITE[m.group(1)], "runs": inline(m.group(2))})

TITLE = re.search(r"\\title\{(.*?)\}\n", t).group(1)
json.dump({"title": TITLE, "refs": REF, "blocks": blocks}, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(REF)
print(sum(1 for b in blocks if b["k"] == "p"), "paragraphs;", sum(1 for b in blocks if b["k"] == "table"), "tables")
