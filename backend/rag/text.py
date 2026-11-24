import re
STOP = set("a an the is are was were be been to of in on for and or what which who whom how when where why do does did can could should would "
           "i my me we our you your it its this that these those with by at from as there many much long about".split())
tokens = lambda t: re.findall(r"\w+(?:[.:\-]\w+)*", t.lower())
stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
terms = lambda t: [stem(w) for w in tokens(t) if w not in STOP]
sentences = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
strip_markers = lambda t: re.sub(r"\s*\[\d+\]", "", t)
