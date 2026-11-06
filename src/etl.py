"""ETL: clean the 13 LKY speech texts -> chunked corpus JSON with metadata.

Input:  ~/workspace/aic-section1/lky-corpus-raw/*.txt  (13 files, 01..13)
Output: corpus/corpus.json  (list of chunks)
        corpus/corpus_stats.json
Rules:
- Drop press-release header blocks (MITA / Tel: lines), page furniture,
  and the SARS WHO annex tables (lines that are mostly numbers).
- Keep footnotes and the speeches' own words intact; do NOT rewrite.
- Split into ~500-word chunks on paragraph boundaries, ~80-word overlap.
- Metadata flags: is_translation (#12), is_draft (#06), is_qa (#02 press conf).
"""
import json, re, os, hashlib

RAW = os.path.expanduser("~/workspace/aic-section1/lky-corpus-raw")
OUT = os.path.expanduser("~/workspace/aic-section1/lky-chatbot/corpus")

SPEECHES = [
    ("01_19661227_education.txt", "Opening address, seminar on 'Education and Nation-Building'", "1966-12-27", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19661227.pdf", {}),
    ("02_19650809_separation.txt", "Transcript of press conference at Broadcasting House", "1965-08-09", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19650809b.pdf", {"is_qa": True}),
    ("03_19880414_press.txt", "Address to the American Society of Newspaper Editors", "1988-04-14", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19880414c.pdf", {}),
    ("04_20040623_culture.txt", "Speech at NTU International Conference on National Boundaries and Cultural Configurations", "2004-06-23", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/2004062302.htm", {}),
    ("05_20030218_youth.txt", "Speech to NUS/NTU/SMU students, Nanyang Auditorium", "2003-02-18", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/20030218_0001.pdf", {}),
    ("06_20030425_sars.txt", "Speech at Standard Chartered 150th Anniversary Charity Gala Dinner", "2003-04-25", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/20030425-SM.pdf", {"is_draft": True}),
    ("07_19991228_economy.txt", "Speech at SCCCI Millennium Celebration Dinner", "1999-12-28", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/1999122802/lky19991228c.pdf", {}),
    ("08_20090412_integration.txt", "Speech at SPH Integration Seminar", "2009-04-12", "PMO Singapore", "https://www.pmo.gov.sg/newsroom/speech-mr-lee-kuan-yew-minister-mentor-sph-integration-seminar-12-april-2009-300-pm-sph/", {}),
    ("09_20090813_demography.txt", "Speech at Tanjong Pagar 44th National Day Celebration Dinner", "2009-08-13", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/20090821003.htm", {}),
    ("10_20110513_ntuc.txt", "Speech at NTUC 50th Anniversary Dinner, Marina Bay Sands", "2011-05-13", "NTUC", "https://www.ntuc.org.sg/uportal/news/Speech-by-Minister-Mentor-Lee-Kuan-Yew/", {}),
    ("11_19720609_australia.txt", "Speech at state dinner for Australian PM William McMahon", "1972-06-09", "Australian Government PM Transcripts", "https://pmtranscripts.pmc.gov.au/sites/default/files/original/00002615.pdf", {}),
    ("12_19660825_geylang.txt", "Translation of speech at National Day Celebration Dinner, Geylang Serai", "1966-08-25", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19660825a.pdf", {"is_translation": True}),
    ("13_19680915_ns.txt", "Speech at dinner in honour of National Servicemen, Tanjong Pagar Community Centre", "1968-09-15", "NAS", "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19680915.pdf", {}),
]

def clean(text: str, fname: str) -> str:
    lines = text.split("\n")
    kept, dropped = [], 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # press-release header / page furniture
        if re.match(r"^(Singapore Government Press Release|Media Relations Division|MITA Building|Tel:\s*\d|SPEECH Report)$", s):
            dropped += 1; continue
        if s.startswith("SPEECH BY ") and len(s) > 60 and "LEE KUAN YEW" in s:
            dropped += 1; continue
        # WHO annex tables in the SARS speech: rows that are mostly numbers/pipes
        if re.search(r"\d{2,}", s) and len(re.findall(r"[A-Za-z]", s)) < 8:
            dropped += 1; continue
        kept.append(s)
    # join hyphenated line-breaks left by pdftotext, collapse whitespace
    text = " ".join(kept)
    text = re.sub(r"\s+", " ", text)
    return text, dropped

def chunk_paragraphs(text: str, target=500, overlap=80):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # our cleaned text is single-line; split on sentence boundaries into ~target-word chunks
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + target, len(words))
        # extend to a sentence end within 60 words if possible
        k = j
        while k < min(j + 60, len(words)) and not words[k - 1].endswith((".", "?", "!")):
            k += 1
        j = min(k, len(words))
        chunks.append(" ".join(words[i:j]))
        if j >= len(words):
            break
        i = max(j - overlap, i + 1)
    return chunks

def main():
    os.makedirs(OUT, exist_ok=True)
    all_chunks, stats = [], []
    for fname, title, date, source, url, flags in SPEECHES:
        raw = open(os.path.join(RAW, fname), encoding="utf-8").read()
        cleaned, dropped = clean(raw, fname)
        chunks = chunk_paragraphs(cleaned)
        sid = fname[:2]
        for idx, ch in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{sid}-{idx:02d}",
                "speech_id": sid,
                "title": title, "date": date, "source": source, "url": url,
                "chunk_index": idx, "n_chunks": len(chunks),
                "text": ch, "words": len(ch.split()),
                **flags,
            })
        stats.append({"speech_id": sid, "title": title, "date": date,
                      "raw_words": len(raw.split()), "clean_words": len(cleaned.split()),
                      "dropped_lines": dropped, "n_chunks": len(chunks)})
    with open(os.path.join(OUT, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"total_chunks": len(all_chunks),
                   "total_words": sum(c["words"] for c in all_chunks),
                   "speeches": stats}, f, ensure_ascii=False, indent=1)
    print(f"chunks: {len(all_chunks)}, words: {sum(c['words'] for c in all_chunks)}")
    for s in stats:
        print(f"  {s['speech_id']} {s['date']} raw={s['raw_words']} clean={s['clean_words']} chunks={s['n_chunks']} dropped_lines={s['dropped_lines']}")

if __name__ == "__main__":
    main()
