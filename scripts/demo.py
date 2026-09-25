"""End-to-end demo / proof of execution.

Uploads the two task files, waits for async processing, runs semantic queries against
both and checks that the top hit comes from the expected file.

    python scripts/demo.py --with-delete | tee proof/demo_output.txt
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import httpx

PDF_QUERIES = [
    "What is the difference between iPaaS and AI orchestration?",
    "What is Model Context Protocol and why does it matter?",
    "What results did Remote get from its IT helpdesk automation?",
    "What are the three phases for implementing AI orchestration?",
]
CODE_QUERIES = [
    "How does a proxy's score recover over time?",
    "What happens when a proxy reports a failure?",
    "How is a user agent selected for a request?",
    "Where is thread safety handled?",
]


def where(r: dict) -> str:
    m = r["metadata"]
    if "symbol" in m:
        return f"{m['symbol']} (lines {m.get('start_line')}-{m.get('end_line')})"
    return f"page {m.get('page')}" + (f" | {m['heading']}" if m.get("heading") else "")


def show(query: str, data: dict, expect: str | None = None) -> bool:
    print(
        f"\nQ: {query}   [{data['timings_ms']['total']} ms, embed cache hit={data['cache_hit']}]"
    )
    for r in data["results"]:
        snippet = " ".join(r["content"].split())[:110]
        print(
            f"  #{r['rank']} {r['score']:.3f}  {r['filename']}  {where(r)}\n       {snippet}..."
        )
    if expect is None:
        return True
    ok = bool(data["results"]) and data["results"][0]["filename"] == expect
    print(f"  -> top hit from {expect}: {'PASS' if ok else 'FAIL'}")
    return ok


def wait_ready(c: httpx.Client, doc_id: str, timeout: int = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get(f"/documents/{doc_id}").json()
        if d["status"] in ("ready", "failed"):
            return d
        time.sleep(2)
    raise TimeoutError(doc_id)


def upload(c: httpx.Client, path: Path, tags: str) -> dict:
    with path.open("rb") as fh:
        r = c.post("/documents", files={"file": (path.name, fh)}, data={"tags": tags})
    r.raise_for_status()
    body = r.json()
    print(
        f"POST /documents {path.name} -> {r.status_code} {body['status']} duplicate={body['duplicate']}"
    )
    t0 = time.time()
    doc = wait_ready(c, body["document_id"])
    print(
        f"   processed in {time.time() - t0:.1f}s: status={doc['status']} chunks={doc['chunk_count']} model={doc['embedding_model']}"
    )
    if doc["status"] != "ready":
        sys.exit(f"ingestion failed: {doc['error']}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--pdf", default="samples/Knowledge_Base_Sample.pdf")
    ap.add_argument("--code", default="samples/Source_Code_Sample.py")
    ap.add_argument("--api-key", default=None)
    ap.add_argument(
        "--with-delete",
        action="store_true",
        help="also demo soft delete, restore and hard delete",
    )
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    c = httpx.Client(base_url=args.base_url, headers=headers, timeout=60)
    pdf, code = Path(args.pdf), Path(args.code)

    print("=== 1. Ingestion ===")
    pdf_doc = upload(c, pdf, "knowledge-base,pdf-sample")
    code_doc = upload(c, code, "codebase,code-sample")

    results = []
    print("\n=== 2. Semantic queries over the PDF ===")
    for q in PDF_QUERIES:
        results.append(
            show(q, c.post("/query", json={"query": q, "top_k": 3}).json(), pdf.name)
        )
    print("\n=== 3. Semantic queries over the code ===")
    for q in CODE_QUERIES:
        results.append(
            show(q, c.post("/query", json={"query": q, "top_k": 3}).json(), code.name)
        )

    print("\n=== 4. Metadata filtering ===")
    q = "How is a score reduced and recovered?"
    print("\n-- filter: file_types=['code']")
    results.append(
        show(
            q,
            c.post(
                "/query",
                json={"query": q, "top_k": 3, "filters": {"file_types": ["code"]}},
            ).json(),
            code.name,
        )
    )
    print("\n-- filter: tags=['knowledge-base']")
    results.append(
        show(
            "What is AI orchestration?",
            c.post(
                "/query",
                json={
                    "query": "What is AI orchestration?",
                    "top_k": 3,
                    "filters": {"tags": ["knowledge-base"]},
                },
            ).json(),
            pdf.name,
        )
    )
    print("\n-- filter: chunk_metadata={'class_name': 'UAFreshnessRotator'}")
    results.append(
        show(
            q,
            c.post(
                "/query",
                json={
                    "query": q,
                    "top_k": 3,
                    "filters": {"chunk_metadata": {"class_name": "UAFreshnessRotator"}},
                },
            ).json(),
            code.name,
        )
    )
    print("\n-- filter: chunk_metadata={'page': 16}  (the MCP page)")
    results.append(
        show(
            "What is MCP?",
            c.post(
                "/query",
                json={
                    "query": "What is MCP?",
                    "top_k": 2,
                    "filters": {"chunk_metadata": {"page": 16}},
                },
            ).json(),
            pdf.name,
        )
    )

    if args.with_delete:
        print("\n=== 5. Delete lifecycle ===")
        q = CODE_QUERIES[1]
        r = c.delete(f"/documents/{code_doc['id']}", params={"mode": "soft"})
        print(
            f"DELETE /documents/{{code}}?mode=soft -> {r.status_code} purge_scheduled_at={r.json()['purge_scheduled_at']}"
        )
        data = c.post("/query", json={"query": q, "top_k": 3}).json()
        gone = all(x["filename"] != code.name for x in data["results"])
        print(f"soft-deleted doc hidden from search: {'PASS' if gone else 'FAIL'}")
        results.append(gone)
        r = c.post(f"/documents/{code_doc['id']}/restore")
        print(
            f"POST /documents/{{code}}/restore -> {r.status_code} status={r.json()['status']}"
        )
        results.append(
            show(q, c.post("/query", json={"query": q, "top_k": 3}).json(), code.name)
        )

        tmp = Path(tempfile.gettempdir()) / "kb_demo_temp.txt"
        tmp.write_text(
            f"Temporary note {time.time()}: the quarterly zebra migration checklist."
        )
        tdoc = upload(c, tmp, "temp")
        r = c.delete(f"/documents/{tdoc['id']}", params={"mode": "hard"})
        print(f"DELETE /documents/{{temp}}?mode=hard -> {r.status_code}")
        for _ in range(30):
            if c.get(f"/documents/{tdoc['id']}").status_code == 404:
                break
            time.sleep(1)
        purged = c.get(f"/documents/{tdoc['id']}").status_code == 404
        print(
            f"hard delete purged chunks, file and metadata: {'PASS' if purged else 'FAIL'}"
        )
        results.append(purged)

    print(f"\n=== Summary: {sum(results)}/{len(results)} checks passed ===")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
