"""Hard delete pipeline. Every step is idempotent, so the job can be retried after a partial failure."""

import logging
import shutil
from pathlib import Path

from sqlalchemy import text

from ..db import engine

log = logging.getLogger("deletion")
BATCH = 1000


def hard_delete_document(document_id: str) -> None:
    with engine.begin() as conn:
        doc = conn.execute(
            text(
                "SELECT id, storage_path, deleted_at FROM documents WHERE id=:id FOR UPDATE"
            ),
            {"id": document_id},
        ).first()
        if doc is None:
            log.info("hard delete %s: already gone", document_id)
            return
        if doc.deleted_at is None:
            log.info("hard delete %s: document was restored, skipping", document_id)
            return
        conn.execute(
            text(
                "UPDATE documents SET status='deleting', updated_at=now() WHERE id=:id"
            ),
            {"id": document_id},
        )

    # Step 1: chunks + embeddings, in batches to keep transactions and WAL small
    total = 0
    while True:
        with engine.begin() as conn:
            n = conn.execute(
                text(
                    "DELETE FROM chunks WHERE id IN (SELECT id FROM chunks WHERE document_id=:id LIMIT :n)"
                ),
                {"id": document_id, "n": BATCH},
            ).rowcount
        total += n
        if n == 0:
            break

    # Step 2: stored file (missing file is fine, we are deleting anyway)
    shutil.rmtree(Path(doc.storage_path).parent, ignore_errors=True)

    # Step 3: metadata row last, so a failure above leaves a visible 'deleting' record to retry
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM documents WHERE id=:id AND deleted_at IS NOT NULL"),
            {"id": document_id},
        )
    log.info("hard delete %s complete (%d chunks removed)", document_id, total)
