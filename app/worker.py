"""Background worker: claims jobs from Postgres with FOR UPDATE SKIP LOCKED.

Run several replicas to scale ingestion horizontally.
"""

import logging
import signal
import time

from sqlalchemy import text

from .config import settings
from .db import engine
from .services.deletion import hard_delete_document
from .services.ingest import ingest_document

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("worker")

_running = True


def _stop(*_):
    global _running
    _running = False


CLAIM_SQL = text("""
    UPDATE jobs SET status='running', locked_at=now(), attempts=attempts+1
    WHERE id = (
        SELECT id FROM jobs
        WHERE (status='queued' AND run_after <= now())
           OR (status='running' AND locked_at < now() - (:stale * interval '1 minute'))  -- crashed worker
        ORDER BY run_after
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, document_id, type, attempts, max_attempts
    """)


def claim_job():
    with engine.begin() as conn:
        return conn.execute(CLAIM_SQL, {"stale": settings.stale_job_minutes}).first()


def finish_job(job_id) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status='done', finished_at=now(), locked_at=NULL WHERE id=:id"
            ),
            {"id": job_id},
        )


def fail_job(job, err: Exception) -> None:
    final = job.attempts >= job.max_attempts
    delay = min(300, 5 * 2 ** (job.attempts - 1))
    msg = f"{type(err).__name__}: {err}"[:2000]
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status=:s, last_error=:e, locked_at=NULL, "
                "run_after = now() + (:d * interval '1 second'), "
                "finished_at = CASE WHEN :final THEN now() END WHERE id=:id"
            ),
            {
                "s": "failed" if final else "queued",
                "e": msg,
                "d": delay,
                "final": final,
                "id": job.id,
            },
        )
        if final and job.type == "ingest":
            conn.execute(
                text(
                    "UPDATE documents SET status='failed', error=:e, updated_at=now() WHERE id=:d"
                ),
                {"e": msg, "d": job.document_id},
            )
    log.error(
        "job %s (%s) failed attempt %d/%d: %s",
        job.id,
        job.type,
        job.attempts,
        job.max_attempts,
        msg,
    )


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log.info("worker started")
    while _running:
        job = claim_job()
        if job is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        log.info(
            "claimed job %s type=%s doc=%s attempt=%d",
            job.id,
            job.type,
            job.document_id,
            job.attempts,
        )
        try:
            if job.type == "ingest":
                ingest_document(str(job.document_id))
            elif job.type == "hard_delete":
                hard_delete_document(str(job.document_id))
            finish_job(job.id)
        except Exception as e:
            fail_job(job, e)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
