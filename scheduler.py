"""
Фоновый планировщик:
   периодически запускает парсер и переиндексацию как subprocess
   после завершения уведомляет о необходимости перезагрузки индекса
   работает в отдельном процессе — не грузит память сервера
"""

import logging
import subprocess
import sys

from apscheduler.schedulers.background import BackgroundScheduler

from config import BASE_DIR, COLLECTION_INTERVAL_HOURS, RUN_PIPELINE_ON_START

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler")

scheduler = BackgroundScheduler()
_reload_callback = None


def _run_script(script: str, timeout: int) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.error(
                "%s failed (code %s): %s",
                script,
                result.returncode,
                (result.stderr or "")[-1000:],
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("%s timeout (%ss)", script, timeout)
    except Exception:
        logger.exception("%s error", script)
    return False


def _run_pipeline_subprocess():
    if not _run_script("parser.py", timeout=3600):
        return
    if not _run_script("create_index.py", timeout=7200):
        return

    if _reload_callback:
        try:
            _reload_callback()
            logger.info("Index reloaded in memory")
        except Exception:
            logger.exception("Reload error")


def start_scheduler(reload_callback=None):
    global _reload_callback
    _reload_callback = reload_callback

    if RUN_PIPELINE_ON_START:
        scheduler.add_job(
            _run_pipeline_subprocess,
            trigger="date",
            id="collect_and_index_startup",
            replace_existing=True,
        )

    scheduler.add_job(
        _run_pipeline_subprocess,
        trigger="interval",
        hours=COLLECTION_INTERVAL_HOURS,
        id="collect_and_index",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("Scheduler started: every %sh", COLLECTION_INTERVAL_HOURS)


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    _run_pipeline_subprocess()
