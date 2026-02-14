"""HTTP client to TRELLIS 3D model generation server.

Adapted from SAGE server/objects/object_generation.py (TrellisClient).
Uses a two-phase protocol: submit job, then poll for completion.
"""

import os
import sys
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger("scene-gen.trellis")


class TrellisClient:
    """Client for the TRELLIS text-to-3D generation server."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or os.environ.get("TRELLIS_URL", "http://localhost:8080")

    def health_check(self) -> Optional[dict]:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            return resp.json()
        except Exception as e:
            logger.warning("TRELLIS health check failed: %s", e)
            return None

    def generate_model(
        self,
        input_text: str,
        seed: int = 1,
        output_file: str = "generated_model.glb",
        max_retries: int = 3,
        poll_interval: float = 1.0,
        max_poll_seconds: int = 200,
    ) -> bool:
        """Generate a 3D model (.glb) from text description.

        Uses a two-phase protocol:
        1. POST /generate → 202 Accepted with job_id
        2. GET /job/{job_id} → poll until 200 (completed) or 500 (failed)

        Returns True if the model was saved successfully.
        """
        for attempt in range(max_retries):
            if attempt > 0:
                logger.info("Retry attempt %d/%d", attempt, max_retries - 1)

            try:
                # Wait for server readiness
                for _ in range(30):
                    if self.health_check():
                        break
                    time.sleep(1)
                else:
                    logger.error("TRELLIS server not ready after 30s")
                    continue

                # Phase 1: submit
                job_id = self._submit_job(input_text, seed)
                if not job_id:
                    continue

                # Phase 2: poll
                if self._poll_and_save(job_id, output_file, poll_interval, max_poll_seconds):
                    return True

            except Exception as e:
                logger.error("Generation attempt failed: %s", e)

        logger.error("Failed after %d attempts", max_retries)
        return False

    def _submit_job(self, input_text: str, seed: int) -> Optional[str]:
        payload = {"input_text": input_text, "seed": seed}
        for trial in range(10):
            try:
                resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=10)
                if resp.status_code == 202:
                    job_id = resp.json().get("job_id")
                    logger.info("Job submitted: %s", job_id)
                    return job_id
                logger.warning("Unexpected status %d: %s", resp.status_code, resp.text)
            except Exception as e:
                logger.warning("Submit error (trial %d): %s", trial, e)
            time.sleep(2)
        return None

    def _poll_and_save(self, job_id: str, output_file: str, interval: float, max_seconds: int) -> bool:
        elapsed = 0.0
        while elapsed < max_seconds:
            try:
                resp = requests.get(f"{self.server_url}/job/{job_id}", timeout=10)
                if resp.status_code == 200:
                    with open(output_file, "wb") as f:
                        f.write(resp.content)
                    logger.info("Model saved to %s", output_file)
                    return True
                elif resp.status_code == 202:
                    pass  # still processing
                elif resp.status_code in (404, 500):
                    logger.error("Job %s failed: %s", job_id, resp.text)
                    return False
            except Exception as e:
                logger.warning("Poll error: %s", e)
            time.sleep(interval)
            elapsed += interval
        logger.error("Poll timeout after %ds", max_seconds)
        return False
