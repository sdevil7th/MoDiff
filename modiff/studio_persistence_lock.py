"""One process-local lock for Studio workflow and reusable-block mutations.

Composite migration must compare source bytes, create exact backups, and
replace several files without racing the ordinary workflow/User Node APIs.
Keeping the lock in a dependency-free module avoids coupling storage modules
to the migration implementation.
"""

from __future__ import annotations

import threading


STUDIO_PERSISTENCE_LOCK = threading.RLock()
