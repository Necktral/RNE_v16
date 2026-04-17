import os
import threading
import time
import logging


import os
import threading
import time
import logging
import sqlite3

class SQLiteRepo:
    def __init__(self, db_path, test_mode=False):
        self.test_mode = test_mode or os.getenv("AEON_TEST_MODE") == "1"
        if self.test_mode:
            self.db_path = ":memory:"
        else:
            self.db_path = db_path
        self._conn = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger("aeon.sqlite_repo")
        self._connect()

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def insert_mutation(self, mutation):
        with self._lock:
            try:
                if self._conn is None:
                    self._connect()
                self._conn.execute(
                    """CREATE TABLE IF NOT EXISTS mutations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        details TEXT
                    )""")
                self._conn.execute(
                    "INSERT INTO mutations (timestamp, details) VALUES (?, ?)",
                    (mutation.get('timestamp', time.time()), str(mutation))
                )
                self._conn.commit()
            except Exception as e:
                self._logger.warning(f"Error insert_mutation: {e}")

    def close(self):
        if self._conn:
            self._conn.close()
