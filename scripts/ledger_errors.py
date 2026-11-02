"""Print the status and error tail of every run in a session.

    conda run -n minute-agent python scripts/ledger_errors.py <session> [ledger.sqlite]
"""
import sqlite3
import sys

session = sys.argv[1]
db = sys.argv[2] if len(sys.argv) > 2 else "runs/ledger.sqlite"
c = sqlite3.connect(db)
for r in c.execute("SELECT run_id, status, eval_split, round(t_total, 0), round(objective, 3), substr(error, -2000) "
                   "FROM runs WHERE session=? ORDER BY created_at", (session,)):
    print(r[:5])
    if r[5]:
        print(r[5])
