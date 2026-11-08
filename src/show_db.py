"""Task 5 - show what the database trail contains (works for SQLite and Postgres).
  python src/show_db.py            (reads DATABASE_URL, default sqlite:///visionguard.db)"""
import os
from sqlalchemy import create_engine, inspect, text

url = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
engine = create_engine(url)
print("Database:", engine.url.render_as_string(hide_password=True))
print("\nTable 'predictions' - columns:")
for col in inspect(engine).get_columns("predictions"):
    print(f"  {col['name']:<15} {col['type']}")

with engine.connect() as conn:
    n = conn.execute(text("select count(*) from predictions")).scalar()
    print(f"\n{n} rows saved. Latest 10 (newest first):")
    print(f"  {'id':<4} {'when (UTC)':<20} {'file':<28} {'decision':<13} {'conf':<7} model")
    rows = conn.execute(text("select id, created_at, filename, label, confidence, model_name, model_version "
                             "from predictions order by id desc limit 10"))
    for r in rows:
        when = str(r[1])[:19]
        print(f"  {r[0]:<4} {when:<20} {str(r[2])[:26]:<28} {r[3]:<13} {r[4]:<7.3f} {r[5]} v{r[6]}")
    print("\nDecisions so far (a query over the whole history):")
    for label, count, avg in conn.execute(text(
            "select label, count(*), avg(confidence) from predictions group by label")):
        print(f"  {label:<13} {count} images, average confidence {avg:.3f}")
