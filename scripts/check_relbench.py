from relbench.datasets import get_dataset
from relbench.tasks import get_task

ds = get_dataset("rel-hm", download=True)
db = ds.get_db()
task = get_task("rel-hm", "user-churn", download=True)

# Check table attributes
split = task.get_table("train")
print("max_timestamp:", repr(split.max_timestamp))
print("type max_timestamp:", type(split.max_timestamp))
print("target_col:", task.target_col)

# Check event table
evt = db.table_dict["transactions"]
print("\nevent table type:", type(evt))
print("event df columns:", evt.df.columns.tolist()[:15])
print("event df dtypes:\n", evt.df.dtypes[:10])
