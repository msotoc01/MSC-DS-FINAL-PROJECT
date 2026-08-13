import sys; from pathlib import Path
R=Path(".").resolve(); sys.path.insert(0,str(R)); sys.path.insert(0,str(R/"src"))

from dataset.loaders import load_evaluation_tasks
from scheduling import milp
from evaluation import metrics
import importlib.util
spec = importlib.util.spec_from_file_location("s2","scripts/02_run_pipeline_sample.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

batch = m.build_batch(load_evaluation_tasks(True), window_days=0.25)
sched, info = milp.solve(list(batch))
print("objetivo que dice el solver:", info["objective"])
print("makespan real:", metrics.makespan(sched))
print("switches real:", metrics.context_switches(sched))
print("lateness real:", metrics.total_lateness(sched))
print("objetivo recalculado:", metrics.makespan(sched) + 5*metrics.context_switches(sched) + metrics.total_lateness(sched))