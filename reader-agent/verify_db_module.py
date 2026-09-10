import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))
import db
import inspect

print("db module loaded from:", db.__file__)
print()
print("=== Actual live source of save_candidate() ===")
print(inspect.getsource(db.save_candidate))