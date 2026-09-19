import reader_agent
import inspect

print("Module loaded from:", reader_agent.__file__)
print()
print("=== Actual live source of process_cv() ===")
print(inspect.getsource(reader_agent.process_cv))