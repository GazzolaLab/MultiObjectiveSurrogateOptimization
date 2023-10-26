from machinable import Execution


class DryExecution(Execution):
    def __call__(self):
        for executable in self.pending_executables:
            print(f"Dry running {executable} ({executable.local_directory()})")
