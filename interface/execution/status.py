from machinable import Execution

from rich.console import Console
from rich.table import Table


class Status(Execution):
    def commit(self):
        return self

    def dispatch(self):
        self.__call__()
        return self

    def __call__(self) -> None:
        c = len(self.executables)
        cp = len(self.pending_executables)
        table = Table(title=f"Status summary ({c - cp} out of {c} cached)")

        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Execution", style="blue")
        table.add_column("Label")
        table.add_column("Job ID")
        table.add_column("Logs")

        for executable in self.executables:
            execution = executable.execution
            if execution.is_finished():
                status = "FINISHED"
            elif execution.is_active():
                status = "ACTIVE"
            elif execution.is_incomplete():
                status = "INCOMPLETE"
            elif execution.is_committed():
                status = "COMMITTED"
            else:
                status = ""
            label = ""
            if hasattr(executable, "label"):
                label = executable.label()
            label += " " + executable.load_attribute("label", "")
            # if hasattr(executable, 'load_h5'):
            #     try:
            #         label = str(len(executable.load_h5()['epochs']))
            #         label += " / "
            #         label += str(executable.num_evals_total)
            #     except:
            #         pass
            state = "🆕" if not executable.is_committed() else "❌"
            if executable.cached():
                state = "✅"
            table.add_row(
                state + " " + repr(executable).replace("interface.", ""),
                status,
                label,
                str(
                    executable.execution.load_file(
                        [executable.id, "slurm.json"], {}
                    ).get("job_id", "")
                ),
                executable.local_directory()
                + ", "
                + executable.execution.output_filepath(),
            )

        console = Console()
        console.print(table)
