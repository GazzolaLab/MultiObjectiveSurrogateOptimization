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
        table.add_column("Status", style="blue")
        table.add_column("Job ID")
        table.add_column("Logs")

        def pb(b):
            return "✅" if b else "❌"

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

            table.add_row(
                pb(executable.cached()) + " " + repr(executable),
                status,
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
