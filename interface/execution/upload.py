from machinable import Execution, Storage

from rich.console import Console
from rich.table import Table


class StorageUpload(Execution):
    class Config:
        cached_only: bool = True
        related: int = 0

    def commit(self):
        return self

    def dispatch(self):
        self.__call__()
        return self

    def __call__(self) -> None:
        c = len(self.executables)
        cp = len(self.pending_executables)

        storage = Storage.get()

        table = Table(title=f"Storing to {storage} ({c - cp} out of {c} cached)")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="blue")
        table.add_column("Staged")
        table.add_column("Local directory")

        def pb(b):
            return "✅" if b else "❌"

        updates = []
        inserts = []

        for executable in self.executables:
            status = ""
            staged = True
            if self.config.cached_only and not executable.cached():
                staged = False
            if executable.is_committed():
                if storage.contains(executable.uuid):
                    status = "STORED"
                    if staged:
                        updates.append(executable)
                else:
                    status = "NOT FOUND"
                    if staged:
                        inserts.append(executable)

            table.add_row(
                pb(executable.cached()) + " " + repr(executable),
                status,
                "Yes" if staged else "No",
                executable.local_directory(),
            )

        console = Console()
        console.print(table)

        if len(updates) + len(inserts) == 0:
            print("Nothing to do ...")
            return

        print(f"{len(updates)} updates and {len(inserts)} inserts. Proceed?")

        choice = input().lower()
        if not {"": True, "yes": True, "y": True, "no": False, "n": False}[choice]:
            print("No ...")
            return

        for i in inserts:
            storage.commit(i)

        for u in updates:
            storage.update(u)

        # TODO: relations
