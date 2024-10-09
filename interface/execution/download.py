from machinable import Execution, Storage

from rich.console import Console
from rich.table import Table


class StorageDownload(Execution):
    class Config:
        related: int = 0

    def commit(self):
        return self

    def dispatch(self):
        self.__call__()
        return self

    def __call__(self) -> None:
        storage = Storage.get()

        table = Table(title=f"Retrieving from {storage}")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="blue")
        table.add_column("Label")
        table.add_column("Directory")

        def pb(b):
            return " ⬇️ " if b else "❔"

        retrievals = []

        for executable in self.executables:
            status = ""
            cached = executable.cached()

            found = []
            if cached:
                status = "DONE"
            else:
                found = storage.search_for(executable)
                if not found:
                    status = "NOT FOUND"
                else:
                    status = f"FOUND ({len(found)}) -> {found[0]}"

                    retrievals.append(found[0])

            table.add_row(
                pb(len(found) > 0) + " " + repr(executable),
                status,
                (
                    executable.label()
                    if hasattr(executable, "label")
                    else repr(executable)
                ),
                executable.local_directory(),
            )

        console = Console()
        console.print(table)

        if len(retrievals) == 0:
            print("Nothing to do ...")
            return

        print(f"{len(retrievals)} retrievals. Proceed?")

        choice = input().lower()
        if not {"": True, "yes": True, "y": True, "no": False, "n": False}[choice]:
            print("No ...")
            return

        for r in retrievals:
            storage.download(r, related=self.config.related)
