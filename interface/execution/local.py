import os
import shutil

import sys
from dataclasses import dataclass
from typing import Optional

from machinable import Execution
from machinable.utils import run_and_stream, chmodx
from interface.execution.slurm import confirm


class LocalExecution(Execution):
    @dataclass
    class Config:
        confirm: bool = False
        mpi: Optional[str] = "mpirun"
        ranks: Optional[int] = None
        nodes: Optional[int] = None

    def on_before_dispatch(self):
        if self.config.confirm:
            return confirm(self)

    def __call__(self):
        for executable in self.pending_executables:
            # automatically infer the ranks and nodes from the executable
            # (if the executable does not expose `ranks`, `nodes` will be ignored)
            if (ranks := self.config.ranks) == -1:
                ranks = executable.config.get("ranks", False)
            if (nodes := self.config.nodes) == -1:
                nodes = executable.config.get("nodes", None)
            if self.config.mpi is None or ranks is False:
                # single-threaded execution
                executable.dispatch()
            else:
                # run using MPI
                script_file = chmodx(
                    self.save_file(
                        f"mpi-{executable.id}.sh",
                        executable.dispatch_code(),
                    )
                )
                cmd = [shutil.which(self.config.mpi)]
                if isinstance(ranks, int):
                    cmd.extend(["-n", str(ranks)])
                if isinstance(nodes, int):
                    cmd.extend(
                        [
                            "-N",
                            str(nodes),
                        ]
                    )
                cmd.append(script_file)
                print(" ".join(cmd))

                with open(executable.local_directory("output.log"), "w") as f:
                    run_and_stream(
                        cmd,
                        stdout_handler=lambda o: [sys.stdout.write(o), f.write(o)],
                        stderr_handler=lambda o: [sys.stderr.write(o), f.write(o)],
                    )
