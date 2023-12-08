import os
import shutil
import stat
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
from dataclasses import dataclass
from typing import Optional

from machinable import Execution
from interface.execution.slurm import confirm


def run_and_stream(
    args,
    *,
    stdout_handler=print,
    stderr_handler=print,
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    **kwargs,
) -> int:
    with subprocess.Popen(
        args, text=text, stdout=stdout, stderr=stderr, **kwargs
    ) as process:
        with ThreadPoolExecutor(2) as pool:

            def _st(stream, handler):
                deque((handler(line) for line in stream), maxlen=0)

            pool.submit(_st, process.stdout, stdout_handler)
            pool.submit(_st, process.stderr, stderr_handler)
        retcode = process.wait()
        if check and retcode:
            raise subprocess.CalledProcessError(retcode, process.args)
    return retcode


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
                script_file = self.save_file(
                    f"mpi-{executable.id}.sh",
                    executable.dispatch_code(),
                )
                st = os.stat(script_file)
                os.chmod(script_file, st.st_mode | stat.S_IEXEC)
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
                        env=os.environ,  # see https://stackoverflow.com/a/60070753
                    )
