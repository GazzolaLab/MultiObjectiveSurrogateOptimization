from machinable import Execution


class Requires(Execution):
    def on_before_dispatch(self):
        raise RuntimeError(
            "Execution is required:\n- "
            + "\n- ".join(
                self.pending_executables.map(lambda x: x.module + " <" + x.id + ">")
            )
        )
