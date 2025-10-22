from models.utils import SwapQueue
from models.model import Model
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python worker.py <queue_directory>")
    sys.exit(1)

queue = sys.argv[1]

q = SwapQueue(queue)

i = 0

while True:
    print(f"Awaiting task {i}")
    task = q.receive()  # await processing

    print(task)

    model = Model.unserialize(task["backbone"])

    x, y, yC, epochs = task["x"], task["y"], task["yC"], task["epochs"]

    if epochs == "request":
        scores = model.autoepoch(x, y, yC, verbose=0)
    else:
        model.autofit(x, y, yC, verbose=1, epochs=epochs)
        scores = model.autoeval(x, y, yC)

        weights_filename = "current.weights.h5"

        if q.is_remote:
            local_weights_path = os.path.join(q.local_temp_dir, weights_filename)
            model.save_weights(local_weights_path)

            q._ensure_remote_dir()
            remote_weights_path = os.path.join(q.remote_path, weights_filename).replace(
                "\\",
                "/",
            )
            q._scp_to_remote(local_weights_path, remote_weights_path)
            os.remove(local_weights_path)
        else:
            model.save_weights(os.path.join(queue, weights_filename))

    print(f"Tasks {i} finished, sending data")

    q.receive(scores)  # done

    i += 1
