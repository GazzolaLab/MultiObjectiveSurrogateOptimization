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

    model.autofit(x, y, yC, verbose=1, epochs=501.0)
    scores = model.autoeval(x, y, yC)

    model.save_weights(os.path.join(queue, f"current.weights.h5"))

    print(f"Tasks {i} finished, sending data")

    q.receive(scores)  # done

    i += 1
