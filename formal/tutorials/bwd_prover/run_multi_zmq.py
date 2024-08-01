# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import zmq
import socket
import os

from params import ConfStore
from evariste.backward.prover.zmq_prover import ZMQProverParams, zmq_prover, Worker
from evariste.backward.prover.prover_args import ProverParams
from evariste.model.transformer_args import DecodingParams
from zmq import ZMQError
from pathlib import Path
import random
from evariste.logger import create_logger
import time

logger = create_logger(None)

import multiprocessing as mp

if __name__ == "__main__":

    # This is used to communicate work units to the provers, and receive results.
    context = zmq.Context()
    sock = context.socket(zmq.ROUTER)
    port = sock.bind_to_random_port(f"tcp://*")
    hostname = socket.gethostname()
    controller_addr = f"tcp://{hostname}:{port}"

    dataset = ConfStore["new3_100"]
    mcts_params = ConfStore["mcts_fast"]
    decoding_params: DecodingParams = ConfStore["decoding_fast"]
    zmq_params = ZMQProverParams(
        ProverParams(n_simultaneous_proofs=5, beam_search_path=None, mcts=mcts_params),
        decoding=decoding_params,
        dataset=dataset,
        # The ZMQ prover expects to be called with a dump path that has been set-up for use.
        # It should contain a file 'checkpoint.-1.pth' which is the model to be loaded.
        dump_path=f"{os.environ.get('USER')}/dump/mm",
    )

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=zmq_prover, args=(zmq_params, controller_addr,))
    p.start()

    from evariste import json as json

    workers = {}
    labels = []
    with open(Path(dataset.data_dir) / "split.valid", "r") as f:
        for label in f:
            labels.append({"label": label.strip()})

    random.seed(43)
    random.shuffle(labels)
    to_send = 25
    labels = labels[:to_send]
    proved, total = 0, 0
    # Iterate while there is work to be done. This is the work done in the Proverhandler class
    completely_done = False
    while not completely_done:
        try:
            worker_id, result_json = sock.recv_multipart(zmq.NOBLOCK)
            result = json.loads(result_json)
            # Handle worker connection. Initialize a new Worker object
            if result["type"] == "connect":
                workers[worker_id] = Worker(worker_id, result["n_provers"])
                print(worker_id, "connected")
            # If we received a result, log it and allow the corresponding worker to do more work
            elif result["type"] == "result":
                print(f"%%%%%%%% {worker_id} finished a job!")
                if worker_id in workers:
                    workers[worker_id].finish_goal(result["label"])
                total += 1
                if result["success"]:
                    proved += 1
                print(proved / total, total)
                if total == len(to_send):
                    completely_done = True  # we've received everything let's exit.
        except ZMQError:
            # no message -> busy wait
            time.sleep(1)
            pass

        # Once incoming messages have been dealt with, we can distribute work to the workers
        for worker_id, worker in workers.items():
            while worker.can_receive():
                # If there is work to do
                if labels:
                    next_job = labels.pop()
                    sock.send_multipart(
                        [worker_id, json.dumps(next_job).encode("utf-8")]
                    )
                    print(f"-------> SENT JOB {next_job}")
                # Otherwise, if this worker hasn't been stopped, tell it to stop
                elif not worker.stop_sent:
                    next_job = "stop"
                    sock.send_multipart(
                        [worker_id, json.dumps(next_job).encode("utf-8")]
                    )

                if next_job != "stop":
                    worker.add_goal(next_job)
                else:
                    worker.stop_sent = True
                    print(f"Stopping worker {worker_id}")
                    break

    print("Received everything, waiting for zmq prover to join")
    p.join()  # This will join once the zmq_prover has actually processed the "stop" command
