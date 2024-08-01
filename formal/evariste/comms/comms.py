# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional
from pathlib import Path
import logging

from evariste.comms.zip import get_other_ranks, ZipReceiver
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.comms.store import Receiver, Sender, MultiReceiver, MultiSender
from evariste.comms.rl_distributed_config import RLDistributedConfig


def get_world_size(worker_type: WorkerType, cfg: RLDistributedConfig) -> int:
    try:
        return {
            WorkerType.GENERATOR_ACTOR: cfg.n_generators,
            WorkerType.PROVER_ACTOR: cfg.n_provers,
            WorkerType.PROVER_TRAINER: cfg.n_prover_trainers,
            WorkerType.GENERATOR_TRAINER: cfg.n_generator_trainers,
        }[worker_type]
    except KeyError:
        raise NotImplementedError(worker_type)


def make_sender(
    rank: int,
    sender_type: WorkerType,
    cfg: RLDistributedConfig,
    debug: bool,
    receiver_type: Optional[WorkerType] = None,
) -> Sender:
    """
    If receiver_type is None, receiver_types are deduced from RLDistributedConfig
    """

    receiver_types = cfg.get_receiver_types(sender_type)
    if receiver_type is not None:
        assert receiver_type in receiver_types
        receiver_types = [receiver_type]

    if not cfg.use_zmq and len(receiver_types) > 1:
        logging.info(
            f"Using zip: only creating one unique sender for receivers: "
            f"{receiver_types}"
        )
        receiver_types = receiver_types[:1]
        assert len(receiver_types) == 1

    logging.info(f"Creating sender -- Src: {sender_type} -- Dst: {receiver_types}")
    senders = []
    for receiver_type in receiver_types:
        senders.append(
            _make_sender(
                rank,
                sender_type=sender_type,
                receiver_type=receiver_type,
                cfg=cfg,
                debug=debug,
            )
        )
    return MultiSender(senders=senders, send_to_all=True)


def _make_sender(
    rank: int,
    sender_type: WorkerType,
    receiver_type: WorkerType,
    cfg: RLDistributedConfig,
    debug: bool,
) -> Sender:
    assert cfg.is_rl_distributed_training
    assert cfg.exp_root_path
    comms_dir = Path(cfg.exp_root_path) / "comms"

    sender_ws = get_world_size(sender_type, cfg)
    receiver_ws = get_world_size(receiver_type, cfg)
    assert sender_ws > 0, f"{sender_ws} {sender_type} {cfg}"
    assert receiver_ws > 0, f"{receiver_ws} {receiver_type} {cfg}"

    if cfg.use_zmq:
        from evariste.comms.zmq import ZMQSender

        queue_name = receiver_type
        # 4 senders -> 8 receivers : [0, 1, 2, 3] -> [(0,4), (1,5), (2,6), (3,7)]
        # 8 senders -> 4 receivers : opposite
        dst_ranks = get_other_ranks(
            rank, world_size=sender_ws, other_world_size=receiver_ws
        )
        if len(dst_ranks) == 0:
            logging.warning(
                f"Dummy client for receiver_type {receiver_type}, since"
                f"receiver_ws={receiver_ws}"
            )
        else:
            logging.info(
                f"Creating ZMQ sender for receiver_type: {receiver_type} "
                f"(receiver_ranks: {dst_ranks})"
            )
        senders = []
        for dst_rank in dst_ranks:
            worker_id = 0  # logic not supported yet for multiple worker ids
            socket_file_path = (
                Path(comms_dir)
                / f"{queue_name}_sockets"
                / f"{dst_rank}_{worker_id}.addr"
            )
            partial_sender: ZMQSender = ZMQSender(
                client_id=f"{queue_name}_{rank}",
                socket_file_path=str(socket_file_path),
            )
            logging.info(
                f"Waiting for ZMQ connection to dst at path {str(socket_file_path)}"
            )
            partial_sender.wait_for_store()
            logging.info(
                f"Connected to dst! (rank={rank}) {sender_type} -> {receiver_type}"
            )
            senders.append(partial_sender)
        sender: Sender = MultiSender(senders, send_to_all=False)
    else:
        from evariste.comms.zip import ZipSender

        queue_name = sender_type
        zip_size = cfg.zip_chunk_size if not debug else 16
        logging.info(f"Creating ZipSender with chunk_size: {zip_size}")
        sender = ZipSender.from_dump_path(
            zip_size=zip_size,
            dump_path=comms_dir,
            store_name=queue_name,
            rank=rank,
            worker_id=0,
        )
    return sender


def make_receiver(
    rank: int, receiver_type: WorkerType, cfg: RLDistributedConfig
) -> Receiver:
    """
    If sender_type is None, sender_type are deduced from RLDistributedConfig
    """
    assert cfg.is_rl_distributed_training
    assert cfg.exp_root_path

    sender_types = cfg.get_sender_types(receiver_type)
    assert len(sender_types) == 1
    sender_type = sender_types[0]

    logging.info(
        f"Creating receiver -- Sender: {sender_type} -- Receiver: {receiver_type}"
    )

    comms_dir = Path(cfg.exp_root_path) / "comms"
    sender_ws = get_world_size(sender_type, cfg)
    receiver_ws = get_world_size(receiver_type, cfg)

    if cfg.use_zmq:
        queue_name = receiver_type
        from evariste.comms.zmq import ZMQReceiver

        receiver: Receiver = ZMQReceiver(
            dump_path=comms_dir, global_rank=rank, name=queue_name
        )
    else:
        queue_name = sender_type
        src_ranks = get_other_ranks(
            rank=rank, world_size=receiver_ws, other_world_size=sender_ws
        )

        assert len(src_ranks) >= 1
        receivers = []
        for src_rank in src_ranks:
            rec = ZipReceiver.from_dump_path(
                dump_path=comms_dir,
                rank=rank,
                name=receiver_type,
                worker_id=0,
                src_rank=src_rank,
                src_name=queue_name,
                src_worker_id=0,
                wait_for_store=receiver_type.is_actor(),  # deadlock for trainer?
                # when checkpointed and restarted if not a trainer
            )
            receivers.append(rec)
        receiver = (
            MultiReceiver(receivers=receivers) if len(receivers) > 1 else receivers[0]
        )

    return receiver
