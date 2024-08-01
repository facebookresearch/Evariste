# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tornado.web import url, RequestHandler
from pathlib import Path
from typing import Dict
import os
import sys
import traceback
import tornado.web
import tornado.ioloop

import evariste.datasets
from params import ConfStore
from evariste import json as json
from evariste.logger import create_logger
from evariste.model_zoo import ZOO, ZOOModel
from evariste.model.data.dictionary import EOS_WORD
from evariste.backward.env.metamath.env import MMEnvGenerator
from evariste.backward.env.equations.env import EQEnvGenerator
from evariste.backward.model.beam_search import DecodingParams
from evariste.backward.model.beam_search import load_beam_search_from_checkpoint
from evariste.backward.prover.dump import MAX_MCTS_DEPTH
from visualizer.session import Session


logger = create_logger(None)


cur_dir = Path(os.path.dirname(__file__))  # TODO: fix this hack
static_path = cur_dir / "static"
template_path = cur_dir / "templates"


SETTINGS = {
    "static_path": static_path,
    "template_path": template_path,
    "debug": True,
    "gzip": True,
    "autoreload": True,
}


MODELS: Dict[str, ZOOModel] = {
    "hl": ZOOModel("YOUR_PATH", dataset="hl_plus_default", lang="hl"),
    "mm": ZOOModel("YOUR_PATH", dataset="new3", lang="mm"),
    "eq": ZOOModel("YOUR_PATH", dataset="eq_dataset_exp_trigo_hyper", lang="eq"),
    "lean": ZOOModel("YOUR_PATH", dataset="lean_v3", lang="lean"),
}
DATASETS = {"mm": ConfStore["new3"], "eq": ConfStore["eq_dataset_exp_trigo_hyper"]}
DECODING_PARAMS: DecodingParams = ConfStore["decoding_fast"]

CURRENT_SESSIONS: Dict[str, Session] = {}

# session language (Metamath / HOL-Light)
session_type = sys.argv[1]
assert session_type in {"mm", "hl", "eq", "lean"}


def to_escaped_json(x):
    return json.dumps(x).replace("\\", "\\\\")


def get_session(name: str) -> Session:
    if name not in CURRENT_SESSIONS:
        logger.info(f"Creating session {name} ...")
        CURRENT_SESSIONS[name] = Session(
            session_name=name,
            session_type=session_type,
            dataset=DATASETS[session_type],
            model=model,
            env=env,
        )
    logger.info(f"Loading session {name} ...")
    return CURRENT_SESSIONS[name]


class InteractiveHandler(RequestHandler):
    def write_error(self, status_code, **kwargs):
        data = {"code": status_code, "reason": self._reason}
        logger.error(f"Error {status_code} detected. Reason: {self._reason}.")
        if self.settings.get("serve_traceback") and "exc_info" in kwargs:
            error_msg = "".join(traceback.format_exception(*kwargs["exc_info"]))
            data["traceback"] = error_msg
            logger.error(f"Traceback: {error_msg}")
        self.set_status(200)
        self.finish(json.dumps({"server_error": data}))

    def get(self, name: str):
        self.render("main.html", session_name=name)

    def post(self, name: str):
        data = json.loads(self.request.body.decode("utf-8"))
        logger.info(f"Input data: {data}")

        # get session / action
        action = data["action"]
        session = get_session(name)
        logger.info(f"Session {name} - action: {action}")

        if action == "initialize":
            self.initialize_session(session, data)
        elif action == "query_model":
            self.query_model(session, data)
        elif action == "grab_state":
            self.return_session_data(session)
        elif action == "update_policy":
            assert session.mcts_dump is not None
            session.update_mcts_policy(
                policy_type=data["policy_type"],
                exploration=data["exploration"],
                max_depth=data["max_reload_depth"],
            )
            self.return_session_data(session)
        else:
            logger.error(f"Unknown action: {action}")
            self.write("invalid")

    def initialize_session(self, session: Session, data: Dict):
        """
        Initialize a new session, from:
        - a custom statement
        - an existing label
        - a MCTS dump path
        - a proof dump path TODO: implement
        """
        assert data.keys() == {
            "action",
            "conclusion",
            "hyps",
            "label",
            "proof_dump",
            "mcts_dump",
        }, data.keys()
        logger.info(f"Initializing session with {data}")

        # initialize from a proof dump
        if data["proof_dump"] != "":
            session.initialize_from_proof(data["proof_dump"])

        # initialize from a MCTS dump
        elif data["mcts_dump"] != "":
            session.initialize_from_mcts(data["mcts_dump"])

        # initialize from a proof label
        elif data["label"] != "":
            session.initialize_from_label(data["label"])

        # initialize from a custom statement
        elif data["conclusion"] != "":
            session.initialize_from_statement(
                statement=data["conclusion"], hyps=data["hyps"]
            )

        else:
            raise Exception(f"Data is required to initialize the session")

        self.return_session_data(session)

    def query_model(self, session: Session, data: Dict):
        res = session.query_model(data)
        logger.info(f"Response: {res}")
        self.write(json.dumps(res))

    def return_session_data(self, session: Session):

        # get session data
        data = session.get_data()
        res = json.dumps(data)

        # log stats
        logger.info(f"Response: {res}")
        logger.info(f"Response size: {len(res) / 1024 ** 2:.2f}MB")
        if session.mcts_dump is not None:
            logger.info(
                f"MCTS contains {len(session.mcts_dump.nodes)} unique nodes, "
                f"{session.mcts_dump.nodes_in_tree} in the tree. "
                f"MAX_MCTS_DEPTH={MAX_MCTS_DEPTH}"
            )

        self.write(res)


def make_app():
    return tornado.web.Application(
        [url(r"/([a-zA-Z0-9_-]+)", InteractiveHandler)], **SETTINGS,
    )


if __name__ == "__main__":

    print(f"static_path: {static_path}")
    print(f"template_path: {template_path}")
    assert static_path.is_dir()
    assert template_path.is_dir()

    if session_type == "eq":
        DECODING_PARAMS.stop_symbol = EOS_WORD

    # reload model
    assert MODELS[session_type].lang == session_type
    model = load_beam_search_from_checkpoint(
        path=Path(MODELS[session_type].path),
        decoding_params=DECODING_PARAMS,
        device="cuda",
    )

    # build environment generator
    if session_type == "mm":
        env = MMEnvGenerator(DATASETS["mm"])()
    elif session_type == "eq":
        env = EQEnvGenerator(DATASETS["eq"], n_async_envs=0)()
    else:
        raise RuntimeError(f"Environment for {session_type} not supported!")

    # server port
    PORTS = {
        "mm": 9097,
        "hl": 9098,
        "eq": 9099,
        "lean": 9100,
    }
    PORT = PORTS[session_type]

    logger.info(f"Starting {session_type} server on port {PORT} ...")
    app = make_app()
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()
    logger.info("Server started.")
