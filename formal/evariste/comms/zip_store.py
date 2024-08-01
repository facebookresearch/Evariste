# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Dict, Any
from logging import getLogger
from pathlib import Path
import pickle
import shutil
import zipfile

from evariste import json as json
from evariste.refac.utils import safe_pkl_load


logger = getLogger(__name__)


class ChunkedZipStore:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        assert self.root_path.exists()
        self.cur_chunk_id: int = -1
        self.cur_zip: Optional[ZipStore] = None
        self.cur_id = 0
        self.uuid = self.root_path.name

    def store_in_pickle_zip(self, obj: Any, zip_name: str):
        assert self.cur_zip
        self.cur_zip.store_in_pickle_zip(obj, zip_name=zip_name)

    def start_chunk(self, chunk_id: Optional[int] = None):
        if chunk_id is None:
            chunk_id = self.cur_chunk_id + 1
        self.cur_chunk_id = chunk_id
        assert (
            self.cur_chunk_id not in self.ready_chunks()
        ), f"{self.cur_chunk_id} in {self.cur_chunk_id}"
        tmp_chunk_path = self.root_path / f"tmp.chunk_{self.cur_chunk_id}"
        if tmp_chunk_path.exists():
            logger.info(
                f"[{(self.__class__.__name__, self.uuid)}] "
                f"Removing existing tmp chunk in {tmp_chunk_path}"
            )
            shutil.rmtree(tmp_chunk_path)
        tmp_chunk_path.mkdir()
        logger.info(
            f"[{(self.__class__.__name__, self.uuid)}] "
            f"Starting chunk: {self.cur_chunk_id}"
        )
        self.cur_zip = ZipStore(tmp_chunk_path)

    def finish_chunk(self):
        assert self.cur_zip
        self.cur_zip.close()
        src = self.cur_zip.root_path
        assert src.name.startswith("tmp.")
        dst: Path = src.parent / (src.name[len("tmp.") :])
        assert not dst.exists(), dst
        shutil.move(str(src), str(dst))
        logger.info(
            f"[{(self.__class__.__name__, self.uuid)}] "
            f"Finished chunk {self.cur_chunk_id}"
        )
        self.cur_zip = None

    def ready_chunks(self) -> List[int]:
        return sorted(
            [
                int(p.name[len("chunk_") :])
                for p in self.root_path.iterdir()
                if p.name.startswith("chunk_") and not p.name.endswith(".tmp")
            ]
        )

    def get_chunk(self, chunk_id: int) -> "ZipStore":
        path = self.root_path / f"chunk_{chunk_id}"
        assert path.exists()
        return ZipStore(path)

    def close(self):
        if self.cur_zip:
            self.cur_zip.close()

    def __del__(self):
        self.close()


class ZipStore:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        assert self.root_path.exists()
        self._open_handlers: Dict[str, zipfile.ZipFile] = {}
        self.cur_id = 0

    def clean_existing(self, zip_names: List[str], jsonl_names: List[str]):
        for zip_name in zip_names:
            zip_path = self.get_zip_path(zip_name)
            if zip_path.exists():
                zip_path.unlink()
                logger.info(f"Deleting already existing {zip_path}")
        for jsonl_name in jsonl_names:
            jsonl_path = self.get_jsonl_path(jsonl_name)
            if jsonl_path.exists():
                jsonl_path.unlink()
                logger.info(f"Deleting already existing {jsonl_path}")

    def cached_zip_handler(self, path: Path):
        key = path.name
        if key in self._open_handlers:
            return self._open_handlers[key]
        assert not path.exists(), path
        handler = zipfile.ZipFile(str(path), mode="w")
        self._open_handlers[key] = handler

        return handler

    def get_zip_path(self, name: str) -> Path:
        return self.root_path / f"{name}.zip"

    def get_jsonl_path(self, name: str) -> Path:
        return self.root_path / f"{name}.jsonl"

    def path(self, name) -> Path:
        return self.root_path / name

    def store_in_pickle_zip(self, obj: Any, zip_name: str):
        path = self.get_zip_path(zip_name)
        gen_zh = self.cached_zip_handler(path=path)
        gen_zh.writestr(f"{self.cur_id}.pkl", pickle.dumps(obj))
        self.cur_id += 1

    def read_pickle_zip(self, zip_name: str) -> List[Any]:
        path = self.get_zip_path(zip_name)
        if not path.exists():
            raise FileNotFoundError(path)
        data = []
        with zipfile.ZipFile(str(path), mode="r") as zh:
            for filename in sorted(zh.namelist()):
                with zh.open(filename, mode="r") as fp:
                    obj = safe_pkl_load(fp)
                data.append(obj)
        return data

    def store_in_jsonl(self, obj: Any, filename: str):
        path = self.get_jsonl_path(filename)
        with path.open("a") as fp:
            fp.write(json.dumps(obj) + "\n")

    def read_jsonl(self, filename: str) -> List[Any]:
        path = self.get_jsonl_path(filename)
        with path.open("r") as fp:
            data = [json.loads(line.strip()) for line in fp.readlines()]
        return data

    def close(self):
        for handler in self._open_handlers.values():
            handler.close()
        self._open_handlers = {}

    def __del__(self):
        self.close()
