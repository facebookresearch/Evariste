# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import TypeVar, Generic, Union, cast

from evariste.forward.core.generation_errors import GenerationError

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class GenericMaybe(Generic[T, E]):
    def __init__(self, wrapped: Union[T, E]):
        self.wrapped = wrapped

    def unwrap(self) -> T:
        if not self.ok():
            raise ValueError
        wrapped = cast(T, self.wrapped)
        return wrapped

    def ok(self) -> bool:
        return not isinstance(self.wrapped, Exception)

    def err(self) -> E:
        if self.ok():
            raise ValueError
        wrapped = cast(E, self.wrapped)
        return wrapped


class Maybe(Generic[T], GenericMaybe[T, GenerationError]):
    pass


class Fail(Generic[T], Maybe[T]):
    def __init__(self, fail: GenerationError):
        super(Fail, self).__init__(fail)
        assert not self.ok()


class Ok(Generic[T], Maybe[T]):
    def __init__(self, ok: T):
        super(Ok, self).__init__(ok)
        assert self.ok()
