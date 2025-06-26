from functools import cached_property
from typing import Optional

from pydantic import BaseModel

from utils import pretty_byte_size

large_array_threshold = 1024 ** 2  # 1MB


class Allocation(BaseModel):
    alloc_id: int
    total_size: int


class ValueDetailed(BaseModel):
    id: int
    name: str
    at: int
    uses: list[str] = []
    positions: list[str] = []
    instruction: Optional[dict] = None
    opt_name: Optional[str] = None

    @property
    def op_name(self):
        try:
            return self.instruction["metadata"]["op_name"]
        except TypeError:
            return None

    @property
    def source(self):
        try:
            source_file = self.instruction["metadata"]["source_file"]
            source_line = self.instruction["metadata"]["source_line"]
        except (TypeError, KeyError):
            return None
        return source_file, source_line

    @property
    def short_source(self):
        if self.source is None:
            return None
        source_file, source_line = self.source
        filename = source_file.split("/")[-1]
        return f"{filename}:{source_line}"


class Value(BaseModel):
    id: int
    name: str
    at: int
    size: int
    offset: int
    array_info: str
    allocation: Allocation
    value_detailed: Optional[ValueDetailed] = None
    live_range: Optional[tuple[int, int]] = None
    sequence: Optional[int] = None

    @property
    def is_large_array(self) -> bool:
        return self.size > large_array_threshold

    @property
    def pretty_size(self) -> str:
        return pretty_byte_size(self.size)


class ModuleStats(BaseModel):
    value_name_to_id: dict[str, int] = {}
    value_id_to_name: dict[int, str] = {}
    values: dict[int, Value] = {}
    used_values: dict[int, ValueDetailed] = {}
    allocations: dict[int, Allocation] = {}

    @cached_property
    def largest_sequence_value(self):
        l = 0
        for v in self.values.values():
            if v.sequence is None:
                continue
            if v.sequence > l:
                l = v.sequence
        return l
