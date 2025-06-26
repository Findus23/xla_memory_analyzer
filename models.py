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


class Value(BaseModel):
    id: int
    name: str
    at: int
    size: int
    offset: int
    array_info: str
    allocation: Allocation
    value_detailed: Optional[ValueDetailed] = None
    live_range:Optional[tuple[int, int]]=None

    @property
    def is_large_array(self) -> bool:
        return self.size > large_array_threshold

    @property
    def pretty_size(self) -> str:
        return pretty_byte_size(self.size)
