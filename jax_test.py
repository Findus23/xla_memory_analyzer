import os
from pathlib import Path

import jax

os.environ["XLA_FLAGS"] = "--xla_dump_to=local_test3 --xla_hlo_profile=true --xla_dump_hlo_as_text=true"


def test(input):
    bla = jax.numpy.sin(input)
    test = jax.numpy.cumsum(bla)
    return bla, test


test = jax.jit(test)
input = jax.numpy.zeros((128, 128, 128))

for file in Path("local_test3").glob("*"):
    file.unlink()

comp=test.lower(input).compile()
print(comp)
