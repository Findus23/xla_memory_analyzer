from pathlib import Path

import click
from click.shell_completion import CompletionItem


class ByteSizeParamType(click.ParamType):
    name = "bytesize"

    def convert(self, value, param, ctx):
        """
        Parse strings like “4K”, “10M”, “1G” (case‑insensitive)
        into an integer number of bytes.
        """
        s = value.strip().upper()
        num_part = ""
        unit_part = ""
        # split numeric vs alpha
        for ch in s:
            if ch.isdigit() or ch == ".":
                num_part += ch
            else:
                unit_part += ch

        try:
            num = float(num_part)
        except ValueError:
            self.fail(f"invalid size value: {value}", param, ctx)

        multipliers = {
            "": 1024 ** 0,
            "B": 1024 ** 0,
            "K": 1024 ** 1,
            "KB": 1024 ** 1,
            "M": 1024 ** 2,
            "MB": 1024 ** 2,
            "G": 1024 ** 3,
            "GB": 1024 ** 3,
            "T": 1024 ** 4,
            "TB": 1024 ** 4,
        }

        if unit_part not in multipliers:
            self.fail(f"unknown size unit: {unit_part}", param, ctx)

        return int(num * multipliers[unit_part])


BYTE_SIZE = ByteSizeParamType()


def module_name_completer(ctx, param, incomplete):
    directory: Path = ctx.params.get("directory")
    if not directory:
        return []

    # your own logic to discover module names under `directory`
    all_suggestions:list[CompletionItem] = []
    try:
        for p in Path(directory).glob("*memory-usage-report.txt"):
            module_name = p.stem.split(".")[1]
            module_id = p.stem.split(".")[0].split("_")[1]
            all_suggestions.append(CompletionItem(module_name, help=module_id))
    except Exception:
        return []
    suggestions = list(filter(lambda c: c.value.startswith(incomplete), all_suggestions))
    print(len(suggestions))
    return suggestions
