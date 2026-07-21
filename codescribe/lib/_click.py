# Copyright (c) 2026 UChicago Argonne LLC
# SPDX-License-Identifier: Apache-2.0
# Full license and notices: see LICENSE and NOTICE in the repo root.

from click import command, option, Option, UsageError, Context
from typing import Any, Dict, List, Set

__all__ = ["MutuallyExclusiveOption"]


class MutuallyExclusiveOption(Option):
    def __init__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        self.mutually_exclusive: Set[str] = set(kwargs.pop("mutually_exclusive", []))
        help: str = kwargs.get("help", "")
        if self.mutually_exclusive:
            ex_str: str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = help + (
                " NOTE: This argument is mutually exclusive with "
                " arguments: [" + ex_str + "]."
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(
        self, ctx: Context, opts: Dict[str, Any], args: List[str]
    ) -> Any:
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(self.name, ", ".join(self.mutually_exclusive))
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(ctx, opts, args)
