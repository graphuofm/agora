"""yamltags.py — load/dump AGORA rule-base YAML preserving serde external `!tags`.

The rule-base YAML uses serde external tagging (``!preferential_attachment {m: 3}``,
``!clustered {n_communities: 8}``, ``!log_normal {mu: 6.5, sigma: 1.0}``, ...).
PyYAML's ``SafeLoader`` cannot construct these unknown tags, and ``safe_dump``
would drop them — yet the Rust loader REQUIRES them. This module provides a
loader that captures any ``!tag`` as a :class:`Tagged` wrapper and a dumper that
re-emits it with the same tag, so a scaffold can be round-tripped losslessly and
only its plain (untagged) fields mutated.
"""
from __future__ import annotations

from typing import Any

import yaml


class Tagged:
    """A YAML node that carried an explicit ``!tag`` (preserved for re-emission)."""

    __slots__ = ("tag", "value")

    def __init__(self, tag: str, value: Any) -> None:
        self.tag = tag  # includes the leading '!'
        self.value = value  # the wrapped mapping/scalar/sequence

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Tagged({self.tag!r}, {self.value!r})"


class TagPreservingLoader(yaml.SafeLoader):
    pass


class TagPreservingDumper(yaml.SafeDumper):
    pass


def _construct_tagged(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Tagged:
    if isinstance(node, yaml.MappingNode):
        value: Any = loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_scalar(node)
    return Tagged("!" + tag_suffix, value)


# Catch every `!...` tag via the multi-constructor on the empty prefix.
TagPreservingLoader.add_multi_constructor("!", _construct_tagged)


def _represent_tagged(dumper: yaml.Dumper, data: Tagged) -> yaml.Node:
    val = data.value
    if isinstance(val, dict) and val:
        return dumper.represent_mapping(data.tag, val)
    if isinstance(val, (list, tuple)) and val:
        return dumper.represent_sequence(data.tag, val)
    if isinstance(val, dict) or isinstance(val, (list, tuple)) or val is None or val == "":
        # A fieldless unit variant (e.g. !ring / !normal / !uniform_random):
        # emitted as `!tag ''`, then post-processed in dump() to a bare `!tag`
        # (serde external tagging null body).
        return dumper.represent_scalar(data.tag, "")
    return dumper.represent_scalar(data.tag, str(val))


TagPreservingDumper.add_representer(Tagged, _represent_tagged)


def load(text: str) -> Any:
    return yaml.load(text, Loader=TagPreservingLoader)


def load_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=TagPreservingLoader)


def dump(obj: Any) -> str:
    text = yaml.dump(
        obj,
        Dumper=TagPreservingDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    # Fieldless unit variants (e.g. `scope: !normal`, `model: !uniform_random`)
    # are emitted by PyYAML as `!normal ''`; serde external tagging wants the
    # bare tag (a null body). Strip the trailing empty-string scalar.
    return _BARE_TAG_RE.sub(r"\1", text)


import re as _re  # noqa: E402

# Matches: <indent>(key: )?!tag '' at end of line -> keep up to the tag.
_BARE_TAG_RE = _re.compile(r"(?m)^(\s*(?:- )?(?:[\w$-]+: )?!\S+) ''\s*$")
