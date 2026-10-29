"""The settings of research runs (#396 decisions 3-5), the `research` object of the
installation's settings file:

- `models`: alias -> the provider's model id (no default: the ids are the provider's, never the
  code's); `default_model`: an alias, or a list of `{"up_to_chars": N, "model": alias}` steps
  ending in one `{"model": alias}` (a brief up to N characters takes that alias), default the
  first alias;
- `provider`: the opencode provider entry (`id`, and `npm`, `name`, `options`, `models` as
  opencode takes them); the bridge adds `options.apiKey` as `{env:<key_env>}` and an entry per id
  of `models`;
- `key_env`: the name of the environment variable holding the provider's key (read at run time,
  never stored; default `RESEARCH_API_KEY`);
- `opencode`: the program (a name or path) or argv prefix that starts opencode, default
  `opencode` on `PATH`;
- `helpers` (false): whether opencode's `task` tool may start helper sessions;
- `minutes` (20) and `stall_minutes` (10): the run's time bound and stall bound;
- `state`: the folder of runs and jobs, default `research` beside the store in orch-it's data
  folder, never inside a repo.

The names check (`python -m orch_it research-check-names`) reads a `settings.json` at a repo's root, its
`research` object, and refuses the values of the provider and model name fields only.

A value of the wrong shape, or a key not in this list (a typo would be silently ignored), raises
`SettingsError("bad_setting", "research.<key>")`; a run the settings cannot serve raises
`ResearchRefused` before any server starts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from orch_it import settings as installation
from orch_it.settings import SettingsError

REFUSAL_CODES = ("research_not_set", "unknown_model", "key_missing", "brief_unreadable",
                 "report_path_outside", "repo_unknown", "opencode_missing", "state_in_git_tree")
KEYS = ("models", "default_model", "provider", "key_env", "opencode", "helpers", "minutes",
        "stall_minutes", "state")
AGENT = "orch-it-research"
DEFAULT_KEY_ENV = "RESEARCH_API_KEY"
MAX_MINUTES = 7 * 24 * 60
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ResearchRefused(Exception):
    """A research run that cannot start as asked. `code` is one of `REFUSAL_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in REFUSAL_CODES:
            raise ValueError(f"not a research refusal code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class ResearchSettings:
    models: dict[str, str] = field(default_factory=dict)
    default_model: tuple[tuple[int | None, str], ...] = ()
    provider: dict | None = None
    key_env: str = DEFAULT_KEY_ENV
    opencode: str | tuple[str, ...] | None = None
    helpers: bool = False
    minutes: float = 20.0
    stall_minutes: float = 10.0
    state: Path = field(default_factory=lambda: installation.data_folder() / "research")

    @classmethod
    def from_settings(cls, raw: dict) -> ResearchSettings:
        research = raw.get("research", {})
        if not isinstance(research, dict):
            raise SettingsError("bad_setting", "research")
        for key in research:
            if key not in KEYS:
                raise SettingsError("bad_setting", f"research.{key}")
        values: dict = {}
        if research.get("models") is not None:
            values["models"] = _models(research["models"])
        aliases = values.get("models", {})
        if research.get("default_model") is not None:
            values["default_model"] = _default_model(research["default_model"], aliases)
        if research.get("provider") is not None:
            values["provider"] = _provider(research["provider"])
        if research.get("key_env") is not None:
            if not (isinstance(research["key_env"], str) and _ENV_NAME.fullmatch(research["key_env"])):
                raise SettingsError("bad_setting", "research.key_env")
            values["key_env"] = research["key_env"]
        if research.get("opencode") is not None:
            values["opencode"] = _program(research["opencode"])
        if research.get("helpers") is not None:
            if not isinstance(research["helpers"], bool):
                raise SettingsError("bad_setting", "research.helpers")
            values["helpers"] = research["helpers"]
        for key in ("minutes", "stall_minutes"):
            if research.get(key) is not None:
                values[key] = _minutes(key, research[key])
        if research.get("state") is not None:
            state = research["state"]
            if not isinstance(state, str) or not state.strip() or not Path(state).expanduser().is_absolute():
                raise SettingsError("bad_setting", "research.state")  # relative would land wherever it runs
            values["state"] = Path(state).expanduser()
        return cls(**values)

    def _require_set(self) -> None:
        missing = [name for name, value in (("models", self.models), ("provider", self.provider)) if not value]
        if missing:
            raise ResearchRefused("research_not_set", "set research." + ", research.".join(missing))

    def model_id(self, alias: str) -> str:
        """The provider's model id of `alias`, or `unknown_model` naming the aliases there are."""

        self._require_set()
        if alias not in self.models:
            raise ResearchRefused("unknown_model", f"{alias!r}; the settings name {', '.join(self.models)}")
        return self.models[alias]

    def alias_for(self, brief_chars: int) -> str:
        """The default alias for a brief of `brief_chars` characters."""

        self._require_set()
        for up_to, alias in self.default_model:
            if up_to is None or brief_chars <= up_to:
                return alias
        return next(iter(self.models))

    def key(self, env) -> str:
        """The provider's key from the variable `key_env` names in `env`, or `key_missing`
        naming the variable."""

        self._require_set()
        value = (env.get(self.key_env) or "").strip()
        if not value:
            raise ResearchRefused("key_missing", f"the environment variable {self.key_env} is not set")
        return value

    def opencode_config(self) -> dict:
        """The `OPENCODE_CONFIG_CONTENT` of a research server: the provider with the key named by
        its variable and an entry per model id, web search and fetch allowed, no shell, no edit,
        no question; the `task` tool only with `helpers`."""

        self._require_set()
        provider = {key: value for key, value in self.provider.items() if key != "id"}
        provider["options"] = dict(provider.get("options") or {}) | {"apiKey": f"{{env:{self.key_env}}}"}
        models = dict(provider.get("models") or {})
        for model in self.models.values():
            models.setdefault(model, {})
        provider["models"] = models
        task = "allow" if self.helpers else "deny"
        return {
            "$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled",
            "provider": {self.provider["id"]: provider},
            "permission": {"bash": "deny", "edit": "deny", "external_directory": "deny",
                           "question": "deny", "webfetch": "allow", "task": task},
            "agent": {AGENT: {"mode": "primary", "description": "web research for orch-it",
                              "permission": {"webfetch": "allow", "task": task}}},
        }


def _models(value) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise SettingsError("bad_setting", "research.models")
    for alias, model in value.items():
        if not (isinstance(alias, str) and _ALIAS.fullmatch(alias) and isinstance(model, str)
                and model and model == model.strip() and "\x00" not in model):
            raise SettingsError("bad_setting", "research.models")
    return dict(value)


def _default_model(value, aliases: dict) -> tuple[tuple[int | None, str], ...]:
    if isinstance(value, str):
        steps = [{"model": value}]
    elif isinstance(value, list) and value:
        steps = value
    else:
        raise SettingsError("bad_setting", "research.default_model")
    out = []
    for n, step in enumerate(steps):
        last = n == len(steps) - 1
        if not isinstance(step, dict) or step.get("model") not in aliases:
            raise SettingsError("bad_setting", "research.default_model")
        up_to = step.get("up_to_chars")
        if last != (up_to is None) or (up_to is not None and (type(up_to) is not int or up_to < 0)):
            raise SettingsError("bad_setting", "research.default_model")
        if up_to is not None and out and up_to <= out[-1][0]:  # the steps rise
            raise SettingsError("bad_setting", "research.default_model")
        out.append((up_to, step["model"]))
    return tuple(out)


def _provider(value) -> dict:
    if not (isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"].strip()
            and isinstance(value.get("options", {}), dict) and isinstance(value.get("models", {}), dict)):
        raise SettingsError("bad_setting", "research.provider")
    return dict(value)


def _program(value) -> str | tuple[str, ...]:
    if isinstance(value, str) and value.strip() and "\x00" not in value:
        return value
    if (isinstance(value, list) and value
            and all(isinstance(a, str) and a and "\x00" not in a for a in value)):
        return tuple(value)
    raise SettingsError("bad_setting", "research.opencode")


def _minutes(key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError("bad_setting", f"research.{key}")
    try:
        minutes = float(value)
    except OverflowError:
        raise SettingsError("bad_setting", f"research.{key}") from None
    if not (math.isfinite(minutes) and 0 < minutes <= MAX_MINUTES):
        raise SettingsError("bad_setting", f"research.{key}")
    return minutes
