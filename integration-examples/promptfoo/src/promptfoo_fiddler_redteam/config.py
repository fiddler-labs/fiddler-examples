"""Environment-driven settings for the Fiddler side of the bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

# Look for .env starting from the current working directory.
load_dotenv(find_dotenv(usecwd=True))


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class FiddlerSettings:
    url: str
    api_key: str
    project: str
    application: str

    @classmethod
    def from_env(cls) -> FiddlerSettings:
        missing = [
            name
            for name in (
                "FIDDLER_URL",
                "FIDDLER_API_KEY",
                "FIDDLER_PROJECT",
                "FIDDLER_APPLICATION",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                f"{', '.join(missing)}. See examples/.env.example."
            )
        return cls(
            url=os.environ["FIDDLER_URL"].rstrip("/"),
            api_key=os.environ["FIDDLER_API_KEY"],
            project=os.environ["FIDDLER_PROJECT"],
            application=os.environ["FIDDLER_APPLICATION"],
        )
