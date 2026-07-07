from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTIES = PROJECT_ROOT / "third_parties"


def third_party(*parts) -> str:
    return str(THIRD_PARTIES.joinpath(*parts))
