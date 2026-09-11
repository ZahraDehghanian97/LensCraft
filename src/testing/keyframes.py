"""Names and configuration for comparable sparse-keyframe evaluations."""

from numbers import Integral


KEYFRAME_MODES = ("key_framing", "key_framing+prompt")
DEFAULT_KEYFRAME_COUNTS = (1, 2, 4, 8, 26)


def validate_keyframe_counts(counts):
    result = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("Keyframe counts must be positive integers")
        if int(count) not in result:
            result.append(int(count))
    return result


def keyframe_mode(mode, count):
    if mode not in KEYFRAME_MODES:
        raise ValueError(f"Not a keyframe mode: {mode}")
    validate_keyframe_counts([count])
    return f"{mode}_k{count}"


def parse_keyframe_mode(item, default_count=4):
    """Return (generation mode, requested K), with None for non-keyframe modes."""
    if item in KEYFRAME_MODES:
        validate_keyframe_counts([default_count])
        return item, default_count
    for mode in KEYFRAME_MODES:
        prefix = f"{mode}_k"
        if item.startswith(prefix):
            suffix = item[len(prefix):]
            if not suffix.isdigit() or int(suffix) < 1:
                raise ValueError(f"Invalid keyframe metric mode: {item}")
            return mode, int(suffix)
    return item, None
