from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class NumericFeature:
    """Description of a bounded scalar conditioning token.

    Values outside the declared range are clipped before encoding. Optional
    non-negative limits use logarithmic scaling so useful low values retain
    more resolution while pathological input cannot produce NaN/inf features.
    """

    minimum: float
    maximum: float
    logarithmic: bool = False

class CameraVerticalAngle(Enum):
    LOW = "low"
    EYE = "eye"
    HIGH = "high"
    OVERHEAD = "overhead"
    BIRDS_EYE = "birdsEye"

class ShotSize(Enum):
    EXTREME_CLOSE_UP = "extremeCloseUp"
    CLOSE_UP = "closeUp"
    MEDIUM_CLOSE_UP = "mediumCloseUp"
    MEDIUM_SHOT = "mediumShot"
    FULL_SHOT = "fullShot"
    LONG_SHOT = "longShot"
    VERY_LONG_SHOT = "veryLongShot"
    EXTREME_LONG_SHOT = "extremeLongShot"

class Scale(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    FULL = "full"

class MovementEasing(Enum):
    LINEAR = "linear"
    EASE_IN_SINE = "easeInSine"
    EASE_OUT_SINE = "easeOutSine"
    EASE_IN_OUT_SINE = "easeInOutSine"
    EASE_IN_QUAD = "easeInQuad"
    EASE_OUT_QUAD = "easeOutQuad"
    EASE_IN_OUT_QUAD = "easeInOutQuad"
    EASE_IN_CUBIC = "easeInCubic"
    EASE_OUT_CUBIC = "easeOutCubic"
    EASE_IN_OUT_CUBIC = "easeInOutCubic"
    EASE_IN_QUART = "easeInQuart"
    EASE_OUT_QUART = "easeOutQuart"
    EASE_IN_OUT_QUART = "easeInOutQuart"
    EASE_IN_QUINT = "easeInQuint"
    EASE_OUT_QUINT = "easeOutQuint"
    EASE_IN_OUT_QUINT = "easeInOutQuint"
    EASE_IN_EXPO = "easeInExpo"
    EASE_OUT_EXPO = "easeOutExpo"
    EASE_IN_OUT_EXPO = "easeInOutExpo"
    EASE_IN_CIRC = "easeInCirc"
    EASE_OUT_CIRC = "easeOutCirc"
    EASE_IN_OUT_CIRC = "easeInOutCirc"
    SMOOTH = "smooth"

class SubjectView(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    THREE_QUARTER_FRONT_LEFT = "threeQuarterFrontLeft"
    THREE_QUARTER_FRONT_RIGHT = "threeQuarterFrontRight"
    THREE_QUARTER_BACK_LEFT = "threeQuarterBackLeft"
    THREE_QUARTER_BACK_RIGHT = "threeQuarterBackRight"

class SubjectInFramePosition(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    TOP_LEFT = "topLeft"
    TOP_RIGHT = "topRight"
    BOTTOM_LEFT = "bottomLeft"
    BOTTOM_RIGHT = "bottomRight"
    OUTER_LEFT = "outerLeft"
    OUTER_RIGHT = "outerRight"
    OUTER_TOP = "outerTop"
    OUTER_BOTTOM = "outerBottom"

class DynamicMode(Enum):
    INTERPOLATION = "interpolation"
    SIMPLE = "simple"


class Randomness(Enum):
    HAND_HELD = "handHeld"
    SHAKY = "shaky"

class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    FORWARD = "forward"
    BACKWARD = "backward"

class MovementMode(Enum):
    TRANSITION = "transition"
    ROTATION = "rotation"
    ARC = "arc"
    CRANE = "crane"
    ROLL = "roll"

class CameraMovementType(Enum):
    STATIC = "static"
    FOLLOW = "follow"
    TRACK = "track"
    DOLLY_IN = "dollyIn"
    DOLLY_OUT = "dollyOut"
    PAN_LEFT = "panLeft"
    PAN_RIGHT = "panRight"
    TILT_UP = "tiltUp"
    TILT_DOWN = "tiltDown"
    TRUCK_LEFT = "truckLeft"
    TRUCK_RIGHT = "truckRight"
    PEDESTAL_UP = "pedestalUp"
    PEDESTAL_DOWN = "pedestalDown"
    ARC_LEFT = "arcLeft"
    ARC_RIGHT = "arcRight"
    CRANE_UP = "craneUp"
    CRANE_DOWN = "craneDown"
    DUTCH_LEFT = "dutchLeft"
    DUTCH_RIGHT = "dutchRight"

class MovementSpeed(Enum):
    SLOW_TO_FAST = "slowToFast"
    FAST_TO_SLOW = "fastToSlow"
    CONSTANT = "constant"
    SMOOTH_START_STOP = "smoothStartStop"
    
class SetupKind(Enum):
    INIT = "init"
    END = "end"

cinematography_struct = [
    ("initial", [
        ("cameraAngle", CameraVerticalAngle),
        ("shotSize", ShotSize),
        ("subjectView", SubjectView),
        ("subjectFraming", SubjectInFramePosition)
    ]),
    ("movement", [
        ("type", CameraMovementType),
        ("speed", MovementSpeed)
    ]),
    ("final", [
        ("cameraAngle", CameraVerticalAngle),
        ("shotSize", ShotSize),
        ("subjectView", SubjectView),
        ("subjectFraming", SubjectInFramePosition)
    ])
]

setup_config_struct = [
    ("cameraAngle", CameraVerticalAngle),
    ("shotSize", ShotSize),
    ("subjectView", SubjectView),
    ("subjectFraming", [
        ("position", SubjectInFramePosition),
        ("dutchAngleScale", Scale)
    ])
]

simulation_struct = [
    ("setup", [
        ("config", setup_config_struct),
        ("kind", SetupKind)
    ]),
    ("dynamic", [
        ("type", DynamicMode),
        ("easing", MovementEasing),
        ("randomness", Randomness),
        ("complementSetup", setup_config_struct),
        ("subjectAwareInterpolation", bool),
        ("scale", Scale),
        ("direction", Direction),
        ("movementMode", MovementMode)
    ]),
    ("constraints", [
        ("allFramesVisibility", bool),
        ("staticDistance", bool),
        ("staticCameraSubjectRotation", bool),
        ("lockedMovement", [
            ("left", bool),
            ("right", bool),
            ("up", bool),
            ("down", bool),
            ("forward", bool),
            ("backward", bool)
        ]),
        ("lockedRotation", [
            ("left", bool),
            ("right", bool),
            ("up", bool),
            ("down", bool),
            ("rollClockwise", bool),
            ("rollNonClockwise", bool)
        ]),
        ("maxAccelerate", NumericFeature(0.0, 20.0, logarithmic=True)),
        ("maxSpeed", NumericFeature(0.0, 20.0, logarithmic=True)),
        ("importance", NumericFeature(1.0, 10.0))
    ])
]
