from enum import Enum

from data.simulation.utils import count_parameters


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
    EASE_IN_BACK = "easeInBack"
    EASE_OUT_BACK = "easeOutBack"
    EASE_IN_OUT_BACK = "easeInOutBack"
    EASE_IN_ELASTIC = "easeInElastic"
    EASE_OUT_ELASTIC = "easeOutElastic"
    EASE_IN_OUT_ELASTIC = "easeInOutElastic"
    EASE_IN_BOUNCE = "easeInBounce"
    EASE_OUT_BOUNCE = "easeOutBounce"
    EASE_IN_OUT_BOUNCE = "easeInOutBounce"
    HAND_HELD = "handHeld"
    ANTICIPATION = "anticipation"
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

class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

class MovementMode(Enum):
    TRANSITION = "transition"
    ROTATION = "rotation"

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
    DOLLY_OUT_ZOOM_IN = "dollyOutZoomIn"
    DOLLY_IN_ZOOM_OUT = "dollyInZoomOut"
    DUTCH_LEFT = "dutchLeft"
    DUTCH_RIGHT = "dutchRight"

class MovementSpeed(Enum):
    SLOW_TO_FAST = "slowToFast"
    FAST_TO_SLOW = "fastToSlow"
    CONSTANT = "constant"
    SMOOTH_START_STOP = "smoothStartStop"
    

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

cinematography_struct_size = count_parameters(cinematography_struct)

simulation_struct = [
    ("initialSetup", [
        ("cameraAngle", CameraVerticalAngle),
        ("shotSize", ShotSize),
        ("subjectView", SubjectView),
        ("subjectFraming", [
            ("position", SubjectInFramePosition),
            ("dutchAngleScale", Scale)
        ])
    ]),
    ("dynamic", [
        ("easing", MovementEasing),
        ("endSetup", [
            ("cameraAngle", CameraVerticalAngle),
            ("shotSize", ShotSize),
            ("subjectView", SubjectView),
            ("subjectFraming", [
                ("position", SubjectInFramePosition),
                ("dutchAngleScale", Scale)
            ])
        ]),
        ("subjectAwareInterpolation", bool),
        ("scale", Scale),
        ("direction", Direction),
        ("movementMode", MovementMode)
    ]),
    ("constraints", [
        ("allFramesVisibility", bool),
        ("staticDistance", bool),
        ("staticCameraSubjectRotation", bool)
    ])
]

simulation_struct_size = count_parameters(simulation_struct)
