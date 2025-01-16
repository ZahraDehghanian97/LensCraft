from enum import Enum

class CameraMovementType(Enum):
    static = "static"
    follow = "follow"
    track = "track"

    dollyIn = "dollyIn"
    dollyOut = "dollyOut"

    panLeft = "panLeft"
    panRight = "panRight"
    tiltUp = "tiltUp"
    tiltDown = "tiltDown"

    truckLeft = "truckLeft"
    truckRight = "truckRight"
    pedestalUp = "pedestalUp"
    pedestalDown = "pedestalDown"

    arcLeft = "arcLeft"
    arcRight = "arcRight"

    craneUp = "craneUp"
    craneDown = "craneDown"

    dollyOutZoomIn = "dollyOutZoomIn"
    dollyInZoomOut = "dollyInZoomOut"

    dutchLeft = "dutchLeft"
    dutchRight = "dutchRight"


class EasingType(Enum):
    linear = "linear"

    easeInSine = "easeInSine"
    easeOutSine = "easeOutSine"
    easeInOutSine = "easeInOutSine"

    easeInQuad = "easeInQuad"
    easeOutQuad = "easeOutQuad"
    easeInOutQuad = "easeInOutQuad"

    easeInCubic = "easeInCubic"
    easeOutCubic = "easeOutCubic"
    easeInOutCubic = "easeInOutCubic"

    easeInQuart = "easeInQuart"
    easeOutQuart = "easeOutQuart"
    easeInOutQuart = "easeInOutQuart"

    easeInQuint = "easeInQuint"
    easeOutQuint = "easeOutQuint"
    easeInOutQuint = "easeInOutQuint"

    easeInExpo = "easeInExpo"
    easeOutExpo = "easeOutExpo"
    easeInOutExpo = "easeInOutExpo"

    easeInCirc = "easeInCirc"
    easeOutCirc = "easeOutCirc"
    easeInOutCirc = "easeInOutCirc"

    easeInBack = "easeInBack"
    easeOutBack = "easeOutBack"
    easeInOutBack = "easeInOutBack"

    easeInElastic = "easeInElastic"
    easeOutElastic = "easeOutElastic"
    easeInOutElastic = "easeInOutElastic"

    easeInBounce = "easeInBounce"
    easeOutBounce = "easeOutBounce"
    easeInOutBounce = "easeInOutBounce"

    handHeld = "handHeld"
    anticipation = "anticipation"
    smooth = "smooth"


class CameraAngle(Enum):
    Low = "low"
    Eye = "eye"
    High = "high"
    Overhead = "overhead"
    BirdsEye = "birdsEye"


class ShotType(Enum):
    ExtremeCloseUp = "extremeCloseUp"
    CloseUp = "closeUp"
    MediumCloseUp = "mediumCloseUp"
    MediumShot = "mediumShot"
    FullShot = "fullShot"
    LongShot = "longShot"
    VeryLongShot = "veryLongShot"
    ExtremeLongShot = "extremeLongShot"


class MovementSpeed(Enum):
    SlowToFast = "slowToFast"
    FastToSlow = "fastToSlow"
    Constant = "constant"
    DeliberateStartStop = "deliberateStartStop"


class DynamicMode(Enum):
    Interpolation = "interpolation"
    Simple = "simple"


class Direction(Enum):
    Left = "left"
    Right = "right"
    Up = "up"
    Down = "down"


class MovementMode(Enum):
    Transition = "transition"
    Rotation = "rotation"


class Scale(Enum):
    Small = "small"
    Medium = "medium"
    Large = "large"
    Full = "full"


class SubjectView(Enum):
    Front = "front"
    Back = "back"
    Left = "left"
    Right = "right"
    ThreeQuarterFrontLeft = "threeQuarterFrontLeft"
    ThreeQuarterFrontRight = "threeQuarterFrontRight"
    ThreeQuarterBackLeft = "threeQuarterBackLeft"
    ThreeQuarterBackRight = "threeQuarterBackRight"


class SubjectInFramePosition(Enum):
    Left = "left"
    Right = "right"
    Top = "top"
    Bottom = "bottom"
    Center = "center"
    TopLeft = "topLeft"
    TopRight = "topRight"
    BottomLeft = "bottomLeft"
    BottomRight = "bottomRight"
    OuterLeft = "outerLeft"
    OuterRight = "outerRight"
    OuterTop = "outerTop"
    OuterBottom = "outerBottom"
