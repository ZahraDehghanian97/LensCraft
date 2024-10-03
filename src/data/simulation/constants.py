from enum import Enum


class CameraMovementType(Enum):
    panLeft = "panLeft"
    panRight = "panRight"
    tiltUp = "tiltUp"
    tiltDown = "tiltDown"
    dollyIn = "dollyIn"
    dollyOut = "dollyOut"
    truckLeft = "truckLeft"
    truckRight = "truckRight"
    pedestalUp = "pedestalUp"
    pedestalDown = "pedestalDown"
    fullZoomIn = "fullZoomIn"
    fullZoomOut = "fullZoomOut"
    halfZoomIn = "halfZoomIn"
    halfZoomOut = "halfZoomOut"
    shortZoomIn = "shortZoomIn"
    shortZoomOut = "shortZoomOut"
    shortArcShotLeft = "shortArcShotLeft"
    shortArcShotRight = "shortArcShotRight"
    halfArcShotLeft = "halfArcShotLeft"
    halfArcShotRight = "halfArcShotRight"
    fullArcShotLeft = "fullArcShotLeft"
    fullArcShotRight = "fullArcShotRight"
    panAndTilt = "panAndTilt"
    dollyAndPan = "dollyAndPan"
    zoomAndTruck = "zoomAndTruck"


class EasingType(Enum):
    linear = "linear"
    easeInQuad = "easeInQuad"
    easeInCubic = "easeInCubic"
    easeInQuart = "easeInQuart"
    easeInQuint = "easeInQuint"
    easeOutQuad = "easeOutQuad"
    easeOutCubic = "easeOutCubic"
    easeOutQuart = "easeOutQuart"
    easeOutQuint = "easeOutQuint"
    easeInOutQuad = "easeInOutQuad"
    easeInOutCubic = "easeInOutCubic"
    easeInOutQuart = "easeInOutQuart"
    easeInOutQuint = "easeInOutQuint"
    easeInSine = "easeInSine"
    easeOutSine = "easeOutSine"
    easeInOutSine = "easeInOutSine"
    easeInExpo = "easeInExpo"
    easeOutExpo = "easeOutExpo"
    easeInOutExpo = "easeInOutExpo"
    easeInCirc = "easeInCirc"
    easeOutCirc = "easeOutCirc"
    easeInOutCirc = "easeInOutCirc"
    easeInBounce = "easeInBounce"
    easeOutBounce = "easeOutBounce"
    easeInOutBounce = "easeInOutBounce"
    easeInElastic = "easeInElastic"
    easeOutElastic = "easeOutElastic"
    easeInOutElastic = "easeInOutElastic"


class CameraAngle(Enum):
    lowAngle = "lowAngle"
    mediumAngle = "mediumAngle"
    highAngle = "highAngle"
    birdsEyeView = "birdsEyeView"


class ShotType(Enum):
    closeUp = "closeUp"
    mediumShot = "mediumShot"
    longShot = "longShot"


movement_descriptions = {
    "panLeft": "panning left",
    "panRight": "panning right",
    "tiltUp": "tilting up",
    "tiltDown": "tilting down",
    "dollyIn": "moving closer",
    "dollyOut": "moving away",
    "truckLeft": "moving left",
    "truckRight": "moving right",
    "pedestalUp": "rising vertically",
    "pedestalDown": "descending vertically",
    "fullZoomIn": "zooming in fully",
    "fullZoomOut": "zooming out fully",
    "halfZoomIn": "zooming in halfway",
    "halfZoomOut": "zooming out halfway",
    "shortZoomIn": "zooming in slightly",
    "shortZoomOut": "zooming out slightly",
    "shortArcShotLeft": "moving in a short arc to the left",
    "shortArcShotRight": "moving in a short arc to the right",
    "halfArcShotLeft": "moving in a half arc to the left",
    "halfArcShotRight": "moving in a half arc to the right",
    "fullArcShotLeft": "moving in a full arc to the left",
    "fullArcShotRight": "moving in a full arc to the right",
    "panAndTilt": "panning and tilting",
    "dollyAndPan": "moving and panning",
    "zoomAndTruck": "zooming and moving sideways",
}

easing_descriptions = {
    "linear": "at a constant speed",
    "easeInQuad": "slowly at first, then accelerating gradually",
    "easeInCubic": "slowly at first, then accelerating more rapidly",
    "easeInQuart": "very slowly at first, then accelerating dramatically",
    "easeInQuint": "extremely slowly at first, then accelerating very dramatically",
    "easeOutQuad": "quickly at first, then decelerating gradually",
    "easeOutCubic": "quickly at first, then decelerating more rapidly",
    "easeOutQuart": "very quickly at first, then decelerating dramatically",
    "easeOutQuint": "extremely quickly at first, then decelerating very dramatically",
    "easeInOutQuad": "gradually accelerating, then gradually decelerating",
    "easeInOutCubic": "slowly accelerating, then decelerating more rapidly",
    "easeInOutQuart": "slowly accelerating, then decelerating dramatically",
    "easeInOutQuint": "very slowly accelerating, then decelerating very dramatically",
    "easeInSine": "with a gentle start, gradually increasing in speed",
    "easeOutSine": "quickly at first, then gently decelerating",
    "easeInOutSine": "with a gentle start and end, faster in the middle",
    "easeInExpo": "starting very slowly, then accelerating exponentially",
    "easeOutExpo": "starting very fast, then decelerating exponentially",
    "easeInOutExpo": "starting and ending slowly, with rapid acceleration and deceleration in the middle",
    "easeInCirc": "starting slowly, then accelerating sharply towards the end",
    "easeOutCirc": "starting quickly, then decelerating sharply towards the end",
    "easeInOutCirc": "with sharp acceleration and deceleration at both ends",
    "easeInBounce": "with a bouncing effect that intensifies towards the end",
    "easeOutBounce": "quickly at first, then bouncing to a stop",
    "easeInOutBounce": "with a bouncing effect at both the start and end",
    "easeInElastic": "with an elastic effect that intensifies towards the end",
    "easeOutElastic": "quickly at first, then oscillating to a stop",
    "easeInOutElastic": "with an elastic effect at both the start and end",
}

angle_descriptions = {
    "lowAngle": "from a low angle",
    "mediumAngle": "from a medium angle",
    "highAngle": "from a high angle",
    "birdsEyeView": "from a bird's eye view",
}

shot_descriptions = {
    "closeUp": "in a close-up shot",
    "mediumShot": "in a medium shot",
    "longShot": "in a long shot",
}
