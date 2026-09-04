enum_descriptions = {
    "CameraVerticalAngle": {
        "low": "from a low angle",
        "eye": "from an eye-level angle",
        "high": "from a high angle",
        "overhead": "from an overhead angle",
        "birdsEye": "from a bird's-eye angle"
    },
    "ShotSize": {
        "extremeCloseUp": "in an extreme close-up",
        "closeUp": "in a close-up",
        "mediumCloseUp": "in a medium close-up",
        "mediumShot": "in a medium shot",
        "fullShot": "in a full shot",
        "longShot": "in a long shot",
        "veryLongShot": "in a very long shot",
        "extremeLongShot": "in an extreme long shot"
    },
    "Scale": {
        "small": "with small intensity",
        "medium": "with medium intensity",
        "large": "with large intensity",
        "full": "with full intensity"
    },
    "MovementEasing": {
        "linear": "with linear movement",
        "easeInSine": "with gradual acceleration using sine function",
        "easeOutSine": "with gradual deceleration using sine function",
        "easeInOutSine": "with gradual acceleration and deceleration using sine function",
        "easeInQuad": "with quadratic acceleration",
        "easeOutQuad": "with quadratic deceleration",
        "easeInOutQuad": "with quadratic acceleration and deceleration",
        "easeInCubic": "with cubic acceleration",
        "easeOutCubic": "with cubic deceleration",
        "easeInOutCubic": "with cubic acceleration and deceleration",
        "easeInQuart": "with quartic acceleration",
        "easeOutQuart": "with quartic deceleration",
        "easeInOutQuart": "with quartic acceleration and deceleration",
        "easeInQuint": "with quintic acceleration",
        "easeOutQuint": "with quintic deceleration",
        "easeInOutQuint": "with quintic acceleration and deceleration",
        "easeInExpo": "with exponential acceleration",
        "easeOutExpo": "with exponential deceleration",
        "easeInOutExpo": "with exponential acceleration and deceleration",
        "easeInCirc": "with circular acceleration",
        "easeOutCirc": "with circular deceleration",
        "easeInOutCirc": "with circular acceleration and deceleration",
        "smooth": "with smooth movement"
    },
    "SubjectView": {
        "front": "from the front",
        "back": "from the back",
        "left": "from the left side",
        "right": "from the right side",
        "threeQuarterFrontLeft": "from the front-left three-quarter view",
        "threeQuarterFrontRight": "from the front-right three-quarter view",
        "threeQuarterBackLeft": "from the back-left three-quarter view",
        "threeQuarterBackRight": "from the back-right three-quarter view"
    },
    "SubjectInFramePosition": {
        "left": "positioned on the left",
        "right": "positioned on the right",
        "top": "positioned at the top",
        "bottom": "positioned at the bottom",
        "center": "positioned in the center",
        "topLeft": "positioned in the top-left",
        "topRight": "positioned in the top-right",
        "bottomLeft": "positioned in the bottom-left",
        "bottomRight": "positioned in the bottom-right",
        "outerLeft": "positioned on the far left",
        "outerRight": "positioned on the far right",
        "outerTop": "positioned at the very top",
        "outerBottom": "positioned at the very bottom"
    },
    "Direction": {
        "left": "towards the left",
        "right": "towards the right",
        "up": "upwards",
        "down": "downwards",
        "forward": "forwards",
        "backward": "backwards"
    },
    "MovementMode": {
        "transition": "using transitional movement",
        "rotation": "using rotational movement",
        "arc": "using arc movement",
        "crane": "using a vertical crane-boom arc",
        "roll": "using camera-axis roll"
    },
    "DynamicMode": {
        "interpolation": "using setup interpolation",
        "simple": "using a simple camera movement"
    },
    "Randomness": {
        "handHeld": "with subtle handheld variation",
        "shaky": "with pronounced camera shake"
    },
    "CameraMovementType": {
        "static": "remaining stationary",
        "follow": "following the subject",
        "track": "tracking alongside the subject",
        "dollyIn": "moving closer to the subject",
        "dollyOut": "moving away from the subject",
        "panLeft": "panning to the left",
        "panRight": "panning to the right",
        "tiltUp": "tilting upward",
        "tiltDown": "tilting downward",
        "truckLeft": "moving laterally left",
        "truckRight": "moving laterally right",
        "pedestalUp": "moving straight up",
        "pedestalDown": "moving straight down",
        "arcLeft": "moving in a leftward arc",
        "arcRight": "moving in a rightward arc",
        "craneUp": "craning upward",
        "craneDown": "craning downward",
        "dutchLeft": "tilting left on the camera axis",
        "dutchRight": "tilting right on the camera axis"
    },
    "MovementSpeed": {
        "slowToFast": "accelerating from slow to fast",
        "fastToSlow": "decelerating from fast to slow",
        "constant": "at a constant speed",
        "smoothStartStop": "smoothly starting and stopping"
    },
    "SetupKind": {
        "init": "from the initial setup",
        "end": "towards the final setup"
    }
}



CAMERA_MOVEMENT_DESCRIPTIONS = {
    "static": "remains stationary", "follow": "follows the subject",
    "track": "tracks alongside the subject", "dollyIn": "pushes in",
    "dollyOut": "pulls out", "panLeft": "pans to the left",
    "panRight": "pans to the right", "tiltUp": "tilts upward",
    "tiltDown": "tilts downward", "truckLeft": "trucks left",
    "truckRight": "trucks right",
    "pedestalUp": "moves straight up with a pedestal movement",
    "pedestalDown": "moves straight down with a pedestal movement",
    "arcLeft": "moves in a leftward arc", "arcRight": "moves in a rightward arc",
    "craneUp": "cranes upward", "craneDown": "cranes downward",
    "dutchLeft": "tilts left on the camera axis",
    "dutchRight": "tilts right on the camera axis",
}

SUBJECT_MOVEMENT_DESCRIPTIONS = {
    "static": "remains stationary", "circular": "moves in a circular path",
    "linear": "moves along a straight path",
    "zigzag": "moves through a rounded zigzag path", "spiral": "moves in a spiral pattern",
    "figureEight": "moves in a figure-eight pattern",
    "wave": "moves in a wave-like sinusoidal pattern",
    "pendulum": "swings back and forth like a pendulum",
    "orbital": "orbits around a center point",
    "bounce": "bounces while moving forward",
}

DETAILED_SUBJECT_DESCRIPTIONS = {
    "static": "remains stationary throughout the entire sequence",
    "circular": "moves in a continuous circular motion",
    "linear": "travels once along a straight path",
    "zigzag": "follows a smooth zigzag trajectory with rounded turns",
    "spiral": "traces an expanding spiral path",
    "figureEight": "traces a figure-eight pattern",
    "wave": "follows a smooth wave-like path",
    "pendulum": "swings in a pendulum motion",
    "orbital": "orbits in a tilted elliptical path",
    "bounce": "bounces rhythmically while progressing forward",
}

SHOT_SIZE_DESCRIPTIONS = {
    "extremeCloseUp": "extreme close-up", "closeUp": "close-up",
    "mediumCloseUp": "medium close-up", "mediumShot": "medium shot",
    "fullShot": "full shot", "longShot": "wide shot",
    "veryLongShot": "very wide shot", "extremeLongShot": "extreme wide shot",
}

ANGLE_DESCRIPTIONS = {
    "low": "low angle", "eye": "eye level", "high": "high angle",
    "overhead": "overhead angle", "birdsEye": "bird's-eye view",
}

SPEED_DESCRIPTIONS = {
    "slowToFast": "gradually accelerating",
    "fastToSlow": "gradually decelerating",
    "smoothStartStop": "with smooth acceleration and deceleration",
}

FRAMING_DESCRIPTIONS = {
    "left": "left side", "right": "right side", "center": "center",
    "topLeft": "top-left", "topRight": "top-right",
    "bottomLeft": "bottom-left", "bottomRight": "bottom-right",
    "outerLeft": "far left", "outerRight": "far right",
    "outerTop": "very top", "outerBottom": "very bottom",
}


def _transition_clause(initial, final, descriptions, template):
    """Describe a setup change only when both values exist and differ."""
    if initial and final and initial != final:
        return template.format(
            descriptions.get(initial, initial),
            descriptions.get(final, final),
        )
    return ""


def _combined_movement_prompt(camera_movement_type, subject_movement,
                              camera_desc, subject_clause, camera_clause):
    """Phrasing for when the camera AND the subject are both moving.
    Wording is tuned per camera-movement family."""
    if camera_movement_type in ("follow", "track"):
        if subject_movement in ("circular", "linear", "spiral"):
            return f"The camera {camera_desc} as {subject_clause}."
        if subject_movement in ("zigzag", "wave"):
            return f"The camera {camera_desc}, matching the rhythm as {subject_clause}."
        if subject_movement == "bounce":
            return f"The camera {camera_desc}, maintaining focus as {subject_clause}."
        return f"The camera {camera_desc} while {subject_clause}."

    if camera_movement_type in ("truckLeft", "truckRight"):
        if subject_movement in ("linear", "wave", "zigzag"):
            same_direction = (
                ("Left" in camera_movement_type and "left" in subject_movement)
                or ("Right" in camera_movement_type and "right" in subject_movement)
            )
            if same_direction:
                return f"The camera {camera_desc} in sync with {subject_clause}."
            return f"The camera {camera_desc} while {subject_clause}."
        if subject_movement == "circular":
            return f"The camera {camera_desc} as {subject_clause}, creating dynamic framing."
        if subject_movement in ("pendulum", "orbital"):
            return f"The camera {camera_desc} alongside {subject_clause}."
        return f"The camera {camera_desc} while {subject_clause}."

    if camera_movement_type in ("dollyIn", "dollyOut"):
        if subject_movement in ("spiral", "circular"):
            return f"As {subject_clause}, {camera_clause}, creating a dynamic relationship."
        if subject_movement == "linear":
            return f"While {subject_clause}, {camera_clause}."
        if subject_movement in ("bounce", "wave"):
            return f"The camera {camera_desc} as {subject_clause}, emphasizing the motion."
        return f"As {subject_clause}, {camera_clause}."

    if camera_movement_type in ("panLeft", "panRight"):
        if subject_movement in ("linear", "zigzag"):
            pan_direction = "left" if "Left" in camera_movement_type else "right"
            if pan_direction in subject_movement.lower():
                return f"The camera {camera_desc} in tandem with {subject_clause}."
            return f"The camera {camera_desc} while {subject_clause}."
        if subject_movement == "circular":
            return f"The camera {camera_desc} to follow {subject_clause}."
        return f"As {subject_clause}, {camera_clause}."

    if camera_movement_type in ("arcLeft", "arcRight", "craneUp", "craneDown"):
        if subject_movement in ("orbital", "spiral"):
            return f"The camera {camera_desc} complementing {subject_clause}."
        if subject_movement == "figureEight":
            return f"The camera {camera_desc} as {subject_clause}, creating complex choreography."
        return f"While {subject_clause}, {camera_clause}."

    # default family (tilt/pedestal/dutch/zoom)
    if subject_movement in ("circular", "spiral", "figureEight"):
        return f"As {subject_clause}, {camera_clause}."
    if subject_movement in ("linear", "zigzag", "wave"):
        return f"While {subject_clause}, {camera_clause}."
    if subject_movement in ("bounce", "pendulum"):
        return f"The camera {camera_desc} as {subject_clause}."
    return f"While {subject_clause}, {camera_clause}."


def extract_text_prompt(cin_params, subject_movement):
    """Render a structured cinematography spec into one English sentence."""
    movement = cin_params.get("movement", {})
    initial = cin_params.get("initial", {})
    final = cin_params.get("final", {})

    camera_movement_type = movement.get("type")
    movement_speed = movement.get("speed")

    camera_desc = CAMERA_MOVEMENT_DESCRIPTIONS.get(camera_movement_type, "moves")
    subject_desc = SUBJECT_MOVEMENT_DESCRIPTIONS.get(subject_movement, "remains stationary")
    detailed_subject_desc = DETAILED_SUBJECT_DESCRIPTIONS.get(subject_movement, "remains stationary")
    is_subject_stationary = subject_movement in ("static", None)

    setup_changes = [
        clause for clause in (
            _transition_clause(initial.get("shotSize"), final.get("shotSize"),
                                SHOT_SIZE_DESCRIPTIONS, "transitioning from a {} to a {}"),
            _transition_clause(initial.get("cameraAngle"), final.get("cameraAngle"),
                                ANGLE_DESCRIPTIONS, "changing from {} to {}"),
        ) if clause
    ]

    if camera_movement_type == "static":
        if is_subject_stationary:
            prompt = "Both the camera and character remain stationary throughout the sequence."
        else:
            prompt = f"While the camera remains stationary, the character {detailed_subject_desc}."
    elif is_subject_stationary:
        camera_clause = f"The camera {camera_desc}"
        if setup_changes:
            camera_clause += f", {' and '.join(setup_changes)},"
        prompt = f"{camera_clause} while the character remains stationary throughout the entire sequence."
    else:
        subject_clause = f"the character {subject_desc}"
        camera_clause = f"the camera {camera_desc}"
        if setup_changes:
            camera_clause += f", {' and '.join(setup_changes)}"
        prompt = _combined_movement_prompt(
            camera_movement_type, subject_movement,
            camera_desc, subject_clause, camera_clause,
        )

    if movement_speed and movement_speed != "constant":
        speed_desc = SPEED_DESCRIPTIONS.get(movement_speed)
        if speed_desc:
            prompt = prompt.rstrip(".") + f", {speed_desc}."

    framing_change = _transition_clause(
        initial.get("subjectFraming"), final.get("subjectFraming"),
        FRAMING_DESCRIPTIONS, "repositioning from {} to {} of the frame",
    )
    if framing_change:
        prompt = prompt.rstrip(".") + f", {framing_change}."

    return prompt
