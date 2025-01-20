from typing import Dict, List, Any, Optional

camera_vertical_angle_descriptions = {
    "low": "from a low angle",
    "eye": "from an eye-level angle",
    "high": "from a high angle",
    "overhead": "from an overhead angle",
    "birdsEye": "from a bird's-eye angle",
}

shot_size_descriptions = {
    "extremeCloseUp": "an extreme close-up shot",
    "closeUp": "a close-up shot",
    "mediumCloseUp": "a medium close-up shot",
    "mediumShot": "a medium shot",
    "fullShot": "a full shot",
    "longShot": "a long shot",
    "veryLongShot": "a very long shot",
    "extremeLongShot": "an extreme long shot",
}

subject_view_descriptions = {
    "front": "front view",
    "back": "back view",
    "left": "left side",
    "right": "right side",
    "threeQuarterFrontLeft": "three-quarter front-left",
    "threeQuarterFrontRight": "three-quarter front-right",
    "threeQuarterBackLeft": "three-quarter back-left",
    "threeQuarterBackRight": "three-quarter back-right",
}

subject_in_frame_position_descriptions = {
    "left": "the left of the frame",
    "right": "the right of the frame",
    "top": "the top of the frame",
    "bottom": "the bottom of the frame",
    "center": "the center of the frame",
    "topLeft": "the top-left of the frame",
    "topRight": "the top-right of the frame",
    "bottomLeft": "the bottom-left of the frame",
    "bottomRight": "the bottom-right of the frame",
    "outerLeft": "far left",
    "outerRight": "far right",
    "outerTop": "the top edge",
    "outerBottom": "the bottom edge",
}

movement_easing_descriptions = {
    "linear": "Constant speed with no acceleration or deceleration.",

    "easeInSine": "Gentle acceleration following a sine curve.",
    "easeOutSine": "Gentle deceleration following a sine curve.",
    "easeInOutSine": "Gentle acceleration and deceleration using sine curves.",

    "easeInQuad": "Moderate acceleration using a square power curve.",
    "easeOutQuad": "Moderate deceleration using a square power curve.",
    "easeInOutQuad": "Moderate acceleration and deceleration using square power curves.",

    "easeInCubic": "Strong acceleration using a cubic power curve.",
    "easeOutCubic": "Strong deceleration using a cubic power curve.",
    "easeInOutCubic": "Strong acceleration and deceleration using cubic power curves.",

    "easeInQuart": "Stronger acceleration using a quartic power curve.",
    "easeOutQuart": "Stronger deceleration using a quartic power curve.",
    "easeInOutQuart": "Stronger acceleration and deceleration using quartic power curves.",

    "easeInQuint": "Very strong acceleration using a quintic power curve.",
    "easeOutQuint": "Very strong deceleration using a quintic power curve.",
    "easeInOutQuint": "Very strong acceleration and deceleration using quintic power curves.",

    "easeInExpo": "Dramatic acceleration using exponential growth.",
    "easeOutExpo": "Dramatic deceleration using exponential decay.",
    "easeInOutExpo": "Dramatic acceleration and deceleration using exponential curves.",

    "easeInCirc": "Gradual start with quick acceleration using circular math.",
    "easeOutCirc": "Maintains speed longer before quickly settling using circular math.",
    "easeInOutCirc": "Mechanical-feeling acceleration and deceleration using circular math.",

    "easeInBack": "Slight backwards motion before accelerating forward.",
    "easeOutBack": "Overshoots target before settling back.",
    "easeInOutBack": "Combines backwards motion and overshooting at both ends.",

    "easeInElastic": "Multiple increasing backwards bounces before springing forward.",
    "easeOutElastic": "Multiple decreasing forward bounces before settling.",
    "easeInOutElastic": "Elastic bouncing at both start and end.",

    "easeInBounce": "Multiple small bounces that build up momentum.",
    "easeOutBounce": "Multiple decreasing bounces until settling.",
    "easeInOutBounce": "Multiple bounces at both start and end.",

    "handHeld": "Random subtle variations simulating handheld camera movement.",
    "anticipation": "Small backwards motion before smooth acceleration forward.",
    "smooth": "Natural-feeling subtle acceleration and deceleration."
}

camera_movement_type_descriptions = {
    "static": "static (no movement)",

    "follow": "following the subject",
    "track": "tracking the subject",

    "dollyIn": "dolly in",
    "dollyOut": "dolly out",

    "panLeft": "pan left",
    "panRight": "pan right",
    "tiltUp": "tilt up",
    "tiltDown": "tilt down",

    "truckLeft": "truck left",
    "truckRight": "truck right",
    "pedestalUp": "pedestal up",
    "pedestalDown": "pedestal down",

    "arcLeft": "arc left",
    "arcRight": "arc right",

    "craneUp": "crane up",
    "craneDown": "crane down",

    "dollyOutZoomIn": "dolly out while zooming in",
    "dollyInZoomOut": "dolly in while zooming out",

    "dutchLeft": "dutch tilt to the left",
    "dutchRight": "dutch tilt to the right",
}

movement_speed_descriptions = {
    "slowToFast": "accelerating from slow to fast",
    "fastToSlow": "decelerating from fast to slow",
    "constant": "at a constant speed",
    "smoothStartStop": "with a smooth start and stop motion",
}

scale_descriptions = {
    "small": "on a small scale",
    "medium": "on a medium scale",
    "large": "on a large scale",
    "full": "on a full scale",
}

direction_descriptions = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
}

movement_mode_descriptions = {
    "transition": "transition",
    "rotation": "rotation",
}


def generate_setup_text(setup_config: Dict[str, Any]) -> Optional[str]:
    if not setup_config:
        return None

    angle = camera_vertical_angle_descriptions.get(
        setup_config.get("cameraAngle"), None
    )
    shot = shot_size_descriptions.get(
        setup_config.get("shotSize"), None
    )
    subject_view = subject_view_descriptions.get(
        setup_config.get("subjectView"), None
    )

    if not angle and not shot and not subject_view and not setup_config.get("subjectFraming"):
        return None

    angle = angle if angle else "an unknown angle"
    shot = shot if shot else "a medium shot"
    subject_view = subject_view if subject_view else "front view"

    framing_conf = setup_config.get("subjectFraming", {})
    framing_pos = framing_conf.get("position")
    if framing_pos:
        framing_str = subject_in_frame_position_descriptions.get(
            framing_pos, "center of the frame"
        )
        framing_text = f" The subject is positioned at {framing_str}."
    else:
        framing_text = ""

    return f"This setup is {shot} {angle} with the subject in {subject_view}.{framing_text}"


def generate_instruction_movement_text(dynamic_config: Dict[str, Any]) -> Optional[str]:
    if not dynamic_config:
        return None

    if dynamic_config.get("type") == "interpolation":
        easing = movement_easing_descriptions.get(
            dynamic_config.get("easing", "linear"),
            "with a linear ease"
        )
        return f"Interpolating camera movement {easing}."
    else:
        scale_key = dynamic_config.get("scale")
        direction_key = dynamic_config.get("direction")
        movement_mode_key = dynamic_config.get("movementMode")
        easing_key = dynamic_config.get("easing")

        if not scale_key and not direction_key and not movement_mode_key and not easing_key:
            return None

        scale_str = scale_descriptions.get(scale_key, "on a medium scale")
        direction_str = direction_descriptions.get(direction_key, "right")
        movement_mode_str = movement_mode_descriptions.get(
            movement_mode_key, "transition")
        easing_str = movement_easing_descriptions.get(
            easing_key, "with a linear ease")

        return (
            f"A simple {movement_mode_str} to the {direction_str}, {scale_str}, {easing_str}."
        )


def generate_constraints_text(constraints_config: Dict[str, Any]) -> Optional[str]:
    if not constraints_config:
        return None

    desc = []
    if constraints_config.get("allFramesVisibility"):
        desc.append("maintaining subject visibility in all frames")
    if constraints_config.get("staticDistance"):
        desc.append("keeping a fixed distance to the subject")
    if constraints_config.get("staticCameraSubjectRotation"):
        desc.append("no camera rotation around the subject")

    locked = constraints_config.get("lockedMovement")
    if locked:
        locked_desc = []
        for direction, is_locked in locked.items():
            if is_locked:
                locked_desc.append(f"{direction} movement locked")
        if locked_desc:
            desc.append(" and ".join(locked_desc))

    locked_rot = constraints_config.get("lockedRotation")
    if locked_rot:
        locked_rot_desc = []
        for direction, is_locked in locked_rot.items():
            if is_locked:
                locked_rot_desc.append(f"{direction} rotation locked")
        if locked_rot_desc:
            desc.append(" and ".join(locked_rot_desc))

    if not desc:
        return None

    return "Constraints: " + ", ".join(desc) + "."


def generate_simulation_sentences(instruction: Dict[str, Any]) -> List[Optional[str]]:
    initial_setup = generate_setup_text(instruction.get("initialSetup", {}))

    dynamic_conf = instruction.get("dynamic", {})
    movement_str = generate_instruction_movement_text(dynamic_conf)

    end_setup_conf = {}
    if dynamic_conf.get("type") == "interpolation":
        end_setup_conf = dynamic_conf.get("endSetup", {})
    end_setup = generate_setup_text(end_setup_conf) if end_setup_conf else None

    constraints = generate_constraints_text(instruction.get("constraints", {}))

    return {
        "init_setup": initial_setup,
        "movement": movement_str,
        "end_setup": end_setup,
        "constraints": constraints,
    }


def generate_cinematography_sentences(prompt: Dict[str, Any]) -> List[Optional[str]]:
    initial_conf = prompt.get("initial", {})
    initial_text = generate_setup_text({
        "cameraAngle": initial_conf.get("cameraAngle"),
        "shotSize": initial_conf.get("shotSize"),
        "subjectView": initial_conf.get("subjectView"),
        "subjectFraming": {"position": initial_conf.get("subjectFraming")},
    })

    mov_conf = prompt.get("movement", {})
    mov_type = mov_conf.get("type")
    mov_speed = mov_conf.get("speed")

    if not mov_type and not mov_speed:
        movement_text = None
    else:
        camera_move_text = camera_movement_type_descriptions.get(
            mov_type, mov_type if mov_type else "static")
        speed_text = movement_speed_descriptions.get(
            mov_speed, mov_speed if mov_speed else "constant")
        movement_text = f"The camera does a {camera_move_text}, {speed_text}."

    final_conf = prompt.get("final", {})
    if final_conf:
        final_text = generate_setup_text({
            "cameraAngle": final_conf.get("cameraAngle"),
            "shotSize": final_conf.get("shotSize"),
            "subjectView": final_conf.get("subjectView"),
            "subjectFraming": {"position": final_conf.get("subjectFraming")},
        })
    else:
        final_text = None

    return {
        "init_setup": initial_text,
        "movement": movement_text,
        "end_setup": final_text,
    }
