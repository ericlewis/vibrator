from vibrator.actions import UserAction, action_prefix


def test_action_prefix_default():
    assert action_prefix("unknown") == "User performed unknown on:"


def test_user_action_instruction_known_prefix():
    action = UserAction("write", "Need help with onboarding docs")
    assert action.instruction() == "User wrote: Need help with onboarding docs"


def test_user_action_instruction_custom_action():
    action = UserAction("bookmark", "Saved a tip about async IO")
    assert action.instruction() == "User performed bookmark on: Saved a tip about async IO"
