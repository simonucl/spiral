from spiral.envs.SimpleNegotiation.five_resource_env import \
    SimpleNegotiationFiveResourceEnv


def test_parse_offer():
    # Create an instance (assuming you have the class)
    env = SimpleNegotiationFiveResourceEnv()  # Replace with actual class name

    # Test cases with expected results - ONLY those that match the regex pattern
    test_cases = [
        # Basic [Offer: ...] format
        (
            "[Offer: 1 Wood -> 2 Sheep]",
            {"offered_resources": {"Wood": 1}, "requested_resources": {"Sheep": 2}},
        ),
        # Multiple resources with space separation
        (
            "[Offer: 1 Wood 2 Brick -> 2 Sheep 1 Gold]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2},
                "requested_resources": {"Sheep": 2, "Gold": 1},
            },
        ),
        # Multiple resources with + separation
        (
            "[Offer: 1 Wood + 2 Brick -> 2 Sheep + 1 Gold]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2},
                "requested_resources": {"Sheep": 2, "Gold": 1},
            },
        ),
        # Multiple resources with comma separation
        (
            "[Offer: 1 Wood, 2 Brick -> 2 Sheep, 1 Gold]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2},
                "requested_resources": {"Sheep": 2, "Gold": 1},
            },
        ),
        # Multiple resources with "and" separation
        (
            "[Offer: 1 Wood and 2 Brick -> 2 Sheep and 1 Gold]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2},
                "requested_resources": {"Sheep": 2, "Gold": 1},
            },
        ),
        # Mixed separators
        (
            "[Offer: 1 Wood + 2 Brick, 1 Gold -> 2 Sheep and 1 Brick]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2, "Gold": 1},
                "requested_resources": {"Sheep": 2, "Brick": 1},
            },
        ),
        # Case insensitive
        (
            "[offer: 1 wood -> 2 sheep]",
            {"offered_resources": {"Wood": 1}, "requested_resources": {"Sheep": 2}},
        ),
        # With extra whitespace
        (
            "[Offer:   1 Wood   ->   2 Sheep  ]",
            {"offered_resources": {"Wood": 1}, "requested_resources": {"Sheep": 2}},
        ),
        # Large quantities
        (
            "[Offer: 10 Wood 25 Brick -> 100 Sheep 5 Gold]",
            {
                "offered_resources": {"Wood": 10, "Brick": 25},
                "requested_resources": {"Sheep": 100, "Gold": 5},
            },
        ),
        # Resource aliases - plural forms
        (
            "[Offer: 1 Woods -> 2 Sheeps]",
            {"offered_resources": {"Wood": 1}, "requested_resources": {"Sheep": 2}},
        ),
        (
            "[Offer: 2 Bricks 1 Golds -> 3 Wheats]",
            {
                "offered_resources": {"Brick": 2, "Gold": 1},
                "requested_resources": {"Wheat": 3},
            },
        ),
        # Mixed singular and plural
        (
            "[Offer: 1 Wood 2 Bricks -> 1 Sheep 2 Golds]",
            {
                "offered_resources": {"Wood": 1, "Brick": 2},
                "requested_resources": {"Sheep": 1, "Gold": 2},
            },
        ),
        # All aliases with different separators
        (
            "[Offer: 1 Woods + 2 Sheeps -> 3 Bricks, 1 Golds and 2 Wheats]",
            {
                "offered_resources": {"Wood": 1, "Sheep": 2},
                "requested_resources": {"Brick": 3, "Gold": 1, "Wheat": 2},
            },
        ),
        # Case insensitive aliases
        (
            "[Offer: 1 woods -> 2 sheeps]",
            {"offered_resources": {"Wood": 1}, "requested_resources": {"Sheep": 2}},
        ),
    ]

    # Edge cases that WILL match regex but should return None in _parse_offer
    error_cases = [
        # Missing arrow
        "[Offer: 1 Wood 2 Sheep]",
        # Multiple arrows
        "[Offer: 1 Wood -> 2 Sheep -> 1 Gold]",
        # No resources offered
        "[Offer: -> 2 Sheep]",
        # No resources requested
        "[Offer: 1 Wood ->]",
        # Invalid quantity (non-numeric)
        "[Offer: abc Wood -> 2 Sheep]",
        # Missing quantity
        "[Offer: Wood -> 2 Sheep]",
        # Missing resource name
        "[Offer: 1 -> 2 Sheep]",
        # Only offer prefix
        "[Offer:]",
        # Invalid resource format
        "[Offer: 1 Wood 2 -> 2 Sheep]",
        # Zero quantities
        "[Offer: 0 Wood -> 2 Sheep]",
        # Negative quantities
        "[Offer: -1 Wood -> 2 Sheep]",
        # Wrong resource name
        "[Offer: 1 Wood -> 2 Stone]",
    ]

    # Cases that WON'T match the regex pattern (never reach _parse_offer)
    regex_no_match_cases = [
        # Without brackets
        "Offer: 1 Wood -> 2 Sheep",
        # Legacy format with "I offer"
        "I offer 1 Wood -> 2 Sheep",
        # Legacy format with "I give"
        "I give 1 Wood -> 2 Sheep",
        # Empty string
        "",
        # Only whitespace
        "   ",
        # Only brackets
        "[]",
        # Malformed brackets
        "[Offer: 1 Wood -> 2 Sheep",
    ]

    print("Testing valid cases:")
    for i, (input_str, expected) in enumerate(test_cases):
        result = env._parse_offer(input_str)
        if result == expected:
            print(f"✅ Test {i+1}: PASS")
        else:
            print(f"❌ Test {i+1}: FAIL")
            print(f"   Input: {input_str}")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")

    print("\nTesting error cases (should return None):")
    for i, input_str in enumerate(error_cases):
        result = env._parse_offer(input_str)
        if result is None:
            print(f"✅ Error test {i+1}: PASS (correctly returned None)")
        else:
            print(f"❌ Error test {i+1}: FAIL (should return None)")
            print(f"   Input: {input_str}")
            print(f"   Got: {result}")

    print("\nTesting cases that won't match regex (should return None):")
    for i, input_str in enumerate(regex_no_match_cases):
        match = env.offer_pattern.match(input_str)
        if match is None:
            print(f"✅ Regex test {i+1}: PASS (correctly returned None)")
        else:
            print(f"❌ Regex test {i+1}: FAIL (should return None)")
            print(f"   Input: {input_str}")
            print(f"   Got: {match}")


# Run the tests
test_parse_offer()
