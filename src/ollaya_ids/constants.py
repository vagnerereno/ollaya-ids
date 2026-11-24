LABEL_COLUMN = "class"
NORMAL_LABEL = "normal"

CLASS_DESCRIPTIONS = {
    "normal": "Legitimate IEC 61850 GOOSE and sampled-value activity.",
    "random_replay": (
        "A previously captured GOOSE message retransmitted at an unrelated or random time."
    ),
    "inverse_replay": (
        "A captured GOOSE message replayed when its status is opposite to the expected state."
    ),
    "masquerade_fake_fault": (
        "Masqueraded GOOSE traffic that reports a false electrical fault during normal operation."
    ),
    "masquerade_fake_normal": (
        "Masqueraded GOOSE traffic that reports normal operation while a fault is present."
    ),
    "injection": "A forged GOOSE message with injected or manipulated field values.",
    "high_StNum": "A GOOSE message carrying an anomalously high status number.",
    "poisoned_high_rate": "Poisoned GOOSE messages transmitted at a flooding rate.",
    "grayhole": "Selective suppression or dropping of GOOSE messages.",
}

DECISION_FEATURES = [
    "SqNum",
    "StNum",
    "cbStatus",
    "goID",
    "timestampDiff",
    "tDiff",
    "timeFromLastChange",
    "delay",
    "stDiff",
    "sqDiff",
    "gooseLengthDiff",
    "cbStatusDiff",
    "apduSizeDiff",
    "frameLengthDiff",
]

QUESTION = {
    "attack_type": {
        "type": "choice",
        "instructions": (
            "Classify this IEC 61850 smart-grid observation as legitimate traffic or one of "
            "the listed cyberattacks. Use both protocol and timing evidence."
        ),
        "criteria": CLASS_DESCRIPTIONS,
    }
}
