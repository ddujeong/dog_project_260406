def build_analysis_summary(
    breed_primary: str,
    breed_secondary: str,
    breed_primary_prob: float,
    breed_secondary_prob: float,
    breed_confidence: str,
    body_primary: str,
    body_secondary: str,
    body_primary_prob: float,
    body_secondary_prob: float,
    heatmap_region_text: str,
):
    return {
        "breed": {
            "primary": breed_primary,
            "secondary": breed_secondary,
            "primary_prob": round(float(breed_primary_prob), 4),
            "secondary_prob": round(float(breed_secondary_prob), 4),
            "confidence": breed_confidence,
        },
        "body": {
            "primary": body_primary,
            "secondary": body_secondary,
            "primary_prob": round(float(body_primary_prob), 4),
            "secondary_prob": round(float(body_secondary_prob), 4),
        },
        "visual_focus": {
            "region": heatmap_region_text,
        },
    }