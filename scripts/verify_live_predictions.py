import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.predict import predict_image

specimens = [
    ("CIFAKE REAL Sample (Airplane)", "samples/real/01_real_airplane.jpg", "REAL"),
    ("CIFAKE FAKE Sample (Mythical Bird)", "samples/ai/02_ai_diffused_fantasy_mythical_bird.jpg", "AI"),
    ("Known Genuine Portrait (Real)", "tests/regression/known_real/verified_real_portrait_1.jpg", "REAL"),
    ("Known AI-Generated Portrait 1 (ONGC)", "tests/regression_data/ai_portrait_ongc_1.png", "AI"),
    ("Known AI-Generated Portrait 2 (ONGC)", "tests/regression_data/ai_portrait_ongc_2.png", "AI"),
]

results = []
for name, path, gt in specimens:
    res = predict_image(path)
    raw_p = res.get("raw_probabilities", res.get("probabilities", {}))
    item = {
        "specimen_name": name,
        "path": path,
        "ground_truth": gt,
        "verdict": res.get("verdict"),
        "label": res.get("label"),
        "primary_driver": res.get("primary_driver"),
        "is_real": res.get("is_real"),
        "is_ai": res.get("is_ai"),
        "is_ood": res.get("is_ood"),
        "ood_score": res.get("ood_score"),
        "raw_probabilities": raw_p,
        "calibrated_confidence": res.get("confidence"),
        "confidence_percent": res.get("confidence_percent"),
        "raw_score": res.get("raw_score"),
        "architecture": res.get("architecture"),
        "correct": (res.get("is_real") and gt == "REAL") or (res.get("is_ai") and gt == "AI")
    }
    results.append(item)
    print(f"=== {name} ===")
    print(f"  Ground Truth:         {gt}")
    print(f"  Model Verdict:        {item['verdict']}")
    print(f"  Raw P(REAL):          {raw_p['REAL']:.6f}")
    print(f"  Raw P(AI):            {raw_p['AI']:.6f}")
    print(f"  Calibrated Conf:      {item['confidence_percent']}")
    print(f"  OOD Score / Is OOD:   {item['ood_score']:.4f} / {item['is_ood']}")
    print(f"  Primary Driver:       {item['primary_driver']}")
    print(f"  Correct:              {item['correct']}\n")

out_file = Path("model/verified_prediction_samples.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved verified specimen results to {out_file}")
