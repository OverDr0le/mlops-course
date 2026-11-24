"""Сверка /vision с EfficientNet B0 из torchvision, выполняется на CPU."""
import argparse
import os
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def main(args):
    image_path = Path(args.image)
    data = image_path.read_bytes()
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights).eval().cpu()
    with Image.open(BytesIO(data)) as opened:
        x = weights.transforms()(opened.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1)[0]

    r = httpx.post(args.base.rstrip("/") + "/vision",
                   headers={"Authorization": f"Bearer {args.token}"},
                   files={"file": (image_path.name, data)}, timeout=60, trust_env=False)
    r.raise_for_status()
    result = r.json()
    predictions = result["predictions"]
    assert result["model"] == "efficientnet_b0", result
    assert len(predictions) == 5, result
    scores = [item["score"] for item in predictions]
    assert scores == sorted(scores, reverse=True), result
    ids = [item["class_id"] for item in predictions]
    assert len(set(ids)) == 5 and all(isinstance(i, int) and 0 <= i < 1000 for i in ids)
    print("torchvision top-5:")
    for i in probabilities.topk(5).indices.tolist():
        print(f"  {i:3d} {weights.meta['categories'][i]:25s} {float(probabilities[i]):.6f}")
    print("gateway top-5 (сравнение с torchvision для каждого class_id):")
    labels = Path("gateway/imagenet_classes.txt").read_text(encoding="utf-8").splitlines()
    for item in predictions:
        i = item["class_id"]
        expected = float(probabilities[i])
        difference = abs(item["score"] - expected)
        print(f"  {i:3d} {item['label']:25s} {item['score']:.6f} "
              f"reference={expected:.6f} difference={difference:.6f}")
        assert 0 <= item["score"] <= 1 and difference <= .02, item
        assert item["label"] == labels[i], item
    assert predictions[0]["class_id"] == int(probabilities.argmax()), result
    print("PASS: top-1 совпадает, скоры top-5 отличаются не более чем на 0.02")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="cat.jpg")
    parser.add_argument("--base", default=os.getenv("BASE", "http://127.0.0.1:8080"))
    parser.add_argument("--token", default=os.getenv("TOKEN"))
    args = parser.parse_args()
    if not args.token:
        parser.error("Задай TOKEN в окружении или передай --token")
    main(args)