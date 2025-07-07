import os
import glob
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="...",  # replace with your actual key (If you need a key to test this project only, please inform me!)
)

def extract_labels_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    true_class = parts[1].split("-")[1]
    pred_class = parts[2].split("-")[1].split(".")[0]
    return true_class, pred_class

def explain_image_with_llm(image_url, true_label, pred_label):
    prompt = f"""
This is a Grad-CAM visualization over a satellite image.
- Predicted class: {pred_label}
- True class: {true_label}
Please analyze the model's attention pattern. 
If correct, explain what the model might have seen.
If incorrect, suggest possible confusion reasons.
Keep your result very straightforward to the point without extra explanation and words! 
just the explainability!!
"""
    completion = client.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    IMAGE_BASE_URL = "https://raw.githubusercontent.com/ell-hosse/geoxai-eurosat/main/results/gradcam"


    image_paths = glob.glob("results/gradcam/*/*.png")

    for path in image_paths:
        true_label, pred_label = extract_labels_from_filename(path)
        relative_path = os.path.relpath(path, "results/gradcam")
        image_url = f"{IMAGE_BASE_URL}/{relative_path.replace(os.sep, '/')}"

        print(f"Explaining: {relative_path} (true: {true_label}, pred: {pred_label})")
        try:
            explanation = explain_image_with_llm(image_url, true_label, pred_label)
            explanation_path = path.replace(".png", "_explanation.txt")
            with open(explanation_path, "w", encoding="utf-8") as f:
                f.write(explanation)
            print(f"Saved explanation to {explanation_path}")
        except Exception as e:
            print(f"Failed for {path}: {e}")
