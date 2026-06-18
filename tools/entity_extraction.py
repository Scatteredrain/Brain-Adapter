import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


CATEGORY_NAME_TRANSLATIONS = {
    "出血部位": "Hemorrhage Location",
    "出血原因": "Hemorrhage Cause",
    "其他": "Brain Lesions",
}

TAXONOMY_TRANSLATIONS = {
    "基底节": "Basal ganglia",
    "丘脑": "Thalamus",
    "脑室": "Ventricle",
    "小脑": "Cerebellum",
    "脑干": "Brainstem",
    "额颞顶叶/额顶叶/枕叶/额叶/颞叶": "Frontal/Temporal/Parietal/Occipital lobes",
    "蛛网膜下腔": "Subarachnoid space",
    "脑实质内": "Intraparenchymal",
    "硬膜下": "Subdural",
    "硬膜外": "Epidural",
    "高血压": "Hypertension",
    "脑动脉瘤破裂": "Cerebral aneurysm rupture",
    "外伤": "Trauma",
    "其他": "Other",
    "脑积水": "Hydrocephalus",
    "脑疝": "Brain herniation",
    "脑血肿": "Intracerebral hematoma",
    "脑动脉瘤": "Cerebral aneurysm",
    "出血破入脑室": "Hemorrhage into ventricle",
    "脑淀粉样变": "Cerebral amyloidosis",
    "脑血管畸形": "Vascular malformation",
    "脑静脉窦血栓形成": "Cerebral venous sinus thrombosis",
    "脑梗死": "Cerebral infarction",
    "脑肿瘤": "Brain tumor",
    "脑挫伤/颅骨骨折": "Brain contusion/Skull fracture",
    "脑室炎": "Ventriculitis",
    "颅骨缺失": "Skull defect",
    "硬脑膜动静脉瘘": "Dural arteriovenous fistula",
}


def load_schema(schema_path: Path):
    with schema_path.open("r", encoding="utf-8") as f:
        raw_schema = json.load(f)["脑出血分类（新）"]

    english_schema = {}
    ordered_labels = []
    for category_name, labels in raw_schema.items():
        english_category_name = CATEGORY_NAME_TRANSLATIONS.get(category_name, category_name)
        english_labels = [TAXONOMY_TRANSLATIONS.get(label, label) for label in labels]
        english_schema[english_category_name] = english_labels
        ordered_labels.extend(english_labels)
    return english_schema, ordered_labels


def build_prompt(schema):
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "You are a medical information extraction model.\n"
        "Read the following clinical report and determine whether each label in the taxonomy is mentioned (1) or not mentioned (0).\n"
        "Only output valid JSON with integers 0 or 1 for each label and an optional list of other findings.\n"
        "Do not provide explanations.\n\n"
        f"Taxonomy:\n{schema_str}\n\n"
        "Output strictly in this JSON format:\n"
        "{\n"
        '  "出血部位": {"...": 0},\n'
        '  "出血原因": {"...": 0},\n'
        '  "其他": {"...": 0},\n'
        '  "Other findings": []\n'
        "}\n"
    )


def extract_json_payload(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])


def report_from_line(line):
    line = line.strip()
    if "###" not in line:
        return line
    parts = line.split("###")
    if len(parts) >= 3:
        return parts[-3].strip()
    return line


def flatten_multilabel(result_json, schema, ordered_labels):
    flat_results = []
    for category_name, labels in schema.items():
        category_result = result_json.get(category_name, {})
        for label in labels:
            flat_results.append(int(category_result.get(label, 0)))
    if len(flat_results) != len(ordered_labels):
        raise ValueError(f"Flattened label length mismatch: {len(flat_results)} vs {len(ordered_labels)}")
    return flat_results


def build_client(api_key, base_url):
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def classify_report(report_text, schema, ordered_labels, prompt_prefix, model_name, api_key, base_url, timeout):
    client = build_client(api_key, base_url)
    completion = client.chat.completions.create(
        model=model_name,
        timeout=timeout,
        messages=[
            {
                "role": "user",
                "content": f"{prompt_prefix}\nClinical report:\n{report_text}",
            }
        ],
    )
    result_json = extract_json_payload(completion.choices[0].message.content)
    multilabel = flatten_multilabel(result_json, schema, ordered_labels)
    return multilabel, result_json


def classify_report_safe(report_text, schema, ordered_labels, prompt_prefix, model_name, api_key, base_url, timeout):
    try:
        return classify_report(
            report_text, schema, ordered_labels, prompt_prefix, model_name, api_key, base_url, timeout
        )
    except Exception as exc:
        return [0] * len(ordered_labels), {"error": str(exc), "report": report_text}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Brain-Adapter logic-set labels from reports.")
    parser.add_argument("--input", required=True, help="Path to input report file. One report per line.")
    parser.add_argument("--output-labels", required=True, help="Path to output flattened label lists.")
    parser.add_argument("--output-json", required=True, help="Path to output raw JSONL results.")
    parser.add_argument("--schema", default="configs/hemorrhage.json", help="Path to taxonomy schema JSON.")
    parser.add_argument("--model", default="qwen3", help="LLM model name. Defaults to qwen3.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""), help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api-key is required.")

    schema_path = Path(args.schema)
    schema, ordered_labels = load_schema(schema_path)
    prompt_prefix = build_prompt(schema)

    with open(args.input, "r", encoding="utf-8") as f:
        reports = [report_from_line(line) for line in f if line.strip()]

    output_labels = [None] * len(reports)
    output_json = [None] * len(reports)

    if args.workers <= 1:
        for idx, report in enumerate(reports):
            output_labels[idx], output_json[idx] = classify_report_safe(
                report, schema, ordered_labels, prompt_prefix, args.model, args.api_key, args.base_url, args.timeout
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    classify_report_safe,
                    report,
                    schema,
                    ordered_labels,
                    prompt_prefix,
                    args.model,
                    args.api_key,
                    args.base_url,
                    args.timeout,
                ): idx
                for idx, report in enumerate(reports)
            }
            for future in as_completed(futures):
                idx = futures[future]
                output_labels[idx], output_json[idx] = future.result()

    with open(args.output_labels, "w", encoding="utf-8") as f:
        for label_vector in output_labels:
            f.write(f"{label_vector}\n")

    with open(args.output_json, "w", encoding="utf-8") as f:
        for item in output_json:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(reports)} reports with model={args.model}.")
    print(f"Output labels -> {args.output_labels}")
    print(f"Output json   -> {args.output_json}")


if __name__ == "__main__":
    main()
