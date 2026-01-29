

import csv
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio


class TaskType(Enum):
    SENTIMENT = "sentiment"
    TOPIC = "topic"
    CLASSIFICATION = "classification"
    NER = "ner"
    INTENT = "intent"
    QA = "qa"
    EXTRACTION = "extraction"
    OTHER = "other"


@dataclass
class LabelDefinition:
    name: str
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class TaskDefinition:
    task_type: TaskType
    task_description: str
    labels: List[LabelDefinition]
    text_column: str
    input_file: str
    output_column: str = "predicted_label"
    id_column: Optional[str] = None


class TaskParser:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def parse_with_inference(
        self,
        prompt: str,
        csv_path: str,
    ) -> TaskDefinition:
        csv_columns = self._get_csv_columns(csv_path)
        llm_prompt = self._build_llm_prompt(prompt, csv_columns)
        result = await self._call_llm(llm_prompt)
        task_def = self._parse_llm_result(result, csv_path)

        if self.llm_client and task_def.labels:
            task_def.labels = await self._infer_label_descriptions(
                task_def.task_type, task_def.labels
            )

        return task_def

    def _get_csv_columns(self, csv_path: str) -> List[str]:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader.fieldnames) if reader.fieldnames else []
        except Exception:
            return []

    def _build_llm_prompt(self, prompt: str, csv_columns: List[str]) -> str:
        columns_str = ", ".join(csv_columns) if csv_columns else "unknown"
        return f"""Parse task annotation. Respond SHORT JSON.

USER: {prompt}

CSV COLUMNS: {columns_str}

Format:
{{"task":"sentiment|topic|classification|ner|intent","desc":"Mô tả ngắn gọn bằng tiếng Việt","labels":["nhãn1","nhãn2","nhãn3"],"text_col":"tên cột text","id_col":"tên cột id hoặc null","out_col":"tên cột output"}}

Example:
{{"task":"sentiment","desc":"Phân loại cảm xúc","labels":["positive","negative","neutral"],"text_col":"Comment","id_col":"id","out_col":"Predicted_Sentiment"}}

Respond ONLY with JSON."""

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.llm_client is None:
            raise RuntimeError("LLM client not configured")

        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.chat(messages)
        return self._parse_json_response(response.content)

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._try_fix_json(content)

    def _try_fix_json(self, content: str) -> Dict[str, Any]:
        result = {}
        lines = content.split("\n")
        for line in lines:
            line = line.strip().rstrip(",")
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().strip('"')
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                result[key] = value[1:-1]
            elif value.startswith("["):
                result[key] = []
            elif value == "true":
                result[key] = True
            elif value == "false":
                result[key] = False
            elif value == "null":
                result[key] = None
            else:
                try:
                    result[key] = float(value) if "." in value else int(value)
                except ValueError:
                    result[key] = value.strip('"')
        return result if result else {}

    def _parse_llm_result(
        self, result: Dict[str, Any], csv_path: str
    ) -> TaskDefinition:
        task_type_str = result.get(
            "task", result.get("task_type", "classification")
        ).lower()
        task_type_map = {
            "sentiment": TaskType.SENTIMENT,
            "topic": TaskType.TOPIC,
            "classification": TaskType.CLASSIFICATION,
            "ner": TaskType.NER,
            "entity": TaskType.NER,
            "intent": TaskType.INTENT,
            "qa": TaskType.QA,
            "question": TaskType.QA,
            "extraction": TaskType.EXTRACTION,
        }
        task_type = task_type_map.get(task_type_str, TaskType.CLASSIFICATION)

        labels = []
        labels_raw = result.get("labels", [])
        if isinstance(labels_raw, list):
            for item in labels_raw:
                if isinstance(item, str):
                    labels.append(
                        LabelDefinition(name=item, description=f"Label: {item}")
                    )
                elif isinstance(item, dict):
                    labels.append(
                        LabelDefinition(
                            name=item.get("name", "unknown"),
                            description=item.get(
                                "description",
                                item.get(
                                    "desc", f"Label: {item.get('name', 'unknown')}"
                                ),
                            ),
                        )
                    )
        elif isinstance(labels_raw, dict):
            for name, desc in labels_raw.items():
                labels.append(
                    LabelDefinition(
                        name=name, description=desc if desc else f"Label: {name}"
                    )
                )

        text_column = result.get("text_col", result.get("text_column", ""))
        id_column = result.get("id_col", result.get("id_column"))
        output_column = result.get(
            "out_col", result.get("output_column", "predicted_label")
        )

        if text_column:
            text_column = self._validate_column(csv_path, text_column)
        if not text_column:
            text_column = self._get_first_column(csv_path)

        return TaskDefinition(
            task_type=task_type,
            task_description=result.get(
                "desc", result.get("task_description", "Phân loại văn bản")
            ),
            labels=labels,
            text_column=text_column or "text",
            input_file=csv_path,
            output_column=output_column,
            id_column=id_column,
        )

    def _validate_column(self, csv_path: str, column: str) -> str:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for col in reader.fieldnames or []:
                    if col.lower() == column.lower():
                        return col
        except Exception:
            pass
        return ""

    def _get_first_column(self, csv_path: str) -> str:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    return reader.fieldnames[0]
        except Exception:
            pass
        return "text"

    async def _infer_label_descriptions(
        self,
        task_type: TaskType,
        labels: List[LabelDefinition],
    ) -> List[LabelDefinition]:
        labels_needing_desc = [
            l for l in labels if not l.description or l.description.startswith("Label:")
        ]
        if not labels_needing_desc:
            return labels

        label_names = [l.name for l in labels_needing_desc]
        prompt = f"""Cho task {task_type.value}, cung cấp mô tả ngắn gọn cho các nhãn.

Labels: {", ".join(label_names)}

Respond JSON array:
[
  {{"name": "tên nhãn", "desc": "1 câu mô tả bằng tiếng Việt"}}
]

Example sentiment: positive: "Thể hiện sự hài lòng hoặc cảm xúc tích cực"

Respond ONLY with JSON:"""

        try:
            if self.llm_client:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm_client.chat(messages)
                try:
                    desc_map = json.loads(response.content)
                    if isinstance(desc_map, list):
                        for item in desc_map:
                            for label in labels_needing_desc:
                                if label.name == item.get("name"):
                                    label.description = item.get(
                                        "desc", f"Label: {label.name}"
                                    )
                    elif isinstance(desc_map, dict):
                        for label in labels_needing_desc:
                            if label.name in desc_map:
                                label.description = desc_map[label.name]
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        return labels

    def _fallback_parse(self, prompt: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["sentiment", "cảm xúc", "feeling"]):
            task_type = "sentiment"
        elif any(kw in prompt_lower for kw in ["ner", "entity", "thực thể"]):
            task_type = "ner"
        elif any(kw in prompt_lower for kw in ["intent", "ý định"]):
            task_type = "intent"
        elif any(kw in prompt_lower for kw in ["topic", "chủ đề", "danh mục"]):
            task_type = "topic"
        else:
            task_type = "classification"

        labels = []
        label_match = re.search(
            r"[:=]\s*(.+?)(?:\.|,|\s+file|\s+column|\s+cột)", prompt
        )
        if label_match:
            label_str = label_match.group(1)
            for label in re.split(r"[,;]", label_str):
                label = label.strip()
                if label and len(label) < 30:
                    labels.append(
                        {
                            "name": re.sub(r"^\d+[\.\)]\s*", "", label).strip(),
                            "desc": "",
                        }
                    )

        return {
            "task": task_type,
            "desc": f"Task: {task_type}",
            "labels": labels if labels else [{"name": "unknown", "desc": ""}],
            "text_col": "text",
            "out_col": "predicted_label",
        }
