"""Optimized MAFA - 4 agents × batch processing.

API calls for 20 samples:
- Tier 1: 1 call (batch 20)
- Tier 2: 4 calls (4 agents × batch 20)
- Tier 3: 1 call (batch consensus)
Total: 6 calls (vs 120 calls original)
"""

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, get_llm_client
from src.agents.tier1.query_expander import QueryExpander
from src.agents.tier3.judge import JudgeAgent
from src.agents.tier4.review import ReviewQueue


DATA_DIR = Path(__file__).parent.parent / "data"


class BatchMAFA:
    def __init__(self, calls_per_minute: int = 40):
        self.config = get_config()
        self.delay = 60.0 / calls_per_minute
        self.last_call_time = 0
        self.llm_client = get_llm_client(self.config)

        print(f"\n{'=' * 60}")
        print(f"Batch MAFA (6 calls per 20 samples)")
        print(f"{'=' * 60}")
        print(f"Provider: {self.config.provider.type}")

        try:
            csv_path = DATA_DIR / "train.csv"
            self.query_expander = (
                QueryExpander(str(csv_path)) if csv_path.exists() else None
            )
        except:
            self.query_expander = None

        self.judge = JudgeAgent()
        self.review_queue = ReviewQueue()

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call_time = time.time()

    def _parse(self, content: str, n: int) -> List[Dict]:
        """Parse JSON from model response. Handles Vietnamese text + JSON format."""
        import re

        # Find JSON array - look for ```json or just [
        json_start = content.find("```json")
        if json_start == -1:
            json_start = content.find("```")
        if json_start == -1:
            json_start = content.find("[")

        if json_start != -1:
            json_content = content[json_start:]
            # Remove markdown code block
            if json_content.startswith("```"):
                lines = json_content.split("\n")
                if len(lines) > 2:
                    json_str = "\n".join(lines[1:-1])
                    if json_str.strip().endswith("```"):
                        json_str = json_str.strip()[:-3]
                else:
                    json_str = json_content
            else:
                json_str = json_content

            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    results = []
                    for i, item in enumerate(data[:n]):
                        label = item.get("label") or item.get("topic") or "0"
                        confidence = item.get("confidence") or 0.5
                        results.append(
                            {
                                "label": str(label),
                                "confidence": float(confidence),
                            }
                        )
                    return results
            except json.JSONDecodeError:
                pass

        # Fallback: regex extract labels and confidence
        labels = re.findall(r'["\']?label["\']?\s*:\s*["\']?(\d+)["\']?', content)
        confs = re.findall(r'["\']?confidence["\']?\s*:\s*([0-9.]+)', content)

        if labels:
            results = []
            for i in range(min(n, len(labels))):
                results.append(
                    {
                        "label": labels[i],
                        "confidence": float(confs[i]) if i < len(confs) else 0.5,
                    }
                )
            return results

        return [{"label": "0", "confidence": 0.5} for _ in range(n)]

    async def _tier1_batch(self, texts: List[str]) -> List[str]:
        prompt = f"Mở rộng query cho {len(texts)} comments:\n"
        for i, t in enumerate(texts):
            prompt += f'{i + 1}. "{t}"\n'
        prompt += '\nJSON: [{"expanded": "..."}, ...]'

        self._rate_limit()
        resp = await self.llm_client.chat([{"role": "user", "content": prompt}])
        content = resp.content.strip()

        try:
            if content.startswith("```json"):
                content = content[7:-3]
            data = json.loads(content)
            if isinstance(data, list):
                return [item.get("expanded", texts[i]) for i, item in enumerate(data)]
        except:
            pass
        return texts

    async def _agent_batch(self, texts: List[str], agent_name: str) -> List[Dict]:
        prompt = f"""STRICTLY FOLLOW THIS FORMAT:

[
  {{"label": "0", "confidence": 0.95}},
  {{"label": "1", "confidence": 0.8}},
  ...
]

DO NOT add any text, explanation, or markdown. Only output the JSON array above.

Task: Classify {len(texts)} comments as 0 (non-toxic) or 1 (toxic).

"""
        for i, t in enumerate(texts):
            prompt += f'{i + 1}. "{t}"\n'

        prompt += "\nOutput ONLY JSON array above, nothing else."

        self._rate_limit()
        resp = await self.llm_client.chat([{"role": "user", "content": prompt}])
        return self._parse(resp.content, len(texts))

    async def process_batch(self, texts: List[str], batch_num: int = 1) -> List[Dict]:
        n = len(texts)
        results = []

        # ===== TIER 1 =====
        print(f"  [Tier 1] 1 call for {n} samples...")
        expanded = await self._tier1_batch(texts)
        for i, t in enumerate(texts):
            results.append(
                {
                    "text": t,
                    "expanded": expanded[i],
                    "batch": batch_num,
                    "task_id": f"b{batch_num}_{i}",
                }
            )

        # ===== TIER 2: 4 AGENTS =====
        print(f"  [Tier 2] Agent 1/4...")
        primary = await self._agent_batch(texts, "primary")

        print(f"  [Tier 2] Agent 2/4...")
        contextual = await self._agent_batch(texts, "contextual")

        print(f"  [Tier 2] Agent 3/4...")
        retrieval = await self._agent_batch(texts, "retrieval")

        print(f"  [Tier 2] Agent 4/4...")
        hybrid = await self._agent_batch(texts, "hybrid")

        for i, r in enumerate(results):
            r["tier2"] = {
                "primary": primary[i],
                "contextual": contextual[i],
                "retrieval": retrieval[i],
                "hybrid": hybrid[i],
            }

        # ===== TIER 3 =====
        print(f"  [Tier 3] Judge consensus (1 call)...")
        prompt = f"""Tổng hợp {n} kết quả từ 4 agents.

Labels: 0=Không toxic, 1=Toxic

KẾT QUẢ:
"""
        for i, r in enumerate(results):
            p = r["tier2"]["primary"]["label"]
            c = r["tier2"]["contextual"]["label"]
            re = r["tier2"]["retrieval"]["label"]
            h = r["tier2"]["hybrid"]["label"]
            prompt += f"{i + 1}. P:{p}, C:{c}, R:{re}, H:{h}\n"

        prompt += """\nDecision: >=0.85 approve, <0.60 escalate, else review
JSON: [{"label": "0", "decision": "approve"}, ...]"""

        self._rate_limit()
        resp = await self.llm_client.chat([{"role": "user", "content": prompt}])
        content = resp.content.strip()

        try:
            if content.startswith("```json"):
                content = content[7:-3]
            decisions = json.loads(content)
        except:
            decisions = [{"label": "0", "decision": "review"} for _ in range(n)]

        for i, (r, decision) in enumerate(zip(results, decisions)):
            confs = [
                r["tier2"]["primary"]["confidence"],
                r["tier2"]["contextual"]["confidence"],
                r["tier2"]["retrieval"]["confidence"],
                r["tier2"]["hybrid"]["confidence"],
            ]
            avg_conf = sum(confs) / 4

            if avg_conf >= 0.85:
                decision_val = "approve"
            elif avg_conf < 0.60:
                decision_val = "escalate"
            else:
                decision_val = "review"

            r["tier3"] = {
                "label": decision.get("label", "0"),
                "confidence": avg_conf,
                "decision": decision_val,
            }

            # ===== TIER 4 =====
            try:
                self.review_queue.add(
                    task_id=r["task_id"],
                    original_text=r["text"],
                    annotation=r["tier3"],
                    consensus_score=avg_conf,
                )
            except:
                pass

        return results


async def run(
    input_file: str = str(DATA_DIR / "train.csv"),
    output_file: str = str(DATA_DIR / "batch_mafa_results.json"),
    batch_size: int = 20,
    max_samples: int = 0,
    calls_per_minute: int = 40,
):
    mafa = BatchMAFA(calls_per_minute)

    texts = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("Comment", "").strip()
            if t:
                texts.append(t)
            if max_samples > 0 and len(texts) >= max_samples:
                break

    print(f"\nLoaded {len(texts)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Rate: {calls_per_minute}/min (1 call per {60 / calls_per_minute:.1f}s)")
    print(f"{'=' * 60}\n")

    all_results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_num = start // batch_size + 1
        total = (len(texts) + batch_size - 1) // batch_size

        print(f"\nBatch {batch_num}/{total} ({len(batch)} samples)")
        results = await mafa.process_batch(batch, batch_num)
        all_results.extend(results)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(all_results)} total")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    decisions = {"approve": 0, "review": 0, "escalate": 0}
    for r in all_results:
        d = r["tier3"].get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1

    total = len(all_results)
    print(f"Total: {total}")
    print(
        f"Approve: {decisions['approve']} ({decisions['approve'] / total * 100:.1f}%)"
    )
    print(f"Review: {decisions['review']} ({decisions['review'] / total * 100:.1f}%)")
    print(
        f"Escalate: {decisions['escalate']} ({decisions['escalate'] / total * 100:.1f}%)"
    )
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/train.csv")
    parser.add_argument("--output", "-o", default="data/batch_mafa_results.json")
    parser.add_argument("--batch-size", "-b", type=int, default=20)
    parser.add_argument("--max", "-m", type=int, default=0)
    parser.add_argument("--rate", "-r", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.batch_size, args.max, args.rate))
