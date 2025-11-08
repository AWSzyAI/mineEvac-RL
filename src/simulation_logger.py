import json
from pathlib import Path
from typing import Iterable, Optional


class SimulationLogger:
    def __init__(self, save_path: str = "logs/trajectories.jsonl"):
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        # 使用文本模式写入 JSON Lines
        self.file = open(save_path, "w", encoding="utf-8")

    def record(self, t: int, occupants: Iterable[object], responders: Optional[Iterable[object]] = None):
        data = {
            "time": t,
            "occupants": [
                {
                    "id": getattr(o, "id", None),
                    "x": float(getattr(o, "pos", getattr(o, "position", (None, None)))[0]),
                    "y": float(getattr(o, "pos", getattr(o, "position", (None, None)))[1]),
                    "state": getattr(o, "state", None),
                }
                for o in occupants
            ],
        }
        if responders is not None:
            data["responders"] = [
                {
                    "id": getattr(r, "id", None),
                    "x": float(getattr(r, "pos", getattr(r, "position", (None, None)))[0]),
                    "y": float(getattr(r, "pos", getattr(r, "position", (None, None)))[1]),
                    "role": getattr(r, "role", None),
                }
                for r in responders
            ]
        self.file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
