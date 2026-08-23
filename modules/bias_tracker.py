import json
from datetime import datetime

class PipelineLogger:
    def __init__(self, log_path="eval/run_log.json"):
        self.log_path = log_path
        self.entries = []
    
    def log_run(self, brief, profile, image_path, fidelity_scores, cdvr_iterations):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "brief_snippet": brief[:100],
            "profile": profile,
            "image_path": image_path,
            "fidelity": fidelity_scores,
            "cdvr_iterations_needed": cdvr_iterations
        }
        self.entries.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)