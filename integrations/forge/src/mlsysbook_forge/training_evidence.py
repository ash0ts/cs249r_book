"""Read explicit printed epoch records without treating them as benchmark scores."""
import hashlib
import re

NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
EPOCH = re.compile(rf"Epoch\s+(\d+)/(\d+)\s+Loss:\s*({NUMBER})\s+Train:\s*({NUMBER})%\s+Test:\s*({NUMBER})%")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def printed_epoch_evidence(log: bytes) -> dict:
    """Numbers retain printed precision; raw logs are not uploaded by default."""
    rows = []
    for line_number, line in enumerate(ANSI.sub("", log.decode(errors="replace")).splitlines(), 1):
        match = EPOCH.search(line)
        if match:
            epoch, total, loss, train, test = match.groups()
            rows.append({"line_number": line_number, "epoch": int(epoch), "total_epochs": int(total),
                         "loss": float(loss), "train_accuracy": float(train), "test_accuracy": float(test)})
    return {"provenance": "canonical printed epoch records", "precision": "printed, not full precision",
            "log_sha256": hashlib.sha256(log).hexdigest(),
            "units": {"loss": "dimensionless", "train_accuracy": "%", "test_accuracy": "%"},
            "epochs": rows}
