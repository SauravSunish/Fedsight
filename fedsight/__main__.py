"""Reproducible CLI for demo, evaluation, verification and exports."""
import argparse
import json
from pathlib import Path
from .core import canonical


def main():
    parser = argparse.ArgumentParser(prog="python -m fedsight")
    parser.add_argument("--data-dir", type=Path, default=Path("data/workbench"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    sub.add_parser("verify")
    dataset = sub.add_parser("sample-dataset")
    dataset.add_argument("output", type=Path)
    train = sub.add_parser("evaluate")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    export = sub.add_parser("export")
    export.add_argument("case_id")
    export.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "sample-dataset":
        from .evaluation import synthetic_dataset
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(synthetic_dataset()))
    elif args.command == "evaluate":
        from .evaluation import train_compare
        artifact, report = train_compare(json.loads(args.dataset.read_text(encoding="utf-8")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical({"artifact": artifact, "report": report}))
        print(json.dumps({"selected_model": report["selected_model"], "synthetic": report["synthetic"], "test": report["held_out_test"]}, indent=2))
    else:
        from .service import FedSight
        service = FedSight(args.data_dir)
        if args.command == "demo":
            from .demo import create_demo
            print(create_demo(service))
        elif args.command == "verify":
            report = service.repo.verify()
            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["ok"] else 1)
        elif args.command == "export":
            from .reports import export_bundle
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(export_bundle(service, args.case_id, "cli-export"))


if __name__ == "__main__":
    main()
