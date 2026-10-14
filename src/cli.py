"""Command-line interface for the full training and inference workflow."""

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(description="HOG + SVM pedestrian detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="Create fixed-size INRIA training samples")
    build.add_argument("--clean", action="store_true", help="Delete existing processed samples first")
    build.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("train", help="Train the baseline SVM")
    subparsers.add_parser("mine", help="Mine hard negatives and train the final SVM")

    detect = subparsers.add_parser("detect", help="Detect pedestrians in an image")
    detect.add_argument("image")
    detect.add_argument("--output-dir", default=".")
    detect.add_argument("--model")
    detect.add_argument("--scaler")
    detect.add_argument("--win-step", type=int, default=16)
    detect.add_argument("--scale-factor", type=float, default=1.15)
    detect.add_argument("--hit-threshold", type=float, default=0.5)
    detect.add_argument("--nms-overlap", type=float, default=0.3)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate on the INRIA test split")
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.add_argument("--max-error-images", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-dataset":
            from src.data.dataset_builder import build_dataset

            result = build_dataset(clean=args.clean, seed=args.seed)
            print(f"Created {result['positive']} positive and {result['negative']} negative samples.")
        elif args.command == "train":
            from src.models.train_baseline import train_baseline_model

            result = train_baseline_model()
            print(result["report"])
            print(f"Saved: {result['model_path']}")
        elif args.command == "mine":
            from src.models.hard_negative_mining import run_hard_negative_mining

            result = run_hard_negative_mining()
            print(result["report"])
            print(f"Mined {result['hard_negatives']} hard negatives; saved: {result['model_path']}")
        elif args.command == "detect":
            from src.inference import run_inference

            result = run_inference(
                args.image,
                output_dir=args.output_dir,
                model_path=args.model,
                scaler_path=args.scaler,
                win_step=args.win_step,
                scale_factor=args.scale_factor,
                hit_threshold=args.hit_threshold,
                nms_overlap=args.nms_overlap,
            )
            print(f"Detected {result['detections']} person(s) from {result['raw_detections']} raw boxes.")
            print(f"Saved: {result['final_output']}")
        elif args.command == "evaluate":
            from src.evaluate import run_evaluation

            result = run_evaluation(args.iou_threshold, args.max_error_images)
            for name, value in result.items():
                print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
