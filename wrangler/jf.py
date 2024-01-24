import argparse
import numpy as np
import httpx
import json
import logging
from pathlib import Path
from utils.utils import setup_logger, compute_metrics
import utils.constants as constants
import utils.data_utils as data_utils
import utils.prompt_utils as prompt_utils

from llama_cpp import Llama

llm = Llama('../../models/Jellyfish/q4_0.bin')


from typing import Union

def request_llm(prompt: str, max_tokens=5):
    output = llm(prompt, max_tokens=max_tokens, echo=False)
    return output['choices'][0]["text"].strip()


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Which data directory to run.',
        default='/home/dseg/cliu/FlexGen/flexgen/apps/data_wrangle/data/datasets/data_imputation/Restaurant',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs_jf'
    )
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="manual",
        choices=["random", "manual", "validation_clusters"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks \
             with random prompts.",
        action="store_true",
    )
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    parser.add_argument(
        "--nan_tok",
        type=str,
        help="Token to represent nan entries. Default is 'nan'.",
        default="nan",
    )
    parser.add_argument(
        "--num_run",
        type=int,
        help="Number examples to run through model.",
        default=200,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number trials to run. Results will be averaged with variance reported.",
        default=1,
    )
    parser.add_argument(
        "--num_print",
        type=int,
        help="Number example prompts to print.",
        default=10,
    )
    parser.add_argument(
        "--add_task_instruction",
        help="Add task instruction to the prompt before examples.",
        action="store_true",
    )
    parser.add_argument("--task_instruction_idx", type=int, default=0)
    parser.add_argument("--do_test", help="Run on test file.", action="store_true")
    parser.add_argument(
        "--stop_token", help="Token to stop on for a given generated response", default="\n"
    )

    # Model args
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to generate.", default=5
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("num_trials must be greater than 0.")
    
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))

    np.random.seed(args.seed)

    test_file = 'test' if args.do_test else 'validation'

    pd_data_files = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_instruction=False,
        max_train_samples=-1,
        max_train_percent=-1,
        sep_tok=args.sep_tok,
        nan_tok=args.nan_tok,
    )

    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")
    
    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    task = constants.DATA2TASK[args.data_dir]
    logger.info(f"Using {args.task_instruction_idx} instruction idx")
    task_instruction = constants.DATA2INSTRUCT[args.data_dir]
    num_run = args.num_run
    if args.num_run == -1:
        num_run = test_data.shape[0]
    num_run = min(num_run, test_data.shape[0])

    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")
    logger.info(f"Running {num_run} examples for {args.num_trials} trials.")

    if args.add_task_instruction:
        prompt = lambda x: f"{task_instruction} \n{x}"
    else:
        prompt = lambda x: f"{x}"
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    saved_prefix: str | None = None
    for trial_num in range(args.num_trials):
        np.random.seed(args.seed + trial_num)
        queries: list[str] = []
        for _, row in test_data.iterrows():
            serialized_r = row['text']
            if args.sample_method == 'manual':
                prefix_exs = prompt_utils.get_manual_prompt(args.data_dir, row)
            elif args.sample_method == 'validation_clusters':
                if saved_prefix is None:
                    logger.info('Generating validation cluster prompt')
                    saved_prefix = prompt_utils.get_validation_prompt(
                        args.validation_path,
                        num_examples=args.k,
                        task=task
                    )
                prefix_exs = saved_prefix
            else:
                if saved_prefix is None:
                    saved_prefix = prompt_utils.get_random_prompt(
                        pd_data_files['train'], num_examples=args.k
                    )
                prefix_exs = saved_prefix
            queries.append((prefix_exs + '\n' + serialized_r).strip())

        gt = test_data['label_str']
        preds = []
        idx = 0
        for _ in range(min(num_run, args.num_print)):
            logger.info(prompt(queries[idx]))
            pred = request_llm(prompt(queries[idx]))
            preds.append(pred)
            logger.info(f'=========> pred: {pred}, gt: {str(gt[idx]).strip()} <=========')
            idx += 1

        for query in queries[idx:num_run]:
            pred = request_llm(prompt(query))
            preds.append(pred)

        save_data = test_data.iloc[:num_run].copy(deep=True).reset_index()
        gt = gt[:num_run]
        save_data['preds'] = preds
        save_data['queries'] = queries[:num_run]

        prec, rec, acc, f1 = compute_metrics(preds, gt, task)

        logger.info(
            f"Metrics Trial {trial_num}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        output_file = (
            Path(args.output_dir)
            / f'{Path(args.data_dir).stem}'
            / f'{test_file}'
            / f'{args.k}'
            f'_{int(args.add_task_instruction)}inst'
            f'_{int(args.class_balanced)}cb'
            f'_{args.sample_method}'
            f'_{args.num_run}run'
            f'_trial_{trial_num}.feather'
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved to {output_file}")

        save_data.to_feather(output_file)

    for k, values in list(trial_metrics.items()):
        trial_metrics[f'{k}_avg'] = np.average(values)
        trial_metrics[f'{k}_std'] = np.std(values)
    
    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))

    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")


if __name__ == '__main__':
    main()
    