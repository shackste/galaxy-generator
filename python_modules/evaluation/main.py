import argparse

from python_modules.evaluation.evaluator import Evaluator


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        required=True,
                        help='Path to config file for evaluation')
    args = parser.parse_args()
    config_path = args.config
    evaluator = Evaluator(config_path)
    evaluator.evaluate()
