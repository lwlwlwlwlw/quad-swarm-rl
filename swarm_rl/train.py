"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys
from argparse import Namespace

from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import maybe_load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.utils.utils import log

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models
from swarm_rl.swanlab_utils import SwanLabObserver, add_swanlab_args


def register_swarm_components():
    register_env("quadrotor_multi", make_quadrotor_env)
    register_models()


def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    add_swanlab_args(parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def run_swarm_training(cfg):
    # Resolve saved settings before choosing the logger, including when resuming a W&B run.
    cli_args = cfg.cli_args.copy()
    if cfg.restart_behavior == 'resume':
        # make_runner loads again using vars(cfg), which is empty for AttrDict.
        cfg = Namespace(**maybe_load_from_checkpoint(cfg))
    cfg.cli_args = cli_args
    if cfg.with_swanlab:
        if cfg.with_wandb:
            log.info('SwanLab enabled: disabling W&B synchronization')
        cfg.with_wandb = False
        cfg.cli_args['with_wandb'] = False

    cfg, runner = make_runner(cfg)
    cfg.cli_args = cli_args
    observer = SwanLabObserver() if cfg.with_swanlab else None
    if observer is not None:
        runner.register_observer(observer)

    status = ExperimentStatus.FAILURE
    error = None
    try:
        status = runner.init()
        if status == ExperimentStatus.SUCCESS:
            status = runner.run()
        return status
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if observer is not None:
                observer.finish(runner, failed=status == ExperimentStatus.FAILURE or error is not None, error=error)
        finally:
            for writer in runner.writers.values():
                writer.close()


def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=False)
    return run_swarm_training(cfg)


if __name__ == '__main__':
    sys.exit(main())
