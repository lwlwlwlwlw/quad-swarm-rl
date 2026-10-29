"""Mirror Sample Factory's TensorBoard scalars to SwanLab in the runner process."""

import json
from pathlib import Path

from sample_factory.algo.runners.runner import AlgoObserver
from sample_factory.cfg.arguments import cfg_dict
from sample_factory.utils.utils import experiment_dir


def add_swanlab_args(parser):
    from sample_factory.utils.utils import str2bool

    parser.add_argument('--with_swanlab', nargs='?', const=True, default=False, type=str2bool,
                        help='Enable SwanLab and disable W&B (also accepts --with_swanlab=True)')
    parser.add_argument('--swanlab_project', default='quad-swarm-rl', help='SwanLab project name')
    parser.add_argument('--swanlab_workspace', default=None, help='Workspace; defaults to your personal workspace')
    parser.add_argument('--swanlab_experiment_name', default=None, help='Display name; defaults to --experiment')
    parser.add_argument('--swanlab_group', default=None, help='Group; falls back to wandb_group or the environment')
    parser.add_argument('--swanlab_job_type', default=None, help='Job type; falls back to wandb_job_type')
    parser.add_argument('--swanlab_tags', nargs='*', default=None, help='Tags; falls back to wandb_tags')
    parser.add_argument('--swanlab_mode', choices=['cloud', 'offline'], default='cloud',
                        help='Cloud synchronization or offline logging for later upload')


class SwanLabSummaryWriter:
    """Keep TensorBoard output and mirror every scalar with its original environment step."""

    def __init__(self, writer, run, prefix=''):
        self.writer = writer
        self.run = run
        self.prefix = prefix
        self.pending = {}
        self.step = None

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        result = self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        step = int(global_step) if global_step is not None else None
        if self.pending and step != self.step:
            self._flush_metrics()
        self.step = step
        self.pending[self.prefix + tag] = float(scalar_value)
        return result

    def _flush_metrics(self):
        if self.pending:
            self.run.log(dict(self.pending), step=self.step)
            self.pending.clear()

    def flush(self):
        self.writer.flush()
        self._flush_metrics()

    def close(self):
        self.flush()
        self.writer.close()


class SwanLabObserver(AlgoObserver):
    def __init__(self):
        self.run = None
        self.sdk = None

    def on_init(self, runner):
        # Import only when enabled, so existing training/evaluation needs no SwanLab login.
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError('Install SwanLab with: pip install "swanlab==0.7.13"') from exc

        self.sdk = swanlab
        cfg = runner.cfg
        logdir = Path(experiment_dir(cfg)) / 'swanlab'
        logdir.mkdir(parents=True, exist_ok=True)
        state_path = logdir / 'run.json'
        identity = dict(project=cfg.swanlab_project, workspace=cfg.swanlab_workspace)
        run_id = None
        if cfg.swanlab_mode == 'cloud' and cfg.restart_behavior == 'resume' and state_path.exists():
            state = json.loads(state_path.read_text())
            if all(state.get(key) == value for key, value in identity.items()):
                run_id = state['id']

        self.run = swanlab.init(
            **identity,
            experiment_name=cfg.swanlab_experiment_name or cfg.experiment,
            group=cfg.swanlab_group or cfg.wandb_group or cfg.env,
            job_type=cfg.swanlab_job_type or cfg.wandb_job_type,
            tags=cfg.swanlab_tags if cfg.swanlab_tags is not None else cfg.wandb_tags,
            config=dict(cfg_dict(cfg)),
            logdir=str(logdir),
            mode=cfg.swanlab_mode,
            id=run_id,
            resume='allow' if cfg.swanlab_mode == 'cloud' else False,
        )
        if cfg.swanlab_mode == 'cloud':
            temporary_path = state_path.with_suffix('.tmp')
            temporary_path.write_text(json.dumps(dict(identity, id=self.run.id)))
            temporary_path.replace(state_path)

        for policy_id, writer in runner.writers.items():
            prefix = f'policy_{policy_id}/' if cfg.num_policies > 1 else ''
            runner.writers[policy_id] = SwanLabSummaryWriter(writer, self.run, prefix)

    def finish(self, runner, failed=False, error=None):
        if self.run is None:
            return
        try:
            for writer in runner.writers.values():
                writer.flush()
        finally:
            if failed:
                self.sdk.finish(state=self.sdk.State.CRASHED, error=str(error or 'Training failed'))
            else:
                self.sdk.finish()
            self.run = None
