"""
Compare original vs optimized training pipeline.

Usage:
    python compare.py -c config_files/nonvisual/MFT_baseline.yaml           # full run
    python compare.py -c config_files/nonvisual/MFT_baseline.yaml -e 5      # 5 epochs only
    python compare.py -c config_files/nonvisual/CMFT.yaml -e 10             # 10 epochs only

Each (pipeline, dataset) pair runs in its own subprocess so GPU memory and
Python heap are fully released between every run. 4 subprocesses total:
  orig x jaad_beh  →  orig x jaad_all  →  fast x jaad_beh  →  fast x jaad_all
"""
import os
import sys
import time
import getopt
import json
import logging
import subprocess
import tempfile
import yaml


def setup_logging(log_path):
    """Tee all stdout/stderr to log_path while still printing to console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        ]
    )

    class TeeStream:
        def __init__(self, original, logger, level):
            self.original = original
            self.logger = logger
            self.level = level
            self.encoding = getattr(original, 'encoding', 'utf-8')

        def write(self, msg):
            self.original.write(msg)
            if msg.strip():
                self.logger.log(self.level, msg.rstrip())

        def flush(self):
            self.original.flush()

        def isatty(self):
            return False

    logger = logging.getLogger()
    sys.stdout = TeeStream(sys.__stdout__, logger, logging.INFO)
    sys.stderr = TeeStream(sys.__stderr__, logger, logging.WARNING)


path_jaad = r"C:\Users\safee\Desktop\WORk\intentformer_github_sript\JAAD"
path_pie = "/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/data/pie"


# ---------------------------------------------------------------------------
# Worker entry point — runs ONE dataset for ONE pipeline, then exits
# ---------------------------------------------------------------------------

def _worker_main():
    """
    Invoked as:  python compare.py --worker
    Env vars consumed:
      WORKER_FAST        '0' or '1'
      WORKER_CONFIG      path to yaml config file
      WORKER_DATASET_IDX index into exp_opts.datasets list (0-based)
      WORKER_EPOCHS      (optional) epoch override
      WORKER_OUT         path to write JSON result for this one dataset
    """
    import os, json, time, yaml, pickle
    import numpy as np

    use_fast      = os.environ['WORKER_FAST'] == '1'
    config_file   = os.environ['WORKER_CONFIG']
    dataset_idx   = int(os.environ['WORKER_DATASET_IDX'])
    epochs_str    = os.environ.get('WORKER_EPOCHS', '')
    out_path      = os.environ['WORKER_OUT']

    epochs_override = int(epochs_str) if epochs_str else None

    if use_fast:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        from action_predict_fast import action_prediction
    else:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('float32')
        from action_predict import action_prediction

    from jaad_data import JAAD
    from pie_data import PIE

    configs_default = 'config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)
    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    for k in ['model_opts', 'net_opts', 'data_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    tte = configs['model_opts']['time_to_event']
    tte = tte if isinstance(tte, int) else tte[1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    dataset = model_configs['exp_opts']['datasets'][dataset_idx]
    configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
    configs['model_opts']['overlap']    = 0.6 if 'pie' in dataset else 0.8
    configs['model_opts']['dataset']    = dataset.split('_')[0]
    configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
    configs['train_opts']['lr']         = model_configs['exp_opts']['lr'][dataset_idx]
    configs['train_opts']['epochs']     = (
        epochs_override if epochs_override is not None
        else model_configs['exp_opts']['epochs'][dataset_idx]
    )

    if configs['model_opts']['dataset'] == 'pie':
        imdb = PIE(data_path=path_pie)
    else:
        imdb = JAAD(data_path=path_jaad)

    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
    beh_seq_val   = imdb.generate_data_trajectory_sequence('val',   **configs['data_opts'])
    beh_seq_test  = imdb.generate_data_trajectory_sequence('test',  **configs['data_opts'])

    configs['net_opts']['input_type_list'] = configs['model_opts']['obs_input_type']
    method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])

    train_start = time.time()
    saved_files_path = method_class.train(
        beh_seq_train, beh_seq_val,
        **configs['train_opts'],
        model_opts=configs['model_opts'],
    )
    train_total = time.time() - train_start

    # Free training data before inference — large datasets exhaust RAM otherwise
    del beh_seq_train, beh_seq_val
    import gc; gc.collect()

    history_path = os.path.join(saved_files_path, 'history.pkl')
    avg_epoch_s = None
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            hist = pickle.load(f)
        n_epochs = len(hist.get('loss', []))
        avg_epoch_s = round(train_total / n_epochs, 2) if n_epochs > 0 else None

    infer_start = time.time()
    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)
    infer_total = time.time() - infer_start

    result = {
        'dataset':       dataset,
        'train_total_s': round(train_total, 2),
        'avg_epoch_s':   avg_epoch_s,
        'infer_total_s': round(infer_total, 2),
        'acc':           round(float(acc),       4),
        'auc':           round(float(auc),       4),
        'f1':            round(float(f1),        4),
        'precision':     round(float(precision), 4),
        'recall':        round(float(recall),    4),
    }

    with open(out_path, 'w') as f:
        json.dump(result, f)


# ---------------------------------------------------------------------------
# Parent helpers
# ---------------------------------------------------------------------------

def _launch_one(config_file, use_fast, dataset_idx, epochs_override, log_fh, xla_flags=''):
    """
    Spawn a single worker subprocess for one (pipeline, dataset) combination.
    Streams output live. Returns the result dict.
    """
    out_fd, out_path = tempfile.mkstemp(suffix='.json', prefix='compare_worker_')
    os.close(out_fd)

    env = os.environ.copy()
    env['WORKER_FAST']        = '1' if use_fast else '0'
    env['WORKER_CONFIG']      = config_file
    env['WORKER_DATASET_IDX'] = str(dataset_idx)
    env['WORKER_OUT']         = out_path
    if epochs_override is not None:
        env['WORKER_EPOCHS'] = str(epochs_override)
    elif 'WORKER_EPOCHS' in env:
        del env['WORKER_EPOCHS']
    if xla_flags:
        env['XLA_FLAGS'] = xla_flags

    cmd = [sys.executable, __file__, '--worker']
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        bufsize=1,
    )

    for line in proc.stdout:
        sys.__stdout__.write(line)
        sys.__stdout__.flush()
        log_fh.write(line)
        log_fh.flush()

    proc.wait()
    if proc.returncode != 0:
        os.unlink(out_path)
        raise RuntimeError(
            f"Worker exited with code {proc.returncode} "
            f"(pipeline={'fast' if use_fast else 'orig'}, dataset_idx={dataset_idx})"
        )

    with open(out_path, 'r') as f:
        result = json.load(f)
    os.unlink(out_path)
    return result


def run_pipeline(config_file, use_fast, epochs_override, log_fh, xla_flags, n_datasets):
    """Run all datasets for one pipeline, one subprocess per dataset."""
    results = []
    for idx in range(n_datasets):
        results.append(_launch_one(config_file, use_fast, idx, epochs_override, log_fh, xla_flags))
    return results


def print_comparison(orig_results, fast_results, tee_fn=None):
    out = tee_fn or print

    out("\n" + "=" * 95)
    out(f"{'Dataset':<15} {'Metric':<25} {'Original':>12} {'Optimized':>12} {'Delta':>10}")
    out("=" * 95)

    for orig, fast in zip(orig_results, fast_results):
        ds = orig['dataset']
        metrics = [
            ('Train time (s)',  'train_total_s'),
            ('Avg epoch (s)',   'avg_epoch_s'),
            ('Infer time (s)',  'infer_total_s'),
            ('Accuracy',        'acc'),
            ('AUC',             'auc'),
            ('F1',              'f1'),
            ('Precision',       'precision'),
            ('Recall',          'recall'),
        ]
        for label, key in metrics:
            o_val = orig.get(key)
            f_val = fast.get(key)
            if o_val is None or f_val is None:
                out(f"{ds:<15} {label:<25} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
                continue
            delta = f_val - o_val
            fmt = '.4f' if isinstance(delta, float) and abs(delta) < 100 else '.2f'
            delta_str = f"{delta:+{fmt}}"
            out(f"{ds:<15} {label:<25} {o_val:>12} {f_val:>12} {delta_str:>10}")
        out("-" * 95)


def usage():
    print("Usage: python compare.py -c <config_file> [-e <epochs>]")
    print("  -c / --config_file : path to yaml config (e.g. config_files/nonvisual/CMFT.yaml)")
    print("  -e / --epochs      : override number of epochs (omit to use config value)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Subprocess worker mode
    if '--worker' in sys.argv:
        _worker_main()
        sys.exit(0)

    # Parent mode
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:e:', ['help', 'config_file=', 'epochs='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    config_file = None
    epochs_override = None
    for o, a in opts:
        if o in ['-h', '--help']:
            usage()
            sys.exit(0)
        elif o in ['-c', '--config_file']:
            config_file = a
        elif o in ['-e', '--epochs']:
            epochs_override = int(a)

    if not config_file:
        print('ERROR: Provide -c <config_file>')
        usage()
        sys.exit(2)

    # Read dataset count from config so the parent knows how many subprocesses to spawn
    with open(config_file, 'r') as f:
        _mc = yaml.safe_load(f)
    n_datasets = len(_mc['exp_opts']['datasets'])

    timestamp = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    log_path = f"compare_log_{timestamp}.txt"
    log_fh = open(log_path, 'w', encoding='utf-8')

    def tee(msg):
        sys.__stdout__.write(msg + '\n')
        sys.__stdout__.flush()
        log_fh.write(msg + '\n')
        log_fh.flush()

    tee(f"Logging to {log_path}")
    tee(f"Config: {config_file}  |  datasets: {_mc['exp_opts']['datasets']}")

    if epochs_override is not None:
        tee(f"\n>>> Quick comparison mode: {epochs_override} epochs per dataset")
    else:
        tee("\n>>> Full comparison mode: using epochs from config")

    xla_flags = os.environ.get('XLA_FLAGS', '')

    tee("\n>>> Running ORIGINAL pipeline (float32) — one subprocess per dataset...")
    orig_results = run_pipeline(config_file, use_fast=False,
                                epochs_override=epochs_override,
                                log_fh=log_fh, xla_flags=xla_flags,
                                n_datasets=n_datasets)

    tee("\n>>> Running OPTIMIZED pipeline (float16 + XLA + batch=8) — one subprocess per dataset...")
    fast_results = run_pipeline(config_file, use_fast=True,
                                epochs_override=epochs_override,
                                log_fh=log_fh, xla_flags=xla_flags,
                                n_datasets=n_datasets)

    print_comparison(orig_results, fast_results, tee_fn=tee)

    out_path = f"comparison_results_{timestamp}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump({'original': orig_results, 'optimized': fast_results}, f)
    tee(f"\nResults saved to {out_path}")
    tee(f"Full log saved to  {log_path}")

    log_fh.close()
