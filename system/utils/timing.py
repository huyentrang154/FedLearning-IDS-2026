"""Wall-clock timing of a simulated FL run, as if the clients trained in parallel.

The simulation trains clients one after another in one process. This module only measures; it
does not change the algorithms. It wraps
  * client-side work (train, collect_protos, get_outputs_and_loss, compute_per_class_losses, relabel),
  * server barriers (receive / aggregate / FedCorr's _fedavg), which end a group of clients that
    could have run in parallel,
  * evaluation (evaluate, _evaluate_and_log), which is measurement overhead and is excluded,
  * bookkeeping of the experiment code (track_best, round_models, write_final_summary,
    save_model_artifacts, save_results: copying / pickling models and writing logs), which a real
    deployment does not do and is excluded too,
  * client-side computation done by a module-level helper (FedCorr's _lid_term: each client computes
    the LID of its own outputs), counted as work of the client that ran just before it.

A "group" is the client work between two barriers; a client that works twice in the same group
(e.g. train + collect_protos) is summed. Then
  sequential time = total - evaluation - bookkeeping
  parallel time   = sequential time - sum(client work) + sum over groups of max(client work)
i.e. server time + the slowest client of every group. FedCorr Stage 1 aggregates after every single
client (frac1 = 1/N), so those groups have one client each and stay sequential, as in the algorithm.

usage (main.py):  timer = install(server); server.train(); timer.dump(path)
"""
import functools
import json
import os
import time

CLIENT_METHODS = ('train', 'collect_protos', 'get_outputs_and_loss', 'compute_per_class_losses', 'relabel')
BARRIER_METHODS = ('receive_models', 'receive_protos', 'aggregate_parameters', 'train_predictor')
EVAL_METHODS = ('evaluate', '_evaluate_and_log')
BOOKKEEPING_FUNCS = ('track_best', 'round_models', 'write_final_summary', 'save_model_artifacts')
BOOKKEEPING_METHODS = ('save_results',)
CLIENT_SIDE_FUNCS = ('_lid_term',)


class FLTimer:
    def __init__(self):
        self.groups = []          # list of {client_id: seconds}
        self.cur = {}
        self.client_total = 0.0
        self.eval_total = 0.0
        self.bookkeeping_total = 0.0
        self._eval_depth = 0          # > 0 while inside evaluation or bookkeeping
        self._last_client = None      # client that last did training work in the current group
        self.t0 = None
        self.t_end = None

    def close_group(self):
        if self.cur:
            self.groups.append(self.cur)
            self.cur = {}
        self._last_client = None

    def _add_client_time(self, cid, d):
        self.cur[cid] = self.cur.get(cid, 0.0) + d
        self.client_total += d

    def wrap_client(self, client, name):
        fn = getattr(client, name)

        @functools.wraps(fn)
        def wrapped(*a, **k):
            s = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if self._eval_depth == 0:          # client work done for evaluation is not training
                    self._add_client_time(client.id, time.perf_counter() - s)
                    self._last_client = client.id
        setattr(client, name, wrapped)

    def wrap_client_side_fn(self, mod, name):
        """Module-level helper that runs client-side computation for the client that just worked."""
        fn = getattr(mod, name)

        @functools.wraps(fn)
        def wrapped(*a, **k):
            s = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if self._eval_depth == 0 and self._last_client is not None:
                    self._add_client_time(self._last_client, time.perf_counter() - s)
        setattr(mod, name, wrapped)

    def wrap_barrier(self, obj, name):
        fn = getattr(obj, name)

        @functools.wraps(fn)
        def wrapped(*a, **k):
            self.close_group()
            return fn(*a, **k)
        setattr(obj, name, wrapped)

    def wrap_eval(self, obj, name, bucket='eval_total'):
        """Excluded from the training time: evaluation (bucket eval_total) or bookkeeping."""
        fn = getattr(obj, name)

        @functools.wraps(fn)
        def wrapped(*a, **k):
            if bucket == 'eval_total':      # evaluation is a sync point; bookkeeping is not
                self.close_group()
            self._eval_depth += 1
            s = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self._eval_depth -= 1
                if self._eval_depth == 0:
                    setattr(self, bucket, getattr(self, bucket) + time.perf_counter() - s)
        setattr(obj, name, wrapped)

    def summary(self):
        self.close_group()
        total = (self.t_end or time.perf_counter()) - self.t0
        seq = total - self.eval_total - self.bookkeeping_total
        par = seq - self.client_total + sum(max(g.values()) for g in self.groups)
        sizes = [len(g) for g in self.groups]
        return {'total_wall_s': total, 'eval_s': self.eval_total, 'bookkeeping_s': self.bookkeeping_total,
                'client_work_s': self.client_total,
                'server_s': seq - self.client_total, 'sequential_s': seq, 'parallel_s': par,
                'n_groups': len(self.groups), 'clients_per_group_min': min(sizes, default=0),
                'clients_per_group_max': max(sizes, default=0),
                'groups': [{str(c): round(t, 4) for c, t in g.items()} for g in self.groups]}

    def dump(self, path, extra=None):
        self.t_end = time.perf_counter()
        out = dict(extra or {}, **self.summary())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        return out


def install(server):
    t = FLTimer()
    for c in server.clients:
        for m in CLIENT_METHODS:
            if callable(getattr(c, m, None)):
                t.wrap_client(c, m)
    for m in BARRIER_METHODS:
        if callable(getattr(server, m, None)):
            t.wrap_barrier(server, m)
    for m in EVAL_METHODS:
        if callable(getattr(server, m, None)):
            t.wrap_eval(server, m)
    for m in BOOKKEEPING_METHODS:
        if callable(getattr(server, m, None)):
            t.wrap_eval(server, m, 'bookkeeping_total')
    # module-level functions used by the server's train(): patch the names in the server's module
    import sys
    mod = sys.modules.get(type(server).__module__)
    if mod is not None:
        if callable(getattr(mod, '_fedavg', None)):        # FedCorr aggregation
            t.wrap_barrier(mod, '_fedavg')
        for f in CLIENT_SIDE_FUNCS:
            if callable(getattr(mod, f, None)):
                t.wrap_client_side_fn(mod, f)
        for f in BOOKKEEPING_FUNCS:
            if callable(getattr(mod, f, None)):
                t.wrap_eval(mod, f, 'bookkeeping_total')
    t.t0 = time.perf_counter()
    return t
