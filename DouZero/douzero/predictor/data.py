"""The offline predictor dataset: decisions of frozen DouZero-ADP self-play with their true hidden hands.

Each game has its own seed, derived from (NAMESPACE, split, game index) with numpy's SeedSequence,
so a game is reproducible on its own and no game is shared between the train, val and test
splits; the namespace keeps them apart from the RL actors' streams and from the evaluation deals,
and every game's deal hash is recorded so that disjointness can be checked exactly.

One sample is one decision, stored for the acting seat:
    x_no_action (319 or 430,) int8   public observation (DouZero's features), the predictor's input
    z (5, 162) int8                  the public move history, the predictor's input
    labels (2, 15) int8              TRUE counts of the next and the other hidden hand (training labels only)
    union (15,) int8                 the actor's count of the unseen cards (= next + other, kind by kind)
    sizes (2,) int8                  the next and other hand sizes (public)
    game (int32), t (int16), game_length (int16), cards_left (int8), phase (int8): bookkeeping
The hidden hands are stored only as these order-free counts: never as seat-ordered card lists.

Files: <dir>/<split>/shard_<k>/<seat>/<field>.npy, plain numpy arrays (byte-reproducible, memory-
mappable, no extra dependency), and <dir>/manifest.json with every file's sha256, the seeds, the
model and the git state.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from douzero.dmc import models
from douzero.dmc.env_utils import _format_observation
from douzero.dmc.run_dir import git_info, machine_info, now
from douzero.env import hand_belief as hb
from douzero.env.env import Env

SCHEMA = 'cs230.predictor_data/1'
SPLITS = ('train', 'val', 'test')
POSITIONS = ('landlord', 'landlord_down', 'landlord_up')
NAMESPACE = 230_4_0  # Stage 4's seed namespace: actors use [seed, device, actor, attempt], evaluation RandomState(seed)
FIELDS = ('x_no_action', 'z', 'labels', 'union', 'sizes', 'game', 't', 'game_length', 'cards_left', 'phase')
PHASES = ('early', 'mid', 'late')
PHASE_RULE = ('decision progress p = t / (game_length - 1), where t counts every decision of the game from 0: '
              'early if p < 1/3, mid if 1/3 <= p < 2/3, late otherwise; fixed before any predictor was trained')


def game_seeds(split, index):
    """(numpy seed, torch seed) of one game: deal and exploration, independent of every other game.

    The numpy seed is four 32-bit words (128 bits), so no two of ~1e5 games share a stream; a single
    32-bit seed would give ~12% odds of a repeated deal among 33,000 games.
    """
    words = [int(w) for w in np.random.SeedSequence([NAMESPACE, SPLITS.index(split), index]).generate_state(6)]
    return tuple(words[:4]), (words[4] << 32) | words[5]


def phase_of(t, game_length):
    progress = t / max(game_length - 1, 1)
    return int(np.digitize(progress, [1 / 3, 2 / 3]))


def deal_hash(deal):
    """A short hash of a deal: each seat's cards and the bottom cards, as order-free counts."""
    counts = [hb.card_counts(deal[key]).tolist() for key in ('landlord', 'landlord_up', 'landlord_down',
                                                              'three_landlord_cards')]
    return hashlib.sha256(json.dumps(counts).encode()).hexdigest()[:16]


def douzero_model(checkpoint_dir, seed=0):
    """The frozen DouZero network that plays every seat, and where its weights came from."""
    model = models.Model('cpu', belief_dim=0)
    if checkpoint_dir:  # a missing file is an error, never replaced silently by random weights
        for position in POSITIONS:
            path = os.path.join(checkpoint_dir, position + '.ckpt')
            if not os.path.isfile(path):
                raise FileNotFoundError('no DouZero checkpoint at {}'.format(path))
            model.get_model(position).load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        source = 'DouZero-ADP from ' + os.path.abspath(checkpoint_dir)
    else:  # an empty checkpoint_dir on purpose: tests
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for position in POSITIONS:
                model.get_model(position).load_state_dict(models.model_dict[position]().state_dict())
        source = 'RANDOM DouZero weights (no checkpoint): for tests only'
    model.eval()
    return model, source


def play_game(model, split, index, exp_epsilon):
    """Plays one seeded game with epsilon-greedy DouZero and returns ({seat: {field: array}}, game record)."""
    numpy_seed, torch_seed = game_seeds(split, index)
    numpy_state = np.random.get_state()
    rows = {p: {field: [] for field in FIELDS} for p in POSITIONS}
    with torch.random.fork_rng(devices=[]):
        np.random.seed(list(numpy_seed))  # the deal (Env.reset) and exploration draw from numpy's global stream
        torch.manual_seed(torch_seed)  # the random legal move when exploring
        try:
            env = Env('adp')
            raw_obs = env.reset()
            flags = SimpleNamespace(exp_epsilon=exp_epsilon)
            t, deal = 0, None
            while True:
                infoset = env.infoset
                position = infoset.player_position
                next_player, other_player = hb.hidden_players(position)
                labels = hb.hand_labels(infoset)  # the true hidden hands: training labels only
                if deal is None:  # the landlord acts first, so this decision sees the whole deal
                    deal = {position: list(infoset.player_hand_cards), next_player: hb_cards(labels[0]),
                            other_player: hb_cards(labels[1]), 'three_landlord_cards': list(infoset.three_landlord_cards)}
                row = rows[position]
                row['x_no_action'].append(np.asarray(raw_obs['x_no_action'], dtype=np.int8))
                row['z'].append(np.asarray(raw_obs['z'], dtype=np.int8))
                row['labels'].append(labels.astype(np.int8))
                row['union'].append(hb.union_counts(infoset).astype(np.int8))
                row['sizes'].append(np.array([infoset.num_cards_left_dict[p] for p in (next_player, other_player)],
                                             dtype=np.int8))
                row['t'].append(t)
                row['cards_left'].append(sum(infoset.num_cards_left_dict.values()))
                _, obs, _, _ = _format_observation(raw_obs, 'cpu')
                with torch.no_grad():
                    choice = int(model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)['action'])
                raw_obs, _, done, _ = env.step(obs['legal_actions'][choice])
                t += 1
                if done:
                    break
            winner, bombs = env._game_winner, env._game_bomb_num
        finally:
            np.random.set_state(numpy_state)
    samples = {}
    for position, row in rows.items():
        count = len(row['t'])
        row['game'] = [index] * count
        row['game_length'] = [t] * count
        row['phase'] = [phase_of(step, t) for step in row['t']]
        dtypes = dict(game=np.int32, t=np.int16, game_length=np.int16, cards_left=np.int8, phase=np.int8)
        samples[position] = {field: (np.stack(row[field]) if field not in dtypes
                                     else np.asarray(row[field], dtype=dtypes[field])) if count else
                             _empty(field, position) for field in FIELDS}
    game = dict(index=index, deal_hash=deal_hash(deal), length=t, winner=winner, bombs=int(bombs))
    return samples, game


def hb_cards(counts):
    return [kind for kind, num in zip(hb.KINDS, counts) for _ in range(int(num))]


def _empty(field, position):
    shapes = dict(x_no_action=(0, 319 if position == 'landlord' else 430), z=(0, 5, 162), labels=(0, 2, 15),
                  union=(0, 15), sizes=(0, 2))
    dtypes = dict(game=np.int32, t=np.int16, game_length=np.int16)
    return np.zeros(shapes.get(field, (0,)), dtype=dtypes.get(field, np.int8))


def _file_sha256(path):
    return models.file_sha256(str(path))


def write_shard(out_dir, split, shard, indices, model, exp_epsilon):
    """Plays the games of one shard and writes <split>/shard_<k>/<seat>/<field>.npy; returns the game records."""
    played = [play_game(model, split, index, exp_epsilon) for index in indices]
    for position in POSITIONS:
        folder = Path(out_dir) / split / 'shard_{:05d}'.format(shard) / position
        folder.mkdir(parents=True, exist_ok=True)
        for field in FIELDS:
            np.save(folder / (field + '.npy'), np.concatenate([samples[position][field] for samples, _ in played]))
    return [game for _, game in played]


def _worker(job):
    out_dir, split, shard, indices, checkpoint_dir, exp_epsilon = job
    torch.set_num_threads(1)
    model, _ = douzero_model(checkpoint_dir)
    return split, shard, write_shard(out_dir, split, shard, indices, model, exp_epsilon)


def generate(out_dir, games, model=None, model_source=None, checkpoint_dir=None, exp_epsilon=0.01, shard_games=500,
             workers=1, forbidden_deals=None, forbidden_source=None):
    """Generates every split (games = {split: number of games}) and writes manifest.json; returns the manifest.

    With workers > 1 each process loads the model from checkpoint_dir; the files do not depend on
    the number of workers, because every game and shard is seeded on its own. Before the manifest is
    written, no deal may repeat and none may be in forbidden_deals (the evaluation deals' hashes);
    otherwise there is no manifest, and training refuses the folder.
    """
    out_dir = Path(out_dir)
    if (out_dir / 'manifest.json').exists():
        raise FileExistsError('{} already holds a dataset'.format(out_dir))
    jobs = [(str(out_dir), split, shard, list(range(start, min(start + shard_games, count))), checkpoint_dir,
             exp_epsilon)
            for split, count in games.items() for shard, start in enumerate(range(0, count, shard_games))]
    if workers > 1:
        if not checkpoint_dir:
            raise ValueError('workers > 1 load the model from checkpoint_dir')
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(workers) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
        model_source = douzero_model(checkpoint_dir)[1]
    else:
        if model is None:
            model, model_source = douzero_model(checkpoint_dir)
        results = [(split, shard, write_shard(out_dir, split, shard, indices, model, exp_epsilon))
                   for _, split, shard, indices, _, _ in jobs]
    splits = {}
    for split, count in games.items():
        shards = sorted((shard, records) for s, shard, records in results if s == split)
        game_list = [game for _, records in shards for game in records]
        samples = {p: sum(len(np.load(out_dir / split / 'shard_{:05d}'.format(shard) / p / 'phase.npy'))
                          for shard, _ in shards) for p in POSITIONS}
        splits[split] = dict(games=count, shards=[shard for shard, _ in shards], samples=samples,
                             game_seed='SeedSequence([{}, {}, game index])'.format(NAMESPACE, SPLITS.index(split)),
                             game_list=game_list)
    hashes = [game['deal_hash'] for split in splits.values() for game in split['game_list']]
    if len(set(hashes)) != len(hashes):
        raise ValueError('{} deals appear more than once; no manifest written'.format(len(hashes) - len(set(hashes))))
    met = set(hashes) & set(forbidden_deals or ())
    if met:
        raise ValueError('{} predictor deals are evaluation deals ({}); no manifest written'.format(
            len(met), forbidden_source))
    deal_check = dict(unique=True, games=len(hashes), evaluation=forbidden_source or 'not checked',
                      evaluation_deals=len(forbidden_deals or ()))
    files = {str(path.relative_to(out_dir)): _file_sha256(path) for path in sorted(out_dir.rglob('*.npy'))}
    manifest = dict(schema=SCHEMA, created=now(), namespace=NAMESPACE, exp_epsilon=exp_epsilon,
                    shard_games=shard_games, model=model_source, fields=list(FIELDS), phase_rule=PHASE_RULE,
                    splits=splits, deal_check=deal_check, files=files, git=git_info(), machine=machine_info())
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest


def read_manifest(data_dir):
    with open(Path(data_dir) / 'manifest.json') as f:
        return json.load(f)


def verify(data_dir):
    """Raises ValueError unless every file listed in the manifest still has its recorded sha256."""
    data_dir = Path(data_dir)
    for name, sha256 in read_manifest(data_dir)['files'].items():
        if _file_sha256(data_dir / name) != sha256:
            raise ValueError('{} has changed since the dataset was generated'.format(name))


def load_split(data_dir, split, seat):
    """{field: array} of one seat's samples in one split, its shards concatenated in game order."""
    data_dir = Path(data_dir)
    shards = read_manifest(data_dir)['splits'][split]['shards']
    return {field: np.concatenate([np.load(data_dir / split / 'shard_{:05d}'.format(shard) / seat / (field + '.npy'))
                                   for shard in shards])
            for field in FIELDS}


def batches(arrays, batch_size, seed, shuffle=True):
    """One epoch of minibatches as torch tensors (inputs as float32, counts as long), each sample once."""
    count = len(arrays['labels'])
    order = np.random.default_rng(seed).permutation(count) if shuffle else np.arange(count)
    for start in range(0, count, batch_size):
        index = order[start:start + batch_size]
        yield dict(index=torch.from_numpy(index),
                   x_no_action=torch.from_numpy(arrays['x_no_action'][index]).float(),
                   z=torch.from_numpy(arrays['z'][index]).float(),
                   labels=torch.from_numpy(arrays['labels'][index]).long(),
                   union=torch.from_numpy(arrays['union'][index]).long(),
                   phase=torch.from_numpy(arrays['phase'][index]).long())
