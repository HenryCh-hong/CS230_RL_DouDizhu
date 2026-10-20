"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import hashlib
import os

import numpy as np

import torch
from torch import nn

from douzero.dmc.belief import BELIEF_DIM, PREDICTOR_TARGETS, legal_count_mask, mask_logits, union_counts_from_obs
from douzero.env.hand_belief import NUM_COUNTS, NUM_KINDS

def _check_belief(belief, belief_dim):
    """A belief must match the network and arrive detached, so that a Q-loss cannot reach a predictor."""
    if (belief is not None) != bool(belief_dim):
        raise ValueError('this Q-network takes {}'.format(
            'a {}-d belief'.format(belief_dim) if belief_dim else 'no belief'))
    if belief is not None and belief.requires_grad:
        raise ValueError('beliefs must be detached; build them with belief.build_belief')

# With belief_dim > 0 a Q-network also takes a (B, belief_dim) belief, appended after x as in
# DouZero+. belief_dim = 0 is the original DouZero network.
class LandlordLstmModel(nn.Module):
    def __init__(self, belief_dim=0):
        super().__init__()
        self.belief_dim = belief_dim
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128 + belief_dim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None, belief=None):
        _check_belief(belief, self.belief_dim)
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        if self.belief_dim:
            x = torch.cat([x, belief], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self, belief_dim=0):
        super().__init__()
        self.belief_dim = belief_dim
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128 + belief_dim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None, belief=None):
        _check_belief(belief, self.belief_dim)
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        if self.belief_dim:
            x = torch.cat([x, belief], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class HandPredictor(nn.Module):
    """DouZero+ hidden-hand predictor for one seat, from public features only.

    The trunk is DouZero+'s: an LSTM over the action history z, then four 512-unit ReLU layers
    on [history, x_no_action]. Each predicted hand has 15 heads in KINDS order: Linear(512, 256)
    -ReLU-Linear(256, 5) for the 13 ranks and Linear(512, 5) for each joker. num_targets = 1
    predicts the next player's hand (M1, M2c); num_targets = 2 predicts (next, other) (M2).
    Counts above a kind's unseen count, decoded from x_no_action, get probability exactly 0.
    """
    def __init__(self, position, num_targets=1):
        super().__init__()
        self.position = position
        self.num_targets = num_targets
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear((319 if position == 'landlord' else 430) + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.heads = nn.ModuleList(nn.ModuleList(_count_head(kind) for kind in range(NUM_KINDS))
                                   for _ in range(num_targets))

    def forward(self, z, x_no_action):
        """z (B, 5, 162) and x_no_action (B, 319 or 430) -> logits, probs (B, num_targets, 15, 5).

        Logits of infeasible counts are -inf; a Q-network takes probs, never logits.
        """
        lstm_out, _ = self.lstm(z)
        x = torch.cat([lstm_out[:, -1, :], x_no_action], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        logits = torch.stack([torch.stack([head(x) for head in heads], dim=1) for heads in self.heads], dim=1)
        logits = mask_logits(logits, legal_count_mask(union_counts_from_obs(x_no_action)).unsqueeze(1))
        return dict(logits=logits, probs=torch.softmax(logits, dim=-1))


def _count_head(kind):
    if kind < 13:  # ranks 3 .. 2 (DouZero+ pre1-pre13)
        return nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, NUM_COUNTS))
    return nn.Linear(512, NUM_COUNTS)  # jokers (DouZero+ presmall, prebig)

# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel

def make_arm(position, hand_model, predictor=None):
    """Q-network and predictor of one arm at one seat.

    Every arm gets the same 150-d belief Q-network. M1 and M2c take a one-hand predictor (the
    same frozen checkpoint in the main experiment), M2 a two-hand one, and M0c and the oracle
    none. The predictor is always passed in, so an arm never runs on one created silently.
    """
    if hand_model not in PREDICTOR_TARGETS:
        raise ValueError('unknown hand_model {!r}'.format(hand_model))
    num_targets = PREDICTOR_TARGETS[hand_model]
    needed = (position, num_targets) if num_targets else None
    given = (predictor.position, predictor.num_targets) if predictor is not None else None
    if given != needed:
        raise ValueError('{} at {} needs {}'.format(hand_model, position, 'a {}-hand predictor for that seat'.format(
            num_targets) if num_targets else 'no predictor'))
    return model_dict[position](belief_dim=BELIEF_DIM), predictor

def warm_start_from_douzero(position, douzero_state_dict):
    """Belief Q-network that computes exactly what the given DouZero network computes.

    Every parameter is copied. dense1 gets BELIEF_DIM extra input columns, initialized to zero,
    so the belief has no effect on Q until training moves those weights.
    """
    state_dict = dict(douzero_state_dict)
    weight = state_dict['dense1.weight']
    state_dict['dense1.weight'] = torch.cat([weight, weight.new_zeros(len(weight), BELIEF_DIM)], dim=1)
    model = model_dict[position](belief_dim=BELIEF_DIM)
    model.load_state_dict(state_dict)
    return model

POSITIONS = ('landlord', 'landlord_up', 'landlord_down')

def file_sha256(path):
    """Hex sha256 of a file; runs record it to pin the exact checkpoints they used."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_hand_predictor(predictor, path, provenance=None):
    """Saves a predictor with its seat, size and provenance; only provenance['trained'] marks it trained."""
    torch.save(dict(position=predictor.position, num_targets=predictor.num_targets,
                    state_dict=predictor.state_dict(), provenance=provenance or dict(trained=False)), path)

def load_hand_predictor(path, freeze=True, position=None, num_targets=None, sha256=None, require_trained=False):
    """Loads a saved HandPredictor, frozen (eval mode, no gradients) unless freeze is False.

    Fails loudly if the file is missing, differs from the expected sha256, was saved for another
    seat or size than expected, is not marked as trained when require_trained is set, or does not
    match the network exactly.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('no predictor checkpoint at {}'.format(path))
    if sha256 is not None and file_sha256(path) != sha256:
        raise ValueError('{} is not the predictor file the run used (sha256 differs)'.format(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    for key, expected in (('position', position), ('num_targets', num_targets)):
        if expected is not None and checkpoint[key] != expected:
            raise ValueError('{} holds a predictor with {} {!r}, expected {!r}'.format(path, key, checkpoint[key], expected))
    provenance = checkpoint.get('provenance') or {}
    if require_trained and not provenance.get('trained'):
        raise ValueError('{} is not marked as trained; pass --allow_untrained_predictors only for smoke runs'.format(path))
    predictor = HandPredictor(checkpoint['position'], checkpoint['num_targets'])
    predictor.load_state_dict(checkpoint['state_dict'])
    predictor.provenance = provenance
    if freeze:
        predictor.eval()
        predictor.requires_grad_(False)
    return predictor

def load_predictors(hand_model, predictor_dir, sha256=None, require_trained=True):
    """The frozen predictors of one arm, {seat: HandPredictor or None}, each loaded strictly.

    M1 and M2c take one-hand predictors and M2 two-hand ones, as <seat>.ckpt in predictor_dir;
    the other arms take none and must not be given a directory. sha256 ({seat: hash}) pins the
    exact files a run used.
    """
    if hand_model != 'baseline' and hand_model not in PREDICTOR_TARGETS:
        raise ValueError('unknown hand_model {!r}'.format(hand_model))
    num_targets = PREDICTOR_TARGETS.get(hand_model, 0)
    if not num_targets:
        if predictor_dir:
            raise ValueError('{} uses no predictor, but a predictor directory was given'.format(hand_model))
        return {position: None for position in POSITIONS}
    if not predictor_dir:
        raise ValueError('{} needs a predictor directory'.format(hand_model))
    return {position: load_hand_predictor(os.path.join(predictor_dir, position + '.ckpt'), position=position,
                                          num_targets=num_targets, sha256=None if sha256 is None else sha256[position],
                                          require_trained=require_trained)
            for position in POSITIONS}

def predictor_record(predictor_dir, predictors):
    """What a run records about its frozen predictors (None if it has none): files, sha256, K, provenance."""
    if all(predictor is None for predictor in predictors.values()):
        return None
    files = {}
    for position, predictor in predictors.items():
        path = os.path.abspath(os.path.join(predictor_dir, position + '.ckpt'))
        files[position] = dict(path=path, sha256=file_sha256(path), provenance=predictor.provenance)
    return dict(dir=os.path.abspath(predictor_dir), num_targets=predictors['landlord'].num_targets, frozen=True,
                files=files)

def load_arm_checkpoint(path, predictor_dir=None, device='cpu', allow_untrained=False):
    """The Q-networks and frozen predictors of a saved run, loaded strictly; the arm comes from its metadata.

    path is a run's model.tar or one of its weights_<frames>.tar files. Fails loudly if the file is
    missing or has no run metadata, if any Q-network state does not match its network exactly, or
    if a predictor file is missing, differs from the one used in training (sha256), has the wrong
    seat or K, or is not marked as trained (allow_untrained is for smoke runs). predictor_dir
    overrides the recorded directory when a run has been copied to another machine.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('no checkpoint at {}'.format(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('cs230') if isinstance(checkpoint, dict) else None
    if not metadata:
        raise ValueError('{} has no run metadata; it is not a checkpoint of this study'.format(path))
    # Runs written before the flag existed count as valid; a fork's flag comes from its parent or its own resume.
    metadata = dict(metadata, invalid_for_final_results=metadata.get('invalid_for_final_results', False))
    q_networks = {}
    for position in POSITIONS:
        q_network = model_dict[position](belief_dim=metadata['belief_dim'])
        q_network.load_state_dict(checkpoint['model_state_dict'][position])
        q_networks[position] = q_network.to(device).eval()
    record = metadata['predictor']
    if record is None:
        predictors = load_predictors(metadata['hand_model'], None)
    else:
        predictors = load_predictors(metadata['hand_model'], predictor_dir or record['dir'],
                                     sha256={p: f['sha256'] for p, f in record['files'].items()},
                                     require_trained=not allow_untrained)
    predictors = {p: None if predictor is None else predictor.to(device) for p, predictor in predictors.items()}
    return dict(hand_model=metadata['hand_model'], metadata=metadata, q_networks=q_networks, predictors=predictors)

class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    belief_dim > 0 builds the belief-input networks of the hand-belief arms.
    """
    def __init__(self, device=0, belief_dim=0):
        self.models = {}
        self.belief_dim = belief_dim
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel(belief_dim).to(torch.device(device))
        self.models['landlord_up'] = FarmerLstmModel(belief_dim).to(torch.device(device))
        self.models['landlord_down'] = FarmerLstmModel(belief_dim).to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, belief=None):
        model = self.models[position]
        return model.forward(z, x, return_value=training, flags=flags, belief=belief)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
