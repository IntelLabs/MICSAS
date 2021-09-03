'''
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import tqdm
import pickle
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import pairwise
from prg import prg

from models.circle_loss import CircleLoss
from models.bof_model import BagOfFeaturesModel
from models.sbt_model import SBTModel
from models.gnn_model import GNNModel
from models.c2v_model import C2VModel
from models.ncc_model import NCCModel
from models.c2s_model import C2SModel
from datasets.bof_dataset import BoFDataset
from datasets.sbt_dataset import SBTDataset
from datasets.gnn_dataset import GNNDataset
from datasets.c2v_dataset import C2VDataset
from datasets.ncc_dataset import NCCDataset
from datasets.c2s_dataset import C2SDataset


def parse_args():
    parser = argparse.ArgumentParser()

    common_parser = argparse.ArgumentParser(add_help=False)

    model_path_group = common_parser.add_mutually_exclusive_group(
        required=True)
    model_path_group.add_argument('--save', type=str, default=None)
    model_path_group.add_argument('--load', type=str, default=None)

    common_parser.add_argument('--split', type=str, required=True)

    common_parser.add_argument('--p', '-p', type=int, default=16)
    common_parser.add_argument('--k', '-k', type=int, default=5)

    common_parser.add_argument('--gamma', '-g', type=float, default=80)
    common_parser.add_argument('--margin', '-m', type=float, default=0.4)

    common_parser.add_argument('--batch-size', '-bs', type=int, default=128)

    common_parser.add_argument('--epoch-num', '-en', type=int, default=100)
    common_parser.add_argument('--train-epoch-size', '-tes',
                               type=int, default=1000)
    common_parser.add_argument('--valid-epoch-size', '-ves',
                               type=int, default=200)
    common_parser.add_argument('--lr', type=float, default=1e-3)

    common_parser.add_argument('--seed', '-s', type=int, default=0)

    common_parser.add_argument('--disable-cuda', action='store_true')

    common_parser.add_argument('--output-size', '-os', type=int, default=128)

    subparsers = parser.add_subparsers(dest='model', required=True)

    bof_parser = subparsers.add_parser('bof', parents=[common_parser])
    bof_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    bof_parser.add_argument('--feature-emb-size', '-fs', type=int, default=128)

    sbt_parser = subparsers.add_parser('sbt', parents=[common_parser])
    sbt_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    sbt_parser.add_argument('--token-emb-size', '-ts', type=int, default=128)
    sbt_parser.add_argument('--hidden-size', '-hs', type=int, default=128)

    gnn_parser = subparsers.add_parser('gnn', parents=[common_parser])
    gnn_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    gnn_parser.add_argument('--node-emb-size', '-ns', type=int, default=128)
    gnn_parser.add_argument('--num-layers', '-nl', type=int, default=3)

    com_parser = subparsers.add_parser('com', parents=[common_parser])
    com_parser.add_argument('--bof-dataset-dir', '-bf',
                            type=str, required=True)
    com_parser.add_argument('--gnn-dataset-dir', '-gf',
                            type=str, required=True)
    com_parser.add_argument('--feature-emb-size', '-fs', type=int, default=128)
    com_parser.add_argument('--node-emb-size', '-ns', type=int, default=128)
    com_parser.add_argument('--num-layers', '-nl', type=int, default=3)

    c2v_parser = subparsers.add_parser('c2v', parents=[common_parser])
    c2v_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    c2v_parser.add_argument('--leaf-emb-size', '-ls', type=int, default=128)
    c2v_parser.add_argument('--path-emb-size', '-ps', type=int, default=128)
    c2v_parser.add_argument('--code-vec-size', '-cs', type=int, default=384)

    ncc_parser = subparsers.add_parser('ncc', parents=[common_parser])
    ncc_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    ncc_parser.add_argument('--i2v-emb', '-ie', type=str,
                            default=os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                'ncc/ncc/published_results/emb.p'))
    ncc_parser.add_argument('--rnn-size', '-rs', type=int, default=200)
    ncc_parser.add_argument('--dense-size', '-ds', type=int, default=32)
    ncc_parser.add_argument('--disable-i2v-emb', '-noi2v',
                            dest='use_i2v_emb', action='store_false')

    c2v_parser = subparsers.add_parser('c2s', parents=[common_parser])
    c2v_parser.add_argument('--dataset-dir', '-f', type=str, required=True)
    c2v_parser.add_argument('--emb-size', '-es', type=int, default=128)
    c2v_parser.add_argument('--rnn-size', '-rs', type=int, default=256)
    c2v_parser.add_argument('--decoder-size', '-ds', type=int, default=320)

    return parser.parse_args()


def pairwise_cosine_similarity(h):
    h_norm = torch.nn.functional.normalize(h, dim=1)
    sim = torch.mm(h_norm, h_norm.transpose(0, 1))
    return sim


def compute_pairwise_scores(h, pids):
    sim = pairwise_cosine_similarity(h)
    inds = torch.triu_indices(len(pids), len(pids), offset=1)
    sim = sim[inds[0], inds[1]]
    positive = pids[inds[0]] == pids[inds[1]]
    s_p = sim[positive]
    s_n = sim[~positive]
    return s_p, s_n


def iterations(args, epoch, model, criterion, optimizer, data_iter, num_iters, training):
    model.train(training)

    total_loss = 0

    with tqdm.trange(num_iters) as progress:
        for _ in progress:
            input, pids = next(data_iter)
            input = [x.to(device=args.device) for x in input]

            code_vecs = model(*input)
            s_p, s_n = compute_pairwise_scores(
                code_vecs, pids.to(device=args.device))
            loss = criterion(s_p, s_n)

            total_loss += loss.item()

            if training:
                model.zero_grad()
                loss.backward()
                optimizer.step()

            progress.set_description(f'Epoch {epoch} loss: {loss.item():.8f}')

    avg_loss = total_loss / num_iters

    if training:
        print(f'- training avg loss: {avg_loss:.8f}')
    else:
        print(f'- validation avg loss: {avg_loss:.8f}')

    return avg_loss


def train(args, model, dataset, train_split, valid_split):
    criterion = CircleLoss(gamma=args.gamma, m=args.margin)
    train_gen_fun = dataset.get_pk_sample_generator_function(
        train_split, args.p, args.k)
    valid_gen_fun = dataset.get_pk_sample_generator_function(
        valid_split, args.p, args.k)
    train_num_iters = args.train_epoch_size
    valid_num_iters = args.valid_epoch_size

    criterion.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = None
    best_epoch = 0

    for epoch in range(1, args.epoch_num + 1):
        iterations(args, epoch, model, criterion, optimizer,
                   train_gen_fun(), train_num_iters, True)

        best_val, best_epoch = validate(args, model, dataset, valid_split, criterion,
                                        epoch, best_val, best_epoch)

        if epoch == best_epoch:
            output_path = os.path.join(args.save, f'model.pt')
            torch.save(model.state_dict(), output_path)


def validate(args, model, dataset, test_split, criterion, epoch, best_val, best_epoch):
    code_vecs, pids = run_test(args, model, dataset, test_split)

    code_vecs = code_vecs
    pids = pids
    sim = pairwise_cosine_similarity(code_vecs)

    map_r = map_at_r(sim, pids)
    if best_val is None or map_r > best_val:
        best_val = map_r
        best_epoch = epoch
    print(
        f'* validation MAP@R: {map_r}, best epoch: {best_epoch}, best MAP@R: {best_val}')

    return best_val, best_epoch


def run_test(args, model, dataset, test_split):
    model.eval()

    test_gen_fun, num_iters = dataset.get_data_generator_function(
        test_split, args.batch_size, shuffle=False)

    code_vecs = []
    pids = []
    with tqdm.tqdm(test_gen_fun(), total=num_iters) as progress:
        for input, pids_batch in progress:
            input = [x.to(device=args.device) for x in input]
            with torch.no_grad():
                v = model(*input)
            code_vecs.append(v.detach().cpu())
            pids.append(pids_batch)
    code_vecs = torch.cat(code_vecs, dim=0)
    pids = torch.cat(pids)

    return code_vecs, pids


def test(args, model, dataset, test_split):
    code_vecs, pids = run_test(args, model, dataset, test_split)
    sim = pairwise_cosine_similarity(code_vecs)
    compute_metrics(sim, pids)


def compute_metrics(sim, pids):
    inds = torch.triu_indices(len(pids), len(pids), offset=1)
    scores = sim[inds[0], inds[1]]
    labels = pids[inds[0]] == pids[inds[1]]

    map_r = map_at_r(sim, pids)
    ap = average_precision(labels, scores)
    # ap = average_precision_score(labels.numpy(), scores.numpy())
    auprg = area_under_prg(labels.numpy(), scores.numpy())

    # print(f'MAP@R: {map_r}, AP: {ap}')
    print(f'MAP@R: {map_r}, AP: {ap}, AUPRG: {auprg}')


def map_at_r(sim, pids):
    r = torch.bincount(pids) - 1
    max_r = r.max()

    mask = torch.arange(max_r)[None, :] < r[pids][:, None]

    sim = sim.clone()
    ind = np.diag_indices(len(sim))
    sim[ind[0], ind[1]] = -np.inf

    _, result = torch.topk(sim, max_r, dim=1, sorted=True)

    tp = (pids[result] == pids[:, None])
    tp[~mask] = False

    valid = r[pids] > 0

    p = torch.cumsum(tp, dim=1).float() / torch.arange(1, max_r+1)[None, :]
    ap = (p * tp).sum(dim=1)[valid] / r[pids][valid]

    return ap.mean().item()


def average_precision(labels, scores):
    assert labels.dtype == torch.bool

    # desc_score_indices = torch.argsort(scores.cuda(), descending=True).cpu()
    desc_score_indices = torch.argsort(scores, descending=True)

    scores = scores[desc_score_indices]
    labels = labels[desc_score_indices]
    tps = labels.cumsum(dim=0)
    fps = torch.arange(len(labels)) + 1 - tps

    tps_float = tps.float()
    precision = tps_float / (tps + fps)
    precision[torch.isnan(precision)] = 0
    recall = tps_float / tps[-1]

    last_ind = torch.searchsorted(tps, tps[-1])
    precision = precision[:last_ind+1]
    recall = recall[:last_ind+1]

    recall_diff = torch.cat((
        torch.tensor([recall[0].item()]),
        recall[1:] - recall[:-1]
    ))
    return (recall_diff * precision).sum().item()


def area_under_prg(labels, scores):
    prg_curve = prg.create_prg_curve(labels, scores)
    auprg = prg.calc_auprg(prg_curve)
    return auprg


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    if args.model == 'bof':
        dataset = BoFDataset(args.dataset_dir)
        model = BagOfFeaturesModel(
            args.feature_emb_size, len(dataset.vocab), args.output_size)
    elif args.model == 'sbt':
        dataset = SBTDataset(args.dataset_dir)
        model = SBTModel(args.token_emb_size, len(dataset.vocab),
                         args.hidden_size, args.output_size)
    elif args.model == 'gnn':
        dataset = GNNDataset(args.dataset_dir)
        model = GNNModel(args.node_emb_size, len(dataset.vocab),
                         args.output_size, args.num_layers)
    elif args.model == 'com':
        dataset = ComDataset(args.bof_dataset_dir, args.gnn_dataset_dir)
        model = ComModel(args.feature_emb_size, len(dataset.bof_dataset.vocab),
                         args.node_emb_size, len(dataset.gnn_dataset.vocab),
                         args.output_size, args.num_layers)
    elif args.model == 'c2v':
        dataset = C2VDataset(args.dataset_dir)
        model = C2VModel(args.leaf_emb_size, len(dataset.leaf_vocab),
                         args.path_emb_size, len(dataset.path_vocab),
                         args.code_vec_size, args.output_size)
    elif args.model == 'ncc':
        with open(args.i2v_emb, 'rb') as f:
            i2v_emb = pickle.load(f)
        i2v_emb = torch.from_numpy(i2v_emb)
        i2v_emb = torch.nn.functional.normalize(i2v_emb)
        dataset = NCCDataset(args.dataset_dir)
        model = NCCModel(i2v_emb, args.rnn_size,
                         args.dense_size, args.output_size, args.use_i2v_emb)
    elif args.model == 'c2s':
        dataset = C2SDataset(args.dataset_dir)
        model = C2SModel(args.emb_size, len(dataset.subtoken_vocab),
                         len(dataset.node_vocab), args.rnn_size,
                         args.decoder_size, args.output_size)
    else:
        raise Exception

    model.to(device=args.device)

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        train(args, model, dataset, split[0], split[1])
    else:
        model.load_state_dict(torch.load(args.load, map_location=args.device))
        test(args, model, dataset, split[2])


if __name__ == "__main__":
    main()
