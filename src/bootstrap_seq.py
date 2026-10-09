"""Paired sequence bootstrap with repeated draws preserved.

Per-image COCO matching is cached; accumulation retains every sampled occurrence.
Checks against literal duplication with remapped COCO IDs precede computation.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']
N_BOOT_50, N_BOOT_AP = 3000, 1000
MAX_DETS = 100  # Explicitly matches sq_eval.py and retained pycocotools defaults.


def rows_test(manifest):
    with Path(manifest).open(encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if r['split'] == 'test']


def build_gt(data_dir):
    data_dir = Path(data_dir)
    rows = rows_test(data_dir / 'manifest.csv')
    images, anns = [], []
    for image_id, row in enumerate(rows, 1):
        with Image.open(data_dir / 'rgb/images/test' / (row['name'] + '.png')) as im:
            w, h = im.size
        images.append({'id': image_id, 'file_name': row['name'], 'width': w, 'height': h})
        label = data_dir / 'rgb/labels/test' / (row['name'] + '.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            anns.append({'id': len(anns)+1, 'image_id': image_id, 'category_id': int(c)+1,
                         'bbox': [(cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h],
                         'area': bw*w*bh*h, 'iscrowd': 0})
    return rows, {'info': {}, 'images': images, 'annotations': anns,
                  'categories': [{'id': i+1, 'name': n} for i, n in enumerate(NAMES)]}


def make_evaluator(gt, dets):
    with redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = copy.deepcopy(gt)
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(copy.deepcopy(dets)), 'bbox')
        evaluator.params.areaRng = [[0, 1e10]]
        evaluator.params.areaRngLbl = ['all']
        evaluator.params.maxDets = [MAX_DETS]
        evaluator.evaluate()
    return evaluator


class CachedAP:
    def __init__(self, gt, dets):
        self.gt, self.dets = gt, dets
        evaluator = make_evaluator(gt, dets)
        self.recall_thresholds = evaluator.params.recThrs
        self.n_iou_thresholds = len(evaluator.params.iouThrs)
        self.cat_ids = evaluator.params.catIds
        self.cache = {}
        entries = np.array(evaluator.evalImgs, dtype=object).reshape(len(self.cat_ids), -1)
        for cat, block in zip(self.cat_ids, entries):
            for image_id, entry in zip(evaluator.params.imgIds, block):
                if entry is not None:
                    self.cache[cat, image_id] = (
                        np.asarray(entry['dtScores']), entry['dtMatches'] != 0,
                        entry['dtIgnore'], int(np.count_nonzero(entry['gtIgnore'] == 0)))

    def evaluate(self, image_ids, ap50_only=False):
        """image_ids is a list of occurrences, not a set; repetitions remain."""
        values = []
        for cat in self.cat_ids:
            entries = [self.cache[cat, i] for i in image_ids if (cat, i) in self.cache]
            n_gt = sum(e[3] for e in entries)
            if not n_gt:
                continue  # COCO excludes categories absent from sampled ground truth.
            scores = np.concatenate([e[0] for e in entries])
            if not len(scores):
                values.extend([0.0] * (1 if ap50_only else self.n_iou_thresholds))
                continue
            order = np.argsort(-scores, kind='mergesort')
            sl = slice(0, 1) if ap50_only else slice(None)
            matched = np.concatenate([e[1][sl] for e in entries], axis=1)[:, order]
            ignored = np.concatenate([e[2][sl] for e in entries], axis=1)[:, order]
            tp = np.cumsum(matched & ~ignored, axis=1).astype(float)
            fp = np.cumsum(~matched & ~ignored, axis=1).astype(float)
            recall = tp / n_gt
            precision = tp / (tp + fp + np.spacing(1))
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            for rc, pr in zip(recall, precision):
                indices = np.searchsorted(rc, self.recall_thresholds, side='left')
                q = np.zeros(len(self.recall_thresholds))
                valid = indices < len(pr)
                q[valid] = pr[indices[valid]]
                values.append(float(q.mean()))
        return float(np.mean(values)) if values else -1.0


def literal_ap(gt, dets, image_ids, ap50_only=False):
    """Independent reference: duplicate/remap actual images, labels and predictions."""
    images = {i['id']: i for i in gt['images']}
    anns, preds = defaultdict(list), defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    for d in dets:
        preds[d['image_id']].append(d)
    sample_gt = {'info': {}, 'categories': copy.deepcopy(gt['categories']), 'images': [], 'annotations': []}
    sample_dets = []
    for new_id, old_id in enumerate(image_ids, 1):
        sample_gt['images'].append({**images[old_id], 'id': new_id})
        for a in anns[old_id]:
            sample_gt['annotations'].append({**a, 'id': len(sample_gt['annotations'])+1, 'image_id': new_id})
        sample_dets.extend({**d, 'image_id': new_id} for d in preds[old_id])
    evaluator = make_evaluator(sample_gt, sample_dets)
    with redirect_stdout(io.StringIO()):
        evaluator.accumulate()
    p = evaluator.eval['precision']
    if ap50_only:
        p = p[:1]
    return float(p[p > -1].mean())


def sampled_frames(sample, frames_by_seq):
    return [i for seq in sample for i in frames_by_seq[seq]]


def validate_cache(cache, frames_by_seq):
    seqs = sorted(frames_by_seq)
    draws = [[seqs[0], seqs[0], seqs[-1]], [seqs[-1], seqs[0], seqs[-1]]]
    errors = []
    for draw in draws:
        ids = sampled_frames(draw, frames_by_seq)
        assert len(ids) == sum(len(frames_by_seq[s]) for s in draw)
        assert len(ids) > len(set(ids))
        for ap50_only in [True, False]:
            expected = literal_ap(cache.gt, cache.dets, ids, ap50_only)
            actual = cache.evaluate(ids, ap50_only)
            errors.append(abs(expected-actual))
    assert max(errors) < 1e-12, errors
    return {'n_literal_checks': len(errors), 'max_absolute_error': max(errors)}


def bootstrap_pair(cache_a, cache_b, frames_by_seq, rng, n50=N_BOOT_50, n_ap=N_BOOT_AP, progress=None):
    seqs = sorted(frames_by_seq)
    original_ids = sorted(i for ids in frames_by_seq.values() for i in ids)
    point50 = cache_b.evaluate(original_ids, True) - cache_a.evaluate(original_ids, True)
    point_ap = cache_b.evaluate(original_ids) - cache_a.evaluate(original_ids)
    arrays = []
    for ap50_only, n in [(True, n50), (False, n_ap)]:
        differences = []
        for iteration in range(n):
            draw = rng.choice(seqs, size=len(seqs), replace=True)
            ids = sampled_frames(draw, frames_by_seq)
            differences.append(cache_b.evaluate(ids, ap50_only) - cache_a.evaluate(ids, ap50_only))
            if progress and (iteration+1) % 250 == 0:
                progress('AP50' if ap50_only else 'AP', iteration+1, n)
        arrays.append(np.array(differences))
    ds50, ds_ap = arrays
    return {'dAP50_point': point50, 'dAP50_mean': float(ds50.mean()),
            'dAP50_ci95': [float(x) for x in np.percentile(ds50, [2.5, 97.5])],
            'dAP50_frac_pos': float((ds50 > 0).mean()), 'n50': n50,
            'dAP_point': point_ap, 'dAP_mean': float(ds_ap.mean()),
            'dAP_ci95': [float(x) for x in np.percentile(ds_ap, [2.5, 97.5])], 'n_ap': n_ap,
            'n_seqs': len(seqs), 'n_frames': len(original_ids)}, ds50, ds_ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sq-data', type=Path, default=Path('datasets/RLiViT/rlivit_sq'))
    parser.add_argument('--msq-data', type=Path, default=Path('datasets/RLiViT/rlivit_multi_sq'))
    parser.add_argument('--sq-runs', type=Path, default=Path('runs/rlivit_sq'))
    parser.add_argument('--msq-runs', type=Path, default=Path('runs/rlivit_msq'))
    parser.add_argument('--output', type=Path, default=Path('runs/rlivit_sq/bootstrap_seq.json'))
    args = parser.parse_args()
    sq_rows, sq_gt = build_gt(args.sq_data)
    msq_rows, msq_gt = build_gt(args.msq_data)
    rng = np.random.default_rng(42)
    paths = [args.sq_runs / f'dets_{exp}_s0.json' for exp in ['sq-4ch', 'sq-4ch-gbf', 'sq-4ch-gsw']]
    paths += [args.msq_runs / f'dets_{exp}_s0.json' for exp in ['msq-4ch', 'msq-5ch']]
    caches, validations = [], {}
    for i, path in enumerate(paths):
        rows, gt = (sq_rows, sq_gt) if i < 3 else (msq_rows, msq_gt)
        print('Preparing COCO matches:', path.name, flush=True)
        cache = CachedAP(gt, json.loads(path.read_text(encoding='utf-8')))
        frames = defaultdict(list)
        for image_id, row in enumerate(rows, 1):
            frames[row['seq']].append(image_id)
        validations[path.name] = validate_cache(cache, frames)
        caches.append(cache)
    comparisons = [('concat_vs_gbf_full', 0, 1, sq_rows, False),
                   ('concat_vs_gbf_night', 0, 1, sq_rows, True),
                   ('concat_vs_gsw_full', 0, 2, sq_rows, False),
                   ('depth_4ch_vs_5ch', 3, 4, msq_rows, False)]
    hash_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = {'generated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
           'method': 'paired_sequence_cluster_bootstrap_preserving_multiplicity',
           'sampling_seed': 42, 'training_seed': 0, 'n_seqs': 40, 'n_frames': len(sq_rows),
           'n_night_frames': sum(r['daytime'].startswith('night') for r in sq_rows),
           'n_boot_ap50': N_BOOT_50, 'n_boot_ap': N_BOOT_AP,
           'evaluation': {'iou_thresholds': [float(x) for x in np.linspace(.5, .95, 10)],
                          'recall_thresholds': 101, 'max_detections_per_image_per_category': MAX_DETS,
                          'matching': 'pycocotools COCO bbox', 'tie_sort': 'stable mergesort',
                          'absent_classes': 'excluded following COCO convention',
                          'point_estimate': 'direct AP difference on original image order',
                          'interval': '2.5th and 97.5th percentiles of paired differences'},
           'validation': validations, 'input_sha256': {p.name: hash_file(p) for p in paths},
           'manifest_sha256': {'sq': hash_file(args.sq_data/'manifest.csv'), 'msq': hash_file(args.msq_data/'manifest.csv')},
           'comparisons': {}}
    draws = {}
    for key, a, b, rows, night in comparisons:
        frames = defaultdict(list)
        for image_id, row in enumerate(rows, 1):
            if not night or row['daytime'].startswith('night'):
                frames[row['seq']].append(image_id)
        started = time.time()
        def progress(metric, iteration, n):
            print(f'{key}: {metric} {iteration}/{n}; elapsed {time.time()-started:.1f}s', flush=True)
        record, ds50, ds_ap = bootstrap_pair(caches[a], caches[b], frames, rng, progress=progress)
        out['comparisons'][key] = record
        draws[key+'__AP50'], draws[key+'__AP'] = ds50, ds_ap
        print(key, json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(args.output.with_name('bootstrap_draws.npz'), **draws)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
