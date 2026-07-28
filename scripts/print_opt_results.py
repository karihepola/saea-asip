import json
import os
from glob import glob

import numpy as np
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.survival.rank_and_crowding.metrics import (
    get_crowding_function,
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BENCHMARKS = [
    "adpcm", "aes", "blowfish", "core", "crc32", "edn", "gsm", "huffbench",
    "jpeg", "md5", "mips", "motion", "nettle-aes", "nsichneu", "picojpeg",
    "primecount", "qrduino", "sglib-combined", "sha", "slre", "statemate",
    "tarfind", "ud", "wikisort", "matmult-int",
]

FINE_TUNE_DATA_AMOUNTS = [0, 50, 400, 1600]
ENSEMBLES = [1, 2, 5, 10]
SAEA_SEEDS = range(1,6)

# First gen is pure model-based sampling, hence +1
SAEA_GENS = [6, 11, 21, 31, 41, 51]
BASELINE_GENS = [100, 200, 300, 500, 700, 900, 1000, 1100, 1200]

OPT_DATA_DIR = "opt_data"


def nsga2_select2(F, max_size: int = 96):
    if F.shape[0] <= max_size:
        return F

    fronts = NonDominatedSorting().do(F)
    selected = []

    for front in fronts:
        front = np.asarray(front)

        if len(selected) + len(front) <= max_size:
            selected.extend(front)
        else:
            cd = get_crowding_function("cd").do(F[front])
            remaining = max_size - len(selected)
            selected.extend(front[np.argsort(-cd)[:remaining]])
            break

    return F[np.asarray(selected)]


def is_pareto_efficient(points):
    nd_indices = NonDominatedSorting().do(
        points, only_non_dominated_front=True
    )
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[nd_indices] = True
    return mask

def print_subtable(table_results):
    col1_width = max(len(str(row[0])) for row in table_results)
    for row in table_results:
        ensemble, hv, gd, igd = row

        hv_mean, hv_std = hv
        gd_mean, gd_std = gd
        igd_mean, igd_std = igd

        gd_mean = 1e2*gd_mean
        gd_std = 1e2*gd_std
        
        igd_mean = 1e2*igd_mean
        igd_std =  1e2*igd_std
        print(
            f"{ensemble:>{col1_width}} & "
            f"{hv_mean:.3f} ± {hv_std:.3f} & "
            f"{gd_mean:.3f} ± {gd_std:.2f} & "
            f"{igd_mean:.3f} ± {igd_std:.2f} \\\\"
        )



def find_nadir_points_fast(json_files):
    benchmark_nadir = {}

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        for benchmark, points in data.items():
            points = np.unique(np.asarray(points).T, axis=0)
            points = points[is_pareto_efficient(points)]

            if points.size == 0:
                continue

            nadir = np.max(points, axis=0)

            if benchmark in benchmark_nadir:
                benchmark_nadir[benchmark] = np.maximum(
                    benchmark_nadir[benchmark], nadir
                )
            else:
                benchmark_nadir[benchmark] = nadir

    return benchmark_nadir


def calculate_metrics(ref_points_dict, json_files, nadir_points):
    all_data = {
        file: json.load(open(file, "r"))
        for file in json_files
    }

    hypervolumes = {}
    igd_pluses = {}
    gd_pluses = {}
    amounts = {}

    for benchmark in BENCHMARKS:
        ref_points = ref_points_dict[benchmark]

        igd_plus_calc = IGDPlus(ref_points)
        gd_calc = GDPlus(ref_points)

        hv_calc = HV(ref_point=nadir_points[benchmark])
        ref_hv = hv_calc.do(ref_points)

        for file, file_data in all_data.items():
            if benchmark not in file_data:
                raise KeyError(f"{benchmark} missing from {file}")

            points = np.unique(
                np.asarray(file_data[benchmark]).T,
                axis=0,
            )
            points = nsga2_select2(points)
            pareto_mask = is_pareto_efficient(points)
            points = points[pareto_mask]

            hypervolumes.setdefault(benchmark, []).append(
                (file, hv_calc.do(points) / ref_hv)
            )
            igd_pluses.setdefault(benchmark, []).append(
                (file, igd_plus_calc(points))
            )
            gd_pluses.setdefault(benchmark, []).append(
                (file, gd_calc(points))
            )
            amounts.setdefault(benchmark, []).append(
                (file, len(points))
            )

    file_metrics = {}

    for file in json_files:
        hv_values = [
            hv
            for values in hypervolumes.values()
            for f, hv in values
            if f == file
        ]
        igd_values = [
            igd
            for values in igd_pluses.values()
            for f, igd in values
            if f == file
        ]
        gd_values = [
            gd
            for values in gd_pluses.values()
            for f, gd in values
            if f == file
        ]
        amount_values = [
            amount
            for values in amounts.values()
            for f, amount in values
            if f == file
        ]

        file_metrics[file] = {
            "hv_mean": np.mean(hv_values),
            "hv_std": np.std(hv_values),
            "igd_mean": np.mean(igd_values),
            "igd_std": np.std(igd_values),
            "gd_mean": np.mean(gd_values),
            "gd_std": np.std(gd_values),
            "amounts_mean": np.mean(amount_values),
            "amounts_std": np.std(amount_values),
        }

    hv_all = [m["hv_mean"] for m in file_metrics.values()]
    igd_all = [m["igd_mean"] for m in file_metrics.values()]
    gd_all = [m["gd_mean"] for m in file_metrics.values()]

    avg_hv, std_hv = np.mean(hv_all), np.std(hv_all)
    avg_igd, std_igd = np.mean(igd_all), np.std(igd_all)
    avg_gd, std_gd = np.mean(gd_all), np.std(gd_all)

    print(
        f"[{json_files[0]:85s} | {len(json_files):2d}] "
        f"Avg HV: {avg_hv:.3f} ± {std_hv:.3f}, "
        f"Avg IGD⁺: {avg_igd:.4f} ± {std_igd:.4f}, "
        f"Avg GD⁺: {avg_gd:.4f} ± {std_gd:.4f}"
    )

    return (
        (avg_hv, std_hv),
        (avg_gd, std_gd),
        (avg_igd, std_igd),
    )


def calc_nadirs():
    json_files = [ref_file]

    for gen in BASELINE_GENS:
        path = f"{OPT_DATA_DIR}/baseline_opt_data/gen_{gen}.json"
        if os.path.exists(path):
            json_files.append(path)

    for gen in SAEA_GENS:
        for ensemble in ENSEMBLES:
            for seed in SAEA_SEEDS:
                json_files.append(
                    f"{OPT_DATA_DIR}/saea_opt_data/"
                    f"{ensemble}_ensembles/run_{seed}/{gen}/results_combined.json"
                )
    for ensemble in ENSEMBLES:
        for amount in FINE_TUNE_DATA_AMOUNTS:
            json_files.extend(glob(f"{OPT_DATA_DIR}/model_opt_data/data_{amount}/run_{ensemble}_*.json"))

    return find_nadir_points_fast(json_files)


# === Main program ===

ref_file = f"{OPT_DATA_DIR}/baseline_opt_data/gen_1200.json"

nadir_points = calc_nadirs()

with open(ref_file, "r") as f:
    data = json.load(f)

ref_points = {}
for benchmark, points in data.items():
    points = np.unique(np.asarray(points).T, axis=0)
    ref_points[benchmark] = nsga2_select2(points)

for amount in FINE_TUNE_DATA_AMOUNTS:
    table_results = []
    print(
        f"------ Fine tune amount = {amount:4d}, "
        f"ensemble size = [1,2,5,10] ------"
    )

    for ensemble in ENSEMBLES:
        json_files = glob(
            f"{OPT_DATA_DIR}/model_opt_data/data_{amount}/run_{ensemble}_*.json"
        )
        hv, gd, igd = calculate_metrics(ref_points, json_files, nadir_points)
        table_results.append([ensemble, hv, gd, igd])
    print_subtable(table_results)

for gen in SAEA_GENS:
    table_results = []
    print(
        f"------ Iterations = {gen:4d}, "
        f"ensemble size = [1,2,5,10] ------"
    )

    for ensemble in ENSEMBLES:
        json_files = [
            f"{OPT_DATA_DIR}/saea_opt_data/"
            f"{ensemble}_ensembles/run_{seed}/{gen}/results_combined.json"
            for seed in SAEA_SEEDS
        ]

        hv, gd, igd = calculate_metrics(ref_points, json_files, nadir_points)
        table_results.append([ensemble, hv, gd, igd])
    print_subtable(table_results)