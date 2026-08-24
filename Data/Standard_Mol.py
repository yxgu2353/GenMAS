import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class StructureResult:
    standardized_smiles: str | None
    status: str
    fragment_count: int = 0
    organic_fragment_count: int = 0


def _contains_carbon(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def standardize_smiles(
    smiles: object,
    *,
    clear_charge: bool = True,
    canonical_tautomer: bool = False,
    isomeric: bool = False,
) -> StructureResult:
    if pd.isna(smiles) or not str(smiles).strip():
        return StructureResult(None, "missing_smiles")

    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return StructureResult(None, "invalid_smiles")

        mol = rdMolStandardize.Cleanup(mol)
        fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
        organic_fragments = [frag for frag in fragments if _contains_carbon(frag)]

        if not organic_fragments:
            return StructureResult(
                None,
                "inorganic",
                fragment_count=len(fragments),
                organic_fragment_count=0,
            )

        if len(organic_fragments) > 1:
            return StructureResult(
                None,
                "organic_mixture",
                fragment_count=len(fragments),
                organic_fragment_count=len(organic_fragments),
            )

        retained = organic_fragments[0]
        if clear_charge:
            retained = rdMolStandardize.Uncharger().uncharge(retained)
        if canonical_tautomer:
            retained = rdMolStandardize.TautomerEnumerator().Canonicalize(retained)

        Chem.SanitizeMol(retained)
        standardized = Chem.MolToSmiles(
            retained,
            canonical=True,
            isomericSmiles=isomeric,
        )
        status = "salt_counterion_removed" if len(fragments) > 1 else "retained"
        return StructureResult(
            standardized,
            status,
            fragment_count=len(fragments),
            organic_fragment_count=1,
        )
    except Exception:
        return StructureResult(None, "standardization_failed")


def _normalized_endpoint_name(endpoint: str) -> str:
    return "".join(ch for ch in endpoint.lower() if ch.isalnum())


def endpoint_scale(
    endpoint: str,
    log_endpoints: set[str],
    linear_endpoints: set[str],
) -> str:
    """Return ``log10`` or ``linear`` for the replicate-consistency rule."""

    if endpoint in log_endpoints:
        return "log10"
    if endpoint in linear_endpoints:
        return "linear"

    normalized = _normalized_endpoint_name(endpoint)
    if normalized.startswith("log") or normalized.startswith("pka"):
        return "log10"
    return "linear"


def values_exceed_two_orders(values: np.ndarray, scale: str) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return False

    minimum = float(values.min())
    maximum = float(values.max())
    if scale == "log10":
        return (maximum - minimum) > 2.0

    if minimum <= 0:
        return minimum != maximum
    return (maximum / minimum) > 100.0


def _distinct_numeric_values(series: pd.Series) -> np.ndarray:

    numeric = pd.to_numeric(series, errors="coerce")
    return pd.unique(numeric.dropna()).astype(float)


def aggregate_endpoint(
    retained: pd.DataFrame,
    endpoint: str,
    *,
    log_endpoints: set[str],
    linear_endpoints: set[str],
) -> tuple[pd.Series, dict[str, object]]:

    available = retained.loc[
        retained[endpoint].notna(), ["standardized_SMILES", endpoint]
    ].copy()
    numeric = pd.to_numeric(available[endpoint], errors="coerce")
    non_numeric_count = int((available[endpoint].notna() & numeric.isna()).sum())
    available[endpoint] = numeric
    available = available.dropna(subset=[endpoint])

    scale = endpoint_scale(endpoint, log_endpoints, linear_endpoints)
    output: dict[str, float] = {}
    groups_with_replicates = 0
    groups_averaged = 0
    unreliable_groups_removed = 0
    exact_duplicate_observations = 0

    for standardized_smiles, group in available.groupby(
        "standardized_SMILES", sort=False
    ):
        values_all = group[endpoint].to_numpy(dtype=float)
        values = _distinct_numeric_values(group[endpoint])
        exact_duplicate_observations += len(values_all) - len(values)

        if len(values_all) > 1:
            groups_with_replicates += 1
        if values_exceed_two_orders(values, scale):
            unreliable_groups_removed += 1
            continue
        if len(values) > 1:
            groups_averaged += 1
        output[standardized_smiles] = float(np.mean(values))

    result = pd.Series(output, name=endpoint, dtype=float)
    result.index.name = "SMILES"
    stats = {
        "endpoint": endpoint,
        "scale_rule": scale,
        "observations_after_structure_filtering": int(len(available)),
        "non_numeric_values_removed": non_numeric_count,
        "compound_endpoint_groups_before_consistency_filter": int(
            available["standardized_SMILES"].nunique()
        ),
        "groups_with_replicates": int(groups_with_replicates),
        "exact_duplicate_observations_ignored": int(exact_duplicate_observations),
        "groups_averaged_from_multiple_distinct_values": int(groups_averaged),
        "unreliable_groups_removed_over_two_orders": int(
            unreliable_groups_removed
        ),
        "final_compound_endpoint_records": int(len(result)),
    }
    return result, stats


def curate_table(
    input_path: Path,
    output_path: Path,
    stats_prefix: Path,
    *,
    smiles_column: str,
    log_endpoints: set[str],
    linear_endpoints: set[str],
    canonical_tautomer: bool,
    isomeric: bool,
) -> None:

    data = pd.read_csv(input_path)
    if smiles_column not in data.columns:
        raise KeyError(f"SMILES column not found: {smiles_column!r}")

    endpoints = [column for column in data.columns if column != smiles_column]
    if not endpoints:
        raise ValueError("The input table does not contain any endpoint columns.")

    structure_results = [
        standardize_smiles(
            smiles,
            canonical_tautomer=canonical_tautomer,
            isomeric=isomeric,
        )
        for smiles in data[smiles_column]
    ]
    data["standardized_SMILES"] = [
        result.standardized_smiles for result in structure_results
    ]
    data["structure_status"] = [result.status for result in structure_results]

    retained = data.loc[data["standardized_SMILES"].notna()].copy()
    endpoint_results: list[pd.Series] = []
    endpoint_stats: list[dict[str, object]] = []
    for endpoint in endpoints:
        result, stats = aggregate_endpoint(
            retained,
            endpoint,
            log_endpoints=log_endpoints,
            linear_endpoints=linear_endpoints,
        )
        endpoint_results.append(result)
        stats["raw_non_missing_observations"] = int(data[endpoint].notna().sum())
        endpoint_stats.append(stats)

    curated = pd.concat(endpoint_results, axis=1, join="outer")
    curated = curated.dropna(how="all").reset_index()

    status_counts = pd.Series(
        [result.status for result in structure_results], dtype="object"
    ).value_counts()
    structure_stats = {
        "input_file": str(input_path),
        "raw_table_rows": int(len(data)),
        "raw_non_missing_endpoint_observations": int(
            data[endpoints].notna().sum().sum()
        ),
        "raw_unique_smiles_strings": int(data[smiles_column].nunique(dropna=True)),
        "missing_smiles_rows": int(status_counts.get("missing_smiles", 0)),
        "invalid_smiles_rows_removed": int(status_counts.get("invalid_smiles", 0)),
        "standardization_failed_rows_removed": int(
            status_counts.get("standardization_failed", 0)
        ),
        "inorganic_rows_removed": int(status_counts.get("inorganic", 0)),
        "organic_mixture_rows_removed": int(
            status_counts.get("organic_mixture", 0)
        ),
        "salt_rows_with_counterions_removed": int(
            status_counts.get("salt_counterion_removed", 0)
        ),
        "rows_retained_after_structure_processing": int(len(retained)),
        "unique_standardized_structures_after_structure_processing": int(
            retained["standardized_SMILES"].nunique()
        ),
        "final_unique_structures_with_at_least_one_endpoint": int(len(curated)),
        "final_compound_endpoint_records": int(
            curated[endpoints].notna().sum().sum()
        ),
        "canonical_tautomer_enabled": canonical_tautomer,
        "isomeric_smiles_enabled": isomeric,
        "two_order_linear_threshold": "max/min > 100",
        "two_order_log10_threshold": "max-min > 2",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_prefix.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(output_path, index=False)
    pd.DataFrame(endpoint_stats).to_csv(
        stats_prefix.with_name(stats_prefix.name + "_endpoints.csv"), index=False
    )
    with stats_prefix.with_name(stats_prefix.name + "_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(structure_stats, handle, indent=2, ensure_ascii=False)

    print(json.dumps(structure_stats, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV file")
    parser.add_argument("--output", required=True, type=Path, help="Curated CSV file")
    parser.add_argument(
        "--stats-prefix",
        required=True,
        type=Path,
        help="Prefix for *_summary.json and *_endpoints.csv audit files",
    )
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument(
        "--log-endpoints",
        nargs="*",
        default=[],
        help="Endpoint columns known to contain base-10 logarithmic values",
    )
    parser.add_argument(
        "--linear-endpoints",
        nargs="*",
        default=[],
        help="Endpoint columns forced to use the linear max/min rule",
    )
    parser.add_argument(
        "--canonical-tautomer",
        action="store_true",
        help="Canonicalize tautomers (disabled by default to match the old script)",
    )
    parser.add_argument(
        "--isomeric",
        action="store_true",
        help="Retain stereochemical information in output SMILES",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overlap = set(args.log_endpoints) & set(args.linear_endpoints)
    if overlap:
        raise ValueError(
            "Endpoints cannot be listed as both logarithmic and linear: "
            + ", ".join(sorted(overlap))
        )

    curate_table(
        args.input,
        args.output,
        args.stats_prefix,
        smiles_column=args.smiles_column,
        log_endpoints=set(args.log_endpoints),
        linear_endpoints=set(args.linear_endpoints),
        canonical_tautomer=args.canonical_tautomer,
        isomeric=args.isomeric,
    )


if __name__ == "__main__":
    main()
