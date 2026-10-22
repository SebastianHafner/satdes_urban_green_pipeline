from __future__ import annotations

import argparse
import logging
from typing import List

from urban_green.config import PipelineConfig
from urban_green.pipeline import run
from urban_green.thresholding import GMM_SAMPLE_SIZE_DEFAULT


def parse_years(spec: str) -> List[int]:
    """Parse '2018:2025' (inclusive range) or '2018,2020,2022' (explicit list)."""
    if ":" in spec:
        start, end = spec.split(":")
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-year urban green statistics pipeline")
    parser.add_argument("--roi", required=True, help="Path to ROI vector file (any GDAL/OGR format)")
    parser.add_argument("--years", required=True, help="Year range 'start:end' or comma list, e.g. 2018:2025")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--tile-store",
        default="./tile_store",
        help=(
            "Shared cache directory for acquired NDVI tiles, reused across runs/ROIs. "
            "Use --force-reacquire or a fresh --tile-store if you change them."
        ),
    )
    parser.add_argument(
        "--force-reacquire", action="store_true", help="Bypass the tile store cache and re-download every tile"
    )
    parser.add_argument("--tile-size", type=int, default=10_000, help="DES request tile size in meters")
    parser.add_argument("--cloud-threshold", type=int, default=70, help="Max scene cloud cover percentage")
    parser.add_argument("--crs", default="EPSG:3006", help="Working CRS (default: SWEREF99 TM)")
    parser.add_argument("--majority-vote-window", type=int, default=3, help="Temporal smoothing window (years)")
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=None,
        help="User-defined NDVI threshold for vegetation; skips the automatic GMM threshold detection",
    )
    parser.add_argument(
        "--gmm-sample-size",
        type=int,
        default=GMM_SAMPLE_SIZE_DEFAULT,
        help=(
            "Maximum number of pixels used to fit the automatic GMM threshold; ROIs with more valid "
            "pixels are randomly subsampled. Ignored when --ndvi-threshold is set."
        ),
    )
    parser.add_argument(
        "--region-id-field", default=None, help="ROI column to group tiles/statistics by named sub-region"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        roi_path=args.roi,
        years=parse_years(args.years),
        output_dir=args.out,
        tile_store_dir=args.tile_store,
        force_reacquire=args.force_reacquire,
        tile_size=args.tile_size,
        cloud_threshold=args.cloud_threshold,
        crs=args.crs,
        majority_vote_window=args.majority_vote_window,
        ndvi_threshold=args.ndvi_threshold,
        gmm_sample_size=args.gmm_sample_size,
        region_id_field=args.region_id_field,
    )
    run(config)


if __name__ == "__main__":
    main()
