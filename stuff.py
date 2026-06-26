import pandas as pd
from pathlib import Path


INPUT_CSV = "orbslam3_feature_observations.csv"
OUTPUT_CSV = "orbslam3_feature_tracks.csv"
OUTPUT_PARQUET = "orbslam3_feature_tracks.parquet"


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Loaded rows:", len(df))
    print(df.head())
    print(df.columns)

    # Keep only valid depth rows
    df = df[df["depth"] > 0].copy()

    # Make sure the IDs are usable
    df["frame_id"] = df["frame_id"].astype(int)
    df["map_point_id"] = df["map_point_id"].astype(int)

    # Current frame observations, this is frame t
    df_t = df.rename(columns={
        "frame_id": "frame_t",
        "timestamp": "timestamp_t",
        "feature_idx": "feature_idx_t",
        "u": "u_t",
        "v": "v_t",
        "depth": "depth_t",
        "u_right": "u_right_t",
        "is_outlier": "is_outlier_t",
    })

    # Next frame observations, this is frame t+1
    df_t1 = df.rename(columns={
        "frame_id": "frame_t1",
        "timestamp": "timestamp_t1",
        "feature_idx": "feature_idx_t1",
        "u": "u_t1",
        "v": "v_t1",
        "depth": "depth_t1",
        "u_right": "u_right_t1",
        "is_outlier": "is_outlier_t1",
    })

    # To join frame t to frame t+1, create expected next frame id
    df_t["frame_t1"] = df_t["frame_t"] + 1

    tracks = df_t.merge(
        df_t1,
        on=["map_point_id", "frame_t1"],
        how="inner"
    )

    # Keep the columns you actually care about
    tracks = tracks[
        [
            "frame_t",
            "frame_t1",
            "timestamp_t",
            "timestamp_t1",
            "map_point_id",
            "u_t",
            "v_t",
            "u_t1",
            "v_t1",
            "depth_t",
            "u_right_t",
            "feature_idx_t",
            "feature_idx_t1",
        ]
    ].copy()

    tracks["dt"] = tracks["timestamp_t1"] - tracks["timestamp_t"]

    print("Final track rows:", len(tracks))
    print(tracks.head())

    tracks.to_csv(OUTPUT_CSV, index=False)
    tracks.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()