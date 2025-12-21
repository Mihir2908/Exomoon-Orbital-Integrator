import os, json, pathlib
import boto3

from exomoon.params import SystemParams
from exomoon.simulation import run_simulation, run_simulation_for_years
from exomoon.eda import traj_to_frame, to_csv_bytes
from exomoon.plotting.anim import build_animation

def _load_params(path: str) -> tuple[SystemParams, float]:
    with open(path, "r") as f:
        d = json.load(f)
    p = SystemParams(
        Ts=float(d.get("Ts", 5772)),
        rs_solar=float(d.get("rs_solar", 1.0)),
        ms_solar=float(d.get("ms_solar", 1.0)),
        mp_earth=float(d.get("mp_earth", 1.0)),
        dp_cgs=float(d.get("dp_cgs", 5.5)),
        ap_AU=float(d.get("ap_AU", 1.0)),
        ep=float(d.get("ep", 0.0)),
        mm_earth=float(d.get("mm_earth", 0.01)),
        am_hill=float(d.get("am_hill", 0.25)),
        em=float(d.get("em", 0.0)),
        moon_retrograde=bool(d.get("moon_retrograde", False)),
    )
    years = float(d.get("years", 0.0))
    return p, years

def _split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), f"bad S3 URI: {uri}"
    rest = uri[5:]
    bucket, key = (rest.split("/", 1) + [""])[:2]
    return bucket, key

def main():
    in_prefix = os.environ["INPUT_S3"].rstrip("/")
    out_prefix = os.environ["OUTPUT_S3"].rstrip("/")
    region = os.getenv("AWS_REGION", "eu-west-2")
    s3 = boto3.client("s3", region_name=region)

    in_bucket, in_key_prefix = _split_s3(in_prefix)
    out_bucket, out_key_prefix = _split_s3(out_prefix)

    workdir = pathlib.Path("/work"); workdir.mkdir(parents=True, exist_ok=True)
    local_params = str(workdir / "params.json")

    s3.download_file(in_bucket, f"{in_key_prefix}/params.json", local_params)

    p, yrs = _load_params(local_params)
    sim = run_simulation_for_years(p, yrs) if yrs > 0 else run_simulation(p)

    frame = traj_to_frame(sim)
    csv_bytes = to_csv_bytes(frame)
    local_csv = str(workdir / "traj.csv")
    with open(local_csv, "wb") as f:
        f.write(csv_bytes)

    summary = {
        "t_end": sim["t_end"],
        "dt": sim["dt"],
        "rhill_AU": sim["state"].get("rhill_AU"),
        "n_steps": len(sim["traj"]["xyzarr_mp"]),
        "years_requested": yrs,
    }
    local_summary = str(workdir / "summary.json")
    with open(local_summary, "w") as f:
        json.dump(summary, f)

    fig = build_animation(sim["traj"], sim["a_inner_au"], sim["a_outer_au"],
                          open_in_browser=False, dt=sim["dt"], t_end=sim["t_end"])
    local_html = str(workdir / "animation.html")
    fig.write_html(local_html, include_plotlyjs="cdn")

    s3.upload_file(local_csv, out_bucket, f"{out_key_prefix}/traj.csv")
    s3.upload_file(local_summary, out_bucket, f"{out_key_prefix}/summary.json")
    s3.upload_file(local_html, out_bucket, f"{out_key_prefix}/animation.html")

    # Create presigned URLs (expire in 24h)
    urls = {}
    for name in ["traj.csv", "summary.json", "animation.html"]:
        key = f"{out_key_prefix}/{name}"
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": out_bucket, "Key": key},
            ExpiresIn=86400
        )
        urls[name] = url
        print(f"[OUTPUT] {name}: {url}", flush=True)

    links_path = str(workdir / "links.json")
    with open(links_path, "w") as f:
        json.dump({"region": region, "bucket": out_bucket, "prefix": out_key_prefix, "urls": urls}, f)
    s3.upload_file(links_path, out_bucket, f"{out_key_prefix}/links.json")

if __name__ == "__main__":
    print('hi')
    main()