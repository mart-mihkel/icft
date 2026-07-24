"""Export MLflow experiment runs as a metrics dataframe."""

from mlflow.tracking import MlflowClient
from polars import DataFrame

from instruct.constants import logdir
from instruct.logging import logger


def collect_metrics(
    experiment: str,
    mlflow_tracking_uri: str,
    write_csv: bool = False,
) -> DataFrame:
    """Collect metrics and params for every run of `experiment` into a dataframe."""
    logger.info("connecting to %s", mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    logger.info("finding experiment %s", experiment)
    exp = client.get_experiment_by_name(experiment)

    if exp is None:
        msg = f"experiment '{experiment}' not found"
        raise RuntimeError(msg)

    logger.info("collecting metrics")
    runs = client.search_runs(exp.experiment_id, "")
    rows = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

        run_data |= run.data.metrics
        run_data |= run.data.params

        rows.append(run_data)

    metricdir = logdir / "metrics"
    path = metricdir / f"{experiment}.csv"
    metricdir.mkdir(parents=True, exist_ok=True)

    df = DataFrame(rows)
    logger.info("found %d runs with %d params", df.shape[0], df.shape[1])
    if write_csv:
        df.write_csv(path)
        logger.info("saved metrics to '%s'", path)

    return df
