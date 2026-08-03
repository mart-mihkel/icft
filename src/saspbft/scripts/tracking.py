"""Export MLflow experiment runs as a metrics dataframe."""

from typing import TYPE_CHECKING, cast

import mlflow
from mlflow.tracking import MlflowClient
from polars import DataFrame

from saspbft.constants import LOGDIR
from saspbft.logging import logger

if TYPE_CHECKING:
    from mlflow.entities import Run


def start_run(
    experiment: str,
    run_name: str,
    mlflow_tracking_uri: str,
    *,
    resume: bool,
) -> None:
    """Start tracking, reattaching to the latest run of `run_name` when resuming."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment)

    if not resume:
        mlflow.start_run(run_name=run_name)
        return

    previous = cast(
        "list[Run]",
        mlflow.search_runs(
            experiment_names=[experiment],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
            output_format="list",
        ),
    )

    if not previous:
        logger.warning("no previous run named '%s', tracking a new one", run_name)
        mlflow.start_run(run_name=run_name)
        return

    run_id = previous[0].info.run_id
    logger.info("reattaching to run '%s'", run_id)
    mlflow.start_run(run_id=run_id)


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

    metricdir = LOGDIR / "metrics"
    path = metricdir / f"{experiment}.csv"
    metricdir.mkdir(parents=True, exist_ok=True)

    df = DataFrame(rows)
    logger.info("found %d runs with %d params", df.shape[0], df.shape[1])
    if write_csv:
        df.write_csv(path)
        logger.info("saved metrics to '%s'", path)

    return df
