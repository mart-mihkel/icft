import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import plotnine as pn
    import polars as pl

    from instruct.constants import logdir
    from instruct.plotting import color, fill, shape, theme
    from instruct.scripts.tracking import collect_metrics

    return collect_metrics, color, fill, logdir, mo, os, pl, pn, shape, theme


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(mo):
    dataset_dropdown = mo.ui.dropdown(
        ["multinerd", "obl"],
        value="multinerd",
        label="Dataset",
    )

    dataset_size_dropdown = mo.ui.dropdown(
        [None, 20000, 1000, 100, 10],
        value=20000,
        label="Trainset size",
    )

    mo.vstack([dataset_dropdown, dataset_size_dropdown], justify="start")
    return dataset_dropdown, dataset_size_dropdown


@app.cell
def _(dataset_dropdown, dataset_size_dropdown, logdir):
    dataset = dataset_dropdown.value
    dataset_size = (
        dataset_size_dropdown.value if dataset_dropdown.value == "multinerd" else None
    )
    figpath = logdir / "fig" / dataset / str(dataset_size or "")

    method_labels = {
        "5-shot": "Näitepõhine (5)",
        "fine-tune": "Peenhäälestus",
        "cls-head": "Klassifitseerimispea",
        "prompt-tune-random": "Prompt-häälestus (juhuslik)",
        "prompt-tune-pretrained": "Prompt-häälestus (eeltreenitud)",
    }

    model_labels = {
        "distilbert": "DistilBERT",
        "modernbert": "mmBERT",
        "deberta-v2": "DeBERTa",
        "eurobert": "EuroBERT",
        "gpt_neox": "GPT-NeoX",
        "qwen3_5_text": "Qwen 3.5",
        "gemma3_text": "Gemma 3",
        "llama": "Llama 3.2",
        "t5": "Flan-T5",
        "t5gemma2": "T5Gemma2",
    }

    arch_labels = {
        "encoder": "Kooder",
        "decoder": "Dekooder",
        "encoder-decoder": "Kooder-dekooder",
    }

    model_order = [
        "distilbert",
        "modernbert",
        "deberta-v2",
        "eurobert",
        "gpt_neox",
        "qwen3_5_text",
        "gemma3_text",
        "llama",
        "t5",
        "t5gemma2",
    ]

    arch_order = ["encoder", "decoder", "encoder-decoder"]
    return (
        arch_labels,
        arch_order,
        dataset,
        dataset_size,
        figpath,
        method_labels,
        model_labels,
        model_order,
    )


@app.cell
def _(
    arch_order,
    collect_metrics,
    dataset,
    dataset_size,
    figpath,
    mo,
    model_order,
    os,
    pl,
):
    os.makedirs(figpath, exist_ok=True)

    df = (
        collect_metrics("instruct", "sqlite:///mlflow.db")
        .filter(pl.col("dataset") == dataset)
        .with_columns(
            pl.col("model_type")
            .replace({"gemma3": "gemma3_text"})
            .cast(pl.Enum(model_order)),
            pl.col("architecture").cast(pl.Enum(arch_order)),
        )
    )

    if dataset_size is not None:
        df = df.filter(pl.col("train_samples").is_in([0, dataset_size]))

    # NOTE: there's duplicate few-shot runs
    # should fix later on mlflow side
    df = df.sort("end_time").unique(subset=["run_name"], keep="last")

    mo.md(f"Collected metrics for {len(df)} runs")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabels
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cost of compute
    """)
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(
            pl.col("end_time")
            .sub(pl.col("start_time"))
            .mul(1 / 1000)
            .alias("total_runtime")
        )
        .select("train_runtime", "test_runtime", "total_runtime")
        .sum()
        .unpivot(variable_name="task", value_name="time")
        .with_columns(
            pl.col("time").mul(0.5 / 3600).round(2).alias("cost_eur"),
            pl.col("time").mul(1 / 3600).round(2).alias("gpu_hours"),
            pl.col("task").str.replace("_runtime", ""),
        )
        .select("task", "gpu_hours", "cost_eur")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameters
    """)
    return


@app.cell
def _(df, pl):
    (
        df.with_columns(pl.col("base_model").str.split("/").list.last().alias("model"))
        .pivot(
            on="method",
            index=[
                "model",
                "method",
                "architecture",
                "base_model",
                "total_parameters",
                "num_virtual_tokens",
            ],
            values="trainable_parameters",
        )
        .with_columns(
            pl.when(pl.col("architecture") == "encoder")
            .then(pl.coalesce("cls-head", "fine-tune"))
            .otherwise(None)
            .alias("head_parameters"),
            pl.coalesce("prompt-tune-pretrained", "prompt-tune-random").alias(
                "prompt_parameters"
            ),
        )
        .select(
            "model",
            "method",
            "architecture",
            "total_parameters",
            "head_parameters",
            "prompt_parameters",
            "num_virtual_tokens",
        )
        .sort(["architecture", "total_parameters"])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance scaling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All
    """)
    return


@app.cell
def _(
    arch_labels,
    color,
    df,
    figpath,
    fill,
    method_labels,
    model_labels,
    pl,
    pn,
    shape,
    theme,
):
    _df = df.filter(pl.col("model_type") != "distilbert")

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="test_f1", fill="method", shape="architecture")
        + pn.labs(x="Parameetrid", y="F1", fill="", color="", shape="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=[0, 1],
        )
        + pn.facet_wrap(
            "model_type",
            labeller=lambda s: model_labels.get(s, s),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3, color="white")
        + color(method_labels)
        + fill(method_labels)
        + shape(arch_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            strip_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
            figure_size=(8, 7),
        )
        + pn.guides(
            color=pn.guide_legend(ncol=2),
            shape=pn.guide_legend(ncol=1, override_aes={"color": "black"}),
        )
    )

    _p.save(figpath / "model-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder models
    """)
    return


@app.cell
def _(color, df, figpath, fill, method_labels, model_labels, pl, pn, theme):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "encoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            pl.col("metric").replace(
                {
                    "test_f1": "F1",
                    "test_precision": "Täpsus",
                    "test_recall": "Saagis",
                }
            ),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=[0, 1],
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="o", stroke=0.3, size=3, color="white")
        + color(method_labels)
        + fill(method_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            strip_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
            figure_size=(8, 7),
        )
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "encoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decoder models
    """)
    return


@app.cell
def _(color, df, figpath, fill, method_labels, model_labels, pl, pn, theme):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "decoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            pl.col("metric").replace(
                {
                    "test_f1": "F1",
                    "test_precision": "Täpsus",
                    "test_recall": "Saagis",
                }
            ),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=[0, 1],
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="s", stroke=0.3, size=3, color="white")
        + color(method_labels)
        + fill(method_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            strip_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
            figure_size=(8, 7),
        )
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "decoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder-decoder models
    """)
    return


@app.cell
def _(color, df, figpath, fill, method_labels, model_labels, pl, pn, theme):
    _idx = [
        c for c in df.columns if c not in ["test_f1", "test_precision", "test_recall"]
    ]

    _df = (
        df.filter(pl.col("architecture") == "encoder-decoder")
        .unpivot(
            on=["test_f1", "test_precision", "test_recall"],
            index=_idx,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            pl.col("metric").replace(
                {
                    "test_f1": "F1",
                    "test_precision": "Täpsus",
                    "test_recall": "Saagis",
                }
            ),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="value", fill="method")
        + pn.labs(x="Parameetrid", y="", fill="", color="")
        + pn.scale_x_log10(
            breaks=[10**i for i in range(6, 11)],
            labels=["1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=[0, 1],
        )
        + pn.facet_grid(
            "metric ~ model_type",
            labeller=lambda s: model_labels.get(s, s),
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(shape="D", stroke=0.3, size=3, color="white")
        + color(method_labels)
        + fill(method_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            strip_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
            figure_size=(8, 7),
        )
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "encoder-decoder-performance-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instructability scaling
    """)
    return


@app.cell
def _(color, df, figpath, fill, method_labels, pl, pn, theme):
    _df = df.filter(
        pl.col("model_type") == "gpt_neox",
        pl.col("total_parameters") > 1e8,
        pl.col("total_parameters") < 6 * 1e9,
    )

    _labels_df = (
        _df.with_columns(
            pl.col("base_model")
            .str.split("-")
            .list.last()
            .str.to_uppercase()
            .alias("label")
        )
        .group_by("base_model")
        .agg(
            pl.col("total_parameters").mean(),
            pl.col("test_f1").max(),
            pl.col("label").first(),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(x="total_parameters", y="test_f1", fill="method")
        + pn.labs(x="Parameetrid", y="F1", fill="", color="")
        + pn.scale_x_log10(labels=lambda ticks: [f"{t / 1e9:.2}B" for t in ticks])
        + pn.scale_y_continuous(
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0%", "25%", "50%", "75%", "100%"],
            limits=[0, 1],
        )
        + pn.geom_line(pn.aes(color="method"))
        + pn.geom_point(stroke=0.3, size=3.5, color="white")
        + pn.geom_text(
            pn.aes(x="total_parameters", y="test_f1", label="label"),
            inherit_aes=False,
            data=_labels_df,
            nudge_y=0.025,
            va="bottom",
            size=9,
        )
        + color(method_labels)
        + fill(method_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            figure_size=(8, 7),
        )
        + pn.guides(color=pn.guide_legend(ncol=2))
    )

    _p.save(figpath / "instructability-scaling.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relative and absolute metrics change

    $$\Delta_{\text{prompt-tune}} = s_{\text{prompt-tune}} - s_{\text{few-shot}}$$

    $$\Delta_{\text{fine-tune}} = s_{\text{fine-tune}} - s_{\text{few-shot}}$$

    $$\delta = \frac{\Delta_{\text{prompt-tune}}}{\Delta_{\text{fine-tune}}}$$
    """)
    return


@app.cell
def _(
    arch_labels,
    color,
    df,
    figpath,
    fill,
    model_labels,
    pl,
    pn,
    shape,
    theme,
):
    _metric_labels = {"f1": "F1", "recall": "Saagis", "precision": "Täpsus"}
    _metric_order = ["f1", "recall", "precision"]

    _df = (
        df.filter(
            pl.col("method").str.contains(
                r"5-shot|cls-head|fine-tune|prompt-tune-pretrained"
            ),
        )
        .pivot(
            index=["base_model", "model_type", "architecture"],
            values=["test_f1", "test_recall", "test_precision"],
            on="method",
        )
        .with_columns(
            pl.coalesce(["test_f1_5-shot", "test_f1_cls-head"]).alias(
                "test_f1_baseline"
            ),
            pl.coalesce(["test_recall_5-shot", "test_recall_cls-head"]).alias(
                "test_recall_baseline"
            ),
            pl.coalesce(["test_precision_5-shot", "test_precision_cls-head"]).alias(
                "test_precision_baseline"
            ),
        )
        .with_columns(
            pl.col("test_f1_prompt-tune-pretrained")
            .sub(pl.col("test_f1_baseline"))
            .alias("pt_abs_delta_f1"),
            pl.col("test_recall_prompt-tune-pretrained")
            .sub(pl.col("test_recall_baseline"))
            .alias("pt_abs_delta_recall"),
            pl.col("test_precision_prompt-tune-pretrained")
            .sub(pl.col("test_precision_baseline"))
            .alias("pt_abs_delta_precision"),
            pl.col("test_f1_fine-tune")
            .sub(pl.col("test_f1_baseline"))
            .alias("ft_abs_delta_f1"),
            pl.col("test_recall_fine-tune")
            .sub(pl.col("test_recall_baseline"))
            .alias("ft_abs_delta_recall"),
            pl.col("test_precision_fine-tune")
            .sub(pl.col("test_precision_baseline"))
            .alias("ft_abs_delta_precision"),
        )
        .with_columns(
            pl.col("pt_abs_delta_f1")
            .mul(1 / pl.col("ft_abs_delta_f1"))
            .alias("rel_delta_f1"),
            pl.col("pt_abs_delta_recall")
            .mul(1 / pl.col("ft_abs_delta_recall"))
            .alias("rel_delta_recall"),
            pl.col("pt_abs_delta_precision")
            .mul(1 / pl.col("ft_abs_delta_precision"))
            .alias("rel_delta_precision"),
        )
        .select(
            "base_model",
            "model_type",
            "architecture",
            "pt_abs_delta_f1",
            "pt_abs_delta_recall",
            "pt_abs_delta_precision",
            "rel_delta_f1",
            "rel_delta_recall",
            "rel_delta_precision",
        )
        .unpivot(
            index=["base_model", "model_type", "architecture"],
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            pl.when(pl.col("metric").str.starts_with("pt_abs_delta"))
            .then(pl.lit("pt_abs_delta"))
            .otherwise(pl.lit("rel_delta"))
            .alias("measure"),
            pl.col("metric")
            .str.replace("pt_abs_delta_|rel_delta_", "")
            .alias("metric"),
        )
        .pivot(
            index=["base_model", "model_type", "architecture", "metric"],
            values="value",
            on="measure",
        )
        .with_columns(
            pl.col("metric").cast(pl.Enum(_metric_order)),
            (
                pl.col("metric").replace(_metric_labels)
                + pl.lit("\n")
                + pl.col("architecture")
            ).alias("facet_label"),
        )
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="pt_abs_delta", y="rel_delta", shape="model_type", fill="architecture"
        )
        + pn.labs(
            x=r"Prompt-häälestus absoluutne $\Delta$",
            y=r"Prompt-häälestus ja peenhäälestus relatiivne $\delta$",
            shape="",
            fill="",
        )
        # + pn.facet_grid(
        #     "metric ~ architecture",
        #     scales="free",
        #     labeller=lambda s: arch_labels.get(s, _metric_labels.get(s, s)),
        # )
        + pn.facet_wrap(
            ["metric", "architecture"],
            scales="free",
            labeller=lambda s: arch_labels.get(s, _metric_labels.get(s, s)),
        )
        + pn.scale_x_continuous(
            expand=(0.15, 0),
            labels=lambda ticks: [f"{int(100 * t)}%" for t in ticks],
        )
        + pn.scale_y_continuous(
            expand=(0.15, 0),
            labels=lambda ticks: [f"{int(100 * t)}%" for t in ticks],
        )
        + pn.geom_point(stroke=0.3, size=3, color="white")
        + color(arch_labels)
        + fill(arch_labels)
        + shape(model_labels)
        + theme()
        + pn.theme(
            legend_position="top",
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            strip_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            panel_border=pn.element_rect(color="#D8D8D8", alpha=0.25),
            panel_spacing_x=0.025,
            panel_spacing_y=0.025,
            figure_size=(10, 9),
        )
        + pn.guides(
            shape=pn.guide_legend(order=1, nrow=3, override_aes={"color": "black"}),
            fill=pn.guide_legend(order=2, ncol=1),
        )
    )

    _p.save(figpath / "relative-absolute-performance.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute time scaling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Side-by-side
    """)
    return


@app.cell
def _(
    color,
    dataset_size,
    df,
    figpath,
    fill,
    model_labels,
    pl,
    pn,
    shape,
    theme,
):
    _df = df.filter(pl.col("method").str.contains(r"fine-tune|prompt-tune-pretrained"))

    if dataset_size is not None:
        _df = _df.filter(pl.col("train_samples").eq(dataset_size))

    _method_labels = {
        "fine-tune": "Peenhäälestus",
        "prompt-tune-pretrained": "Prompt-häälestus",
    }

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            x="trainable_parameters",
            y="train_runtime",
            fill="method",
            shape="model_type",
        )
        + pn.labs(
            x="Treenitavad parameetrid",
            y="Treenimisaeg",
            fill="Meetod",
            shape="Mudel",
        )
        + pn.scale_x_log10(
            breaks=[10**i for i in range(4, 11)],
            labels=["10K", "100K", "1M", "10M", "100M", "1B", "10B"],
        )
        + pn.scale_y_continuous(
            labels=lambda ticks: [f"{t / 3600:.1f}h" for t in ticks]
        )
        + pn.geom_point(size=3.5, stroke=0.3, color="white")
        + color(_method_labels)
        + fill(_method_labels)
        + shape(model_labels)
        + theme()
        + pn.theme(
            legend_position=(0.02, 0.98),
            legend_justification=("left", "top"),
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            figure_size=(8, 7),
        )
        + pn.guides(
            shape=pn.guide_legend(order=1, override_aes={"color": "black"}),
            fill=pn.guide_legend(order=2),
        )
    )

    _p.save(figpath / "runtime-scaling-side-by-side.png", dpi=300)
    _p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison
    """)
    return


@app.cell
def _(
    arch_labels,
    dataset_size,
    df,
    figpath,
    fill,
    model_labels,
    pl,
    pn,
    shape,
    theme,
):
    if dataset_size is not None:
        _df = df.filter(pl.col("train_samples").eq(dataset_size))

    _df = df.filter(
        pl.col("method").str.contains(r"fine-tune|prompt-tune-pretrained"),
    ).pivot(
        index=["base_model", "model_type", "architecture"],
        values=["train_runtime", "total_parameters"],
        on="method",
        aggregate_function="mean",
    )

    _p = (
        pn.ggplot(_df)
        + pn.aes(
            y="train_runtime_fine-tune",
            x="train_runtime_prompt-tune-pretrained",
            shape="model_type",
            fill="architecture",
            size="total_parameters_fine-tune",
        )
        + pn.labs(
            y="Peenhäälestus treenimisaeg",
            x="Prompt-häälestus treenimisaeg",
            shape="Mudel",
            fill="Arhitektuur",
        )
        + pn.scale_x_continuous(
            labels=lambda ticks: [f"{t / 3600:.1f}h" for t in ticks]
        )
        + pn.scale_y_continuous(
            labels=lambda ticks: [f"{t / 3600:.1f}h" for t in ticks]
        )
        + pn.geom_point(stroke=0.3, color="white")
        + pn.scale_size_continuous(range=(3, 7), guide=None)
        + shape(model_labels)
        + fill(arch_labels)
        + theme()
        + pn.theme(
            legend_position=(0.02, 0.98),
            legend_justification=("left", "top"),
            legend_background=pn.element_rect(
                fill="#D8D8D8", color="#FFFFFF", alpha=0.25
            ),
            legend_margin=2,
            figure_size=(8, 7),
        )
        + pn.guides(
            shape=pn.guide_legend(
                order=1, override_aes={"size": 3.5, "color": "black"}
            ),
            fill=pn.guide_legend(order=2, override_aes={"size": 3.5}),
        )
    )

    _p.save(figpath / "runtime-scaling-comparison.png", dpi=300)
    _p
    return


if __name__ == "__main__":
    app.run()
