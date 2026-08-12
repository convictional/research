import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter
import pytz

from .settings import settings
from .get_usage_data import (
    llm_request_query,
    thread_usage,
    decision_process_usage,
    meeting_usage,
    process_object_ids,
    query_bq,
    clear_output_directory,
    calculate_costs,
    process_meeting_data,
    get_recall_usage_data,
)
from .plot_utils import (
    COLORS,
    create_figure,
    create_subplot_figure,
    style_axis,
    style_stacked_bar,
    add_branded_legend,
    get_color_with_alpha,
)

pd.set_option("future.no_silent_downcasting", True)


async def analyze_convictional_costs() -> None:
    """
    Main analysis function that executes the query once and coordinates all analysis functions.
    """
    # Clear output directory
    clear_output_directory(settings.output_path)

    # Execute query once
    df = query_bq(llm_request_query)

    # Process object IDs
    df = process_object_ids(df)

    # Calculate costs once here
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    costs = df.apply(calculate_costs, axis=1, result_type="expand")
    df = pd.concat([df, costs], axis=1)

    # Token usage analysis if needed (commented out, but you can re-enable)
    analyze_token_usage(df)

    # Object creation analysis (includes meeting usage, recall usage, etc.)
    await analyze_object_creations(df)


def analyze_token_usage(df: pd.DataFrame) -> None:
    """
    Analyze daily/weekly/monthly token usage, costs, and request counts by object type.
    Generates plots and saves summary CSVs.
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)

    # Aggregate by object_id to get total tokens & costs per object
    object_interactions = (
        df.groupby(["created_at", "object_type", "object_id"])
        .agg(
            {
                "total_tokens": "sum",
                "input_tokens": "sum",
                "output_tokens": "sum",
                "total_cost": "sum",
                "input_cost": "sum",
                "output_cost": "sum",
                "llm_model": "first",
            }
        )
        .reset_index()
    )

    # Create subplot figures
    fig_obj, axes_obj = create_subplot_figure(nrows=3, ncols=3, figsize=(30, 15))
    fig_cost, axes_cost = create_subplot_figure(nrows=3, ncols=2, figsize=(20, 15))

    # Object-based aggregations
    daily_obj_df = (
        object_interactions.groupby([object_interactions["created_at"].dt.date, "object_type"])
        .agg(
            {
                "total_tokens": ["sum", "mean"],
                "total_cost": ["sum", "mean"],
                "object_id": ["nunique", "count"],
            }
        )
        .reset_index()
    )
    daily_obj_df.columns = [
        "date",
        "object_type",
        "total_tokens",
        "avg_tokens_per_object",
        "total_cost",
        "avg_cost_per_object",
        "unique_objects",
        "request_count",
    ]

    weekly_obj_df = (
        object_interactions.groupby([object_interactions["created_at"].dt.to_period("W"), "object_type"])
        .agg({"total_tokens": ["sum", "mean"], "total_cost": ["sum", "mean"], "object_id": ["nunique", "count"]})
        .reset_index()
    )
    weekly_obj_df.columns = [
        "week",
        "object_type",
        "total_tokens",
        "avg_tokens_per_object",
        "total_cost",
        "avg_cost_per_object",
        "unique_objects",
        "request_count",
    ]

    monthly_obj_df = (
        object_interactions.groupby([object_interactions["created_at"].dt.to_period("M"), "object_type"])
        .agg({"total_tokens": ["sum", "mean"], "total_cost": ["sum", "mean"], "object_id": ["nunique", "count"]})
        .reset_index()
    )
    monthly_obj_df.columns = [
        "month",
        "object_type",
        "total_tokens",
        "avg_tokens_per_object",
        "total_cost",
        "avg_cost_per_object",
        "unique_objects",
        "request_count",
    ]

    # Use standardized colors
    object_colors = COLORS["primary"]
    time_periods = [
        (daily_obj_df, "date", "Daily"),
        (weekly_obj_df, "week", "Weekly"),
        (monthly_obj_df, "month", "Monthly"),
    ]

    # -------------------- TOKEN-BASED PLOTS --------------------
    for idx, (period_df, date_col, title) in enumerate(time_periods):
        # Convert Period to datetime for weekly/monthly
        if date_col in ["week", "month"]:
            if date_col == "week":
                period_df[date_col] = period_df[date_col].astype(str).apply(lambda x: pd.to_datetime(x.split("/")[0]))
            else:
                period_df[date_col] = period_df[date_col].astype(str).apply(lambda x: pd.to_datetime(x + "-01"))

        # 1) Total tokens
        for obj_type in object_colors:
            mask = period_df["object_type"] == obj_type
            axes_obj[idx, 0].plot(
                period_df[mask][date_col],
                period_df[mask]["total_tokens"],
                color=get_color_with_alpha(obj_type),
                label=f"{obj_type} ({period_df[mask]['unique_objects'].sum()} objects)",
                marker="o",
                markersize=4,
                linewidth=2,
            )

        # 2) Average tokens per object
        for obj_type in object_colors:
            mask = period_df["object_type"] == obj_type
            axes_obj[idx, 1].plot(
                period_df[mask][date_col],
                period_df[mask]["avg_tokens_per_object"],
                color=get_color_with_alpha(obj_type),
                label=obj_type,
                marker="o",
                markersize=4,
                linewidth=2,
            )

        # 3) Request count
        for obj_type in object_colors:
            mask = period_df["object_type"] == obj_type
            axes_obj[idx, 2].plot(
                period_df[mask][date_col],
                period_df[mask]["request_count"],
                color=get_color_with_alpha(obj_type),
                label=f"{obj_type} ({period_df[mask]['request_count'].sum()} requests)",
                marker="o",
                markersize=4,
                linewidth=2,
            )

        style_axis(
            axes_obj[idx, 0], title=f"{title} Token Usage (Total)", xlabel=date_col.title(), ylabel="Total Tokens"
        )
        style_axis(
            axes_obj[idx, 1],
            title=f"{title} Token Usage (Per Object)",
            xlabel=date_col.title(),
            ylabel="Average Tokens per Object",
        )
        style_axis(
            axes_obj[idx, 2], title=f"{title} Request Count", xlabel=date_col.title(), ylabel="Number of Requests"
        )

        # Add legends & format dates
        for col in range(3):
            add_branded_legend(axes_obj[idx, col])
            if title == "Daily" or title == "Weekly":
                axes_obj[idx, col].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            else:
                axes_obj[idx, col].xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    fig_obj.suptitle("Token Usage and Request Count Analysis by Object Type", y=1.02)
    fig_obj.tight_layout()
    fig_obj.savefig(settings.output_path / "token_usage_by_object.png", bbox_inches="tight", dpi=300)
    plt.close(fig_obj)

    # -------------------- COST-BASED PLOTS --------------------
    for idx, (period_df, date_col, title) in enumerate(time_periods):
        if date_col in ["week", "month"]:
            if date_col == "week":
                period_df[date_col] = period_df[date_col].astype(str).apply(lambda x: pd.to_datetime(x.split("/")[0]))
            else:
                period_df[date_col] = period_df[date_col].astype(str).apply(lambda x: pd.to_datetime(x + "-01"))

        # Total & avg cost
        for obj_type in object_colors:
            mask = period_df["object_type"] == obj_type
            axes_cost[idx, 0].plot(
                period_df[mask][date_col],
                period_df[mask]["total_cost"],
                color=object_colors[obj_type],
                label=f"{obj_type} ({period_df[mask]['unique_objects'].sum()} objects)",
                marker="o",
                markersize=4,
                linewidth=2,
                alpha=0.6,
            )
            axes_cost[idx, 1].plot(
                period_df[mask][date_col],
                period_df[mask]["avg_cost_per_object"],
                color=object_colors[obj_type],
                label=obj_type,
                marker="o",
                markersize=4,
                linewidth=2,
                alpha=0.6,
            )

        axes_cost[idx, 0].set_title(f"{title} Cost Analysis (Total)")
        axes_cost[idx, 1].set_title(f"{title} Cost Analysis (Per Object)")
        for col in range(2):
            axes_cost[idx, col].set_xlabel(date_col.title())
            axes_cost[idx, col].grid(True, alpha=0.3)
            axes_cost[idx, col].tick_params(axis="x", rotation=45)
            axes_cost[idx, col].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            if col == 0:
                axes_cost[idx, col].set_ylabel("Total Cost ($)")
                axes_cost[idx, col].set_yscale("log")
                axes_cost[idx, col].yaxis.set_major_formatter(ScalarFormatter())
                axes_cost[idx, col].yaxis.get_major_formatter().set_scientific(False)
            else:
                axes_cost[idx, col].set_ylabel("Average Cost per Object ($)")

    fig_cost.suptitle("Cost Analysis by Object Type", y=1.02)
    fig_cost.tight_layout()
    fig_cost.savefig(settings.output_path / "cost_analysis_by_object.png", bbox_inches="tight", dpi=300)
    plt.close(fig_cost)

    # -------------------- SUMMARY TABLES & CSV --------------------
    object_summary = (
        object_interactions.groupby("object_type")
        .agg(
            {
                "total_tokens": ["sum", "mean"],
                "total_cost": ["sum", "mean"],
                "object_id": ["nunique", "count"],
                "llm_model": "first",
            }
        )
        .round(2)
    )
    object_summary.columns = [
        "total_tokens",
        "avg_tokens_per_object",
        "total_cost",
        "avg_cost_per_object",
        "unique_objects",
        "request_count",
        "model",
    ]

    object_summary.to_csv(settings.output_path / "token_usage_summary.csv")
    object_interactions.to_csv(settings.output_path / "object_interactions.csv", index=False)

    # -------------------- WEEKLY STACKED BAR PLOT --------------------
    weekly_totals_by_type = (
        object_interactions.groupby([object_interactions["created_at"].dt.to_period("W"), "object_type"])
        .agg({"total_tokens": "sum", "total_cost": "sum"})
        .reset_index()
    )
    weekly_totals_by_type["created_at"] = (
        weekly_totals_by_type["created_at"].astype(str).apply(lambda x: pd.to_datetime(x.split("/")[0]))
    )

    fig, ax = create_figure(figsize=(12, 8))
    pivot_costs = (
        weekly_totals_by_type.pivot(index="created_at", columns="object_type", values="total_cost")
        .fillna(0)
        .infer_objects(copy=False)
    )
    pivot_costs.plot(kind="bar", stacked=True, ax=ax, color=[get_color_with_alpha(t) for t in pivot_costs.columns])

    # Calculate and plot 4-week moving average
    weekly_total = pivot_costs.sum(axis=1)
    ma_4week = weekly_total.rolling(window=4, min_periods=1).mean()

    # Create a twin axis for the line plot to ensure proper alignment
    ax2 = ax.twinx()
    ax2.plot(
        range(len(weekly_total)),
        ma_4week,
        color="red",
        linewidth=2,
        label="4-Week Moving Average (Total Cost)",  # Updated label
        zorder=5,
    )

    # Match the y-axis limits and scale of the primary axis
    ax2.set_ylim(ax.get_ylim())

    # Add the moving average to the legend with updated positioning
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), title="Object Type & Trend")

    # Hide the right y-axis as it shows the same values
    ax2.set_yticks([])

    style_stacked_bar(
        ax, title="All Organizations: Weekly Costs by Object Type", xlabel="Week Starting", ylabel="Total Cost ($)"
    )
    add_branded_legend(ax, title="Object Type")

    plt.tight_layout()
    plt.savefig(settings.output_path / "weekly_totals_by_type_aggregate.png", bbox_inches="tight", dpi=300)
    plt.close()


async def analyze_object_creations(llm_df: pd.DataFrame) -> None:
    """
    Analyze creation frequency for threads, meetings, and decision processes,
    merged with cost data by object_id to show cost impact.
    """
    df_threads = query_bq(thread_usage)
    df_threads["object_id"] = df_threads["thread_id"]
    df_threads["object_type"] = "thread"
    df_threads["created_at"] = pd.to_datetime(df_threads["thread_created_at"]).dt.tz_convert(pytz.UTC)

    df_decisions = query_bq(decision_process_usage)
    df_decisions["object_id"] = df_decisions["decision_id"]
    df_decisions["object_type"] = "decision_process"
    df_decisions["created_at"] = pd.to_datetime(df_decisions["created_at"]).dt.tz_convert(pytz.UTC)

    df_meetings = query_bq(meeting_usage)
    df_meetings["object_id"] = df_meetings["meeting_id"]
    df_meetings["object_type"] = "meeting"
    df_meetings["created_at"] = pd.to_datetime(df_meetings["created_at"]).dt.tz_convert(pytz.UTC)

    # Get Recall AI usage data for the same period
    start_date = df_meetings["created_at"].min()
    end_date = df_meetings["created_at"].max()
    recall_data = await get_recall_usage_data(start_date, end_date)
    recall_data.to_csv(settings.output_path / "recall_usage.csv", index=False)

    # Process meeting lengths with recall data
    df_meetings, failed_transcripts, recall_usage = process_meeting_data(df_meetings, recall_data)
    if not failed_transcripts.empty:
        failed_transcripts.to_csv(settings.output_path / "failed_transcripts.csv", index=False)

    # -------------------- MEETING HOURS COMPARISON (DAILY, WEEKLY, MONTHLY) --------------------
    time_periods = [("D", "Daily"), ("W", "Weekly"), ("ME", "Monthly")]
    for period, title in time_periods:
        meetings_by_period = (
            df_meetings.groupby(pd.Grouper(key="created_at", freq=period))
            .agg({"meeting_length": ["sum", "count"], "meeting_id": "nunique"})
            .reset_index()
        )
        meetings_by_period.columns = ["date", "total_seconds", "meeting_count", "unique_meetings"]
        meetings_by_period["total_hours"] = meetings_by_period["total_seconds"] / 3600

        # If recall usage is available
        if not recall_usage.empty:
            recall_by_period = (
                recall_usage.groupby(pd.Grouper(key="created_at", freq=period))
                .agg({"recall_meeting_length": ["sum", "count"], "meeting_id": "nunique"})
                .reset_index()
            )
            recall_by_period.columns = [
                "date",
                "recall_total_seconds",
                "recall_meeting_count",
                "recall_unique_meetings",
            ]
            recall_by_period["recall_total_hours"] = recall_by_period["recall_total_seconds"] / 3600

            combined_stats = pd.merge(meetings_by_period, recall_by_period, on="date", how="outer").fillna(0)
        else:
            combined_stats = meetings_by_period

        fig, ax = create_figure(figsize=(12, 6))
        ax.plot(
            combined_stats["date"],
            combined_stats["total_hours"],
            color=get_color_with_alpha("meeting"),
            label="Internal Tracking",
            marker="o",
            markersize=4,
            linewidth=2,
        )
        if not recall_usage.empty:
            ax.plot(
                combined_stats["date"],
                combined_stats["recall_total_hours"],
                color=get_color_with_alpha("decision_process"),
                label="Recall AI",
                marker="o",
                markersize=4,
                linewidth=2,
            )
        style_axis(ax, title=f"{title} Meeting Hours Comparison", xlabel=f"{title} Starting", ylabel="Total Hours")
        add_branded_legend(ax)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(settings.output_path / f"meeting_hours_comparison_{period}.png", bbox_inches="tight", dpi=300)
        plt.close()

        combined_stats.to_csv(settings.output_path / f"meeting_stats_{period}.csv", index=False)

    # -------------------- MEETING LENGTH DISTRIBUTION --------------------
    df_meetings["meeting_length_minutes"] = df_meetings["meeting_length"] / 60
    fig, ax = create_figure(figsize=(10, 6))
    ax.hist(df_meetings["meeting_length_minutes"], bins=20, color=get_color_with_alpha("meeting"), alpha=0.6)
    style_axis(
        ax, title="Distribution of Meeting Lengths", xlabel="Meeting Length (minutes)", ylabel="Number of Meetings"
    )
    plt.savefig(settings.output_path / "meeting_length_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------- WEEKLY TOTAL MEETING HOURS --------------------
    df_meetings["week"] = pd.to_datetime(df_meetings["created_at"]).dt.to_period("W")
    weekly_hours = df_meetings.groupby("week").agg({"meeting_length": "sum", "meeting_id": "count"}).reset_index()
    weekly_hours["total_hours"] = weekly_hours["meeting_length"] / 3600
    weekly_hours["week"] = weekly_hours["week"].astype(str).apply(lambda x: pd.to_datetime(x.split("/")[0]))

    fig, ax = create_figure(figsize=(12, 6))
    ax.plot(
        weekly_hours["week"],
        weekly_hours["total_hours"],
        color=get_color_with_alpha("meeting"),
        marker="o",
        markersize=4,
        linewidth=2,
    )
    style_axis(ax, title="Weekly Total Meeting Hours", xlabel="Week Starting", ylabel="Total Hours")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(settings.output_path / "weekly_meeting_hours.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------- MEETING STATS SUMMARY --------------------
    valid_meetings = df_meetings[df_meetings["meeting_length"] > 0].copy()
    meeting_stats = {
        "total_meetings": len(df_meetings),
        "valid_meetings": len(valid_meetings),
        "total_hours": valid_meetings["meeting_length"].sum() / 3600,
        "avg_length_minutes": valid_meetings["meeting_length"].mean() / 60 if len(valid_meetings) > 0 else 0,
        "median_length_minutes": valid_meetings["meeting_length"].median() / 60 if len(valid_meetings) > 0 else 0,
        "min_length_minutes": valid_meetings["meeting_length"].min() / 60 if len(valid_meetings) > 0 else 0,
        "max_length_minutes": valid_meetings["meeting_length"].max() / 60 if len(valid_meetings) > 0 else 0,
        "std_length_minutes": valid_meetings["meeting_length"].std() / 60 if len(valid_meetings) > 0 else 0,
        "invalid_meetings": len(df_meetings) - len(valid_meetings),
    }
    pd.DataFrame([meeting_stats]).to_csv(settings.output_path / "meeting_stats_summary.csv", index=False)
    weekly_hours.to_csv(settings.output_path / "weekly_meeting_hours.csv", index=False)

    # -------------------- MERGE WITH LLM COST DATA --------------------
    combined_df = pd.concat([df_threads, df_decisions, df_meetings], ignore_index=True)
    combined_df = combined_df[["object_id", "object_type", "created_at"]]

    usage_merged = pd.merge(
        combined_df,
        llm_df[["object_id", "object_type", "total_cost", "total_tokens", "created_at"]],
        on=["object_id", "object_type"],
        how="left",
        suffixes=("", "_usage"),
    )

    # Create weekly stats
    usage_merged["week"] = usage_merged["created_at"].dt.to_period("W")
    weekly_stats = (
        usage_merged.groupby(["week", "object_type"])
        .agg(
            creation_count=("object_id", "count"),
            total_cost=("total_cost", "sum"),
        )
        .reset_index()
    )
    weekly_stats["week"] = weekly_stats["week"].astype(str).apply(lambda x: pd.to_datetime(x.split("/")[0]))

    # -------------------- OBJECT CREATION COUNTS --------------------
    pivot_counts = (
        weekly_stats.pivot(index="week", columns="object_type", values="creation_count")
        .fillna(0)
        .infer_objects(copy=False)
    )
    fig, ax = create_figure(figsize=(12, 6))
    for obj_type in pivot_counts.columns:
        ax.plot(
            pivot_counts.index,
            pivot_counts[obj_type],
            color=get_color_with_alpha(obj_type),
            label=obj_type,
            marker="o",
            markersize=4,
            linewidth=2,
        )
    style_axis(
        ax,
        title="Convictional Weekly Object Creation Counts by Type",
        xlabel="Week Starting",
        ylabel="Number of Objects Created",
    )
    add_branded_legend(ax, title="Object Type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(settings.output_path / "object_creation_counts.png", bbox_inches="tight", dpi=300)
    plt.close()

    # -------------------- OBJECT COSTS (STACKED BAR) --------------------
    pivot_data = (
        weekly_stats.pivot(index="week", columns="object_type", values="total_cost")
        .fillna(0)
        .infer_objects(copy=False)
    )
    fig, ax = create_figure(figsize=(12, 6))
    pivot_data.plot(kind="bar", stacked=True, ax=ax, color=[get_color_with_alpha(t) for t in pivot_data.columns])
    style_stacked_bar(
        ax, title="Convictional Weekly Object Costs by Type", xlabel="Week Starting", ylabel="Total Cost ($)"
    )
    add_branded_legend(ax, title="Object Type")
    plt.tight_layout()
    plt.savefig(settings.output_path / "object_costs.png", bbox_inches="tight", dpi=300)
    plt.close()

    weekly_stats.to_csv(settings.output_path / "object_creation_summary.csv", index=False)
    creation_counts = weekly_stats[["week", "object_type", "creation_count"]].copy()
    creation_counts.to_csv(settings.output_path / "object_creation_counts.csv", index=False)
