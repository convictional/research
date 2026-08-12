-- ======================================================
-- Organizational Velocity and Engagement Patterns
-- ======================================================
-- This query analyzes activity patterns over time to identify
-- overall organizational velocity and engagement trends.

WITH weekly_activity AS (
    SELECT
        DATE_TRUNC('week', a.created_at) AS week,
        COUNT(*) AS activity_count,
        COUNT(DISTINCT a.actor_id) AS active_people,
        COUNT(DISTINCT a.artifact_id) AS active_artifacts,
        COUNT(DISTINCT a.action_type) AS activity_types
    FROM
        activity a
    WHERE
        a.created_at >= NOW() - INTERVAL '12 weeks'
    GROUP BY
        DATE_TRUNC('week', a.created_at)
),
weekly_trends AS (
    SELECT
        week,
        activity_count,
        active_people,
        active_artifacts,
        activity_types,
        LAG(activity_count, 1) OVER (ORDER BY week) AS prev_week_count,
        LAG(active_people, 1) OVER (ORDER BY week) AS prev_week_people,
        LAG(activity_count, 4) OVER (ORDER BY week) AS month_ago_count,
        AVG(activity_count) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS four_week_avg
    FROM
        weekly_activity
)

SELECT
    week,
    activity_count,
    active_people,
    active_artifacts,
    activity_types,
    CASE
        WHEN prev_week_count IS NULL THEN NULL
        ELSE ROUND((activity_count::numeric - prev_week_count) / NULLIF(prev_week_count, 0) * 100, 1)
    END AS weekly_change_pct,
    CASE
        WHEN month_ago_count IS NULL THEN NULL
        ELSE ROUND((activity_count::numeric - month_ago_count) / NULLIF(month_ago_count, 0) * 100, 1)
    END AS monthly_change_pct,
    four_week_avg,
    CASE
        WHEN activity_count > four_week_avg * 1.5 THEN 'SIGNIFICANT INCREASE'
        WHEN activity_count > four_week_avg * 1.2 THEN 'MODERATE INCREASE'
        WHEN activity_count < four_week_avg * 0.5 THEN 'SIGNIFICANT DECREASE'
        WHEN activity_count < four_week_avg * 0.8 THEN 'MODERATE DECREASE'
        ELSE 'STABLE'
    END AS trend_status,
    -- Team engagement volatility
    ROUND((active_people::numeric / (SELECT AVG(active_people) FROM weekly_activity) * 100), 1) AS team_engagement_index,
    -- People per artifact ratio (higher means more collaborative work)
    ROUND(active_people::numeric / NULLIF(active_artifacts, 0), 2) AS people_per_artifact_ratio
FROM
    weekly_trends
ORDER BY
    week DESC;
