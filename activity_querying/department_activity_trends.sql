-- This query analyzes department activity trends over 12 weeks, providing insights on
-- productivity patterns, engagement percentages, and highlighting significant changes
-- to help executives understand departmental performance and resource utilization.

WITH weekly_department_activity AS (
    SELECT
        COALESCE(p.department, 'Unassigned') AS department,
        DATE_TRUNC('week', a.created_at) AS week,
        COUNT(*) AS activity_count,
        COUNT(DISTINCT a.actor_id) AS active_people,
        COUNT(DISTINCT a.artifact_id) AS artifacts_touched,
        p.organization_id,
        -- Number of people in the department (moved outside to avoid ungrouped column error)
        COUNT(DISTINCT p.id) AS dept_count
    FROM
        person p
    JOIN
        activity a ON p.id = a.actor_id
    WHERE
        a.created_at >= NOW() - INTERVAL '12 weeks'
    GROUP BY
        COALESCE(p.department, 'Unassigned'), DATE_TRUNC('week', a.created_at), p.organization_id
),
-- New CTE to calculate department sizes correctly
department_sizes AS (
    SELECT
        COALESCE(department, 'Unassigned') AS department,
        organization_id,
        COUNT(DISTINCT id) AS dept_size
    FROM
        person
    GROUP BY
        COALESCE(department, 'Unassigned'), organization_id
),
department_trends AS (
    SELECT
        wda.department,
        wda.week,
        wda.activity_count,
        wda.active_people,
        wda.artifacts_touched,
        ds.dept_size,
        wda.organization_id,
        -- Calculate week-over-week changes
        LAG(activity_count, 1) OVER (PARTITION BY wda.department ORDER BY week) AS prev_week_count,
        LAG(active_people, 1) OVER (PARTITION BY wda.department ORDER BY week) AS prev_week_people,
        -- Calculate 4-week rolling averages
        AVG(activity_count) OVER (
            PARTITION BY wda.department
            ORDER BY week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS four_week_avg_activity,
        -- Department engagement percentage
        ROUND(active_people::numeric / NULLIF(ds.dept_size, 0) * 100, 2) AS engagement_percentage
    FROM
        weekly_department_activity wda
    JOIN
        department_sizes ds ON wda.department = ds.department AND wda.organization_id = ds.organization_id
)
SELECT
    department,
    week,
    activity_count,
    active_people,
    dept_size,
    engagement_percentage,
    artifacts_touched,
    -- Week-over-week activity change percentage
    CASE
        WHEN prev_week_count IS NULL THEN NULL
        ELSE ROUND((activity_count - prev_week_count)::numeric / NULLIF(prev_week_count, 0) * 100, 2)
    END AS wow_activity_change_pct,
    -- Trend status
    CASE
        WHEN activity_count > four_week_avg_activity * 1.5 THEN 'SIGNIFICANT INCREASE'
        WHEN activity_count > four_week_avg_activity * 1.2 THEN 'MODERATE INCREASE'
        WHEN activity_count < four_week_avg_activity * 0.5 THEN 'SIGNIFICANT DECREASE'
        WHEN activity_count < four_week_avg_activity * 0.8 THEN 'MODERATE DECREASE'
        ELSE 'STABLE'
    END AS trend_status,
    -- Activity per person
    ROUND(activity_count::numeric / NULLIF(active_people, 0), 2) AS activities_per_active_person,
    -- Relative department activity (vs organization average for the week)
    ROUND(
        activity_count::numeric / NULLIF((
            SELECT SUM(wda.activity_count)
            FROM weekly_department_activity wda
            WHERE wda.week = dt.week
            AND wda.organization_id = dt.organization_id
        ), 0) * 100,
    2) AS org_activity_percentage
FROM
    department_trends dt
ORDER BY
    department,
    week DESC;
