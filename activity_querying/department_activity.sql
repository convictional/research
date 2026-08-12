-- This query summarizes department-level productivity metrics over the past 90 days,
-- providing executives with key insights on team engagement, workload distribution,
-- and relative departmental contributions to overall organizational activity.

SELECT
    COALESCE(p.department, 'Unassigned') AS department,
    COUNT(DISTINCT p.id) AS team_size,
    COUNT(DISTINCT a.id) AS total_activities,
    COUNT(DISTINCT a.artifact_id) AS artifacts_engaged,
    COUNT(DISTINCT a.action_type) AS activity_types,
    ROUND(COUNT(DISTINCT a.id)::numeric / COUNT(DISTINCT p.id), 2) AS activities_per_person,
    ROUND(COUNT(DISTINCT a.created_at::date)::numeric / COUNT(DISTINCT p.id), 2) AS active_days_per_person,
    MAX(a.created_at) AS last_activity,
    NOW() - MAX(a.created_at) AS time_since_last_activity,
    -- Department engagement percentage (active people / total people in department)
    ROUND(COUNT(DISTINCT a.actor_id)::numeric / NULLIF(COUNT(DISTINCT p.id), 0) * 100, 2) AS engagement_percentage,
    -- Department's share of organization's total activity
    ROUND(COUNT(DISTINCT a.id)::numeric / (SELECT COUNT(*) FROM activity), 2) * 100 AS org_activity_percentage
FROM
    person p
LEFT JOIN
    activity a ON p.id = a.actor_id
WHERE
    a.created_at >= NOW() - INTERVAL '90 days'
    OR a.created_at IS NULL  -- Include departments with no activity
GROUP BY
    COALESCE(p.department, 'Unassigned')
ORDER BY
    total_activities DESC NULLS LAST;
