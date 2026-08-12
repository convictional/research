-- This query analyzes cross-department collaborations over the past 180 days,
-- identifying the strength, balance, and extent of interdepartmental work
-- to help executives understand team dynamics and organizational connectivity.

WITH department_artifacts AS (
    -- Find which departments have worked on which artifacts
    SELECT
        COALESCE(p.department, 'Unassigned') AS department,
        a.artifact_id,
        COUNT(DISTINCT a.id) AS activity_count,
        COUNT(DISTINCT a.actor_id) AS people_count
    FROM
        person p
    JOIN
        activity a ON p.id = a.actor_id
    WHERE
        a.created_at >= NOW() - INTERVAL '180 days'
    GROUP BY
        COALESCE(p.department, 'Unassigned'), a.artifact_id
)

SELECT
    da1.department AS department_1,
    da2.department AS department_2,
    COUNT(DISTINCT da1.artifact_id) AS shared_artifacts,
    SUM(da1.activity_count) AS dept1_activities,
    SUM(da2.activity_count) AS dept2_activities,
    SUM(da1.people_count) AS dept1_people,
    SUM(da2.people_count) AS dept2_people,
    -- Calculate collaboration intensity score
    ROUND(
        (COUNT(DISTINCT da1.artifact_id)::numeric /
        (SELECT COUNT(DISTINCT artifact_id) FROM department_artifacts
         WHERE department IN (da1.department, da2.department))) * 100,
    2) AS collaboration_percentage,
    -- Collaboration balance (0 = perfectly balanced, 1 = completely one-sided)
    ROUND(
        ABS(SUM(da1.activity_count) - SUM(da2.activity_count))::numeric /
        NULLIF(SUM(da1.activity_count) + SUM(da2.activity_count), 0),
    2) AS collaboration_balance,
    -- Collaboration classification
    CASE
        WHEN COUNT(DISTINCT da1.artifact_id) < 10 THEN 'MINIMAL COLLABORATION'
        WHEN COUNT(DISTINCT da1.artifact_id) < 25 THEN 'LIGHT COLLABORATION'
        WHEN COUNT(DISTINCT da1.artifact_id) < 50 THEN 'MODERATE COLLABORATION'
        ELSE 'STRONG COLLABORATION'
    END AS collaboration_level
FROM
    department_artifacts da1
JOIN
    department_artifacts da2
    ON da1.artifact_id = da2.artifact_id AND da1.department < da2.department
GROUP BY
    da1.department, da2.department
ORDER BY
    shared_artifacts DESC,
    collaboration_percentage DESC;
