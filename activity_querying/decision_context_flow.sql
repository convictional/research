-- Decision Context Flow & Visibility Analysis
-- This query helps identify how effectively decision context flows from leadership
-- through the organization and where critical information might be getting lost

WITH decision_artifacts AS (
    SELECT
        a.id,
        a.title,
        a.type,
        a.created_at,
        a.metadata,
        -- Identify likely decision artifacts based on titles/types
        CASE
            WHEN a.title ILIKE '%decision%' OR
                 a.title ILIKE '%approved%' OR
                 a.title ILIKE '%strategy%' OR
                 a.title ILIKE '%plan%' OR
                 a.type IN ('decision_process', 'meeting', 'google_doc')
            THEN TRUE
            ELSE FALSE
        END AS is_decision_artifact
    FROM
        artifact a
    WHERE
        a.created_at >= NOW() - INTERVAL '180 days'
),
decision_propagation AS (
    SELECT
        da.id AS artifact_id,
        da.title,
        da.type,
        p.id AS actor_id,
        p.name AS actor_name,
        p.email AS actor_email,
        p.department,
        act.action_type,
        act.created_at,
        -- Identify executive actors based on email/department
        CASE
            WHEN p.email LIKE '%@example.com' AND (  -- replace with your internal domain
                p.email LIKE 'ceo%' OR
                p.email LIKE 'cfo%' OR
                p.email LIKE 'cto%' OR
                p.email LIKE 'vp%' OR
                p.email LIKE 'director%' OR
                LOWER(p.department) IN ('leadership', 'executive', 'management', 'ceo')
            ) THEN 'Executive'
            ELSE 'Non-Executive'
        END AS actor_level
    FROM
        decision_artifacts da
    JOIN
        activity act ON da.id = act.artifact_id
    JOIN
        person p ON act.actor_id = p.id
    WHERE
        da.is_decision_artifact = TRUE
),
decision_engagement AS (
    SELECT
        artifact_id,
        title,
        type,
        COUNT(DISTINCT CASE WHEN actor_level = 'Executive' THEN actor_id END) AS executive_engagement,
        COUNT(DISTINCT CASE WHEN actor_level = 'Non-Executive' THEN actor_id END) AS non_executive_engagement,
        MIN(CASE WHEN actor_level = 'Executive' THEN created_at END) AS first_executive_activity,
        MAX(CASE WHEN actor_level = 'Executive' THEN created_at END) AS last_executive_activity,
        MIN(CASE WHEN actor_level = 'Non-Executive' THEN created_at END) AS first_non_executive_activity,
        MAX(CASE WHEN actor_level = 'Non-Executive' THEN created_at END) AS last_non_executive_activity
    FROM
        decision_propagation
    GROUP BY
        artifact_id, title, type
)

SELECT
    title AS "Decision Context",
    type AS "Context Format",
    executive_engagement AS "Executive Engagement",
    non_executive_engagement AS "Team Engagement",
    CASE
        WHEN first_non_executive_activity IS NULL THEN 'INVISIBLE: No Team Visibility'
        WHEN first_non_executive_activity > first_executive_activity + INTERVAL '7 days' THEN 'DELAYED: Slow Propagation'
        WHEN non_executive_engagement < 3 THEN 'LIMITED: Minimal Reach'
        WHEN non_executive_engagement > 10 THEN 'BROAD: Good Visibility'
        ELSE 'MODERATE: Some Visibility'
    END AS "Context Visibility",
    CASE
        WHEN last_non_executive_activity IS NOT NULL THEN
            EXTRACT(DAY FROM last_non_executive_activity - first_executive_activity)
        ELSE NULL
    END AS "Context Propagation Time (Days)",
    CASE
        WHEN non_executive_engagement = 0 THEN 'NO FLOW'
        WHEN non_executive_engagement < executive_engagement THEN 'POOR TRANSMISSION'
        WHEN non_executive_engagement > executive_engagement * 3 THEN 'STRONG AMPLIFICATION'
        ELSE 'ADEQUATE TRANSMISSION'
    END AS "Information Flow Quality"
FROM
    decision_engagement
ORDER BY
    non_executive_engagement DESC;
