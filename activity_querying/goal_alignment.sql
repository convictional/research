-- Goal Alignment & Implementation Analysis
-- This query tackles the alignment challenges by measuring how organizational goals
-- propagate through the company and translate into action
WITH goal_artifacts_base AS (
    SELECT
        a.id,
        a.title,
        a.type,
        a.created_at,
        a.tags,
        a.metadata
    FROM
        artifact a
    WHERE
        a.title ILIKE '%goal%' OR
        a.title ILIKE '%objective%' OR
        a.title ILIKE '%OKR%' OR
        a.title ILIKE '%KPI%' OR
        a.title ILIKE '%target%' OR
        a.title ILIKE '%strategy%' OR
        a.type = 'goal'
),
tag_words AS (
    SELECT
        a.id,
        a.title,
        a.type,
        a.created_at,
        a.metadata,
        tag_value
    FROM
        goal_artifacts_base a,
        LATERAL jsonb_array_elements_text(a.tags) AS tag_value
),
goal_artifacts AS (
    SELECT
        a.id,
        a.title,
        a.type,
        a.created_at,
        a.metadata,
        regexp_split_to_table(
            lower(
                a.title || ' ' ||
                COALESCE(a.metadata::text, '') || ' ' ||
                string_agg(COALESCE(a.tag_value, ''), ' ')
            ),
            '[^a-zA-Z0-9]+'
        ) AS keyword
    FROM
        (
            SELECT
                id,
                title,
                type,
                created_at,
                metadata,
                tag_value
            FROM
                tag_words
            UNION ALL
            SELECT
                id,
                title,
                type,
                created_at,
                metadata,
                NULL as tag_value
            FROM
                goal_artifacts_base
            WHERE
                tags IS NULL OR jsonb_array_length(tags) = 0
        ) a
    GROUP BY
        a.id, a.title, a.type, a.created_at, a.metadata
),
goal_terms AS (
    SELECT keyword
    FROM (VALUES
        ('flock'),
        ('mappedin'),
        ('cloudlinux')
        -- Add more keywords as needed
    ) AS custom_goals(keyword)
),
goal_propagation AS (
    SELECT
        gt.keyword AS goal_term,
        a.id AS artifact_id,
        a.title AS artifact_title,
        a.type AS artifact_type,
        p.id AS person_id,
        p.name AS person_name,
        p.email AS person_email,
        act.created_at AS activity_date,
        act.action_type,
        act.snippet
    FROM
        goal_terms gt
    JOIN
        activity act ON
        act.snippet ILIKE '%' || gt.keyword || '%' OR
        act.metadata::text ILIKE '%' || gt.keyword || '%'
    JOIN
        artifact a ON act.artifact_id = a.id
    JOIN
        person p ON act.actor_id = p.id
    WHERE
        act.created_at >= NOW() - INTERVAL '180 days'
),
goal_adoption AS (
    SELECT
        goal_term,
        COUNT(DISTINCT artifact_id) AS artifacts_mentioning_goal,
        COUNT(DISTINCT person_id) AS people_engaged,
        MIN(activity_date) AS first_mention,
        MAX(activity_date) AS most_recent_mention,
        COUNT(DISTINCT CASE WHEN action_type IN ('create', 'update') THEN artifact_id END) AS implementation_artifacts,
        COUNT(DISTINCT CASE WHEN action_type IN ('comment', 'review') THEN artifact_id END) AS discussion_artifacts
    FROM
        goal_propagation
    GROUP BY
        goal_term
)
SELECT
    goal_term AS "Keyword",
    artifacts_mentioning_goal AS "Mentions Across Artifacts",
    people_engaged AS "People Engaged",
    EXTRACT(DAY FROM NOW() - first_mention) AS "Days Since Introduction",
    EXTRACT(DAY FROM NOW() - most_recent_mention) AS "Days Since Last Mention",
    implementation_artifacts AS "Implementation Artifacts",
    discussion_artifacts AS "Discussion-Only Artifacts",
    ROUND(
        implementation_artifacts * 100.0 / NULLIF(artifacts_mentioning_goal, 0),
        2
    ) AS "Implementation Ratio (%)",
    CASE
        WHEN people_engaged < 4 THEN 'LOW REACH: Minimal Adoption'
        WHEN people_engaged > 8 THEN 'HIGH REACH: Organization-Wide'
        ELSE 'MODERATE REACH: Team-Level Adoption'
    END AS "Reach",
    CASE
        WHEN implementation_artifacts = 0 THEN 'STALLED: Discussion Only'
        WHEN implementation_artifacts > discussion_artifacts * 2 THEN 'EXECUTION FOCUSED: Implementation'
        WHEN discussion_artifacts > implementation_artifacts * 2 THEN 'PLANNING HEAVY: More Talk Than Action'
        ELSE 'BALANCED: Planning & Execution'
    END AS "Implementation Status"
FROM
    goal_adoption
WHERE
    artifacts_mentioning_goal > 2 -- Filter out noise
ORDER BY
    people_engaged DESC;
