-- Organizational Memory & Decision Continuity Analysis
-- This query addresses the "corporate memory problem" by identifying patterns
-- of knowledge retrieval and usage of historical decisions

WITH historical_decisions AS (
    SELECT
        a.id,
        a.title,
        a.type,
        a.created_at,
        (EXTRACT(YEAR FROM AGE(NOW(), a.created_at)) * 12 +
         EXTRACT(MONTH FROM AGE(NOW(), a.created_at))) AS age_in_months
    FROM
        artifact a
    WHERE
        a.type IN ('decision_process', 'meeting', 'google_doc') OR
        a.title ILIKE '%decision%' OR
        a.title ILIKE '%approval%' OR
        a.title ILIKE '%strategy%'
),
artifact_references AS (
    -- Find activities that reference other artifacts
    SELECT
        act.id AS activity_id,
        act.artifact_id AS current_artifact_id,
        act.actor_id,
        act.created_at AS reference_date,
        hd.id AS referenced_artifact_id,
        hd.title AS referenced_title,
        hd.age_in_months
    FROM
        activity act
    JOIN
        historical_decisions hd ON
            (act.snippet ILIKE '%' || hd.title || '%' OR
             act.metadata::text ILIKE '%' || hd.title || '%')
    WHERE
        act.created_at > hd.created_at
        AND act.artifact_id != hd.id -- Only count references to other artifacts
),
memory_metrics AS (
    SELECT
        hd.id,
        hd.title,
        hd.type,
        hd.age_in_months,
        -- References to this decision/artifact
        COUNT(DISTINCT ar.activity_id) AS reference_count,
        COUNT(DISTINCT ar.current_artifact_id) AS referenced_in_artifacts,
        COUNT(DISTINCT ar.actor_id) AS referenced_by_people,
        MAX(ar.reference_date) AS most_recent_reference
    FROM
        historical_decisions hd
    LEFT JOIN
        artifact_references ar ON hd.id = ar.referenced_artifact_id
    GROUP BY
        hd.id, hd.title, hd.type, hd.age_in_months
)

SELECT
    title AS "Historical Decision",
    type AS "Decision Type",
    age_in_months AS "Age (Months)",
    reference_count AS "Reference Count",
    referenced_in_artifacts AS "Referenced In Documents",
    referenced_by_people AS "Referenced By People",
    CASE
        WHEN most_recent_reference IS NULL THEN 'NEVER REFERENCED'
        WHEN EXTRACT(DAY FROM NOW() - most_recent_reference) <= 30 THEN 'RECENT: Active Memory'
        WHEN EXTRACT(DAY FROM NOW() - most_recent_reference) <= 90 THEN 'SEMI-RECENT: Accessible Memory'
        ELSE 'DISTANT: Fading Memory'
    END AS "Memory Status",
    CASE
        WHEN reference_count = 0 THEN 'FORGOTTEN: Institutional Knowledge Loss'
        WHEN age_in_months > 12 AND reference_count > 10 THEN 'FOUNDATIONAL: Enduring Decision'
        WHEN age_in_months <= 6 AND reference_count > 5 THEN 'ACTIVE: Currently Relevant'
        WHEN reference_count > 0 AND referenced_by_people <= 2 THEN 'SILOED: Limited Knowledge'
        ELSE 'MODERATE: Some Continuity'
    END AS "Knowledge Continuity"
FROM
    memory_metrics
ORDER BY
    -- Order by either age or reference patterns
    CASE WHEN reference_count = 0 THEN 0 ELSE 1 END DESC,
    reference_count DESC;
