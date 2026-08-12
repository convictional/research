-- This query identifies potential knowledge bottlenecks within the organization by
-- analyzing individuals who are primary or sole contributors to critical artifacts,
-- helping executives mitigate risk of knowledge loss and improve work distribution.

WITH artifact_activity AS (
    SELECT
        artifact_id,
        COUNT(DISTINCT actor_id) AS contributors,
        COUNT(*) AS total_activities
    FROM
        activity
    GROUP BY
        artifact_id
),
person_centrality AS (
    SELECT
        p.id AS person_id,
        p.name AS person_name,
        -- Count artifacts where this person is the primary contributor (>50% of activity)
        COUNT(DISTINCT CASE
            WHEN (SELECT COUNT(*) FROM activity a2 WHERE a2.artifact_id = a1.artifact_id AND a2.actor_id = p.id) >
                 (SELECT 0.5 * COUNT(*) FROM activity a3 WHERE a3.artifact_id = a1.artifact_id)
            THEN a1.artifact_id
        END) AS primary_contributor_artifacts,
        -- Count artifacts where this person is the sole contributor
        COUNT(DISTINCT CASE
            WHEN aa.contributors = 1 THEN a1.artifact_id
        END) AS sole_contributor_artifacts,
        -- Count of all artifacts this person has contributed to
        COUNT(DISTINCT a1.artifact_id) AS total_artifacts_contributed,
        -- Percent of total organizational artifacts this person has touched
        COUNT(DISTINCT a1.artifact_id)::float / (SELECT COUNT(DISTINCT artifact_id) FROM activity WHERE organization_id = p.organization_id) * 100 AS artifact_coverage_pct
    FROM
        person p
    JOIN
        activity a1 ON p.id = a1.actor_id
    JOIN
        artifact_activity aa ON a1.artifact_id = aa.artifact_id
    GROUP BY
        p.id, p.name, p.organization_id
)

SELECT
    person_name,
    primary_contributor_artifacts,
    sole_contributor_artifacts,
    total_artifacts_contributed,
    artifact_coverage_pct,
    CASE
        WHEN sole_contributor_artifacts > 5 OR (primary_contributor_artifacts > 10 AND artifact_coverage_pct > 25)
            THEN 'CRITICAL BOTTLENECK'
        WHEN sole_contributor_artifacts > 2 OR (primary_contributor_artifacts > 5 AND artifact_coverage_pct > 15)
            THEN 'HIGH RISK'
        WHEN sole_contributor_artifacts > 0 OR (primary_contributor_artifacts > 3 AND artifact_coverage_pct > 10)
            THEN 'MODERATE RISK'
        ELSE 'LOW RISK'
    END AS bottleneck_risk
FROM
    person_centrality
WHERE
    total_artifacts_contributed > 0
ORDER BY
    sole_contributor_artifacts DESC,
    primary_contributor_artifacts DESC,
    artifact_coverage_pct DESC;
