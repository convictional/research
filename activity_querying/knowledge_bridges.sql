-- This query identifies key individuals who bridge different knowledge domains,
-- highlighting employees who can facilitate cross-functional knowledge transfer
-- and serve as valuable resources for organization-wide initiatives requiring
-- expertise across multiple specialized areas.

WITH domain_participation AS (
    SELECT
        p.id AS person_id,
        p.name,
        p.email,
        a.artifact_type,
        COUNT(DISTINCT a.artifact_id) AS artifacts_in_domain,
        MAX(a.created_at) AS last_contribution_to_domain
    FROM
        person p
    JOIN
        activity a ON p.id = a.actor_id
    GROUP BY
        p.id, p.name, p.email, a.artifact_type
),
domain_counts AS (
    SELECT
        person_id,
        name,
        email,
        COUNT(DISTINCT artifact_type) AS domain_count,
        array_agg(DISTINCT artifact_type) AS domains,
        SUM(artifacts_in_domain) AS total_artifacts
    FROM
        domain_participation
    GROUP BY
        person_id, name, email
),
domain_pairs AS (
    SELECT
        dp1.person_id,
        dp1.artifact_type AS domain1,
        dp2.artifact_type AS domain2
    FROM
        domain_participation dp1
    JOIN
        domain_participation dp2 ON dp1.person_id = dp2.person_id AND dp1.artifact_type < dp2.artifact_type
    WHERE
        -- Both domains have meaningful contributions
        dp1.artifacts_in_domain >= 3 AND dp2.artifacts_in_domain >= 3
        -- Recent activity in both domains
        AND dp1.last_contribution_to_domain > NOW() - INTERVAL '6 months'
        AND dp2.last_contribution_to_domain > NOW() - INTERVAL '6 months'
)
SELECT
    p.name,
    p.email,
    dc.domain_count,
    dc.domains,
    dc.total_artifacts,
    COUNT(DISTINCT dp.domain1 || '-' || dp.domain2) AS domain_bridges,
    -- Calculate a bridge score that measures knowledge transfer potential
    ROUND(
        (dc.domain_count * 10) +
        (dc.total_artifacts * 0.5) +
        (COUNT(DISTINCT dp.domain1 || '-' || dp.domain2) * 5)
    ) AS bridge_score,
    CASE
        WHEN dc.domain_count >= 4 AND COUNT(DISTINCT dp.domain1 || '-' || dp.domain2) >= 3 THEN 'CRITICAL BRIDGE'
        WHEN dc.domain_count >= 3 AND COUNT(DISTINCT dp.domain1 || '-' || dp.domain2) >= 2 THEN 'IMPORTANT BRIDGE'
        WHEN dc.domain_count >= 2 THEN 'POTENTIAL BRIDGE'
        ELSE 'SPECIALIST'
    END AS bridge_role
FROM
    domain_counts dc
JOIN
    person p ON dc.person_id = p.id
LEFT JOIN
    domain_pairs dp ON p.id = dp.person_id
GROUP BY
    p.id, p.name, p.email, dc.domain_count, dc.domains, dc.total_artifacts
ORDER BY
    bridge_score DESC;
