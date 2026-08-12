-- ======================================================
-- Network Bridges and Organizational Connectors
-- ======================================================
-- This query identifies people who connect different groups,
-- facilitating cross-functional collaboration and information flow.

WITH person_connections AS (
    -- Find all pairs of people who have worked on the same artifacts
    SELECT
        a1.actor_id AS person_id_1,
        a2.actor_id AS person_id_2,
        a1.artifact_id,
        COUNT(*) AS connection_strength
    FROM
        activity a1
    JOIN
        activity a2 ON a1.artifact_id = a2.artifact_id AND a1.actor_id < a2.actor_id
    GROUP BY
        a1.actor_id, a2.actor_id, a1.artifact_id
),
person_connection_counts AS (
    -- Aggregate to find total number of connections per person
    SELECT
        p.id AS person_id,
        p.name AS person_name,
        COUNT(DISTINCT pc.person_id_2) AS direct_connections,
        SUM(pc.connection_strength) AS total_connection_strength,
        -- Count distinct artifacts the person has connected others through
        COUNT(DISTINCT pc.artifact_id) AS bridge_artifacts
    FROM
        person p
    LEFT JOIN
        person_connections pc ON p.id = pc.person_id_1
    GROUP BY
        p.id, p.name
),
connection_network AS (
    -- Identify groups by looking at who tends to work together
    SELECT
        pc.person_id_1,
        pc.person_id_2,
        COUNT(DISTINCT pc.artifact_id) AS shared_artifacts
    FROM
        person_connections pc
    GROUP BY
        pc.person_id_1, pc.person_id_2
    HAVING
        COUNT(DISTINCT pc.artifact_id) >= 3 -- Meaningful connections only
),
bridging_score AS (
    -- Calculate bridging score based on how many people are connected through this person
    SELECT
        p.id AS person_id,
        p.name AS person_name,
        -- Count pairs of people who only connect through this person
        COUNT(DISTINCT CASE
            WHEN EXISTS (
                SELECT 1 FROM connection_network cn1
                WHERE (cn1.person_id_1 = p.id AND cn1.person_id_2 = cn2.person_id_2)
                OR (cn1.person_id_2 = p.id AND cn1.person_id_1 = cn2.person_id_1)
            ) AND
            NOT EXISTS (
                SELECT 1 FROM connection_network cn3
                WHERE (cn3.person_id_1 = cn2.person_id_1 AND cn3.person_id_2 = cn2.person_id_2)
            )
            THEN CONCAT(LEAST(cn2.person_id_1, cn2.person_id_2), '-', GREATEST(cn2.person_id_1, cn2.person_id_2))
        END) AS bridge_connections
    FROM
        person p
    CROSS JOIN
        connection_network cn2
    WHERE
        p.id != cn2.person_id_1 AND p.id != cn2.person_id_2
    GROUP BY
        p.id, p.name
)
SELECT
    pcc.person_name,
    pcc.direct_connections,
    pcc.total_connection_strength,
    pcc.bridge_artifacts,
    COALESCE(bs.bridge_connections, 0) AS bridge_connections,
    -- Activity spread across different types of work
    (SELECT COUNT(DISTINCT a.action_type) FROM activity a WHERE a.actor_id = pcc.person_id) AS activity_diversity,
    -- Calculate connector score (higher = more central to the organization's work)
    ROUND((
        (pcc.direct_connections::numeric / NULLIF((SELECT MAX(direct_connections) FROM person_connection_counts), 0)) * 40 +
        (pcc.bridge_artifacts::numeric / NULLIF((SELECT MAX(bridge_artifacts) FROM person_connection_counts), 0)) * 30 +
        (COALESCE(bs.bridge_connections, 0)::numeric / NULLIF((SELECT MAX(bridge_connections) FROM bridging_score), 0)) * 30
    ) * 100) AS connector_score,
    CASE
        WHEN pcc.direct_connections > 10 AND COALESCE(bs.bridge_connections, 0) > 5 THEN 'CRITICAL CONNECTOR'
        WHEN pcc.direct_connections > 5 AND COALESCE(bs.bridge_connections, 0) > 2 THEN 'MAJOR CONNECTOR'
        WHEN pcc.direct_connections > 3 OR COALESCE(bs.bridge_connections, 0) > 0 THEN 'CONNECTOR'
        ELSE 'TEAM MEMBER'
    END AS connector_role
FROM
    person_connection_counts pcc
LEFT JOIN
    bridging_score bs ON pcc.person_id = bs.person_id
WHERE
    pcc.direct_connections > 0
ORDER BY
    connector_score DESC,
    bridge_connections DESC;
