-- This query identifies and quantifies knowledge flow patterns between departments,
-- tracking how quickly information created by one team is consumed by others and
-- measuring the efficiency of cross-departmental information transfer to highlight
-- opportunities for improving organizational knowledge sharing.


WITH department_artifacts AS (
    -- Get each department's activity per artifact with creation indicators
    SELECT
        COALESCE(p.department, 'Unassigned') AS department,
        a.artifact_id,
        MIN(a.created_at) AS first_activity,
        MAX(a.created_at) AS last_activity,
        COUNT(*) AS activity_count,
        -- Check if this department created the artifact (first interaction)
        CASE WHEN MIN(a.created_at) = (
            SELECT MIN(created_at) FROM activity WHERE artifact_id = a.artifact_id
        ) THEN 1 ELSE 0 END AS is_creator
    FROM
        person p
    JOIN
        activity a ON p.id = a.actor_id
    GROUP BY
        COALESCE(p.department, 'Unassigned'), a.artifact_id
),
knowledge_flow AS (
    -- Find creator department and consumer departments
    SELECT
        creator.department AS creator_department,
        consumer.department AS consumer_department,
        creator.artifact_id,
        creator.first_activity AS creation_time,
        consumer.first_activity AS first_consumption,
        consumer.activity_count AS consumption_activities,
        consumer.first_activity - creator.first_activity AS time_to_consumption
    FROM
        department_artifacts creator
    JOIN
        department_artifacts consumer
        ON creator.artifact_id = consumer.artifact_id
        AND creator.department != consumer.department
    WHERE
        creator.is_creator = 1
)

SELECT
    creator_department,
    consumer_department,
    COUNT(DISTINCT artifact_id) AS artifacts_created_and_consumed,
    ROUND(AVG(EXTRACT(EPOCH FROM time_to_consumption) / 3600), 2) AS avg_hours_to_consumption,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM time_to_consumption) / 3600) AS median_hours_to_consumption,
    SUM(consumption_activities) AS total_consumption_activities,
    -- For each producer-consumer pair, calculate knowledge flow score
    ROUND(
        (COUNT(DISTINCT artifact_id) * 10) +
        (SUM(consumption_activities) * 2) -
        (AVG(EXTRACT(EPOCH FROM time_to_consumption) / 86400) * 5)  -- Subtract days to consumption
    , 0) AS knowledge_flow_score,
    -- Knowledge flow patterns
    CASE WHEN
        (SELECT COUNT(DISTINCT artifact_id) FROM knowledge_flow kf2
         WHERE kf2.creator_department = creator_department
         AND kf2.consumer_department = consumer_department) >
        (SELECT COUNT(DISTINCT artifact_id) FROM knowledge_flow kf3
         WHERE kf3.creator_department = consumer_department
         AND kf3.consumer_department = creator_department)
    THEN
        creator_department || ' → ' || consumer_department
    ELSE
        'BALANCED'
    END AS flow_direction
FROM
    knowledge_flow
GROUP BY
    creator_department, consumer_department
HAVING
    COUNT(DISTINCT artifact_id) >= 2  -- Only meaningful knowledge flows
ORDER BY
    knowledge_flow_score DESC;
