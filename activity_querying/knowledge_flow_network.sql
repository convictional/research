-- This query analyzes the network of knowledge sharing between individuals,
-- identifying pairs of employees who frequently collaborate on the same artifacts
-- and measuring the efficiency of their information exchange to help executives
-- understand and optimize the organization's informal knowledge transfer processes.

WITH artifact_collaborators AS (
 SELECT
   a.artifact_id,
   a.actor_id,
   p.name,
   p.email,
   a.artifact_type,
   a.action_type,
   a.created_at,
   art.title,
   art.tags
 FROM
   activity a
 JOIN
   person p ON a.actor_id = p.id
 JOIN
   artifact art ON a.artifact_id = art.id
),
collaboration_pairs AS (
 SELECT
   a1.actor_id AS person1_id,
   a1.name AS person1_name,
   a2.actor_id AS person2_id,
   a2.name AS person2_name,
   a1.artifact_id,
   a1.artifact_type,
   a1.title AS artifact_title,
   -- Ensure we don't double-count (order pairs consistently)
   CASE WHEN a1.actor_id < a2.actor_id THEN a1.actor_id ELSE a2.actor_id END AS lower_id,
   CASE WHEN a1.actor_id < a2.actor_id THEN a2.actor_id ELSE a1.actor_id END AS higher_id,
   -- Calculate time between contributions to measure knowledge transfer lag
   ABS(EXTRACT(EPOCH FROM (a1.created_at - a2.created_at)) / 86400) AS days_between_activity
 FROM
   artifact_collaborators a1
 JOIN
   artifact_collaborators a2 ON a1.artifact_id = a2.artifact_id AND a1.actor_id != a2.actor_id
 WHERE
   -- Focus on meaningful collaborations that occur within a reasonable timeframe
   ABS(EXTRACT(EPOCH FROM (a1.created_at - a2.created_at)) / 86400) < 30
   -- Only include one direction of the pair
   AND a1.actor_id < a2.actor_id
)
SELECT
 person1_name,
 person2_name,
 COUNT(DISTINCT artifact_id) AS shared_artifacts,
 ROUND(AVG(days_between_activity), 1) AS avg_transfer_time_days,
 -- Calculate strength of knowledge flow between people
 CASE
   WHEN COUNT(DISTINCT artifact_id) > 10 THEN 'STRONG'
   WHEN COUNT(DISTINCT artifact_id) > 5 THEN 'MODERATE'
   ELSE 'WEAK'
 END AS connection_strength,
 -- Optional: include a list of artifact types they collaborated on
 STRING_AGG(DISTINCT artifact_type, ', ') AS artifact_types
FROM
 collaboration_pairs
GROUP BY
 lower_id, higher_id, person1_name, person2_name
HAVING
 COUNT(DISTINCT artifact_id) > 1
ORDER BY
 shared_artifacts DESC, avg_transfer_time_days;
