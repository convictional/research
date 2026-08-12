-- This query quantifies individual expertise and knowledge contributions,
-- using weighted scoring that factors in contribution type and recency,
-- to help executives identify key knowledge holders and assess potential
-- risk areas if these individuals were to leave the organization.

WITH person_activities AS (
 SELECT
   p.id AS person_id,
   p.name,
   p.email,
   p.aliases,
   a.id AS activity_id,
   a.action_type,
   a.created_at,
   a.artifact_id,
   art.title AS artifact_title,
   art.tags AS artifact_tags,
   -- Weight actions by type (creation/reviews weighted higher than comments)
   CASE
     WHEN a.action_type = 'create' THEN 3.0
     WHEN a.action_type = 'review' THEN 2.0
     WHEN a.action_type = 'decide' THEN 2.5
     WHEN a.action_type = 'comment' THEN 1.0
     ELSE 0.5
   END AS contribution_weight,
   -- Apply temporal decay factor - more recent contributions weighted higher
   CASE
     WHEN a.created_at > NOW() - INTERVAL '3 months' THEN 1.0
     WHEN a.created_at > NOW() - INTERVAL '6 months' THEN 0.8
     WHEN a.created_at > NOW() - INTERVAL '12 months' THEN 0.5
     ELSE 0.2
   END AS recency_factor
 FROM
   person p
 JOIN
   activity a ON p.id = a.actor_id
 JOIN
   artifact art ON a.artifact_id = art.id
),
domain_expertise AS (
 SELECT
   person_id,
   name,
   email,
   COUNT(DISTINCT artifact_id) AS distinct_artifacts,
   SUM(contribution_weight * recency_factor) AS weighted_contribution,
   MIN(created_at) AS first_contribution,
   MAX(created_at) AS last_contribution,
   NOW() - MAX(created_at) AS time_since_last_contribution
 FROM
   person_activities
 GROUP BY
   person_id, name, email
)
SELECT
 name,
 email,
 distinct_artifacts,
 ROUND(weighted_contribution, 2) AS expertise_score,
 first_contribution,
 last_contribution,
 time_since_last_contribution,
 -- Knowledge risk factor - higher when expertise concentrated in few artifacts
 CASE
   WHEN distinct_artifacts < 2 AND weighted_contribution > 10 THEN 'HIGH RISK'
   WHEN distinct_artifacts < 5 AND weighted_contribution > 20 THEN 'MEDIUM RISK'
   ELSE 'LOW RISK'
 END AS knowledge_risk
FROM
 domain_expertise
WHERE
 weighted_contribution > 5 -- Focus on meaningful contributions
ORDER BY
 weighted_contribution DESC;
