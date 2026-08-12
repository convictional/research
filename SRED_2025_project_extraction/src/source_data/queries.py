GITHUB_ISSUES_QUERY = """
-- 1 row per issue comment,
-- with issue metadata attached to each row
WITH issue_comments AS (
    SELECT
        i.id AS issue_id,
        i.number AS issue_number,
        i.title AS issue_title,
        i.body AS issue_body,
        ic.body AS comment_body,
        i.created_at AS issue_created_at,
        i.closed_at as issue_closed_at,
        ic.created_at AS comment_created_at,
        i.user_id AS issue_user_id,
        ic.user_id AS comment_user_id,
        r.full_name AS repository_full_name
    FROM `${GCP_PROJECT}.github.issue` i
    LEFT JOIN `${GCP_PROJECT}.github.issue_comment` ic
        ON i.id = ic.issue_id
    LEFT JOIN `${GCP_PROJECT}.github.repository` r
        ON i.repository_id = r.id
    WHERE
        i.pull_request IS FALSE
        AND i.body IS NOT NULL
        AND ic.body IS NOT NULL
        AND i.created_at >= '2025-01-01'
        AND i.created_at < '2026-01-01'
        AND ic.created_at >= '2025-01-01'
        AND ic.created_at < '2026-01-01'
        AND r.full_name in (
            "convictional/decide",
            "convictional/data"
        )
),

-- 1 row per issue or issue comment
-- there will be duplicates, e.g. for each comment there will be a row for a given issue
issues_and_comments_flattened as (
    -- issues
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        issue_body AS content,
        issue_created_at AS content_created_at,
        issue_user_id AS user_id
    FROM issue_comments

    UNION ALL

    -- issue comments
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        comment_body AS content,
        comment_created_at AS content_created_at,
        comment_user_id AS user_id
    FROM issue_comments
    WHERE comment_body IS NOT NULL
),

-- 1 row per issue or issue comment,
-- no duplication of issues here
issues_and_comments_flattened_deduplicated as (
    SELECT DISTINCT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        content,
        content_created_at,
        user_id
    FROM issues_and_comments_flattened
),

-- 1 row per issue
issues_timestamps as (
    SELECT DISTINCT
        issue_id,
        MIN(issue_created_at) AS issue_created_at,
        MAX(comment_created_at) as last_comment_at,
        MAX(issue_closed_at) AS issue_closed_at,
    FROM issue_comments
    GROUP BY issue_id
),

-- 1 row per issue or issue comment
content_with_user_names AS (
    SELECT
        icfd.issue_id,
        icfd.issue_number,
        icfd.issue_title,
        icfd.repository_full_name,
        icfd.content,
        icfd.content_created_at,
        icfd.user_id,
        u.name AS user_name,
        u.login AS user_login,
        coalesce(u.name, u.login, 'Unknown user') AS username,
        it.issue_created_at,
        it.last_comment_at,
        it.issue_closed_at
    FROM issues_and_comments_flattened_deduplicated icfd
    LEFT JOIN `${GCP_PROJECT}.github.user` u
        ON icfd.user_id = u.id
    LEFT JOIN issues_timestamps it
        ON icfd.issue_id = it.issue_id
),

-- 1 row per issue
aggregated_content as (
    SELECT
        issue_id,
        issue_number,
        issue_title,
        issue_created_at,
        last_comment_at,
        issue_closed_at,
        repository_full_name,
        STRING_AGG(
            CONCAT(username, ': ', content),
            '\\n'
            ORDER BY
                CASE WHEN content_created_at = issue_created_at THEN 0 ELSE 1 END,
                content_created_at ASC
            LIMIT 1000
        ) AS combined_content
    FROM content_with_user_names
    GROUP BY
        issue_id, issue_number, issue_title, issue_created_at, last_comment_at, issue_closed_at, repository_full_name
)

select *
from aggregated_content
order by issue_id
"""


GITHUB_PULL_REQUESTS_QUERY = """
-- 1 row per issue comment,
-- with issue metadata attached to each row
WITH issue_comments AS (
    SELECT
        i.id AS issue_id,
        i.number AS issue_number,
        i.title AS issue_title,
        i.body AS issue_body,
        ic.body AS comment_body,
        i.created_at AS issue_created_at,
        i.closed_at as issue_closed_at,
        ic.created_at AS comment_created_at,
        i.user_id AS issue_user_id,
        ic.user_id AS comment_user_id,
        r.full_name AS repository_full_name
    FROM `${GCP_PROJECT}.github.issue` i
    LEFT JOIN `${GCP_PROJECT}.github.issue_comment` ic
        ON i.id = ic.issue_id
    LEFT JOIN `${GCP_PROJECT}.github.repository` r
        ON i.repository_id = r.id
    WHERE
        i.pull_request IS TRUE
        AND i.body IS NOT NULL
        AND ic.body IS NOT NULL
        AND i.created_at >= '2025-01-01'
        AND i.created_at < '2026-01-01'
        AND ic.created_at >= '2025-01-01'
        AND ic.created_at < '2026-01-01'
        AND r.full_name in (
            "convictional/decide"
        )
),

-- 1 row per issue or issue comment
-- there will be duplicates, e.g. for each comment there will be a row for a given issue
issues_and_comments_flattened as (
    -- issues
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        issue_body AS content,
        issue_created_at AS content_created_at,
        issue_user_id AS user_id
    FROM issue_comments

    UNION ALL

    -- issue comments
    SELECT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        comment_body AS content,
        comment_created_at AS content_created_at,
        comment_user_id AS user_id
    FROM issue_comments
    WHERE comment_body IS NOT NULL
),

-- 1 row per issue or issue comment,
-- no duplication of issues here
issues_and_comments_flattened_deduplicated as (
    SELECT DISTINCT
        issue_id,
        issue_number,
        issue_title,
        repository_full_name,
        content,
        content_created_at,
        user_id
    FROM issues_and_comments_flattened
),

-- 1 row per issue
issues_timestamps as (
    SELECT DISTINCT
        issue_id,
        MIN(issue_created_at) AS issue_created_at,
        MAX(comment_created_at) as last_comment_at,
        MAX(issue_closed_at) AS issue_closed_at,
    FROM issue_comments
    GROUP BY issue_id
),

-- 1 row per issue or issue comment
content_with_user_names AS (
    SELECT
        icfd.issue_id,
        icfd.issue_number,
        icfd.issue_title,
        icfd.repository_full_name,
        icfd.content,
        icfd.content_created_at,
        icfd.user_id,
        u.name AS user_name,
        u.login AS user_login,
        coalesce(u.name, u.login, 'Unknown user') AS username,
        it.issue_created_at,
        it.last_comment_at,
        it.issue_closed_at
    FROM issues_and_comments_flattened_deduplicated icfd
    LEFT JOIN `${GCP_PROJECT}.github.user` u
        ON icfd.user_id = u.id
    LEFT JOIN issues_timestamps it
        ON icfd.issue_id = it.issue_id
),

-- 1 row per issue
aggregated_content as (
    SELECT
        issue_id,
        issue_number,
        issue_title,
        issue_created_at,
        last_comment_at,
        issue_closed_at,
        repository_full_name,
        STRING_AGG(
            CONCAT(username, ': ', content),
            '\\n'
            ORDER BY
                CASE WHEN content_created_at = issue_created_at THEN 0 ELSE 1 END,
                content_created_at ASC
            LIMIT 1000
        ) AS combined_content
    FROM content_with_user_names
    GROUP BY
        issue_id, issue_number, issue_title, issue_created_at, last_comment_at, issue_closed_at, repository_full_name
)

select *
from aggregated_content
order by issue_id
"""


APP_TASKS_QUERY = """
-- might be inefficient, but follows the same pattern as github issues query
WITH

-- get users to join to app objects for attribution
-- 1 row per user
users AS (
  SELECT
    id,
    name,
    email
  FROM `${GCP_PROJECT}.cloudsql_decide_public.user`
),

-- get all tasks, filtered with criteria
-- 1 row per task
tasks AS (
  SELECT
    tasks.id as task_id,
    tasks.created_at as task_created_at,
    tasks.title as task_title,
    tasks.description as task_content,
    coalesce(users.name, users.email, 'Unknown user') as task_username,
    tasks.workspace_id as task_workspace_id,
    GREATEST(tasks.completed_at, tasks.closed_at) as task_closed_at,
  FROM `${GCP_PROJECT}.cloudsql_decide_public.task` as tasks
  LEFT JOIN users
    ON users.id = tasks.creator_id
  WHERE
    tasks.organization_id = '00000000-0000-0000-0000-000000000000'
    AND tasks._fivetran_deleted = false
    AND tasks.created_at >= '2025-01-01'
    AND tasks.created_at < '2026-01-01'
    AND tasks.deleted_at is null
    AND tasks.sharing = 'organization' -- only want public tasks
),

-- get all task comments, filtered with criteria
-- 1 row per task comment
task_comments AS (
  SELECT
    comments.id as comment_id,
    comments.content as comment_content,
    comments.created_at as comment_created_at,
    coalesce(users.name, users.email, 'Unknown user') as comment_username,
    comments.workspace_id as comment_workspace_id
  FROM `${GCP_PROJECT}.cloudsql_decide_public.comment` as comments
  INNER JOIN tasks
    ON tasks.task_workspace_id = comments.workspace_id
  LEFT JOIN users
    ON users.id = comments.user_id
  WHERE
    comments._fivetran_deleted = false
    AND comments.created_at >= '2025-01-01'
    AND comments.created_at < '2026-01-01'
),

-- join tasks with task comments
-- 1 row per task per comment
joined AS (
  SELECT
    tasks.task_id,
    tasks.task_created_at,
    tasks.task_title,
    tasks.task_content,
    tasks.task_username,
    tasks.task_workspace_id,
    tasks.task_closed_at,

    task_comments.comment_id,
    task_comments.comment_content,
    task_comments.comment_created_at,
    task_comments.comment_username,
    task_comments.comment_workspace_id
  FROM tasks
  LEFT JOIN task_comments
    ON task_comments.comment_workspace_id = tasks.task_workspace_id
),

-- 1 row per task
task_timestamps AS (
  SELECT DISTINCT
    task_id,
    MIN(task_created_at) as task_created_at,
    MAX(comment_created_at) as last_comment_at,
    MAX(task_closed_at) as task_closed_at
  FROM joined
  GROUP BY task_id
),

-- 1 row per task or comment
-- there will be duplicates, e.g. for each comment there will be a row for a given task
joined_flattened AS (
  -- tasks
  SELECT
    task_id,
    task_title,
    task_content as content,
    task_created_at as content_created_at,
    task_username as username
  FROM joined

  UNION ALL

  -- task comments
  SELECT
    task_id,
    task_title,
    comment_content as content,
    comment_created_at as content_created_at,
    comment_username as username
  FROM joined
  WHERE comment_content is not null
),

-- 1 row per task or comment
-- no duplication of tasks here
joined_flattened_deduplicated AS (
  SELECT DISTINCT
    task_id,
    task_title,
    content,
    content_created_at,
    username
  FROM joined_flattened
),

-- 1 row per task or comment
timestamps_joined AS (
  SELECT
    jfd.task_id,
    jfd.task_title,
    jfd.content,
    jfd.content_created_at,
    jfd.username,

    task_timestamps.task_created_at,
    task_timestamps.last_comment_at,
    task_timestamps.task_closed_at
  FROM joined_flattened_deduplicated as jfd
  LEFT JOIN task_timestamps
    ON task_timestamps.task_id = jfd.task_id
),

-- 1 row per task
aggregated_content AS (
  SELECT
    task_id,
    task_title,
    TIMESTAMP_TRUNC(task_created_at, SECOND) as task_created_at,
    TIMESTAMP_TRUNC(last_comment_at, SECOND) as last_comment_at,
    TIMESTAMP_TRUNC(task_closed_at, SECOND) as task_closed_at,
    STRING_AGG(
      CONCAT(username, ': ', content),
      '\\n'
      ORDER BY
        CASE WHEN content_created_at = task_created_at THEN 0 ELSE 1 END,
        content_created_at ASC
      LIMIT 1000
    ) as combined_content
  FROM timestamps_joined
  GROUP BY task_id, task_title, task_created_at, last_comment_at, task_closed_at
)

SELECT *
FROM aggregated_content
"""


APP_TASKS_WITH_TIMESTAMPS_QUERY = """
-- might be inefficient, but follows the same pattern as github issues query
WITH

-- get users to join to app objects for attribution
-- 1 row per user
users AS (
  SELECT
    id,
    name,
    email
  FROM `${GCP_PROJECT}.cloudsql_decide_public.user`
),

-- get all tasks, filtered with criteria
-- 1 row per task
tasks AS (
  SELECT
    tasks.id as task_id,
    tasks.created_at as task_created_at,
    tasks.title as task_title,
    tasks.description as task_content,
    coalesce(users.name, users.email, 'Unknown user') as task_username,
    tasks.workspace_id as task_workspace_id,
    GREATEST(tasks.completed_at, tasks.closed_at) as task_closed_at,
  FROM `${GCP_PROJECT}.cloudsql_decide_public.task` as tasks
  LEFT JOIN users
    ON users.id = tasks.creator_id
  WHERE
    tasks.organization_id = '00000000-0000-0000-0000-000000000000'
    AND tasks._fivetran_deleted = false
    AND tasks.created_at >= '2025-01-01'
    AND tasks.created_at < '2026-01-01'
    AND tasks.deleted_at is null
    AND tasks.sharing = 'organization' -- only want public tasks
),

-- get all task comments, filtered with criteria
-- 1 row per task comment
task_comments AS (
  SELECT
    comments.id as comment_id,
    comments.content as comment_content,
    comments.created_at as comment_created_at,
    coalesce(users.name, users.email, 'Unknown user') as comment_username,
    comments.workspace_id as comment_workspace_id
  FROM `${GCP_PROJECT}.cloudsql_decide_public.comment` as comments
  INNER JOIN tasks
    ON tasks.task_workspace_id = comments.workspace_id
  LEFT JOIN users
    ON users.id = comments.user_id
  WHERE
    comments._fivetran_deleted = false
    AND comments.created_at >= '2025-01-01'
    AND comments.created_at < '2026-01-01'
),

-- join tasks with task comments
-- 1 row per task per comment
joined AS (
  SELECT
    tasks.task_id,
    tasks.task_created_at,
    tasks.task_title,
    tasks.task_content,
    tasks.task_username,
    tasks.task_workspace_id,
    tasks.task_closed_at,

    task_comments.comment_id,
    task_comments.comment_content,
    task_comments.comment_created_at,
    task_comments.comment_username,
    task_comments.comment_workspace_id
  FROM tasks
  LEFT JOIN task_comments
    ON task_comments.comment_workspace_id = tasks.task_workspace_id
),

-- 1 row per task
task_timestamps AS (
  SELECT DISTINCT
    task_id,
    MIN(task_created_at) as task_created_at,
    MAX(comment_created_at) as last_comment_at,
    MAX(task_closed_at) as task_closed_at
  FROM joined
  GROUP BY task_id
),

-- 1 row per task or comment
-- there will be duplicates, e.g. for each comment there will be a row for a given task
joined_flattened AS (
  -- tasks
  SELECT
    task_id,
    task_title,
    task_content as content,
    task_created_at as content_created_at,
    task_username as username
  FROM joined

  UNION ALL

  -- task comments
  SELECT
    task_id,
    task_title,
    comment_content as content,
    comment_created_at as content_created_at,
    comment_username as username
  FROM joined
  WHERE comment_content is not null
),

-- 1 row per task or comment
-- no duplication of tasks here
joined_flattened_deduplicated AS (
  SELECT DISTINCT
    task_id,
    task_title,
    content,
    content_created_at,
    username
  FROM joined_flattened
),

-- 1 row per task or comment
timestamps_joined AS (
  SELECT
    jfd.task_id,
    jfd.task_title,
    jfd.content,
    jfd.content_created_at,
    jfd.username,

    task_timestamps.task_created_at,
    task_timestamps.last_comment_at,
    task_timestamps.task_closed_at
  FROM joined_flattened_deduplicated as jfd
  LEFT JOIN task_timestamps
    ON task_timestamps.task_id = jfd.task_id
),

-- 1 row per task
-- HERE IS WHERE THE DIFFERENCE WITH THE OTHER TASK QUERY IS - LOOK AT THE STRING_AGG
aggregated_content AS (
  SELECT
    task_id,
    task_title,
    TIMESTAMP_TRUNC(task_created_at, SECOND) as task_created_at,
    TIMESTAMP_TRUNC(last_comment_at, SECOND) as last_comment_at,
    TIMESTAMP_TRUNC(task_closed_at, SECOND) as task_closed_at,
    STRING_AGG(
      CONCAT('User: ',username, '\\nTimestamp: ', content_created_at, '\\nContent:\\n', content),
      '\\n\\n\\n'
      ORDER BY
        CASE WHEN content_created_at = task_created_at THEN 0 ELSE 1 END,
        content_created_at ASC
      LIMIT 1000
    ) as combined_content
  FROM timestamps_joined
  GROUP BY task_id, task_title, task_created_at, last_comment_at, task_closed_at
)

SELECT *
FROM aggregated_content
"""
