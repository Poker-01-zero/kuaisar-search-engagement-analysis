SELECT source, COUNT(DISTINCT user_id) AS users, COUNT(*) AS sessions,
       SUM(has_click) AS clicked_sessions, SUM(result_rows) AS result_rows,
       SUM(clicked_rows) AS clicked_rows,
       1.0 * SUM(has_click) / COUNT(*) AS session_click_rate
FROM sessions GROUP BY source ORDER BY source;
