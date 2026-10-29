-- t_ready: the edit that put ready-for-agent on ticket N (the first one); t_close: its ticket.closed
WITH bounds AS (
  SELECT
    (SELECT min(at) FROM events WHERE kind = 'ticket.edited' AND repo = :repo AND ticket = :n
       AND field = 'labels' AND after LIKE '%ready-for-agent%' AND before NOT LIKE '%ready-for-agent%') AS t_ready,
    (SELECT max(at) FROM events WHERE kind = 'ticket.closed' AND repo = :repo AND ticket = :n) AS t_close
),
-- a turn spent on N alone, or on N and other tickets held by the same session at that moment
spend AS (
  SELECT role,
         CASE WHEN ticket = :n THEN usd ELSE usd / json_array_length(tickets_held) END AS usd,
         CASE WHEN ticket = :n THEN 'direct' ELSE 'shared' END AS how
  FROM events, bounds
  WHERE kind = 'turn.used' AND repo = :repo AND at BETWEEN t_ready AND t_close
    AND (ticket = :n OR (ticket IS NULL AND :n IN (SELECT value FROM json_each(tickets_held))))
)
SELECT role, how, round(sum(usd), 2) AS usd FROM spend GROUP BY role, how ORDER BY usd DESC;
