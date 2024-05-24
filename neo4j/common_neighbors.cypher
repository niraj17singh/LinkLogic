MATCH (n:Person)-[r]-(x)
WITH n.name as name, count(distinct r) as edge_count, count(distinct x) as neighbor_count
WITH name, edge_count, neighbor_count, edge_count - neighbor_count as difference
WHERE difference > 10
WITH collect(name) as names

MATCH (n:Person)-[r]-(x)
WHERE n.name IN names
RETURN n.name as head_name, x.name as tail_name, collect(type(r)) as rel_types, count(distinct r) as rel_count, labels(x) as tail_type ORDER BY rel_count DESC
