MATCH (p:Person)
WHERE p.name = 'louis_bonaparte'
CALL apoc.path.expand(p, "spouse|parents|children", "Person", 0, 3) yield path
RETURN path
