LOAD CSV FROM 'file:///entity2id.txt' AS row
FIELDTERMINATOR '\t' CREATE (e:Entity { name: row[0] });

CREATE INDEX ON :Entity(name);

LOAD CSV FROM "file:///test.txt" AS line
FIELDTERMINATOR '\t'
WITH line
MATCH(u:Entity {name: line[0]})
MATCH(v:Entity {name : line[2]})
WITH u, v, line
CALL apoc.create.relationship(u, line[1], {split:'test'}, v)
YIELD rel
RETURN rel;

LOAD CSV FROM "file:///valid.txt" AS line
FIELDTERMINATOR '\t'
WITH line
MATCH(u:Entity {name: line[0]})
MATCH(v:Entity {name : line[2]})
WITH u, v, line
CALL apoc.create.relationship(u, line[1], {split:'valid'}, v)
YIELD rel
RETURN rel;

LOAD CSV FROM "file:///train.txt" AS line
FIELDTERMINATOR '\t'
WITH line
MATCH(u:Entity {name: line[0]})
MATCH(v:Entity {name : line[2]})
WITH u, v, line
CALL apoc.create.relationship(u, line[1], {split:'train'}, v)
YIELD rel
RETURN rel;

MATCH (n1)-->(n2)
WITH DISTINCT n1 AS n1
SET n1 :Person;

MATCH (n1)-[:parents|children|spouse]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Person;

MATCH (p1:Person)-[:spouse]->(p2:Person)
WHERE NOT EXISTS( (p2)-[:spouse]->(p1))
CALL apoc.create.relationship(p2, 'spouse', {split:'None'}, p1)
YIELD rel
RETURN rel

MATCH (p1:Person)-[:children]->(p2:Person)
WHERE NOT EXISTS( (p2)-[:parents]->(p1))
CALL apoc.create.relationship(p2, 'parents', {split:'None'}, p1)
YIELD rel
RETURN rel

MATCH (p1:Person)-[:parents]->(p2:Person)
WHERE NOT EXISTS( (p2)-[:children]->(p1))
CALL apoc.create.relationship(p2, 'children', {split:'None'}, p1)
YIELD rel
RETURN rel

MATCH (n1)-[:cause_of_death]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :CauseOfDeath;

MATCH (n1)-[:ethnicity]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Ethnicity;

MATCH (n1)-[:gender]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Gender;

MATCH (n1)-[:institution]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Institution;

MATCH (n1)-[:place_of_birth|place_of_death|location|nationality]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Location;

MATCH (n1)-[:profession]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Profession;

MATCH (n1)-[:religion]->(n2)
WITH DISTINCT n2 AS n2
SET n2 :Religion;

MATCH (n)
REMOVE n:Entity;


MATCH (n) WHERE n.name = 'african_american' REMOVE n:Location;

MATCH (n) WHERE labels(n) = ['Ethnicity', 'Location'] REMOVE n:Ethnicity;

MATCH (n) WHERE n.name IN ["george_washington_university", "harvard_university","stanford_university", "trinity_college_cambridge", "united_states_naval_academy",                "university_of_wisconsin"]
REMOVE n:Location;

MATCH (n) WHERE labels(n) = ["Institution", "Location"]
REMOVE n:Institution;

MATCH (n) WHERE labels(n) = ["Person", "Location"]
REMOVE n:Location;

MATCH (n) WHERE labels(n) = ["CauseOfDeath", "Location"]
REMOVE n:Location;

MATCH (n) WHERE labels(n) = ["Location", "Profession"]
REMOVE n:Profession;

MATCH (n) WHERE labels(n) = ["Ethnicity", "Religion"]
REMOVE n:Ethnicity;

MATCH (n) WHERE labels(n) = ["CauseOfDeath", "Profession"]
REMOVE n:Profession;

MATCH (n) WHERE labels(n) = ["Person", "Profession"]
REMOVE n:Person;

MATCH (n) WHERE labels(n) = ["CauseOfDeath", "Ethnicity", "Location"]
REMOVE n:CauseOfDeath:Ethnicity;

MATCH (n) WHERE labels(n) = ["Ethnicity", "Location", "Religion"]
REMOVE n:Location:Religion;

