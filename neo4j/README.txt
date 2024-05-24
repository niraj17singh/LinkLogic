This cypher script builds the FB13 database in Neo4j and infers (and disambiguates) entity types for each node.

To run it, you will need to do a few things:
- install Neo4j Desktop
- install APOC within your Neo4j Desktop
- Find your $NEO4J_HOME directory. By default it is where Neo4j is installed. On a Mac it will be something like:
~/Library/Application Support/Neo4j Desktop/Application/neo4jDatabases/database-a523751e-d7f6-4bc3-b91c-37e1d0a69021/installation-1.2.34/
(There may be multiple database directories underneath neo4jDatabases. If so, try the most recent one.)
- In $NEO$J_HOME/conf/neo4j.conf set the following values:
dbms.security.procedures.unrestricted=apoc.*
dbms.directories.import=import
apoc.import.file.enabled=true
apoc.import.file.use_neo4j_config=false
- Copy the following files from data/fb13 to $NEO4J_HOME/import:
train.txt, valid.txt, test.txt, entity2id.txt
- Start Neo4j Desktop, and add a new graph
- Launch the graph by hitting the "play" button and then start up the Neo4j browser
- Then in the browser run the following command:
CALL apoc.cypher.runFile("/path/to/build_fb13.cypher")
It should only take about a minute to complete.

You will then be able to interact with the data and query it using cypher. You can also run cypher queries from python using the the neo4j package (just pip install neo4j).

