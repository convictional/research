DROP_CURRENT_GDS_GRAPH = """
    //Drop projection
    CALL gds.graph.drop('{projection_name}');
    """

PROJECT_CURRENT_GRAPH_FOR_GDS = """
    CALL gds.graph.project(
        'testGraph',
        '*',  // Wildcard to include all node labels
        '*'   // Wildcard to include all relationship types
    );
    """

ALL_PATHS_FROM_SOURCE_NODE_TO_LABELLED_NODES_N_OR_LESS_HOPS = """
    // Step 1: Filter nodes first
    MATCH (sourceNode)
    WITH sourceNode
    MATCH (targetNode)
    WHERE ("{target_node_category}" in labels(targetNode) OR "{target_node_category}" in labels(sourceNode))
    AND (sourceNode.name = "{start_node_name}" OR targetNode.name = "{start_node_name}")
    WITH sourceNode, targetNode

    // Step 2: Match paths between filtered nodes
    // Adjust rels*..X to the desired path length; X is the maximum path length
    MATCH path = (sourceNode)-[rels*..{N}]-(targetNode)
    WHERE NONE(node IN nodes(path) WHERE "Content" IN labels(node) OR "ContentChunk" IN labels(node))
    WITH sourceNode, targetNode, path,
        [node in nodes(path) | node.name] as nodeNames,
        [rel in relationships(path) | type(rel)] as edgeTypes,
        [rel in relationships(path) | CASE WHEN startNode(rel) = sourceNode THEN "->" ELSE "<-" END] as edgeDirections
    WITH sourceNode, targetNode, nodeNames, edgeTypes, edgeDirections,
        [i in range(0, size(nodeNames) - 2) | nodeNames[i] + "-[" + edgeTypes[i] + edgeDirections[i] + "]-" + nodeNames[i + 1]] as pathSegments
    RETURN
        apoc.text.join(pathSegments, "") as pathWithDirections,
        labels(sourceNode) as sourceNodeLabels,
        labels(targetNode) as targetNodeLabels,
        sourceNode.description as sourceNodeDescription,
        targetNode.description as targetNodeDescription
    LIMIT {top_k};
    """

ALL_SHORTEST_PATHS_BETWEEN_NODES = """
    MATCH (n) where n.name IN {node_list}
    WITH collect(n) as nodes
    UNWIND nodes as n
    UNWIND nodes as m
    WITH * WHERE ID(n) < ID(m)
    MATCH path = allShortestPaths( (n)-[*..4]-(m) )
    RETURN path
    LIMIT 50;
    """

GRAPH_PATH_STATS = """
    CALL gds.allShortestPaths.stream('testGraph')
    YIELD sourceNodeId, targetNodeId, distance
    RETURN max(distance) AS diameter,
        min(distance) AS minPathDistance,
        avg(distance) AS avgPathDistance,
        count(distance) AS totalShortestPaths;
    """

CALC_PAGERANK = """
    CALL gds.pageRank.write('testGraph', {
        maxIterations: 20,
        dampingFactor: 0.85,
        writeProperty: 'pagerank'
    });
    """

QUERY_NODES_ON_PAGERANK = """
    MATCH (n)
    RETURN n
    ORDER BY n.pagerank DESC
    LIMIT 20;
    """

QUERY_NODES_ON_PAGERANK_WITH_CAT = """
    MATCH (n)
    WHERE n.category = '{category}'
    RETURN n
    ORDER BY n.pagerank DESC
    LIMIT 20;
    """

ONE_HOP_NEIGHBOURS = """
    MATCH (n)-[r]->(m)
    WHERE (n.name = '{node_name}' OR m.name = '{node_name}')
    RETURN
        n.name as source_node,
        m.name as target_node,
        type(r) as edge_type
    """

ENTITY_RESOLUTION_COUNT_ER_RELATIONSHIP_TYPES = """
MATCH ()-[r]->()
WHERE type(r) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH', 'ER_MERGED_NODE_OF']
RETURN DISTINCT type(r) AS RelationshipType, count(*) AS Count
ORDER BY Count desc
"""

ENTITY_RESOLUTION_CREATE_ER_EXACT_MATCHES = """
MATCH (n1), (n2)
WHERE toLower(n1.category) = toLower(n2.category)
    AND toLower(n1.name) = toLower(n2.name)
    AND id(n1) > id(n2)
    AND NOT n1.category IN ['Content', 'ContentChunk']
CREATE (n1)-[r:ER_EXACT_MATCH {name: 'ER_EXACT_MATCH',
                source: n1.name, source_node_id: n1.node_id,
                target: n2.name, target_node_id: n2.node_id}]->(n2)
RETURN count(r) as num_relationships
"""

ENTITY_RESOLUTION_GET_EXACT_MATCH_PAIRS = """
MATCH (n)-[r]->(m)
WHERE type(r) IN ['ER_EXACT_MATCH']
RETURN n.node_id as source_node_id, m.node_id as target_node_id
"""

ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_COSINE_SIMILARITY = """
MATCH (n1)-[r:ER_SIMILAR]->(n2)
WHERE toFloat(apoc.convert.fromJsonMap(r.other_fields).cosine_similarity) > {threshold}
CREATE (n1)-[r2:ER_SIMILAR_MATCH {{name: 'ER_SIMILAR_MATCH',
                source: n1.name, source_node_id: n1.node_id,
                target: n2.name, target_node_id: n2.node_id,
                reason: 'high cosine similarity'}}]->(n2)
RETURN count(r2) as num_relationships
"""

ENTITY_RESOLUTION_GET_SIMILAR_PAIRS_THAT_ARE_NOT_SIMILAR_MATCH = """
MATCH (a)-[r:ER_SIMILAR]->(b)
OPTIONAL MATCH (a)-[s:ER_SIMILAR_MATCH]->(b)
WITH a, b, r
WHERE s IS NULL
RETURN
    a.name as source_name,
    a.description as source_description,
    a.node_id as source_node_id,
    b.name as target_name,
    b.description as target_description,
    b.node_id as target_node_id,
    toFloat(apoc.convert.fromJsonMap(r.other_fields).cosine_similarity) as cosine_similarity
"""

ENTITY_RESOLUTION_CREATE_SIMILAR_MATCHES_BASED_ON_LLM_DECISION = """
MATCH (a {{node_id: '{source_node_id}'}}), (b {{node_id: '{target_node_id}'}})
CREATE (a)-[r:ER_SIMILAR_MATCH {{name: 'ER_SIMILAR_MATCH',
                source: a.name, source_node_id: a.node_id,
                target: b.name, target_node_id: b.node_id,
                reason: "LLM decision: {decision_reason}"}}]->(b)
"""

ENTITY_RESOLUTION_PAIR_IMMEDIATE_NEIGHBOUR_COUNTS = """
MATCH (a {{node_id: '{node1_id}'}}), (b {{node_id: '{node2_id}'}})

OPTIONAL MATCH (a)-[ra]-(na)
WHERE
    NOT type(ra) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH']
    AND id(b) <> id(na)
WITH a, b, count(DISTINCT na) AS source_num_neighbours

OPTIONAL MATCH (b)-[rb]-(nb)
WHERE
    NOT type(rb) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH']
    AND id(a) <> id(nb)
WITH a, b, source_num_neighbours, count(DISTINCT nb) AS target_num_neighbours

OPTIONAL MATCH (a)-[r1]-(n)-[r2]-(b)
WHERE
    NOT type(r1) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH']
    AND NOT type(r2) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH']
    AND id(a) <> id(b)
WITH source_num_neighbours, target_num_neighbours, count(DISTINCT n) AS num_common_neighbours

RETURN source_num_neighbours, target_num_neighbours, num_common_neighbours
"""

ENTITY_RESOLUTION_CREATE_GDS_PROJECTION_QUERY = """
MATCH (n)-[r:ER_EXACT_MATCH|ER_SIMILAR_MATCH]->(m)
RETURN gds.graph.project('{projection_name}', n, m)
"""

ENTITY_RESOLUTION_EXECUTE_WCC_ALGORITHM_QUERY = """
CALL gds.wcc.stream('{projection_name}')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS n, componentId AS golden_id
RETURN golden_id, n.name as name, n.category as category, n.description as description, n.node_id as node_id
ORDER BY golden_id
"""

# This is done in one big (chained) query, with 3 steps:
# 1. Redirect all incoming relationships into duplicates to merged nodes. Delete the old relationships.
# 2. Redirect all outgoing relationships from duplicates to merged nodes. Delete the old relationships.
# 3. Delete all duplicate nodes and associated relationships.
# (4.) Return the counts of everything that has been added and deleted.
ENTITY_RESOLUTION_MERGE_RELATIONSHIPS_TO_MERGED_NODES_AND_DELETE_OLD_ENTITIES = """
// Handle incoming duplicate node relationships
// Search for relationships that point to the duplicate node,
// create new relationships to the merged node,
// and delete the old relationships to the duplicate node
MATCH (other)-[r]->(duplicate)<-[:ER_MERGED_NODE_OF]-(merged)
WHERE NOT type(r) in ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH', 'ER_MERGED_NODE_OF']
WITH other, r, duplicate, merged, type(r) AS relType, properties(r) as props, count(r) as incomingCount
DELETE r
WITH other, merged, relType, props, incomingCount
CALL apoc.merge.relationship(
    other,
    relType,
    {name: props.name},
    {},
    merged,
    {}
) YIELD rel
WITH count(distinct rel) as num_new_incoming_rels, sum(incomingCount) as num_deleted_incoming_relationships

// Handle outgoing duplicate node relationships
// Search for relationships that point to the duplicate node,
// create new relationships to the merged node,
// and delete the old relationships to the duplicate node
MATCH (other)<-[r]-(duplicate)<-[:ER_MERGED_NODE_OF]-(merged)
WHERE NOT type(r) in ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH', 'ER_MERGED_NODE_OF']
WITH other, r, duplicate, merged, type(r) AS relType, properties(r) as props, count(r) as outgoingCount, num_new_incoming_rels, num_deleted_incoming_relationships
DELETE r
WITH other, merged, relType, props, outgoingCount, num_new_incoming_rels, num_deleted_incoming_relationships
CALL apoc.merge.relationship(
    other,
    relType,
    {name: props.name},
    {},
    merged,
    {}
) YIELD rel
WITH count(distinct rel) as num_new_outgoing_rels, sum(outgoingCount) as num_deleted_outgoing_relationships, num_new_incoming_rels, num_deleted_incoming_relationships

// Delete duplicates and their relationships after all related operations are complete
MATCH (duplicate)<-[:ER_MERGED_NODE_OF]-(merged)
WITH duplicate, count(duplicate) as countDuplicate, num_deleted_incoming_relationships, num_new_incoming_rels, num_deleted_outgoing_relationships, num_new_outgoing_rels
DETACH DELETE duplicate

RETURN sum(countDuplicate) as num_deleted_duplicates, num_deleted_incoming_relationships, num_new_incoming_rels, num_deleted_outgoing_relationships, num_new_outgoing_rels
"""

ENTITY_RESOLUTION_DELETE_CUSTOM_RELATIONSHIPS = """
MATCH ()-[r]->()
WHERE type(r) IN ['ER_EXACT_MATCH', 'ER_SIMILAR', 'ER_SIMILAR_MATCH', 'ER_MERGED_NODE_OF']
DELETE r
"""
