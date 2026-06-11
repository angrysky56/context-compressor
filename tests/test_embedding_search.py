"""Integration test for embedding search functionality.

Tests that the Qwen3-Embedding pooling fix works correctly:
- Compresses two texts with clearly different topics
- Queries for one topic and asserts it ranks higher than the other

Skipped unless:
- torch is available (pytest.importorskip)
- RUN_EMBED_TESTS=1 env var is set
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EMBED_TESTS") != "1",
    reason="Set RUN_EMBED_TESTS=1 to run embedding integration tests",
)


NEO4J_TEXT = """# Neo4j Graph Database

## Cypher Query Language

Neo4j is a graph database management system that uses the Cypher query language.
Cypher is a declarative graph query language that allows for efficient querying
and updating of property graphs. It uses ASCII-art patterns to represent graph
structures. The CREATE clause adds nodes and relationships. MATCH clauses find
patterns in the graph.

## Graph Traversal

Graph traversal is the process of visiting nodes in a graph following relationships.
Depth-first and breadth-first strategies are common. The Neo4j engine optimizes
traversal using index-free adjacency.

## Use Cases

Graph databases excel at social networks, recommendation engines, fraud detection,
and knowledge graphs. They handle highly connected data better than relational
databases.
"""

SOURDOUGH_TEXT = """# Sourdough Bread Baking

## Starter Culture

Sourdough bread uses a natural starter culture of wild yeast and lactobacilli.
The starter must be fed regularly with equal parts flour and water by weight.
A healthy starter doubles in size within 4-6 hours of feeding at room temperature.

## Fermentation

Bulk fermentation typically takes 4-6 hours at 78°F. The dough should roughly
double in size. Stretch and folds every 30 minutes for the first 2 hours build
gluten structure. Proper fermentation develops the characteristic tangy flavor.

## Baking

Bake in a preheated Dutch oven at 450°F for 20 minutes covered, then 20-25
minutes uncovered. The crust should be deep golden brown. Internal temperature
should reach 205-210°F for proper crumb structure.
"""


@pytest.fixture(autouse=True)
def clean_chunk_index():
    """Clear the chunk index before each test."""
    import context_compressor.server as srv

    srv._chunk_index.clear()
    yield
    srv._chunk_index.clear()


@pytest.mark.asyncio
class TestEmbeddingSearch:
    async def test_embedding_search_ranks_correct_topic_first(
        self, tmp_path: Path
    ) -> None:
        """Compress Neo4j and sourdough texts, then query for graph database.

        The Neo4j chunk should rank first with a higher score than the baking chunk.
        This test proves the last-token pooling + instruction-prefix fix works:
        with the old mean-pooling approach, the instruction prefix would dilute
        the query embedding and the results would be poor.
        """
        pytest.importorskip("torch")

        from context_compressor.server import compress_pages, search_chunks

        # Write test files
        neo4j_path = tmp_path / "neo4j.md"
        sourdough_path = tmp_path / "sourdough.md"
        neo4j_path.write_text(NEO4J_TEXT)
        sourdough_path.write_text(SOURDOUGH_TEXT)

        # Compress both files
        result1 = json.loads(await compress_pages(paths=[str(neo4j_path)]))
        assert result1["chunks_created"] == 1, f"Failed to compress Neo4j: {result1}"

        result2 = json.loads(await compress_pages(paths=[str(sourdough_path)]))
        assert (
            result2["chunks_created"] == 1
        ), f"Failed to compress sourdough: {result2}"

        # Search with a graph-database query
        search_result = json.loads(
            await search_chunks(query="graph database cypher", top_k=5)
        )

        assert search_result["returned"] >= 2, (
            f"Expected at least 2 results, got {search_result['returned']}. "
            f"index_status={search_result.get('index_status')}"
        )

        results = search_result["results"]

        # Find the Neo4j and sourdough results by source path
        neo4j_result = next(
            (r for r in results if "neo4j" in r["source_path"].lower()), None
        )
        sourdough_result = next(
            (r for r in results if "sourdough" in r["source_path"].lower()), None
        )

        assert neo4j_result is not None, (
            "Neo4j chunk not found in results: "
            f"{[r['source_path'] for r in results]}"
        )
        assert sourdough_result is not None, (
            "Sourdough chunk not found in results: "
            f"{[r['source_path'] for r in results]}"
        )

        # Neo4j must rank higher
        assert neo4j_result["relevance_score"] > sourdough_result["relevance_score"], (
            f"Neo4j (score={neo4j_result['relevance_score']:.4f}) should rank higher "
            f"than sourdough (score={sourdough_result['relevance_score']:.4f}) "
            f"for query 'graph database cypher'"
        )
