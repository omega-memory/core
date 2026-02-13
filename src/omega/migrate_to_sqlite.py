"""
OMEGA Migration: JSON graphs + JSONL sidecar → SQLite + sqlite-vec.

Reads the existing graph JSON files (semantic.json, temporal.json, causal.json,
entity.json) and JSONL store, generates embeddings, and inserts everything into
the new SQLite database.

Usage:
    omega migrate-db           # Interactive migration
    omega migrate-db --force   # Overwrite existing database
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from omega import json_compat as json

logger = logging.getLogger("omega.migrate")

OMEGA_DIR = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))
GRAPHS_DIR = OMEGA_DIR / "graphs"
DB_PATH = OMEGA_DIR / "omega.db"


def _read_json(path: Path) -> Dict:
    """Read a JSON file with orjson."""
    return json.loads(path.read_bytes())


def _load_graph_nodes(graph_name: str) -> List[Dict[str, Any]]:
    """Load nodes from a graph JSON file."""
    path = GRAPHS_DIR / f"{graph_name}.json"
    if not path.exists():
        return []
    try:
        data = _read_json(path)
        nodes = data.get("nodes", [])
        logger.info(f"  {graph_name}.json: {len(nodes)} nodes")
        return nodes
    except Exception as e:
        logger.warning(f"  Failed to read {graph_name}.json: {e}")
        return []


def _load_graph_edges(graph_name: str) -> List[Dict[str, Any]]:
    """Load edges from a graph JSON file."""
    path = GRAPHS_DIR / f"{graph_name}.json"
    if not path.exists():
        return []
    try:
        data = _read_json(path)
        edges = data.get("edges", [])
        logger.info(f"  {graph_name}.json: {len(edges)} edges")
        return edges
    except Exception as e:
        logger.warning(f"  Failed to read {graph_name}.json edges: {e}")
        return []


def _load_jsonl_entries(store_path: Path) -> List[Dict[str, Any]]:
    """Load entries from the JSONL sidecar store."""
    if not store_path.exists():
        return []
    entries = []
    errors = 0
    for line in store_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line.encode() if isinstance(line, str) else line)
            entries.append(entry)
        except Exception:
            errors += 1
    if errors:
        logger.warning(f"  {errors} malformed JSONL lines skipped")
    return entries


def _normalize_metadata(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metadata from graph node format."""
    meta = dict(node_data.get("metadata", {}) or {})
    # Unify "type" -> "event_type"
    if "type" in meta and "event_type" not in meta:
        meta["event_type"] = meta["type"]
    return meta


def migrate(force: bool = False, batch_size: int = 50) -> Dict[str, Any]:
    """Run the full migration from JSON graphs to SQLite.

    Returns a report dict with counts.
    """
    report = {
        "nodes_found": 0,
        "nodes_migrated": 0,
        "edges_migrated": 0,
        "duplicates_skipped": 0,
        "jsonl_only": 0,
        "errors": 0,
        "warnings": [],
    }

    # Check for existing database
    if DB_PATH.exists() and not force:
        # Check if it has data
        import sqlite3

        conn = sqlite3.connect(str(DB_PATH))
        try:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count > 0:
                report["warnings"].append(f"Database already exists with {count} memories. Use --force to overwrite.")
                return report
        except Exception:
            pass  # Table doesn't exist yet, safe to proceed
        finally:
            conn.close()

    if DB_PATH.exists() and force:
        # Back up before overwriting
        backup_path = DB_PATH.with_suffix(".db.bak")
        import shutil

        shutil.copy2(DB_PATH, backup_path)
        DB_PATH.unlink()
        logger.info(f"  Backed up existing database to {backup_path}")

    # ---- Phase 1: Load all nodes from JSON graphs ----
    print("\n[1/4] Loading graph data...")

    semantic_nodes = _load_graph_nodes("semantic")
    temporal_nodes = _load_graph_nodes("temporal")
    causal_nodes = _load_graph_nodes("causal")
    entity_data = {}
    entity_path = GRAPHS_DIR / "entity.json"
    if entity_path.exists():
        try:
            entity_data = _read_json(entity_path)
            print("  entity.json: loaded")
        except Exception as e:
            logger.warning(f"  Failed to read entity.json: {e}")

    # Deduplicate nodes across graphs (semantic is the canonical source)
    all_nodes: Dict[str, Dict[str, Any]] = {}
    for node in semantic_nodes:
        nid = node.get("id")
        if nid:
            all_nodes[nid] = node

    # Add any nodes from temporal/causal that aren't in semantic
    for nodes in [temporal_nodes, causal_nodes]:
        for node in nodes:
            nid = node.get("id")
            if nid and nid not in all_nodes:
                all_nodes[nid] = node

    report["nodes_found"] = len(all_nodes)
    print(f"  Total unique nodes: {len(all_nodes)}")

    # ---- Phase 2: Load JSONL entries not in graphs ----
    store_path = OMEGA_DIR / "store.jsonl"
    jsonl_entries = _load_jsonl_entries(store_path)
    print(f"  JSONL entries: {len(jsonl_entries)}")

    # Find entries in JSONL but not in graphs
    graph_content_hashes: Set[str] = set()
    for node in all_nodes.values():
        content = node.get("content", "")
        if content:
            graph_content_hashes.add(hashlib.sha256(content.encode()).hexdigest())

    jsonl_only_entries = []
    for entry in jsonl_entries:
        content = entry.get("content", "")
        if content:
            h = hashlib.sha256(content.encode()).hexdigest()
            if h not in graph_content_hashes:
                jsonl_only_entries.append(entry)
                graph_content_hashes.add(h)  # Prevent JSONL-internal dupes

    if jsonl_only_entries:
        print(f"  JSONL-only entries (not in graphs): {len(jsonl_only_entries)}")
        report["jsonl_only"] = len(jsonl_only_entries)

    # ---- Phase 3: Generate embeddings and insert into SQLite ----
    print(f"\n[2/4] Generating embeddings for {len(all_nodes) + len(jsonl_only_entries)} entries...")

    from omega.sqlite_store import SQLiteStore

    store = SQLiteStore(db_path=DB_PATH)
    try:
        return _migrate_into_store(store, all_nodes, jsonl_only_entries, entity_data, report)
    finally:
        store.close()


def _migrate_into_store(
    store,
    all_nodes,
    jsonl_only_entries,
    entity_data,
    report,
) -> Dict[str, Any]:
    """Insert nodes, edges, and entities into the store. Called by migrate()."""
    # Prepare all items for batch insertion
    items_to_store: List[Dict[str, Any]] = []

    for node_data in all_nodes.values():
        meta = _normalize_metadata(node_data)
        content = node_data.get("content", "")
        if not content.strip():
            continue

        items_to_store.append(
            {
                "node_id": node_data.get("id"),
                "content": content,
                "metadata": meta,
                "event_type": meta.get("event_type", ""),
                "session_id": meta.get("session_id", ""),
                "project": meta.get("project", ""),
                "created_at": node_data.get("created_at"),
                "access_count": node_data.get("access_count", 0),
                "last_accessed": node_data.get("last_accessed"),
                "ttl_seconds": node_data.get("ttl_seconds"),
                "embedding": node_data.get("embedding"),  # Usually None (not in JSON)
            }
        )

    for entry in jsonl_only_entries:
        content = entry.get("content", "")
        if not content.strip():
            continue
        meta = {}
        for key in ["event_type", "session_id", "project", "type"]:
            if key in entry:
                meta[key] = entry[key]
        if "type" in meta and "event_type" not in meta:
            meta["event_type"] = meta.pop("type")

        items_to_store.append(
            {
                "node_id": entry.get("id"),
                "content": content,
                "metadata": meta,
                "event_type": meta.get("event_type", ""),
                "session_id": meta.get("session_id", entry.get("session_id", "")),
                "project": meta.get("project", entry.get("project", "")),
                "created_at": entry.get("timestamp") or entry.get("created_at"),
                "access_count": 0,
                "last_accessed": None,
                "ttl_seconds": None,
                "embedding": None,
            }
        )

    # Batch-generate embeddings
    items_needing_embedding = [i for i, item in enumerate(items_to_store) if item.get("embedding") is None]
    if items_needing_embedding:
        print(f"  Generating embeddings for {len(items_needing_embedding)} nodes...")
        try:
            from omega.graphs import generate_embedding

            done = 0
            for idx in items_needing_embedding:
                content = items_to_store[idx]["content"]
                try:
                    emb = generate_embedding(content)
                    items_to_store[idx]["embedding"] = emb
                except Exception:
                    pass
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(items_needing_embedding)} embeddings generated...")
            print(f"    {done}/{len(items_needing_embedding)} embeddings generated.")
        except ImportError:
            print("  WARNING: Could not import embedding model. Nodes will be stored without embeddings.")
            report["warnings"].append("Embedding model not available; stored without vectors")

    # Insert into SQLite
    print(f"\n[3/4] Inserting {len(items_to_store)} memories into SQLite...")
    migrated = 0
    dupes = 0
    errors = 0

    for i, item in enumerate(items_to_store):
        try:
            # Build metadata for store()
            meta = dict(item.get("metadata", {}))
            if item.get("event_type"):
                meta["event_type"] = item["event_type"]
            if item.get("session_id"):
                meta["session_id"] = item["session_id"]
            if item.get("project"):
                meta["project"] = item["project"]
            if item.get("created_at"):
                meta["_original_created_at"] = item["created_at"]
            if item.get("access_count"):
                meta["access_count"] = item["access_count"]
            if item.get("last_accessed"):
                meta["last_accessed"] = item["last_accessed"]

            node_id = store.store(
                content=item["content"],
                session_id=item.get("session_id"),
                metadata=meta,
                embedding=item.get("embedding"),
                ttl_seconds=item.get("ttl_seconds"),
                skip_inference=True,  # We already generated embeddings
            )

            if node_id:
                migrated += 1
            else:
                dupes += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Error migrating node: {e}")

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(items_to_store)} inserted...")

    report["nodes_migrated"] = migrated
    report["duplicates_skipped"] = dupes
    report["errors"] = errors

    # ---- Phase 4: Migrate causal edges ----
    # Note: _conn direct access is intentional — migration is single-threaded
    # and offline, so the store's _lock is not needed.
    print("\n[4/4] Migrating edges...")
    causal_edges = _load_graph_edges("causal")
    edges_migrated = 0
    for edge_data in causal_edges:
        try:
            source = edge_data.get("source")
            target = edge_data.get("target")
            if source and target:
                store._conn.execute(
                    "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        source,
                        target,
                        edge_data.get("edge_type", "causal"),
                        edge_data.get("weight", 1.0),
                        json.dumps(edge_data.get("metadata", {})),
                    ),
                )
                edges_migrated += 1
        except Exception as e:
            if edges_migrated == 0:
                logger.warning(f"  Edge migration error: {e}")

    store._conn.commit()
    report["edges_migrated"] = edges_migrated

    # Migrate entity index
    entity_index = entity_data.get("entity_index", {})
    if entity_index:
        for entity_id, node_ids in entity_index.items():
            for nid in node_ids:
                try:
                    store._conn.execute(
                        "INSERT OR IGNORE INTO entity_index (entity_id, node_id) VALUES (?, ?)",
                        (entity_id, nid),
                    )
                except Exception:
                    pass
        store._conn.commit()
        print(f"  Entity index: {len(entity_index)} entities migrated")

    # ---- Rename old files ----
    store_path = OMEGA_DIR / "store.jsonl"
    backed_up = []
    for name in ["semantic.json", "temporal.json", "causal.json", "entity.json"]:
        old = GRAPHS_DIR / name
        if old.exists():
            bak = old.with_suffix(".json.bak")
            old.rename(bak)
            backed_up.append(name)

    # Also back up the JSONL sidecar
    if store_path.exists():
        store_path.rename(store_path.with_suffix(".jsonl.pre-sqlite"))

    # Back up auxiliary index files
    for name in ["stats.json", "type_index.json", "session_index.json", "project_index.json", "feedback_index.json"]:
        old = GRAPHS_DIR / name
        if old.exists():
            old.rename(old.with_suffix(".json.bak"))

    # Clean up lock files and WAL
    for lock in GRAPHS_DIR.glob("*.lock"):
        lock.unlink()
    wal_path = GRAPHS_DIR / "wal.jsonl"
    if wal_path.exists():
        wal_path.rename(wal_path.with_suffix(".jsonl.bak"))

    if backed_up:
        print(f"  Backed up: {', '.join(backed_up)}")

    print(f"\n{'=' * 50}")
    print("Migration complete!")
    print(f"  Migrated:   {migrated} memories")
    print(f"  Duplicates: {dupes} skipped")
    print(f"  Edges:      {edges_migrated}")
    print(f"  Errors:     {errors}")
    print(f"  Database:   {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")
    if backed_up:
        print(f"  Old files renamed to .bak in {GRAPHS_DIR}")

    return report


def auto_migrate_if_needed() -> bool:
    """Auto-migrate on first run if JSON graphs exist but SQLite doesn't.

    Called by bridge._get_store() to handle transparent migration.
    Returns True if migration was performed.
    """
    # Already have a database with data — no migration needed
    if DB_PATH.exists():
        import sqlite3

        conn = sqlite3.connect(str(DB_PATH))
        try:
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count > 0:
                return False
        except Exception:
            pass  # Table doesn't exist yet, proceed with migration
        finally:
            conn.close()

    # Check if there are JSON graphs to migrate
    has_graphs = any((GRAPHS_DIR / f"{name}.json").exists() for name in ["semantic", "temporal", "causal"])
    has_jsonl = (OMEGA_DIR / "store.jsonl").exists()

    if not has_graphs and not has_jsonl:
        return False  # Fresh install, no migration needed

    print("OMEGA: Auto-migrating to SQLite backend...")
    try:
        report = migrate(force=False)
        if report.get("nodes_migrated", 0) > 0:
            print(f"OMEGA: Migrated {report['nodes_migrated']} memories to SQLite.")
            return True
        return False
    except Exception as e:
        logger.error(f"Auto-migration failed: {e}")
        print(f"OMEGA: Auto-migration failed: {e}")
        print("  Run 'omega migrate-db --force' to retry manually.")
        return False
