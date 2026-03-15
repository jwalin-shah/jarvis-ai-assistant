#!/usr/bin/env python3  # noqa: E501
"""Template mining from iMessage conversations using clustering and semantic similarity.  # noqa: E501
  # noqa: E501
This module discovers common message patterns by:  # noqa: E501
1. Loading messages from the iMessage database  # noqa: E501
2. Generating embeddings using sentence transformers  # noqa: E501
3. Clustering similar messages using DBSCAN  # noqa: E501
4. Extracting representative templates from clusters  # noqa: E501
"""  # noqa: E501
  # noqa: E501
from __future__ import annotations  # noqa: E402  # noqa: E501

# noqa: E501
import argparse  # noqa: E501
import json  # noqa: E501
import logging  # noqa: E501
import sqlite3  # noqa: E501
import sys  # noqa: E501
import time  # noqa: E501
from dataclasses import asdict, dataclass  # noqa: E402  # noqa: E501
from pathlib import Path  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
import numpy as np  # noqa: E501
from sklearn.cluster import DBSCAN  # noqa: E402  # noqa: E501
from tqdm import tqdm  # noqa: E402  # noqa: E501

  # noqa: E501
# Conditional imports for sentence transformers  # noqa: E501
try:  # noqa: E501
    from sentence_transformers import SentenceTransformer  # noqa: E501
  # noqa: E501
    HAS_SENTENCE_TRANSFORMERS = True  # noqa: E501
except ImportError:  # noqa: E501
    HAS_SENTENCE_TRANSFORMERS = False  # noqa: E501
    SentenceTransformer = None  # noqa: E501
  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class Template:  # noqa: E501
    """A discovered message template."""  # noqa: E501
  # noqa: E501
    representative: str  # Most central example from cluster  # noqa: E501
    frequency: int  # Number of messages in cluster  # noqa: E501
    cluster_id: int  # Cluster identifier  # noqa: E501
    examples: list[str]  # Sample messages from cluster  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class MiningStats:  # noqa: E501
    """Statistics from template mining run."""  # noqa: E501
  # noqa: E501
    total_messages: int  # noqa: E501
    messages_sampled: int  # noqa: E501
    templates_extracted: int  # noqa: E501
    coverage: float  # Fraction of sampled messages covered by templates  # noqa: E501
    largest_cluster_size: int  # noqa: E501
    embedding_time_seconds: float  # noqa: E501
    clustering_time_seconds: float  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class MiningResult:  # noqa: E501
    """Result from template mining."""  # noqa: E501
  # noqa: E501
    templates: list[Template]  # noqa: E501
    stats: MiningStats  # noqa: E501
    config: dict[str, Any]  # noqa: E501
  # noqa: E501
  # noqa: E501
class TemplateMiner:  # noqa: E501
    """Discovers common message templates using clustering."""  # noqa: E501
  # noqa: E501
    def __init__(  # noqa: E501
        self,  # noqa: E501
        model_name: str = "sentence-transformers/all-mpnet-base-v2",  # noqa: E501
        eps: float = 0.35,  # noqa: E501
        min_samples: int = 5,  # noqa: E501
        min_frequency: int = 8,  # noqa: E501
        db_path: str | None = None,  # noqa: E501
    ):  # noqa: E501
        """Initialize template miner.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_name: Sentence transformer model for embeddings  # noqa: E501
            eps: DBSCAN epsilon - max distance for neighbors (lower = tighter clusters)  # noqa: E501
            min_samples: DBSCAN min samples to form a core point  # noqa: E501
            min_frequency: Minimum cluster size to extract as template  # noqa: E501
            db_path: Path to iMessage database (uses default if None)  # noqa: E501
        """  # noqa: E501
        if not HAS_SENTENCE_TRANSFORMERS:  # noqa: E501
            raise ImportError(  # noqa: E501
                "sentence-transformers is required for template mining. "  # noqa: E501
                "Install with: pip install sentence-transformers"  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        self.model_name = model_name  # noqa: E501
        self.eps = eps  # noqa: E501
        self.min_samples = min_samples  # noqa: E501
        self.min_frequency = min_frequency  # noqa: E501
        self.db_path = db_path or str(Path.home() / "Library/Messages/chat.db")  # noqa: E501
  # noqa: E501
        logger.info("Initializing template miner with model: %s", model_name)  # noqa: E501
        self.model = SentenceTransformer(model_name)  # noqa: E501
  # noqa: E501
    def load_messages(self, sample_size: int | None = None) -> list[str]:  # noqa: E501
        """Load messages from iMessage database.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            sample_size: Maximum number of messages to load (None = all)  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of message texts  # noqa: E501
        """  # noqa: E501
        logger.info("Loading messages from: %s", self.db_path)  # noqa: E501
  # noqa: E501
        try:  # noqa: E501
            # Open read-only with URI  # noqa: E501
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)  # noqa: E501
            cursor = conn.cursor()  # noqa: E501
  # noqa: E501
            # Query sent messages (is_from_me=1) that are text (not attachments)  # noqa: E501
            query = """  # noqa: E501
                SELECT text  # noqa: E501
                FROM message  # noqa: E501
                WHERE is_from_me = 1  # noqa: E501
                  AND text IS NOT NULL  # noqa: E501
                  AND text != ''  # noqa: E501
                  AND length(text) > 0  # noqa: E501
                ORDER BY RANDOM()  # noqa: E501
            """  # noqa: E501
  # noqa: E501
            if sample_size:  # noqa: E501
                query += f" LIMIT {sample_size}"  # noqa: E501
  # noqa: E501
            cursor.execute(query)  # noqa: E501
            messages = [row[0] for row in cursor.fetchall()]  # noqa: E501
            conn.close()  # noqa: E501
  # noqa: E501
            logger.info("Loaded %d messages", len(messages))  # noqa: E501
            return messages  # noqa: E501
  # noqa: E501
        except sqlite3.Error as e:  # noqa: E501
            logger.error("Failed to load messages: %s", e)  # noqa: E501
            raise  # noqa: E501
  # noqa: E501
    def generate_embeddings(self, messages: list[str]) -> np.ndarray:  # noqa: E501
        """Generate embeddings for messages.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            messages: List of message texts  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Numpy array of embeddings (n_messages, embedding_dim)  # noqa: E501
        """  # noqa: E501
        logger.info("Generating embeddings for %d messages", len(messages))  # noqa: E501
        start = time.time()  # noqa: E501
  # noqa: E501
        # Generate embeddings with progress bar  # noqa: E501
        embeddings = self.model.encode(  # noqa: E501
            messages,  # noqa: E501
            show_progress_bar=True,  # noqa: E501
            batch_size=32,  # noqa: E501
            convert_to_numpy=True,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        elapsed = time.time() - start  # noqa: E501
        logger.info("Generated embeddings in %.1fs", elapsed)  # noqa: E501
  # noqa: E501
        return np.asarray(embeddings)  # noqa: E501
  # noqa: E501
    def cluster_messages(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict[int, list[int]]]:  # noqa: E501
        """Cluster messages using DBSCAN.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            embeddings: Message embeddings  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            Tuple of (cluster_labels, cluster_to_indices)  # noqa: E501
        """  # noqa: E501
        logger.info("Clustering with DBSCAN (eps=%.2f, min_samples=%d)", self.eps, self.min_samples)  # noqa: E501
        start = time.time()  # noqa: E501
  # noqa: E501
        clusterer = DBSCAN(  # noqa: E501
            eps=self.eps,  # noqa: E501
            min_samples=self.min_samples,  # noqa: E501
            metric="cosine",  # noqa: E501
            n_jobs=2,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        labels = clusterer.fit_predict(embeddings)  # noqa: E501
        elapsed = time.time() - start  # noqa: E501
  # noqa: E501
        # Group indices by cluster  # noqa: E501
        cluster_to_indices: dict[int, list[int]] = {}  # noqa: E501
        for idx, label in enumerate(labels):  # noqa: E501
            if label != -1:  # Ignore noise (-1)  # noqa: E501
                if label not in cluster_to_indices:  # noqa: E501
                    cluster_to_indices[label] = []  # noqa: E501
                cluster_to_indices[label].append(idx)  # noqa: E501
  # noqa: E501
        n_clusters = len(cluster_to_indices)  # noqa: E501
        n_noise = np.sum(labels == -1)  # noqa: E501
  # noqa: E501
        logger.info(  # noqa: E501
            "Clustering complete in %.1fs: %d clusters, %d noise points",  # noqa: E501
            elapsed,  # noqa: E501
            n_clusters,  # noqa: E501
            n_noise,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        return labels, cluster_to_indices  # noqa: E501
  # noqa: E501
    def extract_templates(  # noqa: E501
        self,  # noqa: E501
        messages: list[str],  # noqa: E501
        embeddings: np.ndarray,  # noqa: E501
        cluster_to_indices: dict[int, list[int]],  # noqa: E501
    ) -> list[Template]:  # noqa: E501
        """Extract templates from clusters.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            messages: Original message texts  # noqa: E501
            embeddings: Message embeddings  # noqa: E501
            cluster_to_indices: Mapping from cluster ID to message indices  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            List of extracted templates  # noqa: E501
        """  # noqa: E501
        templates: list[Template] = []  # noqa: E501
  # noqa: E501
        for cluster_id, indices in tqdm(  # noqa: E501
            cluster_to_indices.items(),  # noqa: E501
            desc="Extracting templates",  # noqa: E501
        ):  # noqa: E501
            # Filter by minimum frequency  # noqa: E501
            if len(indices) < self.min_frequency:  # noqa: E501
                continue  # noqa: E501
  # noqa: E501
            # Get cluster messages and embeddings  # noqa: E501
            cluster_messages = [messages[i] for i in indices]  # noqa: E501
            cluster_embeddings = embeddings[indices]  # noqa: E501
  # noqa: E501
            # Find most central message (closest to cluster centroid)  # noqa: E501
            centroid = cluster_embeddings.mean(axis=0)  # noqa: E501
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)  # noqa: E501
            most_central_idx = np.argmin(distances)  # noqa: E501
            representative = cluster_messages[most_central_idx]  # noqa: E501
  # noqa: E501
            # Sample examples (up to 5)  # noqa: E501
            example_indices = np.random.choice(  # noqa: E501
                len(cluster_messages),  # noqa: E501
                size=min(5, len(cluster_messages)),  # noqa: E501
                replace=False,  # noqa: E501
            )  # noqa: E501
            examples = [cluster_messages[i] for i in example_indices]  # noqa: E501
  # noqa: E501
            templates.append(  # noqa: E501
                Template(  # noqa: E501
                    representative=representative,  # noqa: E501
                    frequency=len(indices),  # noqa: E501
                    cluster_id=cluster_id,  # noqa: E501
                    examples=examples,  # noqa: E501
                )  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Sort by frequency (descending)  # noqa: E501
        templates.sort(key=lambda t: t.frequency, reverse=True)  # noqa: E501
  # noqa: E501
        logger.info("Extracted %d templates", len(templates))  # noqa: E501
        return templates  # noqa: E501
  # noqa: E501
    def mine(self, sample_size: int = 100000) -> MiningResult:  # noqa: E501
        """Run template mining pipeline.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            sample_size: Number of messages to sample  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            MiningResult with templates and statistics  # noqa: E501
        """  # noqa: E501
        # Load messages  # noqa: E501
        messages = self.load_messages(sample_size)  # noqa: E501
        total_messages = len(messages)  # noqa: E501
  # noqa: E501
        if total_messages == 0:  # noqa: E501
            logger.warning("No messages found")  # noqa: E501
            return MiningResult(  # noqa: E501
                templates=[],  # noqa: E501
                stats=MiningStats(  # noqa: E501
                    total_messages=0,  # noqa: E501
                    messages_sampled=0,  # noqa: E501
                    templates_extracted=0,  # noqa: E501
                    coverage=0.0,  # noqa: E501
                    largest_cluster_size=0,  # noqa: E501
                    embedding_time_seconds=0.0,  # noqa: E501
                    clustering_time_seconds=0.0,  # noqa: E501
                ),  # noqa: E501
                config={  # noqa: E501
                    "model_name": self.model_name,  # noqa: E501
                    "eps": self.eps,  # noqa: E501
                    "min_samples": self.min_samples,  # noqa: E501
                    "min_frequency": self.min_frequency,  # noqa: E501
                    "sample_size": sample_size,  # noqa: E501
                },  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        # Generate embeddings  # noqa: E501
        embed_start = time.time()  # noqa: E501
        embeddings = self.generate_embeddings(messages)  # noqa: E501
        embed_time = time.time() - embed_start  # noqa: E501
  # noqa: E501
        # Cluster messages  # noqa: E501
        cluster_start = time.time()  # noqa: E501
        labels, cluster_to_indices = self.cluster_messages(embeddings)  # noqa: E501
        cluster_time = time.time() - cluster_start  # noqa: E501
  # noqa: E501
        # Extract templates  # noqa: E501
        templates = self.extract_templates(messages, embeddings, cluster_to_indices)  # noqa: E501
  # noqa: E501
        # Calculate statistics  # noqa: E501
        messages_in_clusters = sum(len(indices) for indices in cluster_to_indices.values())  # noqa: E501
        coverage = messages_in_clusters / total_messages if total_messages > 0 else 0.0  # noqa: E501
        largest_cluster = max((len(indices) for indices in cluster_to_indices.values()), default=0)  # noqa: E501
  # noqa: E501
        stats = MiningStats(  # noqa: E501
            total_messages=int(total_messages),  # noqa: E501
            messages_sampled=int(total_messages),  # noqa: E501
            templates_extracted=int(len(templates)),  # noqa: E501
            coverage=float(coverage),  # noqa: E501
            largest_cluster_size=int(largest_cluster),  # noqa: E501
            embedding_time_seconds=float(embed_time),  # noqa: E501
            clustering_time_seconds=float(cluster_time),  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        return MiningResult(  # noqa: E501
            templates=templates,  # noqa: E501
            stats=stats,  # noqa: E501
            config={  # noqa: E501
                "model_name": self.model_name,  # noqa: E501
                "eps": self.eps,  # noqa: E501
                "min_samples": self.min_samples,  # noqa: E501
                "min_frequency": self.min_frequency,  # noqa: E501
                "sample_size": sample_size,  # noqa: E501
            },  # noqa: E501
        )  # noqa: E501
  # noqa: E501
  # noqa: E501
def main() -> int:  # noqa: E501
    """CLI entry point for template mining."""  # noqa: E501
    parser = argparse.ArgumentParser(description="Mine templates from iMessage conversations")  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--sample-size",  # noqa: E501
        type=int,  # noqa: E501
        default=100000,  # noqa: E501
        help="Number of messages to sample (default: 100000)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--model",  # noqa: E501
        type=str,  # noqa: E501
        default="sentence-transformers/all-mpnet-base-v2",  # noqa: E501
        help="Sentence transformer model (default: all-mpnet-base-v2)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--eps",  # noqa: E501
        type=float,  # noqa: E501
        default=0.35,  # noqa: E501
        help="DBSCAN epsilon - max distance for neighbors (default: 0.35)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--min-samples",  # noqa: E501
        type=int,  # noqa: E501
        default=5,  # noqa: E501
        help="DBSCAN min samples to form cluster (default: 5)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--min-frequency",  # noqa: E501
        type=int,  # noqa: E501
        default=8,  # noqa: E501
        help="Minimum cluster size to extract as template (default: 8)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--output",  # noqa: E501
        type=str,  # noqa: E501
        required=True,  # noqa: E501
        help="Output JSON file path",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "--db-path",  # noqa: E501
        type=str,  # noqa: E501
        help="Path to iMessage database (default: ~/Library/Messages/chat.db)",  # noqa: E501
    )  # noqa: E501
    parser.add_argument(  # noqa: E501
        "-v",  # noqa: E501
        "--verbose",  # noqa: E501
        action="store_true",  # noqa: E501
        help="Enable verbose logging",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    args = parser.parse_args()  # noqa: E501
  # noqa: E501
    # Setup logging  # noqa: E501
    logging.basicConfig(  # noqa: E501
        level=logging.DEBUG if args.verbose else logging.INFO,  # noqa: E501
        format="%(asctime)s [%(levelname)s] %(message)s",  # noqa: E501
    )  # noqa: E501
  # noqa: E501
    try:  # noqa: E501
        # Create miner  # noqa: E501
        miner = TemplateMiner(  # noqa: E501
            model_name=args.model,  # noqa: E501
            eps=args.eps,  # noqa: E501
            min_samples=args.min_samples,  # noqa: E501
            min_frequency=args.min_frequency,  # noqa: E501
            db_path=args.db_path,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        # Run mining  # noqa: E501
        result = miner.mine(sample_size=args.sample_size)  # noqa: E501
  # noqa: E501
        # Convert to JSON-serializable format  # noqa: E501
        output_data = {  # noqa: E501
            "templates": [  # noqa: E501
                {  # noqa: E501
                    "representative": t.representative,  # noqa: E501
                    "frequency": int(t.frequency),  # Convert numpy int64 to Python int  # noqa: E501
                    "cluster_id": int(t.cluster_id),  # Convert numpy int64 to Python int  # noqa: E501
                    "examples": t.examples,  # noqa: E501
                }  # noqa: E501
                for t in result.templates  # noqa: E501
            ],  # noqa: E501
            "stats": asdict(result.stats),  # noqa: E501
            "config": result.config,  # noqa: E501
        }  # noqa: E501
  # noqa: E501
        # Write output  # noqa: E501
        output_path = Path(args.output)  # noqa: E501
        output_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: E501
  # noqa: E501
        with open(output_path, "w") as f:  # noqa: E501
            json.dump(output_data, f, indent=2)  # noqa: E501
  # noqa: E501
        logger.info("Results written to: %s", output_path)  # noqa: E501
  # noqa: E501
        # Print summary  # noqa: E501
        print("\n" + "=" * 60)  # noqa: E501
        print("TEMPLATE MINING COMPLETE")  # noqa: E501
        print("=" * 60)  # noqa: E501
        print(f"Messages analyzed:     {result.stats.total_messages:,}")  # noqa: E501
        print(f"Templates extracted:   {result.stats.templates_extracted}")  # noqa: E501
        print(f"Coverage:              {result.stats.coverage * 100:.1f}%")  # noqa: E501
        print(f"Largest cluster:       {result.stats.largest_cluster_size}")  # noqa: E501
        print(f"Embedding time:        {result.stats.embedding_time_seconds:.1f}s")  # noqa: E501
        print(f"Clustering time:       {result.stats.clustering_time_seconds:.1f}s")  # noqa: E501
        print("\nTop 10 templates:")  # noqa: E501
        for i, template in enumerate(result.templates[:10], 1):  # noqa: E501
            print(f"  {i}. [{template.frequency:3d} uses] {template.representative[:70]}")  # noqa: E501
        print("=" * 60)  # noqa: E501
  # noqa: E501
        return 0  # noqa: E501
  # noqa: E501
    except Exception as e:  # noqa: E501
        logger.exception("Template mining failed: %s", e)  # noqa: E501
        return 1  # noqa: E501
  # noqa: E501
  # noqa: E501
if __name__ == "__main__":  # noqa: E501
    sys.exit(main())  # noqa: E501
