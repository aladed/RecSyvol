import polars as pl
import gc
import logging
from pathlib import Path
from datetime import timedelta


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

 
RAW_DIR = Path("dataset/full")
PROCESSED_DIR = Path("dataset/processed")
VOCAB_PATH = PROCESSED_DIR / "vocab.parquet"
OUTPUT_DIR = PROCESSED_DIR / "shards"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


NUM_SHARDS = 50 


TEST_DAYS_CUTOFF = 2 

def load_vocab_map():
    """
    Загружаем словарь в память как HashMap (Dict) для мгновенного поиска.
    DataFrame -> Dict { 'MP_item_1': 543, ... }
    """
    logger.info("📖 Loading Vocabulary...")
    vocab_df = pl.read_parquet(VOCAB_PATH)

    return dict(zip(vocab_df["token_str"], vocab_df["token_id"]))

def get_domain_plan(domain_folder: Path, domain_prefix: str, vocab_map: dict):
    """
    Создает ленивый план чтения для конкретного домена.
    Превращает item_id -> token_id.
    """

    file_paths = list((domain_folder / "events").glob("*.pq"))
    if not file_paths:
        return None


    q = pl.scan_parquet(file_paths)


    entity_col = "brand_id" if "reviews" in str(domain_folder) else "item_id"
    

    q = q.select([
        pl.col("user_id"),
        pl.col("timestamp"),
        pl.col(entity_col).cast(pl.Utf8) 
    ])


    q = q.with_columns(
        (pl.lit(domain_prefix) + pl.col(entity_col)).alias("token_key")
    )

    q = q.select([
        pl.col("user_id"),
        pl.col("timestamp"),
        pl.col("token_key").replace(vocab_map, default=4).cast(pl.UInt32).alias("token_id") 

    ])

    return q

def process_shards():
    """
    Главный цикл обработки.
    """
    
    vocab_map = load_vocab_map()
    

    domains = {
        RAW_DIR / "marketplace": "MP_",
        RAW_DIR / "retail": "RT_",
        RAW_DIR / "offers": "OF_",
        RAW_DIR / "reviews": "BR_" 
    }

    for shard_id in range(NUM_SHARDS):
        logger.info(f"🔨 Processing Shard {shard_id + 1}/{NUM_SHARDS}...")

        plans = []
        for domain_path, prefix in domains.items():
            lazy_df = get_domain_plan(domain_path, prefix, vocab_map)
            
            if lazy_df is not None:
                sharded_df = lazy_df.filter(
                    (pl.col("user_id").hash() % NUM_SHARDS) == shard_id
                )
                plans.append(sharded_df)

        if not plans:
            continue

        combined_lazy = pl.concat(plans)

        df_shard = combined_lazy.collect()

        if df_shard.height == 0:
            continue

        df_shard = df_shard.sort(["user_id", "timestamp"])

        sequences = df_shard.group_by("user_id").agg([
            pl.col("token_id").alias("sequence"),
            pl.col("timestamp").alias("timestamps") 
        ])

        max_time = df_shard["timestamp"].max()
        cutoff_time = max_time - timedelta(days=TEST_DAYS_CUTOFF)

        
        output_file = OUTPUT_DIR / f"shard_{shard_id}.parquet"
        sequences.write_parquet(output_file)
        
        logger.info(f"✅ Saved shard {shard_id} to {output_file} (Users: {sequences.height})")

        del df_shard
        del sequences
        gc.collect()

if __name__ == "__main__":
    process_shards()