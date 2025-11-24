import sqlite3
import json
import sys
from pathlib import Path


def create_schema(cursor):

    # 1. Main Models Table
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS models
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       type
                       TEXT
                       NOT
                       NULL,
                       residues
                       TEXT, -- Comma separated, e.g., "S,T"
                       organism
                       TEXT, -- Mapped from meta['tree']
                       kinase
                       TEXT, -- Mapped from meta['kinase']
                       method
                       TEXT, -- Mapped from meta['method']
                       classifier
                       TEXT, -- Mapped from meta['classifier']
                       prior
                       REAL,
                       divisor
                       REAL,

                       -- Sigmoid Parameters
                       sig_slope
                       REAL,
                       sig_inflection
                       REAL,
                       sig_min
                       REAL,
                       sig_max
                       REAL
                   )
                   """)

    # Indices for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_organism ON models(organism)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kinase ON models(kinase)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON models(type)")

    # 2. Components Table (Stores the heavy weights)
    # Handles both PSSM matrices (single component) and NN ensembles (multiple components)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS model_components
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       model_id
                       TEXT,
                       component_index
                       INTEGER,
                       window_size
                       INTEGER,
                       hidden_units
                       INTEGER, -- Null for PSSM
                       weights
                       TEXT,    -- JSON string of the float array

                       FOREIGN
                       KEY
                   (
                       model_id
                   ) REFERENCES models
                   (
                       id
                   )
                       )
                   """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_model ON model_components(model_id)")


def convert_json_to_db(json_path, db_path):
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'models' not in data:
        print("Error: JSON does not contain 'models' list.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    create_schema(cursor)

    models = data['models']
    print(f"Migrating {len(models)} models into {db_path}...")

    # Use a transaction for speed
    try:
        for model in models:
            # --- 1. Prepare Model Metadata ---
            m_id = model.get('id')
            m_type = model.get('type')

            # Join residues list into string "S,T"
            residues = ",".join(model.get('residues', []))

            # Extract Meta
            meta = model.get('meta', {})
            organism = meta.get('tree', 'Unknown')  # 'tree' usually maps to organism in this dataset
            kinase = meta.get('kinase', 'Unknown')
            method = meta.get('method', 'Unknown')
            classifier = meta.get('classifier', 'Unknown')
            prior = meta.get('prior', 0.0)

            # Extract Sigmoid & Divisor
            sig = model.get('sigmoid', {})
            divisor = model.get('divisor', 1.0)

            cursor.execute("""
                INSERT OR REPLACE INTO models 
                (id, type, residues, organism, kinase, method, classifier, prior, divisor, 
                 sig_slope, sig_inflection, sig_min, sig_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                m_id, m_type, residues, organism, kinase, method, classifier, prior, divisor,
                sig.get('slope'), sig.get('inflection'), sig.get('min'), sig.get('max')
            ))

            # --- 2. Prepare Model Components (Weights) ---

            # Case A: Neural Networks (Ensemble of networks)
            if m_type == 'NN' and 'networks' in model:
                for idx, net in enumerate(model['networks']):
                    cursor.execute("""
                                   INSERT INTO model_components
                                       (model_id, component_index, window_size, hidden_units, weights)
                                   VALUES (?, ?, ?, ?, ?)
                                   """, (
                                       m_id,
                                       idx,
                                       net.get('window'),
                                       net.get('hidden'),
                                       json.dumps(net.get('weights'))  # Store array as JSON text
                                   ))

            # Case B: PSSM (Single matrix)
            elif m_type == 'PSSM' or 'weights' in model:
                # PSSMs have weights at the top level
                cursor.execute("""
                               INSERT INTO model_components
                                   (model_id, component_index, window_size, hidden_units, weights)
                               VALUES (?, ?, ?, ?, ?)
                               """, (
                                   m_id,
                                   0,
                                   model.get('window'),
                                   0,  # No hidden units for PSSM
                                   json.dumps(model.get('weights'))
                               ))

        conn.commit()
        print("Success! Database creation complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_sqlite.py <netphorest_atlas.json> [netphorest.db]")
    else:
        atlas_file = sys.argv[1]
        db_file = sys.argv[2] if len(sys.argv) > 2 else "netphorest.db"
        convert_json_to_db(atlas_file, db_file)