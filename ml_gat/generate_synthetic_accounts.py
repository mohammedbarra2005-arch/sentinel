"""
ml_gat/generate_synthetic_accounts.py
---------------------------------------
Génère un jeu de données synthétique PLUS RICHE, au niveau COMPTE INDIVIDUEL
(pas au niveau banque/BIC comme les données originales), avec de VRAIS patterns
de fraude en réseau injectés délibérément.

Pourquoi ce fichier existe :
Le dataset original (data/raw/iso20022_analytics.csv) ne contient que 14 BIC
(14 banques), ce qui est beaucoup trop peu pour qu'un GAT apprenne quoi que ce
soit de significatif (cf. limite documentée dans graph_builder.py). Ce script
simule un scénario plus réaliste : des centaines de COMPTES CLIENTS individuels,
répartis entre les mêmes 13 banques (pour rester cohérent avec les données
existantes), avec :

  1. Des comptes "normaux" qui font des transactions aléatoires entre eux.
  2. Des "anneaux de fraude" (fraud rings) : petits groupes de 3-5 comptes qui se
     transfèrent de l'argent en cercle (A->B->C->A), un pattern classique de
     blanchiment ("layering"). Ce pattern est INVISIBLE pour XGBoost (qui ne voit
     que des transactions isolées) mais DÉTECTABLE par un GAT (qui voit la
     structure du graphe).
  3. Des "comptes collecteurs" (fan-in) : un compte qui reçoit de l'argent de
     nombreux comptes distincts en peu de temps (mules bancaires classiques).

Point important : contrairement au XGBoost (entraîné sur un PROXY de risque, le
statut de rejet), ici on connaît la VRAIE étiquette (on a nous-mêmes injecté les
comptes frauduleux) -> c'est donc un jeu de données idéal pour DÉMONTRER la
capacité du GAT à détecter des patterns structurels, même si ça reste un jeu de
données synthétique et pas des transactions bancaires réelles.
"""
import numpy as np
import pandas as pd
import random
import string
from pathlib import Path
from datetime import datetime, timedelta

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Les mêmes banques que dans le dataset original, pour rester cohérent
BANKS = [
    ("LOYDGB21", "GB"), ("STBGGB2L", "GB"), ("SOGEFRPP", "FR"), ("NWBKGB2L", "GB"),
    ("DEUTDEFF", "DE"), ("HSBCGB2L", "GB"), ("MONZGB2L", "GB"), ("BNPAFRPP", "FR"),
    ("COBADEFF", "DE"), ("BARCGB22", "GB"), ("INGBNL2A", "NL"), ("CHASGB2L", "GB"),
    ("RABONL2U", "NL"),
]

N_ACCOUNTS = 400          # nombre de comptes clients individuels
N_NORMAL_TRANSACTIONS = 6000
N_RINGS = 8               # nombre d'anneaux de fraude (cycles A->B->C->A)
RING_SIZE_RANGE = (3, 5)  # taille de chaque anneau
N_FAN_IN_SCHEMES = 5      # nombre de schémas "compte collecteur"
FAN_IN_SOURCES_RANGE = (6, 12)  # nombre de comptes sources par collecteur


def make_account_id(i: int) -> str:
    return f"ACC-{i:05d}"


def generate_accounts():
    """Crée N_ACCOUNTS comptes individuels, chacun rattaché à une banque (BIC)."""
    accounts = []
    for i in range(N_ACCOUNTS):
        bic, country = random.choice(BANKS)
        accounts.append({"account_id": make_account_id(i), "bic": bic, "country": country})
    return pd.DataFrame(accounts)


def random_timestamp():
    start = datetime(2025, 11, 1)
    return start + timedelta(days=random.randint(0, 90), hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))


def generate_normal_transactions(accounts_df):
    """Transactions normales entre comptes choisis au hasard."""
    ids = accounts_df["account_id"].tolist()
    rows = []
    for _ in range(N_NORMAL_TRANSACTIONS):
        src, dst = random.sample(ids, 2)
        rows.append({
            "source": src, "target": dst,
            "amount": round(np.random.lognormal(mean=6.5, sigma=1.0), 2),
            "timestamp": random_timestamp(),
            "is_fraud_ring": 0,
        })
    return rows


def generate_fraud_rings(accounts_df):
    """Anneaux de fraude : A->B->C->A, répétés plusieurs fois sur quelques jours
    (pattern classique de layering / blanchiment). Marque ces comptes comme frauduleux."""
    ids = accounts_df["account_id"].tolist()
    rows = []
    ring_accounts = set()

    for _ in range(N_RINGS):
        size = random.randint(*RING_SIZE_RANGE)
        ring = random.sample([i for i in ids if i not in ring_accounts], size)
        ring_accounts.update(ring)

        base_time = random_timestamp()
        # Répète le cycle plusieurs fois (2-4 tours) pour simuler un vrai schéma de layering
        for repeat in range(random.randint(2, 4)):
            for i in range(len(ring)):
                src = ring[i]
                dst = ring[(i + 1) % len(ring)]
                amount = round(np.random.uniform(8000, 25000), 2)  # montants ronds, suspects
                rows.append({
                    "source": src, "target": dst, "amount": amount,
                    "timestamp": base_time + timedelta(hours=repeat * 6 + i),
                    "is_fraud_ring": 1,
                })
    return rows, ring_accounts


def generate_fan_in_schemes(accounts_df, exclude_ids):
    """Comptes collecteurs (fan-in) : un compte reçoit de nombreux comptes distincts
    en peu de temps -> pattern de mule bancaire."""
    ids = [i for i in accounts_df["account_id"].tolist() if i not in exclude_ids]
    rows = []
    fan_in_accounts = set()

    for _ in range(N_FAN_IN_SCHEMES):
        collector = random.choice(ids)
        n_sources = random.randint(*FAN_IN_SOURCES_RANGE)
        sources = random.sample([i for i in ids if i != collector], n_sources)
        fan_in_accounts.add(collector)
        fan_in_accounts.update(sources)

        base_time = random_timestamp()
        for i, src in enumerate(sources):
            amount = round(np.random.uniform(500, 3000), 2)  # petits montants, sous les seuils de déclaration
            rows.append({
                "source": src, "target": collector, "amount": amount,
                "timestamp": base_time + timedelta(hours=i * 2),
                "is_fraud_ring": 1,
            })
    return rows, fan_in_accounts


def build_dataset():
    accounts_df = generate_accounts()

    normal_rows = generate_normal_transactions(accounts_df)
    ring_rows, ring_accounts = generate_fraud_rings(accounts_df)
    fanin_rows, fanin_accounts = generate_fan_in_schemes(accounts_df, ring_accounts)

    all_rows = normal_rows + ring_rows + fanin_rows
    tx_df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
    tx_df["message_id"] = [f"MSG-{i:06d}" for i in range(len(tx_df))]

    risky_accounts = ring_accounts | fanin_accounts
    accounts_df["label"] = accounts_df["account_id"].apply(lambda a: 1 if a in risky_accounts else 0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tx_df.to_parquet(PROCESSED_DIR / "gat_accounts_transactions.parquet", index=False)
    accounts_df.to_parquet(PROCESSED_DIR / "gat_accounts_nodes.parquet", index=False)

    return accounts_df, tx_df


if __name__ == "__main__":
    accounts_df, tx_df = build_dataset()
    print(f"Comptes générés : {len(accounts_df)}")
    print(f"Comptes à risque (vraie étiquette) : {accounts_df['label'].sum()} "
          f"({accounts_df['label'].mean():.1%})")
    print(f"Transactions générées : {len(tx_df)}")
    print(f"  - normales : {(tx_df['is_fraud_ring'] == 0).sum()}")
    print(f"  - liées à un pattern frauduleux : {(tx_df['is_fraud_ring'] == 1).sum()}")
    print(f"\nSauvegardé -> {PROCESSED_DIR / 'gat_accounts_nodes.parquet'}")
    print(f"Sauvegardé -> {PROCESSED_DIR / 'gat_accounts_transactions.parquet'}")
