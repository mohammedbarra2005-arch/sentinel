"""
Schémas de réponse de l'API. Les noms de champs sont en camelCase pour
matcher directement les types TypeScript du frontend React (src/api/types.ts) —
aucune transformation supplémentaire n'est nécessaire côté client.
"""
from typing import Literal, Optional
from pydantic import BaseModel


class TransactionOut(BaseModel):
    messageId: str
    debtorBic: str
    creditorBic: str
    debtorCountry: str
    creditorCountry: str
    currency: str
    purposeCode: str
    amount: float
    status: str  # 'ACSP' (accepté) ou 'RJCT' (rejeté) — codes réels du projet
    riskScore: float
    processingTimeSecs: int
    timestamp: str


class AccountOut(BaseModel):
    accountId: str
    label: Literal["normal", "ring", "fan_in"]
    riskScore: float
    totalSent: float
    totalReceived: float
    txCountSent: int
    txCountReceived: int


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    amount: float


class GraphClusterOut(BaseModel):
    id: str
    type: Literal["ring", "fan_in"]
    accountIds: list[str]
    collectorId: Optional[str] = None


class DashboardStatsOut(BaseModel):
    totalTransactions: int
    rejectRate: float
    highRiskAccounts: int
    activeRings: int
    avgRiskScore: float


class ShapFeatureOut(BaseModel):
    feature: str
    importance: float


class XgboostMetricsOut(BaseModel):
    rocAuc: float
    precision: float
    recall: float
    shapFeatures: list[ShapFeatureOut]


class GatMetricsOut(BaseModel):
    f1: float
    precision: float
    recall: float
    logisticBaselineF1: Optional[float] = None


class ModelMetricsOut(BaseModel):
    xgboost: XgboostMetricsOut
    gat: GatMetricsOut


class TransactionPredictionIn(BaseModel):
    amount: float
    currency: str
    purposeCode: str
    debtorCountry: str
    creditorCountry: str
    debtorBic: str
    creditorBic: str
    processingTimeSecs: int = 5
    hour: int = 12
    dayOfWeek: int = 0  # 0 = lundi, ... 6 = dimanche


class PredictionOut(BaseModel):
    score: float
    level: Literal["faible", "moyen", "élevé"]


class AccountPredictionOut(BaseModel):
    accountId: str
    label: Literal[0, 1]
    riskLevel: str
    confidence: float
    riskScore: float  # P(classe "à risque"), utilisable directement comme severity


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str
