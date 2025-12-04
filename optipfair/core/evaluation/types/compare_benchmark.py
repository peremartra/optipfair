from pydantic import BaseModel


class CompareBenchmark(BaseModel):
    speedup: float
    tps_improvement_percent: float
