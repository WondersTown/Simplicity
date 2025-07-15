from typing import Annotated, Type, TypeAlias

from pydantic import Field, TypeAdapter

from simplicity.engines.pardo.engine import PardoEngine, PardoEngineConfig

EngineConfig: TypeAlias = Annotated[PardoEngineConfig, Field(discriminator="engine")]
Engine: TypeAlias = PardoEngine
EngineConfigAdapter = TypeAdapter(EngineConfig)
def validate_engine_config(raw: dict) -> EngineConfig:
    return EngineConfigAdapter.validate_python(raw)

engines_mapping = {
    PardoEngineConfig: PardoEngine,
}

def get_engine(config_type: Type[EngineConfig]) -> Type[Engine]:
    return engines_mapping[config_type]
