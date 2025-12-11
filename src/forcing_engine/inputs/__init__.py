from forcing_engine.exceptions import FEException
from forcing_engine.inputs import input_hrrr, input_rap

FORCING_INPUT_MODULE_MAP = {
    "HRRR": input_hrrr,
    "RAP": input_rap
}

def find(forcing_type: str):
    module = FORCING_INPUT_MODULE_MAP.get(forcing_type)
    if module is None:
        raise FEException(f"No forcing input module found for forcing type: {forcing_type}")
    return module