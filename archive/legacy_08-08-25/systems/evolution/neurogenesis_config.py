# neurogenesis_config.py
# Configuración de capas para neurogénesis en AEON FENIX-Δ

LAYERS_TO_EXPAND = [
    "encoder.0",  # Primera capa lineal del encoder
]

DEPENDENT_LAYERS = {
    "encoder.0": "decoder.0",  # La expansión de encoder.0 afecta a decoder.0
}
