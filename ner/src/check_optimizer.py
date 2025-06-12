import spacy
from spacy.util import registry

print("AdamW.v1 in registry:", registry.has("optimizers", "AdamW.v1"))
print("Все доступные оптимизаторы:", sorted(registry.get_registry("optimizers").keys()))
