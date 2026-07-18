# AGORA (Python bindings)

Agent-based-simulation graph benchmark generator. `pip install agora`, then:

```python
import agora

agora.domains()        # list built-in domains
agora.doctor()         # probe CPU/RAM/GPU/disk

# write a dataset to disk
s = agora.generate("finance", nodes=20000, edges=200000,
                  time_span_days=20, anomaly_rate=0.03, out="./out")

# or get the graph straight into memory as numpy arrays (no disk)
a = agora.generate_arrays("finance", nodes=10000, edges=100000, time_span_days=10)
a["src"], a["dst"], a["t"], a["label_codes"], a["anomaly_id"]
```
