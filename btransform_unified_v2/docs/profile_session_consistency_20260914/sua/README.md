# DANDI SUA coverage boundary

This analysis intentionally does **not** report a cross-session SUA unit-profile trajectory. The available SUA caches use sorted unit columns within sessions, but do not provide a validated cross-session unit-ID or physical-electrode mapping. Matching columns by sort order would create a false identity assumption.

The paired DANDI PMUA analysis is retained because it matches observed pooled channels to canonical hardware electrode indices. A future SUA analysis requires a source-provenanced cross-session identity map, or a separately justified aggregate defined on a physical-electrode coordinate system. It must not repurpose the PMUA result as an SUA result.
