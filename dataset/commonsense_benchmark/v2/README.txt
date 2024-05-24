This version differs from v1 in the following ways:
- Added confidence level per explanatory path (manually defined per path type). 1 = explanatory path with high uncertainty, 2 = explanatory path with low uncertainty, 3 = deterministic explanatory path
- Removed 'deterministic' key as this is subsumed by the above confidence level
- Added sibling-based paths to the parents triples. (Note that this sometimes adds many explanatory triples if someone has many siblings!)
- Resampled analysis and tuning set to ensure that each triple has at least one explanatory path, and also ensure that some of the location-based triples have at least 3 explanatory paths.
