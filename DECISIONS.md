Phase 1:

Actual column names differ from NBA Stats API standard: SHOT_MADE (boolean) not SHOT_MADE_FLAG (int), BASIC_ZONE not SHOT_ZONE_BASIC, ZONE_NAME not SHOT_ZONE_AREA. SEASON_1 exists as a clean integer so no date parsing needed. SHOT_MADE must be cast to integer before MLlib.

Phase 2:

