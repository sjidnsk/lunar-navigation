uint8 ACTIVE=0
uint8 INVALIDATED=1

unique_identifier_msgs/UUID session_id
uint64 segment_revision
uint64 traversability_revision
uint8 state
bool reaches_final_goal
nav_msgs/Path path
