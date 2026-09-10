"""ROS-free acceptance predicates for recorded execution evidence."""
import math


def plan_found_with_active_reference(session, references, feedback, diagnostics):
    revisions = {revision for (reference_session, revision) in references
                 if reference_session == session}
    # Feedback belongs to this Action callback; diagnostics are a shared topic.
    return bool(revisions) and (
        any(f['reason_code'] == 'PLAN_FOUND' and f['active_segment_revision'] in revisions
            for f in feedback) or
        any(d.get('cycle_result') == 'PLAN_FOUND' and d.get('session_id') == session and
            d.get('segment_revision') in {str(revision) for revision in revisions} and
            d.get('path_state') == 'ACTIVE' for d in diagnostics))


def matching_stopped_completion(item, session, revision, references, completed_state):
    key = (item['session_id'], item['segment_revision'])
    linear, angular = item['linear_speed_mps'], item['angular_speed_radps']
    return (item['state'] == completed_state and item['session_id'] == session and
            item['segment_revision'] == revision and key in references and
            references[key]['reaches_final_goal'] and
            math.isfinite(linear) and math.isfinite(angular) and
            abs(linear) <= .01 and abs(angular) <= .03)
