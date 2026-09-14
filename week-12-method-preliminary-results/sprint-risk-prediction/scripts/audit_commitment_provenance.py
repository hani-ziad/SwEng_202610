"""Create a commitment-level provenance audit for reconstructed sprint scope.

The main reconstruction uses Change_Log when available and falls back to the
static Issue fields only when no relevant field history exists. This audit
records, for each reconstructed committed issue-sprint pair, whether Sprint
membership and Story Points came from observed history or static fallback.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pandas as pd

from src.data.load_tawos import load_sprints, load_issues, load_change_log
from src.data.planning_snapshot import _field_kind, _project_maps, _parse_change_sprint_set, _story_after, _story_before
from src.labels import add_labels
from src.features import add_features, build_model_frame, ALL_FEATURES, PLANNING_TIME_FEATURES
from scripts.run_baseline import time_ordered_split
from scripts.run_validation_sensitivity_analyses import fit_eval

ROOT=Path(__file__).resolve().parents[1]
RAW_DIR=ROOT/'data'/'raw'/'tawos'
RESULTS_DIR=ROOT/'results'


def build_commitments():
    sprints=load_sprints(RAW_DIR, modeling_only=True).sort_values(['project','start_date','sprint_id']).copy()
    issues=load_issues(RAW_DIR).copy()
    change=load_change_log(RAW_DIR, required=True).copy()
    jira_maps, raw_maps, name_maps = _project_maps(sprints)
    sprint_rows_by_project={p:list(g.sort_values('start_date', ascending=False).itertuples(index=False)) for p,g in sprints.groupby('project')}
    rel=change[change['field'].map(_field_kind).notna()].dropna(subset=['raw_issue_id','change_date']).copy()
    sort_cols=['raw_issue_id','change_date'] + (['change_id'] if 'change_id' in rel.columns else [])
    rel=rel.sort_values(sort_cols, ascending=[True,False]+([False] if 'change_id' in rel.columns else []))
    changes_by_issue=defaultdict(list)
    for row in rel.itertuples(index=False):
        changes_by_issue[row.raw_issue_id].append(row)
    rows=[]
    for issue in issues.itertuples(index=False):
        project=issue.project
        if project not in sprint_rows_by_project: continue
        events=changes_by_issue.get(issue.raw_issue_id, [])
        jira_map= jira_maps[project]; raw_map=raw_maps[project]; name_map=name_maps[project]
        sprint_events=[e for e in events if _field_kind(e.field)=='sprint']
        story_events=[e for e in events if _field_kind(e.field)=='story_point']
        sprint_source='observed_sprint_history' if sprint_events else 'static_sprint_fallback'
        story_source='observed_story_history' if story_events else 'static_story_fallback'
        if sprint_events:
            membership,_=_parse_change_sprint_set(sprint_events[0].to_value, sprint_events[0].to_string, jira_map=jira_map, name_map=name_map)
        else:
            membership=set()
            if not pd.isna(issue.raw_sprint_id):
                key=str(int(float(issue.raw_sprint_id)))
                if key in raw_map: membership={raw_map[key]}
        if story_events:
            story_point=_story_after(story_events[0])
        else:
            story_point=None if pd.isna(issue.story_point) else float(issue.story_point)
        created=issue.creation_date; resolved=issue.resolution_date; idx=0
        for sprint in sprint_rows_by_project[project]:
            t=sprint.start_date
            while idx < len(events) and events[idx].change_date > t:
                ev=events[idx]; kind=_field_kind(ev.field)
                if kind=='sprint':
                    membership,_=_parse_change_sprint_set(ev.from_value, ev.from_string, jira_map=jira_map, name_map=name_map)
                elif kind=='story_point':
                    story_point=_story_before(ev)
                idx += 1
            if sprint.sprint_id not in membership: continue
            if created is not None and not pd.isna(created) and created > t: continue
            if resolved is not None and not pd.isna(resolved) and resolved < t: continue
            rows.append({'project':project,'sprint_id':sprint.sprint_id,'raw_sprint_id':sprint.raw_sprint_id,'raw_issue_id':issue.raw_issue_id,'issue_key':issue.issue_key,'sprint_source':sprint_source,'story_point_source':story_source,'story_point_at_start':story_point if story_point is not None else 0.0,'completed_by_end': bool(resolved is not None and not pd.isna(resolved) and resolved <= sprint.end_date)})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    commitments=build_commitments()
    commitments.to_csv(RESULTS_DIR/'commitment_provenance.csv', index=False)
    summary=commitments.groupby('project').agg(
        commitments=('raw_issue_id','count'),
        commitments_with_observed_sprint_history=('sprint_source',lambda s:int((s=='observed_sprint_history').sum())),
        commitments_with_static_sprint_fallback=('sprint_source',lambda s:int((s=='static_sprint_fallback').sum())),
        commitments_with_observed_story_history=('story_point_source',lambda s:int((s=='observed_story_history').sum())),
        commitments_with_static_story_fallback=('story_point_source',lambda s:int((s=='static_story_fallback').sum())),
    ).reset_index()
    for c in ['commitments_with_observed_sprint_history','commitments_with_static_sprint_fallback','commitments_with_observed_story_history','commitments_with_static_story_fallback']:
        summary[c+'_rate']=summary[c]/summary['commitments']
    summary.to_csv(RESULTS_DIR/'commitment_provenance_by_project.csv', index=False)
    # High-confidence sprint subset: nonempty sprints with all committed issues sourced from observed sprint history.
    by_sprint=commitments.groupby(['project','sprint_id']).agg(
        commitments=('raw_issue_id','count'),
        static_sprint_fallback_commitments=('sprint_source',lambda s:int((s=='static_sprint_fallback').sum())),
        static_story_fallback_commitments=('story_point_source',lambda s:int((s=='static_story_fallback').sum())),
    ).reset_index()
    high_sprints=set(by_sprint[by_sprint.static_sprint_fallback_commitments==0].sprint_id)
    from src.data.load_tawos import load_reconstructed_sprints
    sprints=load_reconstructed_sprints(); labeled=add_labels(sprints); featured=add_features(labeled); frame=build_model_frame(featured,'label_spillover')
    subset=frame[(frame.sprint_id.isin(high_sprints)) & (frame.committed_issue_count>0)].copy()
    rows=[]
    for rule, sub in [('all_committed_items_have_observed_sprint_history', subset), ('all_committed_items_have_observed_sprint_and_story_history', frame[frame.sprint_id.isin(set(by_sprint[(by_sprint.static_sprint_fallback_commitments==0)&(by_sprint.static_story_fallback_commitments==0)].sprint_id)) & (frame.committed_issue_count>0)].copy())]:
        if len(sub)>30 and sub.project.nunique()>1:
            train,test=time_ordered_split(sub)
            for model, feats in [('scope_logreg', PLANNING_TIME_FEATURES),('random_forest', ALL_FEATURES)]:
                _,_,_,m=fit_eval(train,test,'label_spillover',feats,model)
                rows.append({'subset_rule':rule,'modeled_sprints':len(sub),'test_sprints':len(test),'test_positives':int(test.label_spillover.sum()),'model':model,**m})
    pd.DataFrame(rows).to_csv(RESULTS_DIR/'commitment_provenance_sensitivity.csv', index=False)
    print('commitments', commitments.shape)
    print(summary.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
