"""Independent check of every number in the CCECE paper against the frozen
artifact's result files. Prints PASS/FAIL per claim."""
import json, sys
import pandas as pd
import numpy as np

import os
from pathlib import Path
# Repo root (the folder holding week-12-..., phase-2-..., phase-3-..., external-validation).
# Override with the SPRINT_RISK_ROOT environment variable if the paper folder is moved.
R = os.environ.get("SPRINT_RISK_ROOT", str(Path(__file__).resolve().parents[1]))
P1 = f"{R}/week-12-method-preliminary-results/sprint-risk-prediction/results"
P2 = f"{R}/phase-2-explainable-sprint-advisor/results"
P3 = f"{R}/phase-3-in-sprint-early-warning/results"

fails = []
def check(name, claimed, actual, tol=0.0006):
    ok = abs(float(claimed) - float(actual)) <= tol
    print(("PASS " if ok else "FAIL ") + f"{name}: paper={claimed} artifact={actual:.4f}")
    if not ok: fails.append(name)

# ---------- cohort ----------
flow = pd.read_csv(f"{P1}/project_flow_table.csv")
check("raw sprint rows 2657", 2657, flow.raw_sprints.sum(), 0)
check("future+active 44", 44, flow.future.sum() + flow.active.sum(), 0)
check("invalid 3", 3, flow.invalid_closed_timestamps.sum(), 0)
check("modeled 2602", 2602, flow.modeled.sum(), 0)
check("zero scope 911", 911, flow.zero_scope.sum(), 0)
check("nonempty 1691", 1691, flow.nonempty.sum(), 0)
check("zero-scope share 35.0%", 35.0, 100 * flow.zero_scope.sum() / flow.modeled.sum(), 0.05)
check("cold starts 8", 8, flow.eligible_closed.sum() - flow.modeled.sum(), 0)
check("Titanium Mobile late 48/212", 48, flow.loc[flow.project == "Titanium Mobile Platform", "late_positive"].item(), 0)
check("Confluence Cloud zero scope 267/464", 267, flow.loc[flow.project == "Atlassian Confluence Cloud", "zero_scope"].item(), 0)

prov = pd.read_csv(f"{P1}/commitment_provenance_by_project.csv")
tot = prov.commitments.sum()
check("commitments 20603", 20603, tot, 0)
check("static sprint fallback 7.2%", 7.2, 100 * prov.commitments_with_static_sprint_fallback.sum() / tot, 0.05)
check("static SP fallback 45.1%", 45.1, 100 * prov.commitments_with_static_story_fallback.sum() / tot, 0.05)
check("Confluence Cloud sprint history 54.1%", 54.1, 100 * prov.loc[prov.project == "Atlassian Confluence Cloud", "commitments_with_observed_sprint_history_rate"].item(), 0.05)
check("Confluence Server sprint history 99.1%", 99.1, 100 * prov.loc[prov.project == "Atlassian Confluence Server", "commitments_with_observed_sprint_history_rate"].item(), 0.05)

frame = pd.read_csv(f"{P2}/phase2_modeling_frame.csv", parse_dates=["start_date"])
check("full cohort late 38.7%", 38.7, 100 * frame.label_delay.mean(), 0.05)
check("nonempty spill 93.1%", 93.1, 100 * frame[frame.nonempty].label_spillover.mean(), 0.05)
check("median days late 0.21", 0.21, frame.late_days.median(), 0.005)
check("p90 days late 10", 10, frame.late_days.quantile(0.9), 0.1)
check("max days late 2603", 2603, frame.late_days.max(), 0.5)
check("close early 34.5%", 34.5, 100 * (frame.late_days < 0).mean(), 0.05)
check("concurrent mean 1.9", 1.9, frame.n_concurrent_sprints.mean(), 0.05)
check("concurrent max 11", 11, frame.n_concurrent_sprints.max(), 0)
# split sizes and drift, using the artifact's split rule
tr_n = te_n = 0; rates = {}
for p, g in frame.groupby("project"):
    g = g.sort_values(["start_date", "sprint_id"]); cut = max(1, int(len(g) * 0.7))
    tr, te = g.iloc[:cut], g.iloc[cut:]
    tr_n += len(tr); te_n += len(te)
    rates[p] = (tr.label_delay.mean(), te.label_delay.mean())
check("train 1816", 1816, tr_n, 0); check("test 786", 786, te_n, 0)
check("Titanium SDK train late 71%", 71, 100 * rates["The Titanium SDK"][0], 0.5)
check("Titanium SDK test late 33%", 33, 100 * rates["The Titanium SDK"][1], 0.5)
check("Apache Mesos train late 38%", 38, 100 * rates["Apache Mesos"][0], 0.5)
check("Apache Mesos test late 9%", 9, 100 * rates["Apache Mesos"][1], 0.5)

cmp = pd.read_csv(f"{P1}/assignment_comparison.csv")
check("test late prevalence 31.3%", 31.3, 100 * cmp.loc[cmp.task == "late_closure", "positive_rate"].iloc[0], 0.05)
check("test spill prevalence 94.3%", 94.3, 100 * cmp.loc[cmp.task == "spillover", "positive_rate"].iloc[0], 0.05)
check("spill train 1180", 1180, cmp.loc[cmp.task == "spillover", "n_train"].iloc[0], 0)
check("spill test 511", 511, cmp.loc[cmp.task == "spillover", "n_test"].iloc[0], 0)

# ---------- Table I (phase 2 strict, time split) ----------
m2 = pd.read_csv(f"{P2}/model_results.csv")
def m2v(task, fs, prot, model, col):
    r = m2[(m2.task == task) & (m2.feature_set == fs) & (m2.protocol == prot) & (m2.model == model)]
    assert len(r) == 1, (task, fs, prot, model); return r[col].item()
T1 = {("late_closure", "always_positive"): (.477, None, None),
      ("late_closure", "scope_only_logreg"): (.336, .530, .244),
      ("late_closure", "project_base_rate"): (.470, .640, .221),
      ("late_closure", "logreg"): (.495, .680, .224),
      ("late_closure", "random_forest"): (.502, .715, .207),
      ("late_closure", "gradient_boosting"): (.553, .751, .176),
      ("spillover", "always_positive"): (.971, None, None),
      ("spillover", "scope_only_logreg"): (.696, .885, .230),
      ("spillover", "logreg"): (.842, .827, .161),
      ("spillover", "random_forest"): (.880, .840, .115),
      ("spillover", "gradient_boosting"): (.974, .812, .049)}
for (t, m), (f1, auc, br) in T1.items():
    check(f"T1 {t} {m} F1", f1, m2v(t, "strict", "time_ordered", m, "f1"))
    if auc is not None: check(f"T1 {t} {m} AUC", auc, m2v(t, "strict", "time_ordered", m, "roc_auc"))
    if br is not None: check(f"T1 {t} {m} Brier", br, m2v(t, "strict", "time_ordered", m, "brier"))

b2 = pd.read_csv(f"{P2}/model_bootstrap.csv")
def b2v(task, fs, prot, model, base):
    r = b2[(b2.task == task) & (b2.feature_set == fs) & (b2.protocol == prot) & (b2.model == model) & (b2.baseline == base)]
    assert len(r) == 1; return r.iloc[0]
r = b2v("late_closure", "strict", "time_ordered", "gradient_boosting", "scope_only_logreg")
check("GB vs scope-only delta +.221", .221, r.delta); check("CI low .098", .098, r.ci_low); check("CI high .369", .369, r.ci_high)
r = b2v("late_closure", "strict", "time_ordered", "gradient_boosting", "project_base_rate")
check("GB vs base rate +.111", .111, r.delta); check("CI .028", .028, r.ci_low); check("CI .258", .258, r.ci_high)
for m in ["logreg", "random_forest"]:
    r = b2v("late_closure", "strict", "time_ordered", m, "project_base_rate"); print(f"   {m} vs base rate excludes zero: {r.ci_excludes_zero}")
r = b2v("spillover", "strict", "time_ordered", "logreg", "scope_only_logreg")
check("spill LR delta -.058", -.058, r.delta); check("CI -.103", -.103, r.ci_low); check("CI -.021", -.021, r.ci_high)
r = b2v("spillover", "strict", "time_ordered", "gradient_boosting", "scope_only_logreg")
check("spill GB delta -.073", -.073, r.delta); check("CI -.117", -.117, r.ci_low); check("CI -.007", -.007, r.ci_high)
r = b2v("spillover", "strict", "time_ordered", "random_forest", "scope_only_logreg"); print(f"   RF vs scope-only includes zero: {not r.ci_excludes_zero}")

perm = pd.read_csv(f"{P1}/permutation_importance.csv")
sp = perm[(perm.label == "label_spillover")].sort_values("auc_drop_mean", ascending=False)
check("perm committed count .220", .220, sp.iloc[0].auc_drop_mean); check("perm next .021", .021, sp.iloc[1].auc_drop_mean)

# legacy numbers (phase 1 baseline / conditional)
base = json.load(open(f"{P1}/baseline_results.json")); cond = json.load(open(f"{P1}/conditional_spillover.json"))
check("legacy late RF .705", .705, base["label_delay"]["time_ordered"]["random_forest"]["roc_auc"])
check("legacy late XGB .695", .695, base["label_delay"]["time_ordered"]["xgboost"]["roc_auc"])
check("legacy spill RF .852", .852, cond["time_ordered"]["random_forest"]["roc_auc"])
check("legacy spill XGB .862", .862, cond["time_ordered"]["xgboost"]["roc_auc"])
check("full-pop spill RF .986", .986, base["label_spillover"]["time_ordered"]["random_forest"]["roc_auc"])

cu = pd.read_csv(f"{P1}/cluster_uncertainty.csv")
r = cu[(cu.label == "label_delay") & (cu.model == "random_forest")].iloc[0]
check("late RF CI .596", .596, r.roc_auc_cluster_ci_low); check("late RF CI .786", .786, r.roc_auc_cluster_ci_high)
r = cu[(cu.label == "label_spillover") & (cu.model == "scope_logreg")].iloc[0]
check("scope LR CI .818", .818, r.roc_auc_cluster_ci_low); check("scope LR CI .922", .922, r.roc_auc_cluster_ci_high)

roll = pd.read_csv(f"{P1}/rolling_temporal_splits.csv")
rl = roll[(roll.label == "label_delay") & (roll.model == "random_forest")].roc_auc
check("rolling late RF min .669", .669, rl.min()); check("rolling late RF max .711", .711, rl.max())
rs = roll[(roll.label == "label_spillover") & (roll.population == "nonempty") & (roll.model == "scope_logreg")].roc_auc
check("rolling scope LR min .865", .865, rs.min()); check("rolling scope LR max .887", .887, rs.max())

# ---------- strict vs legacy ----------
for (prot, m, leg, st) in [("lopo", "logreg", .542, .629), ("lopo", "random_forest", .600, .642),
                           ("time_ordered", "logreg", .636, .680), ("time_ordered", "gradient_boosting", .726, .751)]:
    check(f"legacy {prot} {m} {leg}", leg, m2v("late_closure", "legacy", prot, m, "roc_auc"))
    check(f"strict {prot} {m} {st}", st, m2v("late_closure", "strict", prot, m, "roc_auc"))
sd = [abs(m2v("spillover", "legacy", p, m, "roc_auc") - m2v("spillover", "strict", p, m, "roc_auc")) for p in ["time_ordered", "lopo"] for m in ["logreg", "random_forest", "gradient_boosting"]]
print(f"   max spillover strict-legacy |delta| = {max(sd):.3f} (paper: at most 0.02)");
if max(sd) > 0.0225: fails.append("spill strict-legacy delta")

# ---------- provenance ----------
ps = pd.read_csv(f"{P1}/commitment_provenance_sensitivity.csv")
def psv(rule, m, col): return ps[(ps.subset_rule == rule) & (ps.model == m)][col].item()
a = "all_committed_items_have_observed_sprint_history"; b = "all_committed_items_have_observed_sprint_and_story_history"
check("prov A sprints 1174", 1174, psv(a, "scope_logreg", "modeled_sprints"), 0); check("prov A test 356", 356, psv(a, "scope_logreg", "test_sprints"), 0)
check("prov A scope LR .875", .875, psv(a, "scope_logreg", "roc_auc")); check("prov A RF .850", .850, psv(a, "random_forest", "roc_auc"))
check("prov B sprints 358", 358, psv(b, "scope_logreg", "modeled_sprints"), 0); check("prov B test 110", 110, psv(b, "scope_logreg", "test_sprints"), 0)
check("prov B scope LR .772", .772, psv(b, "scope_logreg", "roc_auc")); check("prov B RF .652", .652, psv(b, "random_forest", "roc_auc"))

ls = json.load(open(f"{P1}/label_sensitivity.json"))
print("   label_sensitivity keys:", list(ls.keys())[:10])

# ---------- LOPO ----------
lopo = pd.read_csv(f"{P1}/lopo_with_denominators.csv")
ld = lopo[(lopo.label == "label_delay") & (lopo.model == "random_forest")].set_index("project").roc_auc
check("LOPO late RF min .531 (Titanium Mobile)", .531, ld["Titanium Mobile Platform"]); check("LOPO late RF max .780 (Appcelerator)", .780, ld["Appcelerator Studio"])
check("LOPO late RF min overall", ld.min(), .531); check("LOPO late RF max overall", ld.max(), .780)
lsp = lopo[(lopo.label == "label_spillover") & (lopo.population == "nonempty") & (lopo.model == "random_forest")].set_index("project").roc_auc
check("LOPO spill RF min .706 (Cloud)", .706, lsp["Atlassian Confluence Cloud"]); check("LOPO spill RF max .950 (Appcelerator)", .950, lsp["Appcelerator Studio"])
check("LOPO spill RF min overall", lsp.min(), .706); check("LOPO spill RF max overall", lsp.max(), .950)
check("LOPO always-late F1 .558", .558, m2v("late_closure", "strict", "lopo", "always_positive", "f1"))
for m in ["logreg", "random_forest", "gradient_boosting"]:
    assert m2v("late_closure", "strict", "lopo", m, "f1") < .558, m

# ---------- Shapley ----------
imp = pd.read_csv(f"{P2}/explanation_group_importance.csv")
def share(model, g): return 100 * imp[(imp.model == model) & (imp.group == g)].share_of_total.item()
for g, v in [("past lateness", 38), ("project", 22), ("sprint length", 13), ("capacity", 12), ("scope", 9)]:
    check(f"shap late {g} {v}", v, share("late_closure", g), 0.5)
check("late past lateness + project = 60", 60, share("late_closure", "past lateness") + share("late_closure", "project"), 0.5)
for g, v in [("scope", 41), ("capacity", 37)]: check(f"shap spill {g} {v}", v, share("spillover", g), 0.5)
for g, v in [("project", 26), ("past lateness", 26), ("sprint length", 22)]: check(f"shap late_days {g} {v}", v, share("late_days", g), 0.5)
check("shap issues_done scope 71", 71, share("issues_done", "scope"), 0.5)

abl = pd.read_csv(f"{P1}/assignment_ablation.csv")
def ab(t, c): return abl[(abl.task == t) & (abl.rf_features == c)].roc_auc.item()
check("ablation spill full .852", .852, ab("spillover", "full")); check("ablation spill no_scope .652", .652, ab("spillover", "no_scope"))
check("ablation late full .705", .705, ab("late_closure", "full")); check("ablation late no_scope .690", .690, ab("late_closure", "no_scope"))

cal = pd.read_csv(f"{P1}/calibration_bins.csv")
check("cal late .7-.8 observed 35%", 35, 100 * cal[(cal.label == "label_delay") & (cal.bin == "(0.7, 0.8]")].observed_rate.item(), 0.5)
check("cal spill .3-.4 observed 90%", 90, 100 * cal[(cal.label == "label_spillover") & (cal.bin == "(0.3, 0.4]")].observed_rate.item(), 0.5)

# ---------- numeric + advisor ----------
check("issues_done scope_x_rate MAE 2.10", 2.10, m2v("issues_done", "strict", "time_ordered", "scope_x_rate", "mae"), 0.005)
check("issues_done scope_x_rate R2 .42", .42, m2v("issues_done", "strict", "time_ordered", "scope_x_rate", "r2"), 0.005)
check("issues_done gb_median MAE 1.96", 1.96, m2v("issues_done", "strict", "time_ordered", "gb_median", "mae"), 0.005)
check("issues_done gb_median R2 .46", .46, m2v("issues_done", "strict", "time_ordered", "gb_median", "r2"), 0.005)
r = b2v("issues_done", "strict", "time_ordered", "gb_median", "scope_x_rate")
check("issues_done delta -.13", -.13, r.delta, 0.005); check("CI -.31", -.31, r.ci_low, 0.005); check("CI 0.00", 0.00, r.ci_high, 0.005)
check("carry train_median .232", .232, m2v("carry_share", "strict", "time_ordered", "train_median", "mae")); check("carry gb_median .201", .201, m2v("carry_share", "strict", "time_ordered", "gb_median", "mae"))
r = b2v("carry_share", "strict", "time_ordered", "gb_median", "train_median"); check("carry CI -.050", -.050, r.ci_low); check("carry CI -.012", -.012, r.ci_high)
check("late_days train_median 2.66", 2.66, m2v("late_days", "strict", "time_ordered", "train_median", "mae"), 0.005)
check("late_days gb_median 2.48", 2.48, m2v("late_days", "strict", "time_ordered", "gb_median", "mae"), 0.005)
sq = [m2v("late_days", "strict", "time_ordered", m, "mae") for m in ["ridge", "random_forest", "gradient_boosting"]]
print(f"   squared-error late_days MAE range {min(sq):.2f}-{max(sq):.2f} (paper 3.7 to 4.0)")
if not (3.65 <= min(sq) and max(sq) <= 4.05): fails.append("late_days sq range")
r = b2v("late_days", "strict", "time_ordered", "gb_median", "train_median"); print(f"   late_days gb_median vs median excludes zero: {r.ci_excludes_zero}")

adv = pd.read_csv(f"{P2}/advisor_done_forecast_quality.csv")
def av(prot, f, col): return adv[(adv.protocol == prot) & (adv.forecaster == f)][col].item()
check("coverage time 84%", 84, 100 * av("time_ordered", "quantile_gradient_boosting", "coverage_80"), 0.5)
check("coverage lopo 85%", 85, 100 * av("lopo", "quantile_gradient_boosting", "coverage_80"), 0.5)
check("width 5.4", 5.4, av("time_ordered", "quantile_gradient_boosting", "mean_interval_width"), 0.05)
check("baseline width 8.4", 8.4, av("time_ordered", "scope_x_rate_quantiles", "mean_interval_width"), 0.05)
check("coverage 13+ 70%", 70, 100 * av("time_ordered", "quantile_gradient_boosting", "coverage_80__13+ issues"), 0.5)
late = pd.read_csv(f"{P2}/advisor_late_forecast_quality.csv")
def lv(prot, f, col): return late[(late.protocol == prot) & (late.forecaster == f)][col].item()
check("q90 exceeded time 10%", 10, 100 * (1 - lv("time_ordered", "gradient_boosting", "share_below_q90_forecast")), 0.5)
check("q90 exceeded lopo 11%", 11, 100 * (1 - lv("lopo", "gradient_boosting", "share_below_q90_forecast")), 0.6)
calb = pd.read_csv(f"{P2}/advisor_late_calibration.csv")
r = calb[(calb.protocol == "time_ordered") & (calb.bin == 3)].iloc[0]; check("top quartile pred .66", .66, r.mean_predicted, 0.005); check("top quartile obs .65", .65, r.observed_rate, 0.005)
r = calb[(calb.protocol == "lopo") & (calb.bin == 0)].iloc[0]; check("lopo bottom pred .16", .16, r.mean_predicted, 0.005); check("lopo bottom obs .30", .30, r.observed_rate, 0.005)
summ = json.load(open(f"{P2}/advisor_summary.json"))
o = summ["overload"]
check("carry quartile diff 3.2pp", 3.2, 100 * o["carry_ratio"]["heaviest_minus_lightest_quartile"], 0.05)
check("carry CI -2.3", -2.3, 100 * o["carry_ratio"]["ci95_cluster_bootstrap"][0], 0.05); check("carry CI 8.4", 8.4, 100 * o["carry_ratio"]["ci95_cluster_bootstrap"][1], 0.05)
check("late quartile diff 5.6pp", 5.6, 100 * o["label_delay"]["heaviest_minus_lightest_quartile"], 0.05)
check("late CI -5.2", -5.2, 100 * o["label_delay"]["ci95_cluster_bootstrap"][0], 0.05); check("late CI 16.4", 16.4, 100 * o["label_delay"]["ci95_cluster_bootstrap"][1], 0.05)
check("carry positive projects 2/8", 2, o["carry_ratio"]["projects_with_positive_spearman"], 0); check("late positive projects 5/8", 5, o["label_delay"]["projects_with_positive_spearman"], 0)
wi = pd.read_csv(f"{P2}/advisor_whatif_examples.csv"); l = wi[wi.project == "Lsstcorp Data management"]
check("LSST p_late at 8 issues .555", .555, l[l.planned_issues == 8].p_late_closure.item(), 0.005); check("LSST p_late at 38 .31", .31, l[l.planned_issues == 38].p_late_closure.item(), 0.005)

# ---------- phase 3 ----------
m3 = pd.read_csv(f"{P3}/checkpoint_model_results.csv")
def m3v(task, cp, prot, fs, model, col):
    r = m3[(m3.task == task) & (np.isclose(m3.checkpoint, cp)) & (m3.protocol == prot) & (m3.feature_set == fs) & (m3.model == model)]
    assert len(r) == 1, (task, cp, prot, fs, model); return r[col].item()
T2 = {("spillover", "time_ordered", "planning", "gradient_boosting"): [.830, .835, .827],
      ("spillover", "time_ordered", "progress", "gradient_boosting"): [.926, .957, .953],
      ("spillover", "time_ordered", "rule", "rule_remaining_share"): [.667, .767, .802],
      ("spillover", "lopo", "planning", "gradient_boosting"): [.801, .815, .804],
      ("spillover", "lopo", "progress", "gradient_boosting"): [.860, .897, .927],
      ("spillover", "lopo", "rule", "rule_remaining_share"): [.610, .715, .827],
      ("late_closure", "time_ordered", "planning", "gradient_boosting"): [.751, .751, .751, .717],
      ("late_closure", "time_ordered", "progress", "gradient_boosting"): [.754, .760, .749, .704],
      ("late_closure", "time_ordered", "rule", "rule_remaining_issues"): [.479, .483, .486, .484],
      ("late_closure", "lopo", "planning", "gradient_boosting"): [.645, .652, .647, .672],
      ("late_closure", "lopo", "progress", "gradient_boosting"): [.658, .667, .664, .657],
      ("late_closure", "lopo", "rule", "rule_remaining_issues"): [.518, .517, .521, .510]}
for (t, prot, fs, m), vals in T2.items():
    for cp, v in zip([0.25, 0.5, 0.75, 1.0], vals):
        check(f"T2 {t} {prot} {fs} {cp}", v, m3v(t, cp, prot, fs, m, "roc_auc"))
b3 = pd.read_csv(f"{P3}/checkpoint_model_bootstrap.csv")
def b3v(task, cp, prot, model, vs):
    r = b3[(b3.task == task) & (np.isclose(b3.checkpoint, cp)) & (b3.protocol == prot) & (b3.model == model) & (b3.versus == vs)]
    assert len(r) == 1; return r.iloc[0]
for cp, d in zip([0.25, 0.5, 0.75], [.10, .12, .13]):
    r = b3v("spillover", cp, "time_ordered", "gradient_boosting", "planning_same_algo"); check(f"spill gain time {cp} {d}", d, r.delta, 0.006); assert r.ci_excludes_zero
r = b3v("spillover", 0.25, "time_ordered", "gradient_boosting", "planning_same_algo"); check("CI .054", .054, r.ci_low); check("CI .163", .163, r.ci_high)
for cp, d in zip([0.25, 0.5, 0.75], [.06, .08, .12]):
    r = b3v("spillover", cp, "lopo", "gradient_boosting", "planning_same_algo"); check(f"spill gain lopo {cp} {d}", d, r.delta, 0.006); assert r.ci_excludes_zero
for m in ["logreg", "random_forest"]:
    r = b3v("spillover", 0.25, "lopo", m, "planning_same_algo"); print(f"   lopo 25% {m} vs planning excludes zero: {r.ci_excludes_zero} (paper: does not)")
    if r.ci_excludes_zero: fails.append(f"lopo25 {m}")
rules = [b3v("spillover", cp, p, "gradient_boosting", "rule_remaining_share").delta for cp in [0.25, 0.5, 0.75] for p in ["time_ordered", "lopo"]]
print(f"   GB over rule range {min(rules):.2f}-{max(rules):.2f} (paper 0.10 to 0.26)")
if not (0.095 <= min(rules) and max(rules) <= 0.265): fails.append("rule range")
for cp, v in zip([0.25, 0.5, 0.75], [.146, .112, .070]): check(f"carry progress gb_median {cp}", v, m3v("carry_share", cp, "time_ordered", "progress", "gb_median", "mae"))
for cp, v in zip([0.25, 0.5, 0.75], [.165, .115, .064]): check(f"carry rule {cp}", v, m3v("carry_share", cp, "time_ordered", "rule", "rule_remaining_share", "mae"))
for cp in [0.25, 0.5, 0.75]: check(f"carry planning ~.20 {cp}", .20, m3v("carry_share", cp, "time_ordered", "planning", "gb_median", "mae"), 0.005)
# late closure: all GB intervals include zero; RF 25% excludes
for cp in [0.25, 0.5, 0.75, 1.0]:
    r = b3v("late_closure", cp, "time_ordered", "gradient_boosting", "planning_same_algo"); assert not r.ci_excludes_zero, cp
r = b3v("late_closure", 0.25, "time_ordered", "random_forest", "planning_same_algo"); check("RF 25% +.02", .02, r.delta, 0.005); check("CI .002", .002, r.ci_low); check("CI .049", .049, r.ci_high)
check("100% positive rate 51%", 51, 100 * m3v("late_closure", 1.0, "time_ordered", "progress", "gradient_boosting", "positive_rate"), 0.5)
check("100% always-late F1 .68", .68, m3v("late_closure", 1.0, "time_ordered", "baseline", "always_positive", "f1"), 0.005)
check("100% GB F1 .68", .68, m3v("late_closure", 1.0, "time_ordered", "progress", "gradient_boosting", "f1"), 0.005)
i3 = pd.read_csv(f"{P3}/explanation_group_importance.csv")
def s3(model, g): return 100 * i3[(i3.model == model) & (i3.group == g)].share_of_total.item()
for g, v in [("progress so far", 34), ("time pressure", 23), ("scope", 16), ("recent history", 15)]: check(f"p3 spill {g} {v}", v, s3("spillover", g), 0.5)
for g, v in [("past lateness", 37), ("project", 17), ("progress so far", 6)]: check(f"p3 late {g} {v}", v, s3("late_closure", g), 0.5)
cf = pd.read_csv(f"{P3}/phase3_checkpoint_frame.csv", usecols=["sprint_id", "checkpoint", "open_at_checkpoint"])
for cp, v in [(0.25, 98.6), (0.75, 96), (1.0, 65)]:
    check(f"open at {cp} {v}%", v, 100 * cf[np.isclose(cf.checkpoint, cp)].open_at_checkpoint.mean(), 0.5)
check("sprint-checkpoints 10408", 10408, len(cf), 0)

# ---------- leakage ----------
lk = json.load(open(f"{P1}/leakage_case_studies.json"))
d, s_ = lk["label_delay"], lk["label_spillover"]
check("leak late post safe .463", .463, d["case_study_a_post_outcome_feature"]["leakage_safe (this project's protocol)"]["f1"])
check("leak late post leaky .473", .473, d["case_study_a_post_outcome_feature"]["leaky (+ this sprint's own completion_ratio)"]["f1"])
check("leak late over correct .457", .457, d["case_study_b_oversample_order"]["correct order (oversample train only, after split)"]["f1"])
check("leak late over leaky .646", .646, d["case_study_b_oversample_order"]["incorrect order (oversample whole pool, then split)"]["f1"])
check("leak spill post safe .756", .756, s_["case_study_a_post_outcome_feature"]["leakage_safe (this project's protocol)"]["f1"])
check("leak spill post leaky .991", .991, s_["case_study_a_post_outcome_feature"]["leaky (+ this sprint's own completion_ratio)"]["f1"])
check("leak spill post AUC .834->1.000", .834, s_["case_study_a_post_outcome_feature"]["leakage_safe (this project's protocol)"]["roc_auc"])
check("leak spill post AUC 1.000", 1.000, s_["case_study_a_post_outcome_feature"]["leaky (+ this sprint's own completion_ratio)"]["roc_auc"])
check("leak spill over .972/.972", .972, s_["case_study_b_oversample_order"]["correct order (oversample train only, after split)"]["f1"])
check("leak spill over leaky .972", .972, s_["case_study_b_oversample_order"]["incorrect order (oversample whole pool, then split)"]["f1"])

env = json.load(open(f"{P1}/environment_versions.json")); print("   env:", env)
print("\nFAILS:", fails if fails else "none")

# ---------- headline table (Table II, 31 Oct revision) ----------
mb = pd.read_csv(f"{P2}/model_bootstrap.csv")
def mbv(task, prot, model, base):
    r = mb[(mb.task == task) & (mb.feature_set == "strict") & (mb.protocol == prot) & (mb.model == model) & (mb.baseline == base) & (mb.metric.str.startswith("delta"))]
    assert len(r) == 1; return r.iloc[0]
r = mbv("late_closure", "lopo", "random_forest", "scope_only_logreg"); check("T2 row3 delta .110", .110, r.delta); check("T2 row3 lo .018", .018, r.ci_low); check("T2 row3 hi .215", .215, r.ci_high)
check("T2 row3 RF lopo .642", .642, m2v("late_closure", "strict", "lopo", "random_forest", "roc_auc")); check("T2 row3 scope lopo .532", .532, m2v("late_closure", "strict", "lopo", "scope_only_logreg", "roc_auc"))
check("T2 row4 legacy RF lopo .600", .600, m2v("late_closure", "legacy", "lopo", "random_forest", "roc_auc"))
r = mbv("spillover", "time_ordered", "gradient_boosting", "scope_only_logreg"); check("T2 row5 GB-scope -.073", -.073, r.delta); check("T2 row5 lo -.117", -.117, r.ci_low); check("T2 row5 hi -.007", -.007, r.ci_high)
r = mbv("spillover", "lopo", "logreg", "scope_only_logreg"); check("T2 row6 LR-scope +.017", .017, r.delta); check("T2 row6 lo -.007", -.007, r.ci_low); check("T2 row6 hi .051", .051, r.ci_high)
check("T2 row6 scope lopo .837", .837, m2v("spillover", "strict", "lopo", "scope_only_logreg", "roc_auc")); check("T2 row6 LR lopo .854", .854, m2v("spillover", "strict", "lopo", "logreg", "roc_auc"))
check("T2 row7 legacy RF lopo .815", .815, m2v("spillover", "legacy", "lopo", "random_forest", "roc_auc"))
cu = pd.read_csv(f"{P1}/cluster_uncertainty.csv")
r = cu[(cu.label == "label_spillover") & (cu.model == "random_forest")].iloc[0]; check("T2 row8 nonempty lo .774", .774, r.roc_auc_cluster_ci_low); check("T2 row8 nonempty hi .911", .911, r.roc_auc_cluster_ci_high)
sb = pd.read_csv(f"{P1}/scope_only_baselines.csv"); r = sb[(sb.label == "label_spillover") & (sb.population == "full") & (sb.model == "random_forest")].iloc[0]; check("T2 row8 full n 786", 786, r.n, 0); check("T2 row8 full pos 453", 453, r.positives, 0)
ps = pd.read_csv(f"{P1}/commitment_provenance_sensitivity.csv"); r = ps[(ps.subset_rule.str.contains("story_history")) & (ps.model == "scope_logreg")].iloc[0]
check("T2 row9 n 110", 110, r.n, 0); check("T2 row9 pos 94", 94, r.positives, 0); check("T2 row9 auc .772", .772, r.roc_auc)
import math
A, npos, nneg = r.roc_auc, 94, 16; Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A); se = math.sqrt((A * (1 - A) + (npos - 1) * (Q1 - A * A) + (nneg - 1) * (Q2 - A * A)) / (npos * nneg))
check("row9 Hanley-McNeil SE ~.05", .05, se, .01); check("row9 lo ~.67", .67, A - 1.96 * se, .01); check("row9 hi ~.88", .88, A + 1.96 * se, .01)
r = ps[(ps.subset_rule.str.contains("story_history") == False) & (ps.model == "scope_logreg")].iloc[0]; check("subset1 pos 335", 335, r.positives, 0)
m3b = pd.read_csv(f"{P3}/checkpoint_model_bootstrap.csv")
def m3b_(task, cp, prot):
    r = m3b[(m3b.task == task) & np.isclose(m3b.checkpoint, cp) & (m3b.protocol == prot) & (m3b.model == "gradient_boosting") & (m3b.versus == "planning_same_algo") & (m3b.metric == "delta_auc")]; assert len(r) == 1; return r.iloc[0]
r = m3b_("spillover", .25, "time_ordered"); check("T2 row10 +.097", .097, r.delta); r = m3b_("spillover", .5, "time_ordered"); check("T2 row11 +.123", .123, r.delta); check("row11 lo .068", .068, r.ci_low); check("row11 hi .188", .188, r.ci_high)
r = m3b_("late_closure", .5, "time_ordered"); check("T2 row12 +.008", .008, r.delta); check("row12 lo -.008", -.008, r.ci_low); check("row12 hi .031", .031, r.ci_high)
for task, cp, n, pos in [("spillover", .25, 507, 478), ("spillover", .5, 501, 472), ("late_closure", .5, 775, 246)]:
    check(f"T2 {task} {cp} n {n}", n, m3v(task, cp, "time_ordered", "progress", "gradient_boosting", "n"), 0)
    check(f"T2 {task} {cp} pos {pos}", pos, round(m3v(task, cp, "time_ordered", "progress", "gradient_boosting", "n") * m3v(task, cp, "time_ordered", "progress", "gradient_boosting", "positive_rate")), 0)
r = mb[(mb.task == "carry_share") & (mb.feature_set == "strict") & (mb.protocol == "time_ordered") & (mb.model == "gb_median") & (mb.baseline == "train_median")].iloc[0]; check("T2 row14 -.031", -.031, r.delta)
r = mb[(mb.task == "late_days") & (mb.feature_set == "strict") & (mb.protocol == "time_ordered") & (mb.model == "gb_median") & (mb.baseline == "train_median")].iloc[0]; check("T2 row15 -.18", -.18, r.delta, .005); check("row15 lo -.34", -.34, r.ci_low, .005); check("row15 hi .03", .03, r.ci_high, .005)
check("late positives 1006", 1006, frame.label_delay.sum(), 0); check("spill positives 1575", 1575, frame[frame.committed_issue_count > 0].label_spillover.sum(), 0)
# ---------- external check on MariaDB (Section IV-D, Table II row 16) ----------
E = f"{R}/external-validation/results"
ev = pd.read_csv(f"{E}/external_validation_results.csv")
def evr(arm, task, model, comp=None):
    r = ev[(ev.arm == arm) & (ev.task == task) & (ev.model == model)]
    if comp is not None: r = r[r.comparator == comp]
    assert len(r) == 1, (arm, task, model, comp); return r.iloc[0]
T = "Transfer TAWOS->MariaDB"
r = evr(T, "Late closure", "GB, strict, no SP")
check("MDB row16 n 391", 391, r.n, 0); check("MDB row16 pos 106", 106, r.pos, 0)
check("MDB row16 GB .710", .710, r.auc); check("MDB GB lo .637", .637, r.auc_lo); check("MDB GB hi .773", .773, r.auc_hi)
check("MDB row16 diff .134", .134, r["diff"]); check("MDB diff lo .034", .034, r.diff_lo); check("MDB diff hi .223", .223, r.diff_hi)
check("MDB scope-only .576", .576, r.comp_auc)
check("MDB RF .716", .716, evr(T, "Late closure", "RF, strict, no SP").auc)
check("TAWOS no-SP GB late .751", .751, evr("TAWOS, no SP (reference)", "Late closure", "GB, strict, no SP").auc)
check("MDB within RF .750", .750, evr("MariaDB", "Late closure", "RF, strict").auc)
check("MDB project rate .745", .745, evr("MariaDB", "Late closure", "GB, strict", "Project rate").comp_auc)
r = evr("MariaDB", "Late closure, 50%", "GB, strict + progress")
check("MDB 50% gain +.019", .019, r["diff"]); check("MDB 50% lo -.050", -.050, r.diff_lo); check("MDB 50% hi .089", .089, r.diff_hi)
desc = json.load(open(f"{E}/cohort_description.json")); m = [d for d in desc if d["source"] == "MariaDB"][0]
check("MDB sprints 391", 391, m["modeled_sprints"], 0)
check("MDB spill 99%", 99, 100 * m["spillover_rate_nonempty"], 0.5)
mf = pd.read_csv(f"{E}/mariadb_modeling_frame.csv"); lr = mf.groupby("project").label_delay.mean()
check("MDB ColumnStore late 54%", 54, 100 * lr["MariaDB ColumnStore"], 0.5); check("MDB MaxScale late 15%", 15, 100 * lr["MariaDB MaxScale"], 0.5)
check("MDB first year 2016", 2016, int(m["first_start"][:4]), 0); check("MDB last year 2026", 2026, int(m["last_start"][:4]), 0)
# ---------- Phase 4: repeated splits and ten outside projects (Table II rows 17-18, Sections IV-B, IV-D, V) ----------
F4 = f"{R}/phase-4-fixes/results"
e1 = pd.read_csv(f"{F4}/e1_summary.csv")
def e1v(task, scheme, model, col):
    r = e1[(e1.task == task) & (e1.scheme.str.startswith(scheme)) & (e1.model == model)]
    assert len(r) == 1, (task, scheme, model); return r[col].item()
GB = "Gradient boosting"
check("P4 time x5 late GB .696", .696, e1v("late_closure", "C.", GB, "auc_mean")); check("P4 time x5 late SD .054", .054, e1v("late_closure", "C.", GB, "auc_std"))
check("P4 shuffled 70/30 late GB .765", .765, e1v("late_closure", "A.", GB, "auc_mean")); check("P4 shuffled 70/30 late SD .017", .017, e1v("late_closure", "A.", GB, "auc_std"))
check("P4 shuffling adds .07", .07, e1v("late_closure", "A.", GB, "auc_mean") - e1v("late_closure", "C.", GB, "auc_mean"), .005)
check("P4 time x5 spill GB .806", .806, e1v("spillover", "C.", GB, "auc_mean")); check("P4 time x5 spill SD .028", .028, e1v("spillover", "C.", GB, "auc_std"))
check("P4 shuffled 70/30 spill GB .813", .813, e1v("spillover", "A.", GB, "auc_mean"))
rep = pd.read_csv(f"{F4}/e1_repeats.csv")
tl = rep[(rep.task == "late_closure") & rep.scheme.str.startswith("C.") & (rep.model == GB)].sort_values("repeat")
check("P4 time repeat 1 (40% train) .62", .62, tl.auc.iloc[0], .005); check("P4 time repeat 4 (70% train) .75", .75, tl.auc.iloc[3], .005)
check("P4 time test sprints ~520", 520, tl.n.mean(), 5); check("P4 time test late ~177", 177, tl.n_pos.mean(), 1)
e3 = pd.read_csv(f"{F4}/e3_external_per_project.csv")
lg = e3[(e3.task == "Late closure") & (e3.model == GB) & e3.interpretable].set_index("project")
ls = e3[(e3.task == "Late closure") & (e3.model == "Scope-only LR (baseline)") & e3.interpretable].set_index("project")
check("P4 outside late testable 8", 8, len(lg), 0); check("P4 outside late GB mean .620", .620, lg.auc.mean())
check("P4 outside late GB SD .090", .090, lg.auc.std(), .001)
check("P4 above chance in 7", 7, (lg.auc > .5).sum(), 0); check("P4 above scope-only in 8", 8, (lg.auc > ls.loc[lg.index].auc).sum(), 0)
check("P4 scope-only mean .362", .362, ls.auc.mean())
check("P4 sensitivity min .11", .11, lg.sensitivity.min(), .005); check("P4 sensitivity max .86", .86, lg.sensitivity.max(), .005)
sg = e3[(e3.task == "Spillover") & (e3.model == GB) & e3.interpretable]
ss = e3[(e3.task == "Spillover") & (e3.model == "Scope-only LR (baseline)") & e3.interpretable]
check("P4 outside spill testable 4", 4, len(sg), 0); check("P4 outside spill GB .734", .734, sg.auc.mean()); check("P4 outside spill scope .742", .742, ss.auc.mean())
pj = pd.read_csv(f"{F4}/e3_external_projects.csv")
check("P4 MongoDB sprints 526", 526, pj[pj.instance == "mongodb"].sprints.sum(), 0)
lt = pj[pj.project.isin(lg.index)].late_rate
check("P4 late rate min 7%", 7, 100 * lt.min(), 0.5); check("P4 late rate max 59%", 59, 100 * lt.max(), 0.5)
# ---------- review pass (Nov 2026): figures added or made explicit in the text ----------
r = mb[(mb.task == "spillover") & (mb.feature_set == "strict") & (mb.protocol == "time_ordered") & (mb.model == "random_forest") & (mb.baseline == "scope_only_logreg")].iloc[0]
check("RF vs scope-only spill delta -.045", -.045, r.delta); check("RF spill CI lo -.088", -.088, r.ci_low); check("RF spill CI hi .010", .010, r.ci_high)
ckf = pd.read_csv(f"{P3}/phase3_checkpoint_frame.csv")
op = ckf.groupby("checkpoint").open_at_checkpoint.mean()
check("open at 25% 98.6%", 98.6, 100 * op[0.25], 0.05); check("open at 75% 96.2%", 96.2, 100 * op[0.75], 0.05); check("open at 100% 65.5%", 65.5, 100 * op[1.0], 0.05)
A, n1, n0 = .772, 94, 16
Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
se = np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))
check("Hanley-McNeil SE .054", .054, se, .0006); check("HM range lo .67", .67, A - 1.96 * se, .005); check("HM range hi .88", .88, A + 1.96 * se, .005)
check("Confluence Cloud nonempty 197", 197, flow.loc[flow.project == "Atlassian Confluence Cloud", "nonempty"].item(), 0)
cr = pd.read_csv(f"{P3}/checkpoint_model_results.csv")
q = cr[(cr.task == "late_closure") & (cr.protocol == "time_ordered") & (cr.feature_set == "planning") & (cr.model == "gradient_boosting")].set_index("checkpoint").roc_auc
check("late planning-only 25% .751", .751, q[0.25]); check("late planning-only 75% .751", .751, q[0.75])
assert abs(q[0.25] - q[0.75]) < 0.001 and abs(q[0.5] - q[0.75]) < 0.001, "planning-only late AUCs differ beyond the fourth decimal"
ins = pd.read_csv(f"{P3}/checkpoint_model_bootstrap.csv")
for m in ("logreg", "random_forest"):
    r = ins[(ins.task == "spillover") & (ins.checkpoint == 0.25) & (ins.protocol == "lopo") & (ins.model == m) & (ins.versus == "planning_same_algo")].iloc[0]
    check(f"LOPO 25% {m} gain includes zero", 1, int(r.ci_low < 0 < r.ci_high), 0)
# ---------- 8 Nov 2026 review: count baselines, timing confound, label defect, outside CIs ----------
# Recomputed here from the frames where possible; the timing rerun is read from reviewer_checks/.
from sklearn.metrics import roc_auc_score as _auc
RC = f"{R}/ccece-2027-paper/reviewer_checks"
ne = frame[frame.nonempty]
te_ne = []
for p, g in ne.groupby("project"):
    g = g.sort_values(["start_date", "sprint_id"]); te_ne.append(g.iloc[int(len(g) * 0.7):])
te_ne = pd.concat(te_ne)
check("spill test negatives 29", 29, (te_ne.label_spillover == 0).sum(), 0)
check("raw committed count spill AUC .881", .881, _auc(te_ne.label_spillover, te_ne.committed_issue_count))
for c, (tv, lv) in {0.25: (.925, .876), 0.5: (.945, .902), 0.75: (.954, .938)}.items():
    b = ckf[(ckf.checkpoint == c) & ckf.nonempty & ckf.open_at_checkpoint]
    t = b[b.split_nonempty == "test"]
    check(f"open-count rule time {c}", tv, _auc(t.label_spillover, t.remaining_n))
    check(f"open-count rule LOPO pool {c}", lv, _auc(b.label_spillover, b.remaining_n))
end = ckf[ckf.nonempty & (ckf.checkpoint == 1.0)]
zd = end.groupby("project").done_share.apply(lambda v: (v == 0).mean())
check("Confluence Cloud zero done 74%", 74, 100 * zd["Atlassian Confluence Cloud"], 0.5)
check("Confluence Server zero done 48%", 48, 100 * zd["Atlassian Confluence Server"], 0.5)
rest = zd.drop(["Atlassian Confluence Cloud", "Atlassian Confluence Server"])
check("other projects zero done min 11%", 11, 100 * rest.min(), 0.5); check("other projects zero done max 31%", 31, 100 * rest.max(), 0.5)
pc = pd.read_csv(f"{RC}/planning_checks.csv").set_index("check").value
check("spill scope-only without Confluence Cloud .908", .908, pc["spill_time_auc_scope_only_lr_without_confluence_cloud"])
check("zero scope filled within 1h 33%", 33, 100 * pc["zero_scope_first_issue_added_within_1h_after_start"], 0.5)
check("zero scope filled within 24h 42%", 42, 100 * pc["zero_scope_first_issue_added_within_24h_after_start"], 0.5)
check("outside late GB CI above .5 in 2", 2, (lg.auc_lo > .5).sum(), 0)
check("outside late sprints min 21", 21, lg.n.min(), 0); check("outside late sprints max 269", 269, lg.n.max(), 0)
tc = pd.read_csv(f"{RC}/timing_confound.csv").set_index(["feature_set", "model"]).lopo_pooled_auc_late
check("confound LOPO strict-timed LR .519", .519, tc[("strict timing, same content", "LR")])
check("confound LOPO strict-timed RF .563", .563, tc[("strict timing, same content", "RF")])
check("confound LOPO legacy+lateness LR .645", .645, tc[("legacy + past lateness", "LR")])
check("confound LOPO legacy+lateness RF .682", .682, tc[("legacy + past lateness", "RF")])
check("confound rerun full strict LR within .007 of .629", .629, tc[("full strict", "LR")], .0075)
check("confound rerun full strict RF within .007 of .642", .642, tc[("full strict", "RF")], .0075)
tt = pd.read_csv(f"{RC}/timing_confound_time.csv").set_index(["feature_set", "model"]).time_split_auc_late
check("confound time strict-timed LR .661", .661, tt[("strict timing, same content", "LR")])
check("confound time legacy GB .732", .732, tt[("legacy (scope + old history)", "GB")])
check("confound time strict-timed GB .731", .731, tt[("strict timing, same content", "GB")])
print("FAILS:", fails)
