"""Held-out neural applicability calibration with conservative abstention.

This is a host evidence gate, not a second cognitive model or decoder.
"""
from __future__ import annotations
from hashlib import sha256
from math import sqrt
import torch
from .neural_inferencing import NeuralSkillHead


def _wilson_upper(errors: int, count: int, z: float = 1.96) -> float:
    if count <= 0:
        return 1.0
    p=errors/count
    return min(1.0,(p+z*z/(2*count)+z*sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count))


def calibrate_and_activate(head: NeuralSkillHead, heldout: dict, *,
                           independently_labelled: bool,
                           minimum_recall: float=.8,
                           maximum_false_application: float=.02,
                           minimum_each_label: int=40) -> dict:
    """Assess score threshold on heldout labels; never certify train loss alone.

    Callers must establish that labels are independent and unseen in training.
    No learning runs during this function; the base model is never modified.
    """
    head.activation_certified=False
    head.calibration_report=None
    if not independently_labelled:
        return {"approved":False,"reason":"independent_holdout_labels_not_established"}
    if set(heldout)!={"current","prior","metadata","label"}:
        raise ValueError("Incomplete heldout fields")
    if not (0<minimum_recall<=1 and 0<=maximum_false_application<1):
        raise ValueError("Invalid reliability bounds")
    n=len(heldout["label"])
    if any(len(heldout[k])!=n for k in ("current","prior","metadata")):
        raise ValueError("Inconsistent heldout row count")
    labels=heldout["label"].detach().cpu()
    if not bool(((labels==0)|(labels==1)).all()):
        raise ValueError("Expected binary labels")
    pos=int((labels==1).sum())
    neg=int((labels==0).sum())
    if min(pos,neg)<minimum_each_label:
        return {"approved":False,"reason":"insufficient_independent_positives_or_negatives",
                "positive_count":pos,"negative_count":neg}
    if not int(head.trained_examples.item()):
        return {"approved":False,"reason":"neural_parameters_untrained"}
    seen={}
    collisions=0
    for i in range(n):
        payload=b"".join(heldout[k][i].detach().contiguous().cpu().numpy().tobytes()
                         for k in ("current","prior","metadata"))
        key=sha256(payload).hexdigest()
        label=int(labels[i].item())
        if key in seen and seen[key]!=label:
            collisions+=1
        seen[key]=label
    if collisions:
        report={"approved":False,"reason":"indistinguishable_opposing_outcomes",
                "feature_collisions":collisions,"positive_count":pos,"negative_count":neg}
        head.calibration_report=report
        return report
    head.eval()
    device=next(head.parameters()).device
    with torch.no_grad():
        scores=head.score(*(heldout[k].to(device)
                            for k in ("current","prior","metadata"))).detach().cpu()
    thresholds=sorted(set([head.config.threshold,1.0]+[
                max(head.config.threshold,float(x.item())) for x in scores]))
    best=None
    for threshold in thresholds:
        prediction=scores>=threshold
        recall=float(prediction[labels==1].float().mean())
        false=int(prediction[labels==0].sum())
        upper=_wilson_upper(false,neg)
        if recall>=minimum_recall and upper<=maximum_false_application:
            best=(threshold,recall,false/neg,upper)
            break
    report={"approved":bool(best),
            "reason":"calibration_passed" if best else
                "no_threshold_meets_recall_and_false_application_bounds",
            "positive_count":pos,"negative_count":neg,"feature_collisions":0,
            "confidence_interval":"95% Wilson upper bound",
            "brier":round(float(((scores-labels)**2).mean()),8),
            "threshold":round(best[0],8) if best else None,
            "useful_recall":round(best[1],8) if best else None,
            "observed_false_application":round(best[2],8) if best else None,
            "false_application_upper_95":round(best[3],8) if best else None}
    if best:
        head.activation_certified=True
        head.activation_threshold=float(best[0])
    head.calibration_report=report
    return report
