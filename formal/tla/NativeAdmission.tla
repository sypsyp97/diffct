-------------------------- MODULE NativeAdmission --------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Ranks, Mutation
VARIABLES pitch, request, fast, hybrid, parallel3d, nvidia, dtype, selected, accepted, warned, pc
vars == <<pitch, request, fast, hybrid, parallel3d, nvidia, dtype, selected, accepted, warned, pc>>
\* Cases: scalar, lower edge, upper edge, ratio edge, below, above, excess ratio,
\* ordinary anisotropy. Real thresholds are observed separately by the harness.
InRange(k) == k \notin {4, 5, 6}
Compatible == fast /\ ~hybrid /\ ~parallel3d
Native(r) == Compatible /\ nvidia[r] /\ (InRange(pitch[r]) \/ Mutation = "range")
Choice(r) == IF request = "torch" THEN "torch"
             ELSE IF Native(r) THEN "numba"
             ELSE IF request = "numba" THEN "local-error" ELSE "torch"
Same(f) == Cardinality({f[r] : r \in Ranks}) = 1
Init == /\ pitch \in [Ranks -> 0..7] /\ request \in {"auto", "numba", "torch"}
        /\ fast \in BOOLEAN /\ hybrid \in BOOLEAN /\ parallel3d \in BOOLEAN
        /\ nvidia \in [Ranks -> BOOLEAN] /\ dtype \in [Ranks -> {32, 64}]
        /\ selected = [r \in Ranks |-> "none"] /\ warned = [r \in Ranks |-> FALSE]
        /\ accepted = FALSE /\ pc = 0
Select == /\ pc = 0 /\ pc' = 1 /\ selected' = [r \in Ranks |-> Choice(r)]
          /\ UNCHANGED <<pitch, request, fast, hybrid, parallel3d, nvidia, dtype, accepted, warned>>
Agree == LET admits == /\ \A r \in Ranks : selected[r] # "local-error"
                       /\ (Same(selected) \/ Mutation = "selection")
                       /\ (Same(pitch) \/ Mutation = "spacing")
                       /\ (Same(dtype) \/ Mutation = "dtype")
         IN /\ pc = 1 /\ pc' = 2 /\ accepted' = admits
            /\ warned' = [r \in Ranks |-> admits /\ request = "auto" /\ nvidia[r]
                                          /\ selected[r] = "torch" /\ Mutation # "warning"]
            /\ UNCHANGED <<pitch, request, fast, hybrid, parallel3d, nvidia, dtype, selected>>
Next == Select \/ Agree \/ (pc = 2 /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
NativeRange == pc > 0 => \A r \in Ranks : selected[r] = "numba" =>
                           Compatible /\ nvidia[r] /\ InRange(pitch[r])
SelectionAgreement == accepted => Same(selected)
SpacingAgreement == accepted => Same(pitch)
TrajectoryAgreement == accepted => Same(dtype)
FallbackWarning == pc = 2 /\ accepted /\ request = "auto" =>
                     \A r \in Ranks : nvidia[r] /\ selected[r] = "torch" => warned[r]
Termination == <>(pc = 2)
=============================================================================
